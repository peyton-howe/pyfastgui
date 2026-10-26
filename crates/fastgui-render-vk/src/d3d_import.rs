//! Import a shared Direct3D 11 texture and sample it as a viewport image (Windows).
//!
//! The caller dups an NT handle (`IDXGIResource1::CreateSharedHandle` on a texture created with
//! `D3D11_RESOURCE_MISC_SHARED_NTHANDLE`) into [`fastgui_core::D3d11Import`]. `vkAllocateMemory`
//! with `VkImportMemoryWin32HandleInfoKHR` references that handle without taking it, so the
//! `D3d11Import` closes its duplicate after the import. The image is optimally tiled 8-bit RGBA or
//! BGRA, and the driver asks for a dedicated allocation. The D3D device that owns the texture must
//! be on the same adapter as the Vulkan device.
//!
//! Sync is an optional shared fence (`ID3D11Fence` / `ID3D12Fence`, imported as a timeline
//! semaphore of handle type `D3D12_FENCE`). The submission waits for `wait_value` on the render
//! thread, bounded by [`FENCE_WAIT`], before the texture is sampled; if the fence is late the
//! submission is dropped and the previous frame stays, so a stalled producer cannot hang the window.
//! `signal_value` is signaled when the frame that first draws the submission has finished on the
//! GPU. Submissions that fail or are replaced before being drawn never signal their `signal_value`.
//! Producers waiting on `signal_value` must check `d3d11_status` and use a timeout.
//! Without a fence the producer must have finished (flushed and waited) before the call.

use std::collections::VecDeque;

use ash::{khr, vk, Device, Instance};
use fastgui_core::{D3d11Import, SharedFormat};

use crate::dmabuf::ImportedImage;
use crate::error::VkRendererError as Error;
use crate::texture::find_memory_type_index;

const HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE;
const FENCE_HANDLE_TYPE: vk::ExternalSemaphoreHandleTypeFlags = vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE;

/// How long a submission waits for its fence on the render thread.
pub(crate) const FENCE_WAIT: std::time::Duration = std::time::Duration::from_millis(100);

pub(crate) fn vk_format(format: SharedFormat) -> vk::Format {
    match format {
        SharedFormat::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        SharedFormat::Bgra8 => vk::Format::B8G8R8A8_UNORM,
    }
}

/// What this device can import, and the loaders to do it. Built once at device creation.
#[derive(Clone)]
pub(crate) struct D3dSupport {
    pub memory: khr::external_memory_win32::Device,
    /// `Some` when the device can import a D3D fence as a timeline semaphore.
    pub semaphore: Option<khr::external_semaphore_win32::Device>,
    pub rgba8: bool,
    pub bgra8: bool,
    /// Opens a shared handle on a D3D11 device on this adapter to read the texture's real size and
    /// format. Vulkan accepts any size for an imported texture, so without this a wrong `width` or
    /// `height` would make the GPU read past the allocation.
    #[cfg(windows)]
    pub probe: Result<crate::d3d_probe::D3dProbe, String>,
}

impl D3dSupport {
    pub(crate) fn supports(&self, format: SharedFormat) -> bool {
        match format {
            SharedFormat::Rgba8 => self.rgba8,
            SharedFormat::Bgra8 => self.bgra8,
        }
    }
}

/// `true` when this device can import a sampled optimal image of `format` from a D3D11 NT handle.
pub(crate) unsafe fn texture_importable(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    format: SharedFormat,
) -> bool {
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(HANDLE_TYPE);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(vk_format(format))
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .push_next(&mut external);
    let mut external_props = vk::ExternalImageFormatProperties::default();
    let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
    instance
        .get_physical_device_image_format_properties2(pdevice, &info, &mut props)
        .is_ok()
        && external_props
            .external_memory_properties
            .external_memory_features
            .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
}

/// `true` when this device can import a D3D fence as a timeline semaphore.
pub(crate) unsafe fn fence_importable(instance: &Instance, pdevice: vk::PhysicalDevice) -> bool {
    let mut timeline = vk::SemaphoreTypeCreateInfo::default().semaphore_type(vk::SemaphoreType::TIMELINE);
    let info = vk::PhysicalDeviceExternalSemaphoreInfo::default()
        .handle_type(FENCE_HANDLE_TYPE)
        .push_next(&mut timeline);
    let mut props = vk::ExternalSemaphoreProperties::default();
    instance.get_physical_device_external_semaphore_properties(pdevice, &info, &mut props);
    props
        .external_semaphore_features
        .contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE)
}

/// Imports kept per viewport besides the one on screen. A producer that cycles a few textures
/// (double or triple buffering) finds each one here instead of re-importing it every frame.
pub(crate) const PARKED_IMPORTS: usize = 3;

/// Inactive imports for one viewport, most recently shown last. Generic so the eviction order
/// can be tested without a GPU.
pub(crate) struct ImportRing<T>(VecDeque<T>);

impl<T> Default for ImportRing<T> {
    fn default() -> Self {
        Self(VecDeque::new())
    }
}

impl<T> ImportRing<T> {
    pub(crate) fn take(&mut self, matches: impl Fn(&T) -> bool) -> Option<T> {
        let index = self.0.iter().position(matches)?;
        self.0.remove(index)
    }

    pub(crate) fn find_mut(&mut self, matches: impl Fn(&T) -> bool) -> Option<&mut T> {
        self.0.iter_mut().find(|item| matches(item))
    }

    /// Park `item`; returns the oldest entries pushed out past [`PARKED_IMPORTS`].
    pub(crate) fn park(&mut self, item: T) -> Vec<T> {
        self.0.push_back(item);
        let mut evicted = Vec::new();
        while self.0.len() > PARKED_IMPORTS {
            evicted.extend(self.0.pop_front());
        }
        evicted
    }

    pub(crate) fn drain(&mut self) -> impl Iterator<Item = T> + '_ {
        self.0.drain(..)
    }
}

#[cfg(windows)]
mod sys {
    #[link(name = "kernel32")]
    extern "system" {
        pub fn CloseHandle(handle: isize) -> i32;
        pub fn CompareObjectHandles(first: isize, second: isize) -> i32;
    }
}

pub(crate) fn close_handle(handle: isize) {
    #[cfg(windows)]
    unsafe {
        sys::CloseHandle(handle);
    }
    #[cfg(not(windows))]
    let _ = handle;
}

/// `true` when both handles refer to the same kernel object, which is how a duplicated handle
/// to the texture (or fence) we already imported is recognized.
pub(crate) fn same_object(first: isize, second: isize) -> bool {
    #[cfg(windows)]
    {
        first == second || unsafe { sys::CompareObjectHandles(first, second) != 0 }
    }
    #[cfg(not(windows))]
    {
        let _ = (first, second);
        false
    }
}

/// A D3D fence imported as a timeline semaphore, with the handle kept so a later submit of the
/// same fence is recognized.
pub(crate) struct ImportedFence {
    pub(crate) semaphore: vk::Semaphore,
    handle: isize,
}

impl ImportedFence {
    pub(crate) fn is_import_of(&self, handle: isize) -> bool {
        same_object(self.handle, handle)
    }

    pub(crate) unsafe fn destroy(self, device: &Device) {
        device.destroy_semaphore(self.semaphore, None);
        close_handle(self.handle);
    }
}

/// Import `fence` (an NT handle to a shared `ID3D11Fence`/`ID3D12Fence`). On success the
/// imported fence keeps the handle (taken from `fence`); on failure `fence` still owns it.
pub(crate) unsafe fn import_fence(
    device: &Device,
    loader: &khr::external_semaphore_win32::Device,
    fence: &mut fastgui_core::D3d11Fence,
) -> Result<ImportedFence, Error> {
    let mut timeline = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let semaphore = device.create_semaphore(&vk::SemaphoreCreateInfo::default().push_next(&mut timeline), None)?;
    let info = vk::ImportSemaphoreWin32HandleInfoKHR::default()
        .semaphore(semaphore)
        .handle_type(FENCE_HANDLE_TYPE)
        .handle(fence.handle());
    if let Err(err) = loader.import_semaphore_win32_handle(&info) {
        device.destroy_semaphore(semaphore, None);
        return Err(err.into());
    }
    Ok(ImportedFence { semaphore, handle: fence.take_handle() })
}

/// Wait until `semaphore` reaches `value`, for at most [`FENCE_WAIT`]. `Ok(false)` is a timeout.
pub(crate) unsafe fn wait_fence(device: &Device, semaphore: vk::Semaphore, value: u64) -> Result<bool, Error> {
    let semaphores = [semaphore];
    let values = [value];
    let info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
    match device.wait_semaphores(&info, FENCE_WAIT.as_nanos() as u64) {
        Ok(()) => Ok(true),
        Err(vk::Result::TIMEOUT) => Ok(false),
        Err(err) => Err(err.into()),
    }
}

impl ImportedImage {
    /// `true` when `texture` is the shared texture this image already imported.
    pub(crate) fn is_import_of(&self, texture: &D3d11Import) -> bool {
        self.width == texture.width
            && self.height == texture.height
            && texture.format.is_none_or(|format| self.format == vk_format(format))
            && self.shared_handle.is_some_and(|kept| same_object(kept, texture.handle()))
    }
}

pub(crate) unsafe fn import(
    device: &Device,
    memory_win32: &khr::external_memory_win32::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    mut texture: D3d11Import,
    format: SharedFormat,
) -> Result<ImportedImage, Error> {
    if texture.width == 0 || texture.height == 0 {
        return Err(Error::D3d11Extent);
    }
    let format = vk_format(format);
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(HANDLE_TYPE);
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: texture.width, height: texture.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            // The spec requires UNDEFINED for an image with external memory; the producer's pixels are
            // kept by acquiring the image from the external queue family before its first sample.
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut external),
        None,
    )?;

    let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
    let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
    device.get_image_memory_requirements2(
        &vk::ImageMemoryRequirementsInfo2::default().image(image),
        &mut requirements2,
    );
    let requirements = requirements2.memory_requirements;

    let handle = texture.handle();
    let mut handle_props = vk::MemoryWin32HandlePropertiesKHR::default();
    if let Err(err) = memory_win32.get_memory_win32_handle_properties(HANDLE_TYPE, handle, &mut handle_props) {
        device.destroy_image(image, None);
        return Err(err.into());
    }
    let type_bits = requirements.memory_type_bits & handle_props.memory_type_bits;
    let Some(memory_type_index) =
        find_memory_type_index(memory_properties, type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
            .or_else(|| find_memory_type_index(memory_properties, type_bits, vk::MemoryPropertyFlags::empty()))
    else {
        device.destroy_image(image, None);
        return Err(Error::D3d11Memory);
    };

    let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::default()
        .handle_type(HANDLE_TYPE)
        .handle(handle);
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let mut alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index)
        .push_next(&mut import_info);
    if dedicated_req.requires_dedicated_allocation == vk::TRUE
        || dedicated_req.prefers_dedicated_allocation == vk::TRUE
    {
        alloc_info = alloc_info.push_next(&mut dedicated_info);
    }
    let memory = match device.allocate_memory(&alloc_info, None) {
        Ok(memory) => memory,
        Err(err) => {
            device.destroy_image(image, None);
            return Err(err.into());
        }
    };
    if let Err(err) = device.bind_image_memory(image, memory, 0) {
        device.destroy_image(image, None);
        device.free_memory(memory, None);
        return Err(err.into());
    }
    let view = device.create_image_view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            ),
        None,
    );
    let view = match view {
        Ok(view) => view,
        Err(err) => {
            device.destroy_image(image, None);
            device.free_memory(memory, None);
            return Err(err.into());
        }
    };
    Ok(ImportedImage {
        image,
        memory,
        view,
        width: texture.width,
        height: texture.height,
        format,
        current_layout: vk::ImageLayout::UNDEFINED,
        acquire_external: true,
        // Keep the duplicate so a later submit of the same texture can be recognized.
        shared_handle: Some(texture.take_handle()),
        fence: None,
        pending_signal: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_reuses_a_match_and_evicts_the_oldest() {
        let mut ring = ImportRing::default();
        for n in 1..=PARKED_IMPORTS {
            assert!(ring.park(n).is_empty());
        }
        // A hit leaves the others in place.
        assert_eq!(ring.take(|&n| n == 2), Some(2));
        assert_eq!(ring.take(|&n| n == 2), None);
        // Parking past the limit pushes out the entry parked longest ago, not the newest.
        assert!(ring.park(10).is_empty());
        assert_eq!(ring.park(11), vec![1]);
        assert_eq!(ring.drain().collect::<Vec<_>>(), vec![3, 10, 11]);
    }

    #[test]
    fn formats_map_to_the_vulkan_formats_the_shader_samples_as_rgba() {
        assert_eq!(vk_format(SharedFormat::Rgba8), vk::Format::R8G8B8A8_UNORM);
        assert_eq!(vk_format(SharedFormat::Bgra8), vk::Format::B8G8R8A8_UNORM);
    }
}
