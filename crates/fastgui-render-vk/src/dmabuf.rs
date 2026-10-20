//! Import a linear RGBA8 dma-buf and sample it as a viewport image.
//!
//! The caller dups the fd into [`fastgui_core::DmaBufImport`]. `vkAllocateMemory` with
//! `VkImportMemoryFdInfoKHR` consumes that fd. Non-zero DRM modifiers and NV12 are not
//! accepted. The image is `R8G8B8A8_UNORM` with modifier 0. Drivers that reject
//! `VK_IMAGE_TILING_LINEAR` for dma-buf, including Mali, are asked again with
//! `VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT` and `DRM_FORMAT_MOD_LINEAR` (still modifier 0).
//! The row stride must match the image's row pitch. The producer finishes writing before
//! `submit_dmabuf`; there is no sync-file wait yet.

use ash::{khr, vk, Device, Instance};
use fastgui_core::DmaBufImport;

use crate::error::VkRendererError as Error;
use crate::texture::find_memory_type_index;

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// How a modifier-0 RGBA8 dma-buf is created. `DrmLinear` is still modifier 0; it is the
/// explicit-layout tiling some drivers require instead of `VK_IMAGE_TILING_LINEAR`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ImportTiling {
    Linear,
    DrmLinear,
}

/// `Some` when this device can import the linear RGBA8 image `submit_dmabuf` accepts.
pub(crate) unsafe fn rgba8_import_tiling(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    drm_modifier_ext: bool,
) -> Option<ImportTiling> {
    if dma_external_features(instance, pdevice, FORMAT, vk::ImageTiling::LINEAR, None)
        .is_ok_and(|features| features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE))
    {
        return Some(ImportTiling::Linear);
    }
    if drm_modifier_ext
        && dma_external_features(
            instance,
            pdevice,
            FORMAT,
            vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
            Some(0),
        )
        .is_ok_and(|features| features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE))
    {
        return Some(ImportTiling::DrmLinear);
    }
    None
}

pub(crate) unsafe fn dma_external_features(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    format: vk::Format,
    tiling: vk::ImageTiling,
    modifier: Option<u64>,
) -> Result<vk::ExternalMemoryFeatureFlags, vk::Result> {
    let mut modifier_info = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(modifier.unwrap_or(0))
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external_format = vk::PhysicalDeviceExternalImageFormatInfo::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let mut format_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(tiling)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .push_next(&mut external_format);
    if modifier.is_some() {
        format_info = format_info.push_next(&mut modifier_info);
    }
    let mut external_props = vk::ExternalImageFormatProperties::default();
    let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
    instance.get_physical_device_image_format_properties2(pdevice, &format_info, &mut props)?;
    Ok(external_props.external_memory_properties.external_memory_features)
}

pub(crate) unsafe fn create_dma_image(
    device: &Device,
    format: vk::Format,
    width: u32,
    height: u32,
    stride: u32,
    tiling: ImportTiling,
) -> Result<vk::Image, vk::Result> {
    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let byte_size = u64::from(stride) * u64::from(height);
    let planes = [vk::SubresourceLayout {
        offset: 0,
        size: byte_size,
        row_pitch: u64::from(stride),
        array_pitch: byte_size,
        depth_pitch: byte_size,
    }];
    let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
        .drm_format_modifier(0)
        .plane_layouts(&planes);
    let mut info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::PREINITIALIZED)
        .push_next(&mut external);
    info = match tiling {
        ImportTiling::Linear => info.tiling(vk::ImageTiling::LINEAR),
        ImportTiling::DrmLinear => info
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .push_next(&mut explicit),
    };
    device.create_image(&info, None)
}

pub struct ImportedImage {
    pub(crate) image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    pub current_layout: vk::ImageLayout,
}

impl ImportedImage {
    pub unsafe fn destroy(self, device: &Device) {
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

pub unsafe fn import(
    device: &Device,
    memory_fd: &khr::external_memory_fd::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    mut buf: DmaBufImport,
    format: vk::Format,
    tiling: ImportTiling,
) -> Result<ImportedImage, Error> {
    if buf.modifier != 0 {
        return Err(Error::DmaBufModifier);
    }
    if buf.width == 0 || buf.height == 0 {
        return Err(Error::DmaBufExtent);
    }
    let stride = if buf.stride == 0 {
        buf.width.checked_mul(4).ok_or(Error::DmaBufExtent)?
    } else {
        buf.stride
    };

    let image = create_dma_image(device, format, buf.width, buf.height, stride, tiling)?;
    let layout = device.get_image_subresource_layout(
        image,
        vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
    );
    if layout.offset != 0 || layout.row_pitch != u64::from(stride) {
        device.destroy_image(image, None);
        return Err(Error::DmaBufStride { stride, row_pitch: layout.row_pitch });
    }

    let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
    let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
    device.get_image_memory_requirements2(
        &vk::ImageMemoryRequirementsInfo2::default().image(image),
        &mut requirements2,
    );
    let requirements = requirements2.memory_requirements;

    let fd = buf.take_fd();
    let mut fd_props = vk::MemoryFdPropertiesKHR::default();
    if let Err(err) = memory_fd.get_memory_fd_properties(
        vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
        fd,
        &mut fd_props,
    ) {
        close_fd(fd);
        device.destroy_image(image, None);
        return Err(err.into());
    }
    let type_bits = requirements.memory_type_bits & fd_props.memory_type_bits;
    let memory_type_index = match find_memory_type_index(
        memory_properties,
        type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .or_else(|| find_memory_type_index(memory_properties, type_bits, vk::MemoryPropertyFlags::empty()))
    {
        Some(index) => index,
        // `take_fd` already disarmed `DmaBufImport`'s drop, and allocate has not consumed `fd`.
        None => {
            close_fd(fd);
            device.destroy_image(image, None);
            return Err(Error::DmaBufMemory);
        }
    };

    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
        .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
        .fd(fd);
    let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
    let mut alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type_index)
        .push_next(&mut import_info);
    let dedicated = dedicated_req.requires_dedicated_allocation == vk::TRUE
        || dedicated_req.prefers_dedicated_allocation == vk::TRUE;
    if dedicated {
        alloc_info = alloc_info.push_next(&mut dedicated_info);
    }
    // `VkImportMemoryFdInfoKHR` consumes `fd` on both success and failure.
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
        width: buf.width,
        height: buf.height,
        current_layout: vk::ImageLayout::PREINITIALIZED,
    })
}

fn close_fd(fd: i32) {
    if fd < 0 {
        return;
    }
    #[cfg(unix)]
    {
        extern "C" {
            fn close(fd: i32) -> i32;
        }
        unsafe { close(fd) };
    }
    #[cfg(not(unix))]
    let _ = fd;
}
