use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use ash::{khr, vk, Device};

use crate::error::VkRendererError as Error;

/// A GPU-resident RGBA8 image whose backing memory is exported (via a Windows NT handle) for
/// import into CUDA, plus a timeline semaphore exported the same way so the render thread can
/// wait for a CUDA kernel's writes to finish before sampling it — no CPU copy, no shared
/// staging buffer, in either direction.
///
/// **Verification status**: the Vulkan-side half of this — image + dedicated-allocation
/// export, timeline semaphore creation + export, `vkGetMemoryWin32HandleKHR`/
/// `vkGetSemaphoreWin32HandleKHR` — has been exercised with validation layers enabled on a
/// real GPU (AMD integrated; this machine has no NVIDIA hardware) and runs clean, no
/// validation errors. What's *not* verified is the other half of the handshake: whether a real
/// CUDA driver actually accepts these exact handles/sizes and maps them correctly, and whether
/// the timeline-semaphore wait genuinely provides correct cross-device synchronization. See
/// `fastgui-interop-cuda`'s crate docs for that half's caveat.
pub struct CudaSharedTexture {
    pub image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub semaphore: vk::Semaphore,
    pub current_layout: vk::ImageLayout,
    pub last_waited_value: u64,
    pub target_value: Arc<AtomicU64>,
}

/// Everything `fastgui-interop-cuda` needs to import the Vulkan side of a `CudaSharedTexture`.
/// Deliberately just raw handles/integers plus one plain `std` type — this crate has no CUDA
/// dependency, and `fastgui-interop-cuda` has no Vulkan dependency; they only share these.
pub struct CudaExportHandles {
    pub memory_win32_handle: isize,
    pub memory_size: u64,
    pub semaphore_win32_handle: isize,
    pub row_pitch: u64,
    pub width: u32,
    pub height: u32,
    /// The render thread polls this every frame to know the highest timeline value a CUDA
    /// producer has signalled; `CudaSurface.signal_ready()` (fastgui-py) is what bumps it.
    pub target_value: Arc<AtomicU64>,
}

impl CudaSharedTexture {
    pub unsafe fn new(
        device: &Device,
        external_memory_win32: &khr::external_memory_win32::Device,
        external_semaphore_win32: &khr::external_semaphore_win32::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
    ) -> Result<(Self, CudaExportHandles), Error> {
        let format = vk::Format::R8G8B8A8_UNORM;

        // LINEAR tiling (not OPTIMAL) so the byte layout CUDA writes into is well-defined
        // row-major-with-pitch, exactly like `ViewportTexture`'s CPU-upload path — the
        // difference here is DEVICE_LOCAL, unmapped memory instead of HOST_VISIBLE mapped
        // memory, since CUDA (not the CPU) is what writes into it.
        let mut external_image_info = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
        let image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::LINEAR)
                .usage(vk::ImageUsageFlags::SAMPLED)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut external_image_info),
            None,
        )?;

        // Exported (VkExportMemoryAllocateInfo{OPAQUE_WIN32}) image memory needs its actual
        // dedicated-allocation requirement queried via the "2" variant — the plain
        // `get_image_memory_requirements` doesn't report whether a dedicated allocation is
        // required, and binding without one when it's required is a validation error (and,
        // per spec, undefined behavior without validation layers to catch it).
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 =
            vk::MemoryRequirements2::default().push_next(&mut dedicated_requirements);
        device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut requirements2,
        );
        let requirements = requirements2.memory_requirements;
        let memory_type_index = find_memory_type_index(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(Error::NoDeviceLocalTextureMemory)?;

        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
        let mut dedicated_alloc_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let mut alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_info);
        if dedicated_requirements.prefers_dedicated_allocation == vk::TRUE
            || dedicated_requirements.requires_dedicated_allocation == vk::TRUE
        {
            alloc_info = alloc_info.push_next(&mut dedicated_alloc_info);
        }
        let memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_image_memory(image, memory, 0)?;

        let layout = device.get_image_subresource_layout(
            image,
            vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
        );

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
        )?;

        let mut memory_handle: vk::HANDLE = 0;
        let result = (external_memory_win32.fp().get_memory_win32_handle_khr)(
            device.handle(),
            &vk::MemoryGetWin32HandleInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32),
            &mut memory_handle,
        );
        if result != vk::Result::SUCCESS {
            return Err(Error::Vk(result));
        }

        let mut semaphore_type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let mut export_semaphore_info = vk::ExportSemaphoreCreateInfo::default()
            .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
        let semaphore = device.create_semaphore(
            &vk::SemaphoreCreateInfo::default()
                .push_next(&mut semaphore_type_info)
                .push_next(&mut export_semaphore_info),
            None,
        )?;

        let mut semaphore_handle: vk::HANDLE = 0;
        let result = (external_semaphore_win32.fp().get_semaphore_win32_handle_khr)(
            device.handle(),
            &vk::SemaphoreGetWin32HandleInfoKHR::default()
                .semaphore(semaphore)
                .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32),
            &mut semaphore_handle,
        );
        if result != vk::Result::SUCCESS {
            return Err(Error::Vk(result));
        }

        let target_value = Arc::new(AtomicU64::new(0));
        let texture = Self {
            image,
            memory,
            view,
            semaphore,
            current_layout: vk::ImageLayout::UNDEFINED,
            last_waited_value: 0,
            target_value: target_value.clone(),
        };
        let handles = CudaExportHandles {
            memory_win32_handle: memory_handle as isize,
            memory_size: requirements.size,
            semaphore_win32_handle: semaphore_handle as isize,
            row_pitch: layout.row_pitch,
            width,
            height,
            target_value,
        };
        Ok((texture, handles))
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_semaphore(self.semaphore, None);
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.memory, None);
    }
}

fn find_memory_type_index(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..properties.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && properties.memory_types[i as usize].property_flags.contains(flags)
    })
}
