use std::sync::Arc;

use ash::{khr, vk, Device};
use fastgui_app::{CudaLayerShared, PublishedSlot, CUDA_SLOTS};

use crate::error::VkRendererError as Error;
use crate::texture::find_memory_type_index;

pub use fastgui_app::CudaExportHandles;

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
/// Slot starts are aligned to this so CUDA gets nicely aligned pointers (texel copies only need 4).
const SLOT_ALIGN: u64 = 256;

/// One CUDA-fed `Viewport` layer on the Vulkan side. See `fastgui_app::cuda_handles` for the
/// protocol.
///
/// CUDA writes frames into [`CUDA_SLOTS`] slots of one exported linear buffer (tightly packed
/// RGBA8, so there is no tiling or row-pitch question to get wrong on either API). Picking up a
/// frame is a GPU copy from its slot into an ordinary optimal-tiled image, which is what gets
/// sampled; the slot is free again as soon as that copy completes.
///
/// **Verification status**: the Vulkan half — export, semaphores, the pickup copy and its
/// queue-family transfers — runs clean under the validation layers (`cuda_test.rs`, which plays
/// the CUDA side with Vulkan itself). The CUDA half has not been run on an NVIDIA GPU.
pub struct CudaLayer {
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    image: vk::Image,
    image_memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    /// Signalled by CUDA when a slot is written.
    ready: vk::Semaphore,
    /// Signalled by Vulkan when a copy out of a slot completes.
    release: vk::Semaphore,
    release_value: u64,
    image_initialized: bool,
    has_content: bool,
    width: u32,
    height: u32,
    slot_stride: u64,
    shared: Arc<CudaLayerShared>,
    /// Slot copied by the frame being recorded, until `finish_submit`.
    copying: Option<PublishedSlot>,
}

impl CudaLayer {
    pub unsafe fn new(
        device: &Device,
        external_memory_win32: &khr::external_memory_win32::Device,
        external_semaphore_win32: &khr::external_semaphore_win32::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        device_uuid: [u8; 16],
        width: u32,
        height: u32,
    ) -> Result<(Self, CudaExportHandles), Error> {
        let frame_bytes = u64::from(width) * u64::from(height) * 4;
        let slot_stride = frame_bytes.div_ceil(SLOT_ALIGN) * SLOT_ALIGN;

        // Exported slot buffer. TRANSFER_DST only so tests can stand in for CUDA.
        let mut external_buffer_info = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(slot_stride * CUDA_SLOTS as u64)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut external_buffer_info),
            None,
        )?;
        // The "2" query is the one that reports whether the export needs a dedicated allocation.
        let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_requirements);
        device.get_buffer_memory_requirements2(
            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
            &mut requirements2,
        );
        let requirements = requirements2.memory_requirements;
        let dedicated = dedicated_requirements.prefers_dedicated_allocation == vk::TRUE
            || dedicated_requirements.requires_dedicated_allocation == vk::TRUE;
        let memory_type_index = find_memory_type_index(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(Error::NoDeviceLocalTextureMemory)?;
        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
        let mut alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut export_info);
        if dedicated {
            alloc_info = alloc_info.push_next(&mut dedicated_info);
        }
        let buffer_memory = device.allocate_memory(&alloc_info, None)?;
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;

        // Sampled image, private to Vulkan.
        let image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(FORMAT)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                // TRANSFER_SRC only so tests can read the picked-up frame back.
                .usage(
                    vk::ImageUsageFlags::SAMPLED
                        | vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::TRANSFER_SRC,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED),
            None,
        )?;
        let image_requirements = device.get_image_memory_requirements(image);
        let image_memory_type = find_memory_type_index(
            memory_properties,
            image_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(Error::NoDeviceLocalTextureMemory)?;
        let image_memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(image_requirements.size)
                .memory_type_index(image_memory_type),
            None,
        )?;
        device.bind_image_memory(image, image_memory, 0)?;
        let view = device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(FORMAT)
                .subresource_range(color_range()),
            None,
        )?;

        let ready = create_exported_timeline(device)?;
        let release = create_exported_timeline(device)?;

        let shared = Arc::new(CudaLayerShared::default());
        let layer = Self {
            buffer,
            buffer_memory,
            image,
            image_memory,
            view,
            ready,
            release,
            release_value: 0,
            image_initialized: false,
            has_content: false,
            width,
            height,
            slot_stride,
            shared: shared.clone(),
            copying: None,
        };
        // From here on `layer` owns everything; a failed export destroys it on the way out. A
        // handle exported before the failure is closed when the partial `CudaExportHandles` drops.
        let mut handles = CudaExportHandles {
            memory_win32_handle: 0,
            memory_size: requirements.size,
            memory_dedicated: dedicated,
            slot_stride,
            width,
            height,
            ready_win32_handle: 0,
            release_win32_handle: 0,
            device_uuid,
            shared,
        };
        let exported = (|| -> Result<(), Error> {
            handles.memory_win32_handle = export_memory(external_memory_win32, device, buffer_memory)?;
            handles.ready_win32_handle = export_semaphore(external_semaphore_win32, device, ready)?;
            handles.release_win32_handle = export_semaphore(external_semaphore_win32, device, release)?;
            Ok(())
        })();
        match exported {
            Ok(()) => Ok((layer, handles)),
            Err(err) => {
                layer.destroy(device);
                Err(err)
            }
        }
    }

    /// Frame size in pixels, for letterboxing.
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// True once a frame has been copied in; until then there is nothing to draw.
    pub fn has_content(&self) -> bool {
        self.has_content
    }

    /// Record the pickup of the newest published frame, if any, into `cmd`. Returns the `ready`
    /// value the submission must wait on (at the TRANSFER stage) before the copy runs. Include
    /// [`Self::release_signal`] in that submission and call [`Self::finish_submit`] after it.
    ///
    /// Leaves the image in `GENERAL` (what the viewport pipeline's descriptor expects), visible
    /// to fragment-shader reads recorded later in `cmd`.
    pub unsafe fn record_pickup(
        &mut self,
        device: &Device,
        cmd: vk::CommandBuffer,
        queue_family: u32,
    ) -> Option<(vk::Semaphore, u64)> {
        let slot = self.shared.take_published()?;
        let offset = slot.index as u64 * self.slot_stride;
        let size = u64::from(self.width) * u64::from(self.height) * 4;

        // Take the slot's bytes over from CUDA (queue family ownership acquire) and make the
        // image writable after any earlier frame's sampling of it.
        let acquire = vk::BufferMemoryBarrier::default()
            .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(queue_family)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .buffer(self.buffer)
            .offset(offset)
            .size(size);
        let old_layout =
            if self.image_initialized { vk::ImageLayout::GENERAL } else { vk::ImageLayout::UNDEFINED };
        let to_transfer = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .image(self.image)
            .subresource_range(color_range());
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[acquire],
            &[to_transfer],
        );
        device.cmd_copy_buffer_to_image(
            cmd,
            self.buffer,
            self.image,
            vk::ImageLayout::GENERAL,
            &[vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 })],
        );
        // Hand the slot back to CUDA (ownership release) and publish the copy to the shader.
        let release = vk::BufferMemoryBarrier::default()
            .src_queue_family_index(queue_family)
            .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .buffer(self.buffer)
            .offset(offset)
            .size(size);
        let to_shader = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .image(self.image)
            .subresource_range(color_range());
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[release],
            &[to_shader],
        );

        self.image_initialized = true;
        self.copying = Some(slot);
        Some((self.ready, slot.ready))
    }

    /// The `release` signal the submission carrying a pickup must include, if it carries one.
    pub fn release_signal(&self) -> Option<(vk::Semaphore, u64)> {
        self.copying?;
        Some((self.release, self.release_value + 1))
    }

    /// After the submission carrying a pickup: hand the slot back to producers. With
    /// `submitted == false` nothing will run, so the slot is freed without a new `release`.
    pub fn finish_submit(&mut self, submitted: bool) {
        let Some(slot) = self.copying.take() else { return };
        if submitted {
            self.release_value += 1;
            self.has_content = true;
            self.shared.copied(slot.index, self.release_value);
        } else {
            self.shared.copy_failed(slot.index);
        }
    }

    /// Destroy every Vulkan object and tell producers the layer is gone. The device must be idle
    /// with respect to this layer.
    pub unsafe fn destroy(&self, device: &Device) {
        self.shared.close();
        device.destroy_semaphore(self.ready, None);
        device.destroy_semaphore(self.release, None);
        device.destroy_image_view(self.view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.image_memory, None);
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.buffer_memory, None);
    }

    #[cfg(test)]
    pub(crate) fn test_parts(&self) -> (vk::Buffer, vk::Image, vk::Semaphore, vk::Semaphore, u64) {
        (self.buffer, self.image, self.ready, self.release, self.slot_stride)
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

unsafe fn create_exported_timeline(device: &Device) -> Result<vk::Semaphore, Error> {
    let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
    let mut export_info = vk::ExportSemaphoreCreateInfo::default()
        .handle_types(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32);
    Ok(device.create_semaphore(
        &vk::SemaphoreCreateInfo::default().push_next(&mut type_info).push_next(&mut export_info),
        None,
    )?)
}

unsafe fn export_memory(
    loader: &khr::external_memory_win32::Device,
    device: &Device,
    memory: vk::DeviceMemory,
) -> Result<isize, Error> {
    let mut handle: vk::HANDLE = 0;
    let result = (loader.fp().get_memory_win32_handle_khr)(
        device.handle(),
        &vk::MemoryGetWin32HandleInfoKHR::default()
            .memory(memory)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32),
        &mut handle,
    );
    if result != vk::Result::SUCCESS {
        return Err(Error::Vk(result));
    }
    Ok(handle as isize)
}

unsafe fn export_semaphore(
    loader: &khr::external_semaphore_win32::Device,
    device: &Device,
    semaphore: vk::Semaphore,
) -> Result<isize, Error> {
    let mut handle: vk::HANDLE = 0;
    let result = (loader.fp().get_semaphore_win32_handle_khr)(
        device.handle(),
        &vk::SemaphoreGetWin32HandleInfoKHR::default()
            .semaphore(semaphore)
            .handle_type(vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32),
        &mut handle,
    );
    if result != vk::Result::SUCCESS {
        return Err(Error::Vk(result));
    }
    Ok(handle as isize)
}
