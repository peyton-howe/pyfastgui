//! Import a linear RGBA8 dma-buf and sample it as a viewport image.
//!
//! The caller dups the fd into [`fastgui_core::DmaBufImport`]. `vkAllocateMemory` with
//! `VkImportMemoryFdInfoKHR` consumes that fd. DRM modifiers and NV12 are not accepted: the
//! image is `R8G8B8A8_UNORM`, linear, and the row stride must match the image's row pitch.
//! The producer finishes writing before `submit_dmabuf`; there is no sync-file wait yet.

use ash::{khr, vk, Device};
use fastgui_core::DmaBufImport;

use crate::error::VkRendererError as Error;
use crate::texture::find_memory_type_index;

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

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

    let mut external = vk::ExternalMemoryImageCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
    let image = device.create_image(
        &vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(FORMAT)
            .extent(vk::Extent3D { width: buf.width, height: buf.height, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::LINEAR)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::PREINITIALIZED)
            .push_next(&mut external),
        None,
    )?;
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
            .format(FORMAT)
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
