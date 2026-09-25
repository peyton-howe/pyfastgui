use ash::{vk, Device};
use fastgui_core::PixelRect;

use crate::error::VkRendererError as Error;

/// A GPU-sampled RGBA8 texture the CPU can write into directly: host-visible, linearly
/// tiled, persistently mapped. Simpler than a staging-buffer + optimal-tiling upload path
/// (no extra command buffer / fence just to copy pixels), at the cost of being single-
/// buffered — see the comment on `upload` for what that trades away.
pub struct ViewportTexture {
    pub image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub width: u32,
    pub height: u32,
    row_pitch: u64,
    mapped: *mut u8,
    pub current_layout: vk::ImageLayout,
}

impl ViewportTexture {
    pub unsafe fn new(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let format = vk::Format::R8G8B8A8_UNORM;
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
                .initial_layout(vk::ImageLayout::PREINITIALIZED),
            None,
        )?;

        let requirements = device.get_image_memory_requirements(image);
        let memory_type_index = find_memory_type_index(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .ok_or(Error::NoHostVisibleTextureMemory)?;
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index),
            None,
        )?;
        device.bind_image_memory(image, memory, 0)?;

        let layout = device.get_image_subresource_layout(
            image,
            vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
        );
        let mapped = device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
            as *mut u8;

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

        Ok(Self {
            image,
            memory,
            view,
            width,
            height,
            row_pitch: layout.row_pitch,
            mapped,
            current_layout: vk::ImageLayout::PREINITIALIZED,
        })
    }

    /// Copy `data`'s tightly packed RGBA8 frame into the mapped image memory, respecting the driver-
    /// reported row pitch (linear images may pad rows; a tightly-packed `memcpy` would
    /// corrupt the image on hardware that pads).
    ///
    /// This does not wait for the GPU to finish reading the previous contents first, so a
    /// producer submitting faster than the display refreshes can occasionally show a torn
    /// frame — visually cheap, never unsound (`HOST_COHERENT` memory, plain byte reads/writes
    /// on both sides). Multi-buffering this texture would remove that, at real extra
    /// complexity; not worth it until the tearing is actually a problem someone hits.
    pub unsafe fn upload(&self, data: &[u8]) {
        self.upload_rect(data, PixelRect { x: 0, y: 0, width: self.width, height: self.height });
    }

    /// Copy just `rect` of `data` (a whole tightly packed `width`x`height` RGBA8 frame) into the
    /// same rect of the image, clipped to it. The rest of the image keeps its contents — linear,
    /// host-coherent memory in `GENERAL` layout, which nothing else writes.
    pub unsafe fn upload_rect(&self, data: &[u8], rect: PixelRect) {
        debug_assert_eq!(data.len(), self.width as usize * self.height as usize * 4);
        let x1 = rect.right().min(self.width) as usize;
        let y1 = rect.bottom().min(self.height) as usize;
        let (x0, y0) = (rect.x as usize, rect.y as usize);
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let stride = self.width as usize * 4;
        let row_bytes = (x1 - x0) * 4;
        for y in y0..y1 {
            let src = &data[y * stride + x0 * 4..][..row_bytes];
            let dst = self.mapped.add(y * self.row_pitch as usize + x0 * 4);
            std::ptr::copy_nonoverlapping(src.as_ptr(), dst, row_bytes);
        }
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_image_view(self.view, None);
        device.unmap_memory(self.memory);
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
