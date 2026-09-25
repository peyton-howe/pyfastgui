//! Chrome as GPU quads on Vulkan: `QuadPipeline` (an instanced, premultiplied-blended quad
//! pipeline, shaders in `shaders/quad.{vert,frag}`) and `QuadChrome` (the glyph atlas, the
//! current quads, and one host-visible instance buffer per frame in flight). Mirrors the Metal
//! backend's quad path; see `fastgui-chrome`'s `gpu` module for what the quads mean.

use ash::{vk, Device};
use fastgui_core::{ChromeQuad, ChromeQuads};

use crate::error::VkRendererError as Error;
use crate::pipeline::ViewportPipeline;
use crate::texture::{find_memory_type_index, ViewportTexture};

const VERT_SPV: &[u8] = include_bytes!("../shaders/quad.vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../shaders/quad.frag.spv");

/// Instanced quads: per-instance `ChromeQuad` vertex attributes, a 4-vertex triangle strip per
/// instance, the atlas bound through `ViewportPipeline`'s descriptor set layout (one combined
/// image sampler), and the quads' pixel-space size as a push constant.
pub struct QuadPipeline {
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl QuadPipeline {
    pub unsafe fn new(
        device: &Device,
        color_format: vk::Format,
        set_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, Error> {
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8)];
        let set_layouts = [set_layout];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges),
            None,
        )?;

        let vert_code = ash::util::read_spv(&mut std::io::Cursor::new(VERT_SPV))?;
        let frag_code = ash::util::read_spv(&mut std::io::Cursor::new(FRAG_SPV))?;
        let vert_module =
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&vert_code), None)?;
        let frag_module =
            device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&frag_code), None)?;
        let entry_point = c"main";
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(entry_point),
        ];

        let bindings = [vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(std::mem::size_of::<ChromeQuad>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE)];
        let vec4 = vk::Format::R32G32B32A32_SFLOAT;
        let attributes = [
            vk::VertexInputAttributeDescription::default().location(0).binding(0).format(vec4).offset(0),
            vk::VertexInputAttributeDescription::default().location(1).binding(0).format(vec4).offset(16),
            vk::VertexInputAttributeDescription::default().location(2).binding(0).format(vec4).offset(32),
            vk::VertexInputAttributeDescription::default()
                .location(3)
                .binding(0)
                .format(vk::Format::R32_UINT)
                .offset(48),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&bindings)
            .vertex_attribute_descriptions(&attributes);
        // Primitive restart only affects indexed draws, which this never issues; enabled anyway
        // because Metal can't disable it and MoltenVK warns when asked to.
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
            .primitive_restart_enable(true);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample =
            vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Premultiplied source-over.
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let color_formats = [color_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);
        let pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
            .map_err(|(_, err)| err)?[0];

        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);
        Ok(Self { pipeline_layout, pipeline })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
    }
}

/// A persistently mapped, host-coherent vertex buffer.
struct HostBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    capacity: u64,
}

impl HostBuffer {
    unsafe fn new(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        capacity: u64,
    ) -> Result<Self, Error> {
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(capacity)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let Some(memory_type_index) = find_memory_type_index(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            device.destroy_buffer(buffer, None);
            return Err(Error::NoHostVisibleBufferMemory);
        };
        let memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type_index),
            None,
        )?;
        device.bind_buffer_memory(buffer, memory, 0)?;
        let mapped = device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())? as *mut u8;
        Ok(Self { buffer, memory, mapped, capacity })
    }

    unsafe fn destroy(&self, device: &Device) {
        device.unmap_memory(self.memory);
        device.destroy_buffer(self.buffer, None);
        device.free_memory(self.memory, None);
    }
}

/// The chrome's GPU state between frames: the atlas (sampled through a descriptor set from
/// `ViewportPipeline`'s pool), the latest quads, and an instance buffer per frame slot so a
/// slot's buffer is only rewritten once that slot's fence says the GPU is done with it.
pub struct QuadChrome {
    atlas: Option<(ViewportTexture, vk::DescriptorSet)>,
    quads: Vec<ChromeQuad>,
    /// Pixel space the quads were built in (their push-constant viewport).
    size: (u32, u32),
    instance_buffers: Vec<Option<HostBuffer>>,
}

impl QuadChrome {
    pub fn new(frames_in_flight: usize) -> Self {
        Self {
            atlas: None,
            quads: Vec::new(),
            size: (1, 1),
            instance_buffers: (0..frames_in_flight).map(|_| None).collect(),
        }
    }

    pub fn is_active(&self) -> bool {
        self.atlas.is_some() && !self.quads.is_empty()
    }

    /// Take a new frame from `fastgui_chrome::ChromeRenderer::build_quads`. Waits for the device
    /// to go idle only when atlas slots are being reused (a repack) or the atlas is replaced;
    /// ordinary uploads land in slots no in-flight frame samples.
    pub unsafe fn update(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        sets: &ViewportPipeline,
        frame: &ChromeQuads<'_>,
    ) -> Result<(), Error> {
        let replace = self.atlas.as_ref().is_none_or(|(atlas, _)| atlas.width != frame.atlas_size);
        if self.atlas.is_some() && (replace || frame.atlas_repacked) {
            device.device_wait_idle()?;
        }
        if replace {
            if let Some((atlas, set)) = self.atlas.take() {
                atlas.destroy(device);
                sets.free_set(device, set);
            }
            let atlas = ViewportTexture::new(device, memory_properties, frame.atlas_size, frame.atlas_size)?;
            let set = sets.alloc_set(device)?;
            sets.bind_texture(device, set, atlas.view);
            self.atlas = Some((atlas, set));
        }
        let (atlas, _) = self.atlas.as_ref().expect("ensured above");
        for upload in &frame.atlas_uploads {
            atlas.write_region(upload.rect, upload.pixels);
        }
        self.quads.clear();
        self.quads.extend_from_slice(frame.quads);
        self.size = (frame.width, frame.height);
        Ok(())
    }

    /// Copy the quads into frame slot `slot`'s instance buffer (growing it if needed). Call only
    /// after that slot's fence has been waited on.
    pub unsafe fn prepare(
        &mut self,
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        slot: usize,
    ) -> Result<(), Error> {
        if !self.is_active() {
            return Ok(());
        }
        let bytes = std::mem::size_of_val(self.quads.as_slice()) as u64;
        let buffer = &mut self.instance_buffers[slot];
        if buffer.as_ref().is_none_or(|b| b.capacity < bytes) {
            if let Some(old) = buffer.take() {
                old.destroy(device);
            }
            // Headroom so a growing UI doesn't reallocate every frame.
            *buffer = Some(HostBuffer::new(device, memory_properties, bytes.next_power_of_two().max(64 * 1024))?);
        }
        let buffer = buffer.as_ref().expect("ensured above");
        std::ptr::copy_nonoverlapping(self.quads.as_ptr().cast::<u8>(), buffer.mapped, bytes as usize);
        Ok(())
    }

    /// Barrier publishing host writes to the atlas to the fragment shader (and taking it out of
    /// `PREINITIALIZED` on first use), for the frame's pre-render barrier batch.
    pub fn atlas_barrier(&mut self, subresource: vk::ImageSubresourceRange) -> Option<vk::ImageMemoryBarrier<'static>> {
        if !self.is_active() {
            return None;
        }
        let (atlas, _) = self.atlas.as_mut()?;
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(atlas.current_layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(atlas.image)
            .subresource_range(subresource)
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        atlas.current_layout = vk::ImageLayout::GENERAL;
        Some(barrier)
    }

    /// Record the instanced draw into `cmd` (inside dynamic rendering, over `extent`), reading
    /// slot `slot`'s instance buffer.
    pub unsafe fn record(
        &self,
        device: &Device,
        cmd: vk::CommandBuffer,
        pipeline: &QuadPipeline,
        slot: usize,
        extent: vk::Extent2D,
    ) {
        let (Some((_, set)), Some(buffer)) = (&self.atlas, &self.instance_buffers[slot]) else { return };
        if self.quads.is_empty() {
            return;
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
        device.cmd_set_viewport(
            cmd,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        device.cmd_set_scissor(cmd, 0, &[vk::Rect2D { offset: vk::Offset2D::default(), extent }]);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline_layout, 0, &[*set], &[]);
        let size = [self.size.0 as f32, self.size.1 as f32];
        let push: [u8; 8] = std::mem::transmute(size);
        device.cmd_push_constants(cmd, pipeline.pipeline_layout, vk::ShaderStageFlags::VERTEX, 0, &push);
        device.cmd_bind_vertex_buffers(cmd, 0, &[buffer.buffer], &[0]);
        device.cmd_draw(cmd, 4, self.quads.len() as u32, 0, 0);
    }

    /// Free everything. The caller makes sure the GPU is idle first.
    pub unsafe fn destroy(&mut self, device: &Device, sets: &ViewportPipeline) {
        if let Some((atlas, set)) = self.atlas.take() {
            atlas.destroy(device);
            sets.free_set(device, set);
        }
        for buffer in self.instance_buffers.iter_mut().filter_map(Option::take) {
            buffer.destroy(device);
        }
        self.quads.clear();
    }
}
