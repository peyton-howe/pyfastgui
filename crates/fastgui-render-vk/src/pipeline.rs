use ash::{vk, Device};

use crate::error::VkRendererError as Error;

const VERT_SPV: &[u8] = include_bytes!("../shaders/viewport.vert.spv");
const FRAG_SPV: &[u8] = include_bytes!("../shaders/viewport.frag.spv");

/// Chrome plus in-tree `Viewport` widgets each need a sampled descriptor set.
const MAX_SAMPLED_TEXTURES: u32 = 64;

/// Draws the viewport's texture as a full-window quad: a vertex shader that generates a
/// full-screen triangle from `gl_VertexIndex` alone (no vertex buffer), a fragment shader
/// that samples the texture, built via dynamic rendering so there's no render pass or
/// framebuffer to manage.
pub struct ViewportPipeline {
    sampler: vk::Sampler,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl ViewportPipeline {
    pub unsafe fn new(device: &Device, color_format: vk::Format) -> Result<Self, Error> {
        let sampler = device.create_sampler(
            &vk::SamplerCreateInfo::default()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(1.0),
            None,
        )?;

        let bindings = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
        let descriptor_set_layout = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(MAX_SAMPLED_TEXTURES)];
        let descriptor_pool = device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
                .pool_sizes(&pool_sizes)
                .max_sets(MAX_SAMPLED_TEXTURES),
            None,
        )?;

        let set_layouts = [descriptor_set_layout];
        let pipeline_layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts),
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

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state =
            vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        // No cull mode: the full-screen-triangle trick's winding direction doesn't matter
        // when nothing gets culled, and it saves working out which way `gl_VertexIndex`
        // formula winds.
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [color_format];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
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
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            .map_err(|(_, err)| err)?[0];

        device.destroy_shader_module(vert_module, None);
        device.destroy_shader_module(frag_module, None);

        Ok(Self {
            sampler,
            descriptor_set_layout,
            descriptor_pool,
            pipeline_layout,
            pipeline,
        })
    }

    pub unsafe fn alloc_set(&self, device: &Device) -> Result<vk::DescriptorSet, Error> {
        let set_layouts = [self.descriptor_set_layout];
        Ok(device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&set_layouts),
        )?[0])
    }

    pub unsafe fn free_set(&self, device: &Device, set: vk::DescriptorSet) {
        let _ = device.free_descriptor_sets(self.descriptor_pool, &[set]);
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }

    /// One combined image sampler at binding 0; also what `QuadPipeline` binds its atlas with.
    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }

    /// Point `set` at a new texture view. Sampled in `GENERAL` layout, which is what
    /// `ViewportTexture` stays in for its whole lifetime (see its `current_layout`).
    pub unsafe fn bind_texture(&self, device: &Device, set: vk::DescriptorSet, view: vk::ImageView) {
        let image_info = [vk::DescriptorImageInfo::default()
            .sampler(self.sampler)
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL)];
        let write = [vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info)];
        device.update_descriptor_sets(&write, &[]);
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        device.destroy_sampler(self.sampler, None);
    }
}
