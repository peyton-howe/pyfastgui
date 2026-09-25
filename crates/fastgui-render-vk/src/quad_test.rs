//! GPU check of the chrome quad pipeline on Vulkan: the renderer's own `QuadPipeline` and
//! `QuadChrome` draw `build_quads` output into an offscreen image, which is read back and compared
//! with the CPU painter (`fastgui_chrome::testing::check_gpu_backend`). Headless (no window or
//! swapchain), so it runs wherever a Vulkan 1.3 driver does:
//!
//!   cargo test -p fastgui-render-vk
//!
//! It skips (passing) when no Vulkan loader or 1.3-capable device is found. With the Khronos
//! validation layer installed, any validation error fails the test. macOS (MoltenVK via Homebrew):
//!
//!   DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
//!   VK_LAYER_PATH=/opt/homebrew/opt/vulkan-validationlayers/share/vulkan/explicit_layer.d \
//!   cargo test -p fastgui-render-vk

use std::ffi::{c_char, CStr};
use std::sync::Mutex;

use ash::{ext::debug_utils, vk, Device, Entry, Instance};

use crate::pipeline::ViewportPipeline;
use crate::quad::{QuadChrome, QuadPipeline};
use crate::texture::find_memory_type_index;

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

static VALIDATION_ERRORS: Mutex<Vec<String>> = Mutex::new(Vec::new());

unsafe extern "system" fn on_validation_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) && !(*data).p_message.is_null() {
        let message = CStr::from_ptr((*data).p_message).to_string_lossy().into_owned();
        VALIDATION_ERRORS.lock().unwrap_or_else(|e| e.into_inner()).push(message);
    }
    vk::FALSE
}

/// A headless device plus everything one offscreen chrome draw needs.
struct Gpu {
    _entry: Entry,
    instance: Instance,
    messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    device: Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    viewport_pipeline: ViewportPipeline,
    quad_pipeline: QuadPipeline,
    chrome: QuadChrome,
    target: (vk::Image, vk::DeviceMemory, vk::ImageView),
    readback: (vk::Buffer, vk::DeviceMemory, *const u8),
    width: u32,
    height: u32,
}

impl Gpu {
    /// `None` when this machine has no usable Vulkan 1.3 device.
    unsafe fn new(width: u32, height: u32) -> Option<Self> {
        let entry = Entry::load().ok()?;
        let available_layers = entry.enumerate_instance_layer_properties().ok()?;
        let has = |name: &CStr| available_layers.iter().any(|l| l.layer_name_as_c_str() == Ok(name));
        let validation = c"VK_LAYER_KHRONOS_validation";
        let layers: Vec<*const c_char> = if has(validation) { vec![validation.as_ptr()] } else { Vec::new() };
        let available_extensions = entry.enumerate_instance_extension_properties(None).ok()?;
        let has_ext = |name: &CStr| available_extensions.iter().any(|e| e.extension_name_as_c_str() == Ok(name));
        let mut extensions = vec![debug_utils::NAME.as_ptr()];
        let mut flags = vk::InstanceCreateFlags::empty();
        // MoltenVK is a "portability" driver: only listed when asked for.
        if has_ext(ash::khr::portability_enumeration::NAME) {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions)
                    .flags(flags),
                None,
            )
            .ok()?;
        let messenger = (!layers.is_empty()).then(|| {
            let loader = debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(on_validation_message));
            let messenger = loader.create_debug_utils_messenger(&info, None).expect("debug messenger");
            (loader, messenger)
        });
        if messenger.is_none() {
            eprintln!("note: Vulkan validation layer not found; running without it");
        }

        let (pdevice, family) = instance.enumerate_physical_devices().ok()?.into_iter().find_map(|pd| {
            if instance.get_physical_device_properties(pd).api_version < vk::API_VERSION_1_3 {
                return None;
            }
            let family = instance
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))?;
            Some((pd, family as u32))
        })?;
        let device_extensions = instance.enumerate_device_extension_properties(pdevice).ok()?;
        let portability_subset = c"VK_KHR_portability_subset";
        let device_extension_names: Vec<*const c_char> = device_extensions
            .iter()
            .any(|e| e.extension_name_as_c_str() == Ok(portability_subset))
            .then(|| portability_subset.as_ptr())
            .into_iter()
            .collect();
        let priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let device = instance
            .create_device(
                pdevice,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&device_extension_names)
                    .push_next(&mut features13),
                None,
            )
            .ok()?;
        let queue = device.get_device_queue(family, 0);
        let memory_properties = instance.get_physical_device_memory_properties(pdevice);
        let command_pool = device
            .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None)
            .ok()?;
        let viewport_pipeline = ViewportPipeline::new(&device, FORMAT).expect("viewport pipeline");
        let quad_pipeline =
            QuadPipeline::new(&device, FORMAT, viewport_pipeline.descriptor_set_layout()).expect("quad pipeline");

        let image = device
            .create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(FORMAT)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC),
                None,
            )
            .ok()?;
        let requirements = device.get_image_memory_requirements(image);
        let index = find_memory_type_index(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let image_memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index),
                None,
            )
            .ok()?;
        device.bind_image_memory(image, image_memory, 0).ok()?;
        let view = device
            .create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(FORMAT)
                    .subresource_range(color_range()),
                None,
            )
            .ok()?;

        let bytes = u64::from(width) * u64::from(height) * 4;
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::default().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_DST),
                None,
            )
            .ok()?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let index = find_memory_type_index(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        let buffer_memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index),
                None,
            )
            .ok()?;
        device.bind_buffer_memory(buffer, buffer_memory, 0).ok()?;
        let mapped = device.map_memory(buffer_memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()).ok()? as *const u8;

        Some(Self {
            _entry: entry,
            instance,
            messenger,
            device,
            memory_properties,
            queue,
            command_pool,
            viewport_pipeline,
            quad_pipeline,
            chrome: QuadChrome::new(1),
            target: (image, image_memory, view),
            readback: (buffer, buffer_memory, mapped),
            width,
            height,
        })
    }

    /// Apply `frame` (if any), draw the chrome, and read the image back as RGBA8.
    unsafe fn draw(&mut self, frame: Option<fastgui_core::ChromeQuads<'_>>) -> Vec<u8> {
        let device = &self.device;
        if let Some(frame) = frame {
            self.chrome.update(device, &self.memory_properties, &self.viewport_pipeline, &frame).expect("update");
        }
        self.chrome.prepare(device, &self.memory_properties, 0).expect("prepare");

        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.command_pool).command_buffer_count(1),
            )
            .expect("command buffer")[0];
        device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
            .unwrap();
        let (image, _, view) = self.target;
        let mut barriers = vec![vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
        barriers.extend(self.chrome.atlas_barrier(color_range()));
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barriers,
        );
        let extent = vk::Extent2D { width: self.width, height: self.height };
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0; 4] } })];
        device.cmd_begin_rendering(
            cmd,
            &vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent })
                .layer_count(1)
                .color_attachments(&attachments),
        );
        self.chrome.record(device, cmd, &self.quad_pipeline, 0, extent);
        device.cmd_end_rendering(cmd);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
        );
        device.cmd_copy_image_to_buffer(
            cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            self.readback.0,
            &[vk::BufferImageCopy::default()
                .image_subresource(vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1))
                .image_extent(vk::Extent3D { width: self.width, height: self.height, depth: 1 })],
        );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
            &[],
        );
        device.end_command_buffer(cmd).unwrap();
        let cmds = [cmd];
        device.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], vk::Fence::null()).unwrap();
        device.queue_wait_idle(self.queue).unwrap();
        device.free_command_buffers(self.command_pool, &cmds);

        let len = (self.width * self.height * 4) as usize;
        std::slice::from_raw_parts(self.readback.2, len).to_vec()
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let device = &self.device;
            let _ = device.device_wait_idle();
            self.chrome.destroy(device, &self.viewport_pipeline);
            self.quad_pipeline.destroy(device);
            self.viewport_pipeline.destroy(device);
            device.unmap_memory(self.readback.1);
            device.destroy_buffer(self.readback.0, None);
            device.free_memory(self.readback.1, None);
            device.destroy_image_view(self.target.2, None);
            device.destroy_image(self.target.0, None);
            device.free_memory(self.target.1, None);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_device(None);
            if let Some((loader, messenger)) = &self.messenger {
                loader.destroy_debug_utils_messenger(*messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1)
}

#[test]
fn vulkan_quads_match_cpu_painter() {
    if unsafe { Gpu::new(4, 4) }.is_none() {
        eprintln!("skipping: no Vulkan 1.3 device (or loader) available");
        return;
    }
    fastgui_chrome::testing::check_gpu_backend(
        |w, h| unsafe { Gpu::new(w, h) }.expect("Vulkan device went away"),
        |gpu, frame| unsafe { gpu.draw(frame) },
    );
    let errors = VALIDATION_ERRORS.lock().unwrap_or_else(|e| e.into_inner());
    assert!(errors.is_empty(), "Vulkan validation errors:\n{}", errors.join("\n"));
}
