//! Headless timings for the Vulkan CPU upload and, on Linux, a linear RGBA8 dma-buf sample.
//!
//! No window, no swapchain, and no validation layer. Prints the median of 40 frames after 8
//! warmup frames at 1920×1080 and 3840×2160. Exits 0 when this machine has no Vulkan 1.3 device,
//! and skips the dma-buf row when the extensions are missing (Windows never has them). It also
//! prints probes for Windows D3D11/D3D12 handle import and host-pointer import; those only ask
//! the driver and do not import anything yet.
//!
//! ```text
//! cargo run -p fastgui-render-vk --release --example upload_bench
//! ```

use std::ffi::{c_char, CStr};
use std::time::Instant;

use ash::{vk, Device, Entry, Instance};
#[cfg(target_os = "linux")]
use fastgui_core::DmaBufImport;

#[cfg(target_os = "linux")]
use crate::dmabuf::{self, ImportedImage};
use crate::error::VkRendererError as Error;
use crate::pipeline::ViewportPipeline;
use crate::texture::{find_memory_type_index, ViewportTexture};

const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const WARMUP: usize = 8;
const FRAMES: usize = 40;
const SIZES: [(u32, u32); 2] = [(1920, 1080), (3840, 2160)];

pub fn run_upload_bench() {
    match unsafe { Bench::open() } {
        Some(bench) => unsafe { bench.run() },
        None => {}
    }
}

struct Bench {
    _entry: Entry,
    instance: Instance,
    pdevice: vk::PhysicalDevice,
    #[cfg(all(windows, test))]
    family: u32,
    #[cfg(windows)]
    #[cfg_attr(not(test), allow(dead_code))]
    d3d: Option<crate::d3d_import::D3dSupport>,
    device: Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    pipeline: ViewportPipeline,
    name: String,
    software: bool,
    readback: Option<(vk::Buffer, vk::DeviceMemory, *mut u8)>,
    #[cfg(target_os = "linux")]
    memory_fd: Option<ash::khr::external_memory_fd::Device>,
    #[cfg(target_os = "linux")]
    drm_modifier: bool,
}

#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
struct DmaChoice {
    format: vk::Format,
    tiling: crate::dmabuf::ImportTiling,
    label: &'static str,
}

impl Bench {
    unsafe fn open() -> Option<Self> {
        let entry = match Entry::load() {
            Ok(entry) => entry,
            Err(err) => {
                eprintln!("upload bench skipped: failed to load the Vulkan library: {err}");
                return None;
            }
        };
        let available_extensions = match entry.enumerate_instance_extension_properties(None) {
            Ok(extensions) => extensions,
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                return None;
            }
        };
        let has_ext = |name: &CStr| {
            available_extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(name))
        };
        let mut extensions = Vec::new();
        let mut flags = vk::InstanceCreateFlags::empty();
        if has_ext(ash::khr::portability_enumeration::NAME) {
            extensions.push(ash::khr::portability_enumeration::NAME.as_ptr());
            flags |= vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR;
        }
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = match entry.create_instance(
            &vk::InstanceCreateInfo::default()
                .application_info(&app)
                .enabled_extension_names(&extensions)
                .flags(flags),
            None,
        ) {
            Ok(instance) => instance,
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                return None;
            }
        };

        let devices = match instance.enumerate_physical_devices() {
            Ok(devices) => devices,
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                instance.destroy_instance(None);
                return None;
            }
        };
        let mut graphics = Vec::new();
        for pdevice in devices {
            let props = instance.get_physical_device_properties(pdevice);
            if props.api_version < vk::API_VERSION_1_3 {
                continue;
            }
            let family = instance
                .get_physical_device_queue_family_properties(pdevice)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS));
            let Some(family) = family else { continue };
            let name = device_name(&props);
            let software = is_software(&name);
            graphics.push((pdevice, family as u32, name, software));
        }
        graphics.sort_by_key(|(_, _, _, software)| *software);
        let Some((pdevice, family, name, software)) = graphics.into_iter().next() else {
            eprintln!("upload bench skipped: no Vulkan 1.3 device with a graphics queue");
            instance.destroy_instance(None);
            return None;
        };

        let device_extensions = match instance.enumerate_device_extension_properties(pdevice) {
            Ok(extensions) => extensions,
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                instance.destroy_instance(None);
                return None;
            }
        };
        let has_dev = |extension: &CStr| {
            device_extensions
                .iter()
                .any(|e| e.extension_name_as_c_str() == Ok(extension))
        };
        let mut device_extension_names: Vec<*const c_char> = Vec::new();
        let portability_subset = c"VK_KHR_portability_subset";
        if has_dev(portability_subset) {
            device_extension_names.push(portability_subset.as_ptr());
        }
        #[cfg(target_os = "linux")]
        let dmabuf = has_dev(ash::khr::external_memory::NAME)
            && has_dev(ash::khr::external_memory_fd::NAME)
            && has_dev(ash::ext::external_memory_dma_buf::NAME);
        #[cfg(target_os = "linux")]
        let drm_modifier = dmabuf && has_dev(ash::ext::image_drm_format_modifier::NAME);
        #[cfg(target_os = "linux")]
        if dmabuf {
            device_extension_names.extend([
                ash::khr::external_memory::NAME.as_ptr(),
                ash::khr::external_memory_fd::NAME.as_ptr(),
                ash::ext::external_memory_dma_buf::NAME.as_ptr(),
            ]);
        }
        #[cfg(target_os = "linux")]
        if drm_modifier {
            device_extension_names.push(ash::ext::image_drm_format_modifier::NAME.as_ptr());
        }

        #[cfg(windows)]
        let rgba8 = crate::d3d_import::texture_importable(&instance, pdevice, fastgui_core::SharedFormat::Rgba8);
        #[cfg(windows)]
        let bgra8 = crate::d3d_import::texture_importable(&instance, pdevice, fastgui_core::SharedFormat::Bgra8);
        #[cfg(windows)]
        let d3d11 = has_dev(ash::khr::external_memory::NAME)
            && has_dev(ash::khr::external_memory_win32::NAME)
            && (rgba8 || bgra8);
        #[cfg(windows)]
        let d3d_fence = d3d11
            && has_dev(ash::khr::external_semaphore::NAME)
            && has_dev(ash::khr::external_semaphore_win32::NAME)
            && crate::d3d_import::fence_importable(&instance, pdevice);
        #[cfg(windows)]
        if d3d11 {
            device_extension_names.extend([
                ash::khr::external_memory::NAME.as_ptr(),
                ash::khr::external_memory_win32::NAME.as_ptr(),
            ]);
        }
        #[cfg(windows)]
        if d3d_fence {
            device_extension_names.extend([
                ash::khr::external_semaphore::NAME.as_ptr(),
                ash::khr::external_semaphore_win32::NAME.as_ptr(),
            ]);
        }

        let priorities = [1.0];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let mut features13 = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let device = match instance.create_device(
            pdevice,
            &vk::DeviceCreateInfo::default()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extension_names)
                .push_next(&mut features13)
                .push_next(&mut features12),
            None,
        ) {
            Ok(device) => device,
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                instance.destroy_instance(None);
                return None;
            }
        };
        let queue = device.get_device_queue(family, 0);
        let memory_properties = instance.get_physical_device_memory_properties(pdevice);
        let command_pool = match device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .queue_family_index(family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
            None,
        ) {
            Ok(pool) => pool,
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                device.destroy_device(None);
                instance.destroy_instance(None);
                return None;
            }
        };
        let cmd = match device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .command_buffer_count(1),
        ) {
            Ok(buffers) => buffers[0],
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                device.destroy_command_pool(command_pool, None);
                device.destroy_device(None);
                instance.destroy_instance(None);
                return None;
            }
        };
        let pipeline = match ViewportPipeline::new(&device, FORMAT) {
            Ok(pipeline) => pipeline,
            Err(err) => {
                eprintln!("upload bench skipped: {err}");
                device.destroy_command_pool(command_pool, None);
                device.destroy_device(None);
                instance.destroy_instance(None);
                return None;
            }
        };
        let readback = match create_readback(&device, &memory_properties) {
            Ok(readback) => Some(readback),
            Err(err) => {
                eprintln!("readback unavailable: {err}");
                None
            }
        };
        #[cfg(target_os = "linux")]
        let memory_fd =
            dmabuf.then(|| ash::khr::external_memory_fd::Device::new(&instance, &device));

        #[cfg(windows)]
        let d3d = d3d11.then(|| crate::d3d_import::D3dSupport {
            memory: ash::khr::external_memory_win32::Device::new(&instance, &device),
            semaphore: d3d_fence.then(|| ash::khr::external_semaphore_win32::Device::new(&instance, &device)),
            rgba8,
            bgra8,
            probe: crate::d3d_probe::for_device(&instance, pdevice),
        });

        Some(Self {
            _entry: entry,
            instance,
            pdevice,
            #[cfg(all(windows, test))]
            family,
            #[cfg(windows)]
            d3d,
            device,
            memory_properties,
            queue,
            command_pool,
            cmd,
            pipeline,
            name,
            software,
            readback,
            #[cfg(target_os = "linux")]
            memory_fd,
            #[cfg(target_os = "linux")]
            drm_modifier,
        })
    }

    unsafe fn run(&self) {
        let kind = if self.software { " (software)" } else { "" };
        eprintln!("device: {}{kind}", self.name);
        eprintln!("headless; no swapchain and no validation layer");
        eprintln!("median of {FRAMES} frames after {WARMUP} warmup");
        eprintln!("cpu copy+upload is a host to_vec plus the mapped linear write, and does not wait for the GPU");
        eprintln!("upload+sample adds a fullscreen sample and queue_wait_idle");
        eprintln!(
            "dma-buf sample imports a modifier-0 image this process exported, then samples it"
        );
        let dma = self.choose_dmabuf();
        self.probe_windows_interop();
        self.probe_host_pointer();
        for (width, height) in SIZES {
            if let Err(err) = self.time_size(width, height, dma.as_ref()) {
                eprintln!("{width}x{height}: {err}");
            }
        }
    }

    unsafe fn time_size(
        &self,
        width: u32,
        height: u32,
        dma: Option<&DmaChoice>,
    ) -> Result<(), Error> {
        let pixels = solid(width, height);
        let texture = ViewportTexture::new(&self.device, &self.memory_properties, width, height)?;
        let texture = OwnedTexture {
            device: self.device.clone(),
            texture,
        };
        let target = OwnedTarget::new(&self.device, &self.memory_properties, width, height)?;
        self.transition_target(target.image)?;
        let set = SetGuard::new(&self.device, &self.pipeline)?;
        self.pipeline
            .bind_texture(&self.device, set.set, texture.texture.view);

        let label = format!("{width}x{height} cpu copy+upload");
        time(&label, || {
            let copied = pixels.to_vec();
            texture.texture.upload(&copied);
            Ok(())
        })?;

        let mut layout = vk::ImageLayout::PREINITIALIZED;
        let label = format!("{width}x{height} upload+sample");
        time(&label, || {
            let copied = pixels.to_vec();
            texture.texture.upload(&copied);
            self.sample(
                texture.texture.image,
                &mut layout,
                vk::AccessFlags::HOST_WRITE,
                target.view,
                set.set,
                width,
                height,
            )
        })?;
        self.check_readback(&pixels, target.image, width, height, "upload+sample");

        self.time_dmabuf(width, height, &pixels, target.view, target.image, dma);
        Ok(())
    }

    /// One fullscreen sample of `image`, then `queue_wait_idle`. `src_access` is `HOST_WRITE` for a
    /// CPU texture and empty for an imported dma-buf, matching `sync_sampled_image`.
    unsafe fn sample(
        &self,
        image: vk::Image,
        layout: &mut vk::ImageLayout,
        src_access: vk::AccessFlags,
        target_view: vk::ImageView,
        set: vk::DescriptorSet,
        width: u32,
        height: u32,
    ) -> Result<(), Error> {
        let device = &self.device;
        device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(
            self.cmd,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(*layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range())
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
        *layout = vk::ImageLayout::GENERAL;
        device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE | vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        let extent = vk::Extent2D { width, height };
        let attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(target_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            })];
        device.cmd_begin_rendering(
            self.cmd,
            &vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent,
                })
                .layer_count(1)
                .color_attachments(&attachments),
        );
        device.cmd_bind_pipeline(
            self.cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline(),
        );
        device.cmd_set_viewport(
            self.cmd,
            0,
            &[vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: width as f32,
                height: height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        );
        device.cmd_set_scissor(
            self.cmd,
            0,
            &[vk::Rect2D {
                offset: vk::Offset2D::default(),
                extent,
            }],
        );
        device.cmd_bind_descriptor_sets(
            self.cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.pipeline_layout(),
            0,
            &[set],
            &[],
        );
        device.cmd_draw(self.cmd, 3, 1, 0, 0);
        device.cmd_end_rendering(self.cmd);
        device.end_command_buffer(self.cmd)?;
        let cmds = [self.cmd];
        device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default().command_buffers(&cmds)],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(self.queue)?;
        Ok(())
    }

    unsafe fn transition_target(&self, image: vk::Image) -> Result<(), Error> {
        let device = &self.device;
        device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())?;
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range())
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];
        device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        device.end_command_buffer(self.cmd)?;
        let cmds = [self.cmd];
        device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default().command_buffers(&cmds)],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(self.queue)?;
        Ok(())
    }

    unsafe fn check_readback(
        &self,
        pixels: &[u8],
        image: vk::Image,
        width: u32,
        height: u32,
        label: &str,
    ) {
        let Some((buffer, _, mapped)) = self.readback else {
            eprintln!("{width}x{height} {label} readback: skipped");
            return;
        };
        let x = width / 2;
        let y = height / 2;
        if self.copy_pixel(image, buffer, x, y).is_err() {
            eprintln!("{width}x{height} {label} readback: copy failed");
            return;
        }
        let got = std::slice::from_raw_parts(mapped, 4);
        let i = ((y as usize * width as usize) + x as usize) * 4;
        let expected = &pixels[i..i + 4];
        if got == expected {
            eprintln!("{width}x{height} {label} readback: ok");
        } else {
            eprintln!("{width}x{height} {label} readback: got {got:?} expected {expected:?}");
        }
    }

    unsafe fn copy_pixel(
        &self,
        image: vk::Image,
        buffer: vk::Buffer,
        x: u32,
        y: u32,
    ) -> Result<(), Error> {
        let device = &self.device;
        device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())?;
        let to_src = [vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range())
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &to_src,
        );
        device.cmd_copy_image_to_buffer(
            self.cmd,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: x as i32,
                    y: y as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: 1,
                    height: 1,
                    depth: 1,
                })],
        );
        // Put the target back so the next sample can render to it. The buffer barrier publishes the
        // one pixel that was just copied.
        device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST | vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_range())
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)],
        );
        device.end_command_buffer(self.cmd)?;
        let cmds = [self.cmd];
        device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default().command_buffers(&cmds)],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(self.queue)?;
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    unsafe fn time_dmabuf(
        &self,
        width: u32,
        height: u32,
        _pixels: &[u8],
        _view: vk::ImageView,
        _target: vk::Image,
        _dma: Option<&DmaChoice>,
    ) {
        let why = if cfg!(windows) {
            "Windows has no dma-buf import"
        } else {
            "dma-buf import is Linux-only"
        };
        eprintln!("{width}x{height} dma-buf sample: skipped ({why})");
    }

    #[cfg(target_os = "linux")]
    unsafe fn time_dmabuf(
        &self,
        width: u32,
        height: u32,
        pixels: &[u8],
        target_view: vk::ImageView,
        target: vk::Image,
        choice: Option<&DmaChoice>,
    ) {
        let Some(loader) = &self.memory_fd else {
            eprintln!(
                "{width}x{height} dma-buf sample: skipped (device does not advertise VK_KHR_external_memory_fd and VK_EXT_external_memory_dma_buf)"
            );
            return;
        };
        let Some(choice) = choice else {
            eprintln!("{width}x{height} dma-buf sample: skipped (no exportable and importable image)");
            return;
        };
        let imported = match self.export_chosen(loader, width, height, pixels, choice) {
            Ok(imported) => imported,
            Err(err) => {
                eprintln!("{width}x{height} dma-buf sample: skipped ({err})");
                return;
            }
        };
        // The imported image owns the dma-buf from here. Drop the export allocation first.
        let set = match SetGuard::new(&self.device, &self.pipeline) {
            Ok(set) => set,
            Err(err) => {
                eprintln!("{width}x{height} dma-buf sample: skipped ({err})");
                return;
            }
        };
        self.pipeline
            .bind_texture(&self.device, set.set, imported.image.as_ref().unwrap().view);
        let mut layout = vk::ImageLayout::PREINITIALIZED;
        let label = format!("{width}x{height} dma-buf sample");
        let image = imported.image.as_ref().unwrap().image;
        if let Err(err) = time(&label, || {
            self.sample(
                image,
                &mut layout,
                vk::AccessFlags::empty(),
                target_view,
                set.set,
                width,
                height,
            )
        }) {
            eprintln!("{width}x{height} dma-buf sample: {err}");
            return;
        }
        self.check_readback(pixels, target, width, height, "dma-buf sample");
    }

    #[cfg(not(windows))]
    unsafe fn probe_windows_interop(&self) {}

    /// Ask the driver which Windows handle types an RGBA8 sampled image can be imported from
    /// or exported to. D3D11 and D3D12 producers (Media Foundation, D3D11VA, DirectML) hand
    /// out these handles, so import support here is what a `submit_d3d11` would rely on.
    #[cfg(windows)]
    unsafe fn probe_windows_interop(&self) {
        let handle_types = [
            ("opaque win32", vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32),
            ("opaque win32 kmt", vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KMT),
            ("d3d11 texture", vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE),
            ("d3d11 texture kmt", vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT),
            ("d3d12 resource", vk::ExternalMemoryHandleTypeFlags::D3D12_RESOURCE),
        ];
        let tilings = [
            ("optimal", vk::ImageTiling::OPTIMAL),
            ("linear", vk::ImageTiling::LINEAR),
        ];
        for (name, handle_type) in handle_types {
            for (tiling_name, tiling) in tilings {
                let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default()
                    .handle_type(handle_type);
                let info = vk::PhysicalDeviceImageFormatInfo2::default()
                    .format(FORMAT)
                    .ty(vk::ImageType::TYPE_2D)
                    .tiling(tiling)
                    .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                    .push_next(&mut external);
                let mut external_props = vk::ExternalImageFormatProperties::default();
                let mut props = vk::ImageFormatProperties2::default().push_next(&mut external_props);
                match self
                    .instance
                    .get_physical_device_image_format_properties2(self.pdevice, &info, &mut props)
                {
                    Ok(()) => eprintln!(
                        "win32 probe: {name} {tiling_name} rgba8: {}",
                        feature_text(external_props.external_memory_properties.external_memory_features)
                    ),
                    Err(err) => eprintln!("win32 probe: {name} {tiling_name} rgba8: {err}"),
                }
            }
        }
        let sync = [
            ("d3d12 fence", vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE),
            ("opaque win32 semaphore", vk::ExternalSemaphoreHandleTypeFlags::OPAQUE_WIN32),
        ];
        for (name, handle_type) in sync {
            let info = vk::PhysicalDeviceExternalSemaphoreInfo::default().handle_type(handle_type);
            let mut props = vk::ExternalSemaphoreProperties::default();
            self.instance
                .get_physical_device_external_semaphore_properties(self.pdevice, &info, &mut props);
            let mut parts = Vec::new();
            if props.external_semaphore_features.contains(vk::ExternalSemaphoreFeatureFlags::EXPORTABLE) {
                parts.push("export");
            }
            if props.external_semaphore_features.contains(vk::ExternalSemaphoreFeatureFlags::IMPORTABLE) {
                parts.push("import");
            }
            let text = if parts.is_empty() { "none".to_string() } else { parts.join(" ") };
            eprintln!("win32 probe: {name}: {text}");
        }
    }

    /// `VK_EXT_external_memory_host` imports an existing allocation, such as a numpy buffer,
    /// as device memory. Print whether it is advertised, the required pointer alignment, and
    /// whether a host-allocation buffer can be imported.
    unsafe fn probe_host_pointer(&self) {
        let advertised = self
            .instance
            .enumerate_device_extension_properties(self.pdevice)
            .map(|list| {
                list.iter().any(|e| {
                    e.extension_name_as_c_str() == Ok(ash::ext::external_memory_host::NAME)
                })
            })
            .unwrap_or(false);
        if !advertised {
            eprintln!("host pointer probe: VK_EXT_external_memory_host is not advertised");
            return;
        }
        let mut host = vk::PhysicalDeviceExternalMemoryHostPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut host);
        self.instance
            .get_physical_device_properties2(self.pdevice, &mut props);
        eprintln!(
            "host pointer probe: min imported pointer alignment {} bytes",
            host.min_imported_host_pointer_alignment
        );
        let info = vk::PhysicalDeviceExternalBufferInfo::default()
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
        let mut external = vk::ExternalBufferProperties::default();
        self.instance
            .get_physical_device_external_buffer_properties(self.pdevice, &info, &mut external);
        eprintln!(
            "host pointer probe: host allocation buffer: {}",
            feature_text(external.external_memory_properties.external_memory_features)
        );
    }

    #[cfg(not(target_os = "linux"))]
    fn choose_dmabuf(&self) -> Option<DmaChoice> {
        None
    }

    /// Print the dma-buf images this device will export and import, and pick the first one
    /// `submit_dmabuf` can sample. Modifier 0 with DRM-modifier tiling is still linear RGBA.
    #[cfg(target_os = "linux")]
    unsafe fn choose_dmabuf(&self) -> Option<DmaChoice> {
        if self.memory_fd.is_none() {
            eprintln!(
                "dma-buf probe: device does not advertise VK_KHR_external_memory_fd and VK_EXT_external_memory_dma_buf"
            );
            return None;
        }
        if self.drm_modifier {
            self.print_modifiers();
        } else {
            eprintln!("dma-buf probe: VK_EXT_image_drm_format_modifier is not advertised");
        }
        let mut chosen = None;
        for candidate in self.dma_candidates() {
            let features = crate::dmabuf::dma_external_features(
                &self.instance,
                self.pdevice,
                candidate.format,
                candidate.query_tiling,
                candidate.modifier,
            );
            match features {
                Ok(features) => {
                    eprintln!("dma-buf probe: {}: {}", candidate.label, feature_text(features));
                    let usable = features.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE)
                        && features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE);
                    if usable && chosen.is_none() && candidate.selectable {
                        chosen = Some(DmaChoice {
                            format: candidate.format,
                            tiling: candidate.tiling,
                            label: candidate.label,
                        });
                    }
                }
                Err(err) => eprintln!("dma-buf probe: {}: {err}", candidate.label),
            }
        }
        match &chosen {
            Some(choice) => eprintln!("dma-buf sample uses {}", choice.label),
            None => eprintln!("dma-buf sample: no probed image is both exportable and importable"),
        }
        chosen
    }

    #[cfg(target_os = "linux")]
    fn dma_candidates(&self) -> Vec<DmaProbe> {
        use crate::dmabuf::ImportTiling;
        let mut candidates = vec![
            DmaProbe::new(
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::LINEAR,
                None,
                ImportTiling::Linear,
                "R8G8B8A8_UNORM linear",
                true,
            ),
            DmaProbe::new(
                vk::Format::B8G8R8A8_UNORM,
                vk::ImageTiling::LINEAR,
                None,
                ImportTiling::Linear,
                "B8G8R8A8_UNORM linear",
                true,
            ),
            DmaProbe::new(
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::OPTIMAL,
                None,
                ImportTiling::Linear,
                "R8G8B8A8_UNORM optimal",
                false,
            ),
        ];
        if self.drm_modifier {
            candidates.push(DmaProbe::new(
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                Some(0),
                ImportTiling::DrmLinear,
                "R8G8B8A8_UNORM drm modifier 0",
                true,
            ));
            candidates.push(DmaProbe::new(
                vk::Format::B8G8R8A8_UNORM,
                vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                Some(0),
                ImportTiling::DrmLinear,
                "B8G8R8A8_UNORM drm modifier 0",
                true,
            ));
        }
        candidates
    }

    #[cfg(target_os = "linux")]
    unsafe fn print_modifiers(&self) {
        let format = vk::Format::R8G8B8A8_UNORM;
        let mut count_list = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut props = vk::FormatProperties2::default().push_next(&mut count_list);
        self.instance
            .get_physical_device_format_properties2(self.pdevice, format, &mut props);
        let count = count_list.drm_format_modifier_count;
        if count == 0 {
            eprintln!("dma-buf modifiers for R8G8B8A8_UNORM: none");
            return;
        }
        let mut modifiers = vec![vk::DrmFormatModifierPropertiesEXT::default(); count as usize];
        let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
            .drm_format_modifier_properties(&mut modifiers);
        let mut props = vk::FormatProperties2::default().push_next(&mut list);
        self.instance
            .get_physical_device_format_properties2(self.pdevice, format, &mut props);
        eprintln!("dma-buf modifiers for R8G8B8A8_UNORM: {count}");
        for modifier in modifiers.iter().take(12) {
            let sampled = modifier
                .drm_format_modifier_tiling_features
                .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE);
            eprintln!(
                "  modifier {:#x} planes {} sampled {sampled}",
                modifier.drm_format_modifier, modifier.drm_format_modifier_plane_count
            );
        }
    }

    /// Export a sampled modifier-0 image as a dma-buf and import it through `dmabuf::import`.
    /// `vkAllocateMemory` on the import consumes the fd.
    #[cfg(target_os = "linux")]
    unsafe fn export_chosen(
        &self,
        loader: &ash::khr::external_memory_fd::Device,
        width: u32,
        height: u32,
        pixels: &[u8],
        choice: &DmaChoice,
    ) -> Result<OwnedImport, Error> {
        let image = self.create_export_image(choice, width, height)?;
        let mut export = Exported {
            device: self.device.clone(),
            image,
            memory: vk::DeviceMemory::null(),
        };
        let mut dedicated_req = vk::MemoryDedicatedRequirements::default();
        let mut requirements2 = vk::MemoryRequirements2::default().push_next(&mut dedicated_req);
        self.device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(image),
            &mut requirements2,
        );
        let requirements = requirements2.memory_requirements;
        let index = memory_type_for_export(&self.memory_properties, requirements.memory_type_bits)
            .ok_or(Error::DmaBufMemory)?;
        let mut export_info = vk::ExportMemoryAllocateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(image);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index)
            .push_next(&mut export_info)
            .push_next(&mut dedicated_info);
        let memory = self.device.allocate_memory(&alloc_info, None)?;
        // `Exported` implements `Drop`, so the memory field is filled in place.
        export.memory = memory;
        self.device.bind_image_memory(image, memory, 0)?;

        let layout = self.device.get_image_subresource_layout(
            image,
            vk::ImageSubresource::default().aspect_mask(vk::ImageAspectFlags::COLOR),
        );
        let stride = u32::try_from(layout.row_pitch).map_err(|_| Error::DmaBufStride {
            stride: 0,
            row_pitch: layout.row_pitch,
        })?;
        if layout.offset != 0 || u64::from(stride) < u64::from(width) * 4 {
            return Err(Error::DmaBufStride {
                stride,
                row_pitch: layout.row_pitch,
            });
        }
        let flags = self.memory_properties.memory_types[index as usize].property_flags;
        if flags.contains(
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            let mapped =
                self.device
                    .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                    as *mut u8;
            let base = mapped.add(layout.offset as usize);
            let tight = width as usize * 4;
            for y in 0..height as usize {
                std::ptr::copy_nonoverlapping(
                    pixels[y * tight..].as_ptr(),
                    base.add(y * layout.row_pitch as usize),
                    tight,
                );
            }
            self.device.unmap_memory(memory);
        } else {
            eprintln!("{width}x{height} dma-buf memory is not host-visible; the sample reads uninitialized pixels");
        }

        let fd = loader.get_memory_fd(
            &vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT),
        )?;
        let imported = dmabuf::import(
            &self.device,
            loader,
            &self.memory_properties,
            DmaBufImport::new(fd, width, height, stride, 0),
            choice.format,
            choice.tiling,
        )?;
        drop(export);
        Ok(OwnedImport {
            device: self.device.clone(),
            image: Some(imported),
        })
    }

    #[cfg(target_os = "linux")]
    unsafe fn create_export_image(
        &self,
        choice: &DmaChoice,
        width: u32,
        height: u32,
    ) -> Result<vk::Image, Error> {
        if choice.tiling == crate::dmabuf::ImportTiling::Linear {
            return Ok(crate::dmabuf::create_dma_image(
                &self.device,
                choice.format,
                width,
                height,
                0,
                choice.tiling,
            )?);
        }
        let packed = width.saturating_mul(4);
        let mut strides = vec![packed];
        for align in [64u32, 256] {
            let aligned = packed.saturating_add(align - 1) & !(align - 1);
            if !strides.contains(&aligned) {
                strides.push(aligned);
            }
        }
        let mut last = vk::Result::ERROR_UNKNOWN;
        for stride in strides {
            match crate::dmabuf::create_dma_image(
                &self.device,
                choice.format,
                width,
                height,
                stride,
                choice.tiling,
            ) {
                Ok(image) => return Ok(image),
                Err(err) => last = err,
            }
        }
        Err(last.into())
    }
}

#[cfg(target_os = "linux")]
struct DmaProbe {
    format: vk::Format,
    query_tiling: vk::ImageTiling,
    modifier: Option<u64>,
    tiling: crate::dmabuf::ImportTiling,
    label: &'static str,
    selectable: bool,
}

#[cfg(target_os = "linux")]
impl DmaProbe {
    fn new(
        format: vk::Format,
        query_tiling: vk::ImageTiling,
        modifier: Option<u64>,
        tiling: crate::dmabuf::ImportTiling,
        label: &'static str,
        selectable: bool,
    ) -> Self {
        Self { format, query_tiling, modifier, tiling, label, selectable }
    }
}

#[cfg(any(target_os = "linux", windows))]
fn feature_text(flags: vk::ExternalMemoryFeatureFlags) -> String {
    let mut parts = Vec::new();
    if flags.contains(vk::ExternalMemoryFeatureFlags::EXPORTABLE) {
        parts.push("export");
    }
    if flags.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE) {
        parts.push("import");
    }
    if flags.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY) {
        parts.push("dedicated");
    }
    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(" ")
    }
}

impl Drop for Bench {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some((buffer, memory, _)) = self.readback {
                self.device.unmap_memory(memory);
                self.device.destroy_buffer(buffer, None);
                self.device.free_memory(memory, None);
            }
            self.pipeline.destroy(&self.device);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

struct OwnedTexture {
    device: Device,
    texture: ViewportTexture,
}

impl Drop for OwnedTexture {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.texture.destroy(&self.device);
        }
    }
}

struct OwnedTarget {
    device: Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

impl OwnedTarget {
    unsafe fn new(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let image = device.create_image(
            &vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(FORMAT)
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC),
            None,
        )?;
        let requirements = device.get_image_memory_requirements(image);
        let index = find_memory_type_index(
            memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .ok_or(Error::NoDeviceLocalTextureMemory)?;
        let memory = match device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(index),
            None,
        ) {
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
        let view = match device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(FORMAT)
                .subresource_range(color_range()),
            None,
        ) {
            Ok(view) => view,
            Err(err) => {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
                return Err(err.into());
            }
        };
        Ok(Self {
            device: device.clone(),
            image,
            memory,
            view,
        })
    }
}

impl Drop for OwnedTarget {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_image_view(self.view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

struct SetGuard<'a> {
    device: &'a Device,
    pipeline: &'a ViewportPipeline,
    set: vk::DescriptorSet,
}

impl<'a> SetGuard<'a> {
    unsafe fn new(device: &'a Device, pipeline: &'a ViewportPipeline) -> Result<Self, Error> {
        Ok(Self {
            device,
            pipeline,
            set: pipeline.alloc_set(device)?,
        })
    }
}

impl Drop for SetGuard<'_> {
    fn drop(&mut self) {
        unsafe { self.pipeline.free_set(self.device, self.set) };
    }
}

#[cfg(target_os = "linux")]
struct Exported {
    device: Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

#[cfg(target_os = "linux")]
impl Drop for Exported {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_image(self.image, None);
            if self.memory != vk::DeviceMemory::null() {
                self.device.free_memory(self.memory, None);
            }
        }
    }
}

#[cfg(target_os = "linux")]
struct OwnedImport {
    device: Device,
    image: Option<ImportedImage>,
}

#[cfg(target_os = "linux")]
impl Drop for OwnedImport {
    fn drop(&mut self) {
        if let Some(image) = self.image.take() {
            unsafe {
                let _ = self.device.device_wait_idle();
                image.destroy(&self.device);
            }
        }
    }
}

fn time(label: &str, mut body: impl FnMut() -> Result<(), Error>) -> Result<(), Error> {
    let mut samples = Vec::with_capacity(FRAMES);
    for i in 0..WARMUP + FRAMES {
        let start = Instant::now();
        body()?;
        if i >= WARMUP {
            samples.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
    }
    summarize(label, samples);
    Ok(())
}

fn summarize(label: &str, mut samples: Vec<f64>) {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    eprintln!(
        "{label}: n={} min={:.3} median={:.3} mean={:.3} max={:.3} ms",
        samples.len(),
        samples[0],
        samples[samples.len() / 2],
        mean,
        samples[samples.len() - 1],
    );
}

fn solid(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = vec![0u8; width as usize * height as usize * 4];
    for px in pixels.chunks_exact_mut(4) {
        px.copy_from_slice(&[220, 40, 40, 255]);
    }
    pixels
}

fn device_name(props: &vk::PhysicalDeviceProperties) -> String {
    unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn is_software(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("llvmpipe") || name.contains("lavapipe") || name.contains("swiftshader")
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1)
}

unsafe fn create_readback(
    device: &Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8), Error> {
    let buffer = device.create_buffer(
        &vk::BufferCreateInfo::default()
            .size(256)
            .usage(vk::BufferUsageFlags::TRANSFER_DST),
        None,
    )?;
    let requirements = device.get_buffer_memory_requirements(buffer);
    let Some(index) = find_memory_type_index(
        memory_properties,
        requirements.memory_type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    ) else {
        device.destroy_buffer(buffer, None);
        return Err(Error::NoHostVisibleBufferMemory);
    };
    let memory = match device.allocate_memory(
        &vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index),
        None,
    ) {
        Ok(memory) => memory,
        Err(err) => {
            device.destroy_buffer(buffer, None);
            return Err(err.into());
        }
    };
    if let Err(err) = device.bind_buffer_memory(buffer, memory, 0) {
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);
        return Err(err.into());
    }
    let mapped = match device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) {
        Ok(mapped) => mapped as *mut u8,
        Err(err) => {
            device.destroy_buffer(buffer, None);
            device.free_memory(memory, None);
            return Err(err.into());
        }
    };
    Ok((buffer, memory, mapped))
}

#[cfg(target_os = "linux")]
fn memory_type_for_export(
    properties: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
) -> Option<u32> {
    find_memory_type_index(
        properties,
        type_bits,
        vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    )
    .or_else(|| {
        find_memory_type_index(properties, type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
    })
    .or_else(|| find_memory_type_index(properties, type_bits, vk::MemoryPropertyFlags::empty()))
}

#[cfg(all(windows, test))]
#[path = "d3d_test.rs"]
mod d3d11_tests;
