//! Minimal winit+Vulkan program that mirrors `fastgui-render-vk`'s actual architecture (the
//! self-perpetuating `ControlFlow::Poll` render loop, deferred-resize-until-settled, per-image
//! views/semaphores, real `acquire`→`submit`→`present` each frame) but with none of fastgui's
//! own widget/chrome/CUDA code — exists to answer one question during M6 debugging (see
//! ROADMAP.md's M6 status): is the ~56-second live-resize stall specific to fastgui's own code,
//! or does *anything* built this way, on this machine, have it?
//!
//! Run with `cargo run --example resize_probe -p fastgui-render-vk`, then drag the window's
//! border. Prints when a swapchain recreation happens and how long it took.

use std::time::Instant;

use ash::khr::{surface, swapchain};
use ash::{vk, Device, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

const FRAMES_IN_FLIGHT: usize = 2;

struct Vulkan {
    _entry: Entry,
    _instance: Instance,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
    device: Device,
    queue: vk::Queue,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    // Kept alive for its owned `command_buffers` (destroying the pool would invalidate them);
    // never read directly.
    _command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
    frame: usize,
}

unsafe fn create_image_view(device: &Device, image: vk::Image, format: vk::Format) -> vk::ImageView {
    let create_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    device.create_image_view(&create_info, None).expect("create_image_view")
}

impl Vulkan {
    unsafe fn new(window: &Window, width: u32, height: u32) -> Self {
        let entry = Entry::load().expect("load Vulkan");
        let display_handle = window.display_handle().unwrap().as_raw();
        let window_handle = window.window_handle().unwrap().as_raw();

        let extension_names = ash_window::enumerate_required_extensions(display_handle).unwrap().to_vec();
        let app_name = c"resize_probe";
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .engine_name(app_name)
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names);
        let instance = entry.create_instance(&instance_create_info, None).expect("create instance");

        let surface = ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
            .expect("create surface");
        let surface_loader = surface::Instance::new(&entry, &instance);

        let (pdevice, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .into_iter()
            .filter(|&pdevice| {
                let props = instance.get_physical_device_properties(pdevice);
                if props.api_version < vk::make_api_version(0, 1, 3, 0) {
                    return false;
                }
                let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
                let mut features2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
                instance.get_physical_device_features2(pdevice, &mut features2);
                features13.dynamic_rendering == vk::TRUE
            })
            .find_map(|pdevice| {
                instance
                    .get_physical_device_queue_family_properties(pdevice)
                    .iter()
                    .enumerate()
                    .find(|(index, info)| {
                        info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                            && surface_loader
                                .get_physical_device_surface_support(pdevice, *index as u32, surface)
                                .unwrap_or(false)
                    })
                    .map(|(index, _)| (pdevice, index as u32))
            })
            .expect("no suitable physical device");

        let queue_priorities = [1.0f32];
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&queue_priorities);
        let device_extension_names_raw = vec![swapchain::NAME.as_ptr()];
        let mut dynamic_rendering_features = vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&device_extension_names_raw)
            .push_next(&mut dynamic_rendering_features);
        let device = instance.create_device(pdevice, &device_create_info, None).expect("create device");
        let queue = device.get_device_queue(queue_family_index, 0);
        let swapchain_loader = swapchain::Device::new(&instance, &device);

        let surface_formats = surface_loader.get_physical_device_surface_formats(pdevice, surface).unwrap();
        let surface_format = surface_formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_UNORM && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or(surface_formats.first())
            .copied()
            .expect("no surface format");

        let command_pool_create_info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_index);
        let command_pool = device.create_command_pool(&command_pool_create_info, None).expect("create_command_pool");
        let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(FRAMES_IN_FLIGHT as u32);
        let command_buffers = device.allocate_command_buffers(&command_buffer_alloc_info).expect("allocate_command_buffers");

        let image_available = (0..FRAMES_IN_FLIGHT)
            .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
            .collect::<Vec<_>>();
        let in_flight = (0..FRAMES_IN_FLIGHT)
            .map(|_| {
                device
                    .create_fence(&vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED), None)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let mut vk = Self {
            _entry: entry,
            _instance: instance,
            surface_loader,
            surface,
            pdevice,
            device,
            queue,
            swapchain_loader,
            swapchain: vk::SwapchainKHR::null(),
            surface_format,
            extent: vk::Extent2D { width, height },
            images: Vec::new(),
            image_views: Vec::new(),
            _command_pool: command_pool,
            command_buffers,
            image_available,
            render_finished: Vec::new(),
            in_flight,
            images_in_flight: Vec::new(),
            frame: 0,
        };
        vk.recreate_swapchain(width, height);
        vk
    }

    /// Mirrors `VulkanRenderer::recreate_swapchain` exactly, including per-image views and
    /// `render_finished` semaphores this simplified first version of the probe skipped.
    unsafe fn recreate_swapchain(&mut self, width: u32, height: u32) {
        self.recreate_swapchain_ex(width, height, true, "resize");
    }

    /// `use_old_swapchain_hint`: when false, destroys the old swapchain *before* creating the new
    /// one (passing `old_swapchain: null`) instead of the usual smooth-handoff pattern (create
    /// new with `old_swapchain` set, then destroy old) — isolates whether requesting that handoff
    /// is itself what costs the ~2s external delay, vs. any create/destroy cycle. `label` is just
    /// for the printed log line, to tell scripted test recreates apart from real resizes.
    unsafe fn recreate_swapchain_ex(&mut self, width: u32, height: u32, use_old_swapchain_hint: bool, label: &str) {
        if width == 0 || height == 0 {
            return;
        }
        let t0 = Instant::now();
        self.device.queue_wait_idle(self.queue).expect("queue_wait_idle");
        let t_idle = t0.elapsed();

        let capabilities =
            self.surface_loader.get_physical_device_surface_capabilities(self.pdevice, self.surface).unwrap();
        let extent = if capabilities.current_extent.width == u32::MAX {
            vk::Extent2D { width, height }
        } else {
            capabilities.current_extent
        };
        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
            image_count = capabilities.max_image_count;
        }

        let old_swapchain = self.swapchain;
        if !use_old_swapchain_hint && old_swapchain != vk::SwapchainKHR::null() {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }
        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(self.surface)
            .min_image_count(image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(if use_old_swapchain_hint { old_swapchain } else { vk::SwapchainKHR::null() });
        let t_create = Instant::now();
        let new_swapchain = self.swapchain_loader.create_swapchain(&create_info, None).expect("create_swapchain");
        let t_create = t_create.elapsed();

        if use_old_swapchain_hint && old_swapchain != vk::SwapchainKHR::null() {
            self.swapchain_loader.destroy_swapchain(old_swapchain, None);
        }
        self.swapchain = new_swapchain;
        self.extent = extent;
        self.images = self.swapchain_loader.get_swapchain_images(self.swapchain).unwrap();

        for &view in &self.image_views {
            self.device.destroy_image_view(view, None);
        }
        self.image_views =
            self.images.iter().map(|&image| create_image_view(&self.device, image, self.surface_format.format)).collect();

        for &s in &self.render_finished {
            self.device.destroy_semaphore(s, None);
        }
        self.render_finished = (0..self.images.len())
            .map(|_| self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None).unwrap())
            .collect();
        self.images_in_flight = vec![vk::Fence::null(); self.images.len()];

        let total = t0.elapsed();
        println!(
            "[resize_probe] recreate_swapchain[{label}]({width}x{height}, old_swapchain_hint={use_old_swapchain_hint}): queue_wait_idle={t_idle:?} create={t_create:?} total={total:?}"
        );
    }

    /// Mirrors `VulkanRenderer::render_frame` — clear-color only, no draw calls (matches
    /// `basic_window.py`'s `ActiveContent::None` case, which is what showed the ~56s stall).
    unsafe fn render_frame(&mut self) {
        if self.images.is_empty() {
            return;
        }
        let fence = self.in_flight[self.frame];
        self.device.wait_for_fences(&[fence], true, u64::MAX).unwrap();

        let image_index = match self.swapchain_loader.acquire_next_image(
            self.swapchain,
            u64::MAX,
            self.image_available[self.frame],
            vk::Fence::null(),
        ) {
            Ok((index, _suboptimal)) => index,
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                println!("[resize_probe] acquire_next_image: OUT_OF_DATE, recreating");
                return self.recreate_swapchain(self.extent.width, self.extent.height);
            }
            Err(e) => panic!("acquire_next_image: {e:?}"),
        };

        let image_fence = self.images_in_flight[image_index as usize];
        if image_fence != vk::Fence::null() {
            self.device.wait_for_fences(&[image_fence], true, u64::MAX).unwrap();
        }
        self.images_in_flight[image_index as usize] = fence;
        self.device.reset_fences(&[fence]).unwrap();

        let cmd = self.command_buffers[self.frame];
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty()).unwrap();
        self.device
            .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
            .unwrap();

        let image = self.images[image_index as usize];
        let color_subresource =
            vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let to_color = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_subresource)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_color],
        );

        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(self.image_views[image_index as usize])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.05, 0.06, 0.08, 1.0] } })];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent })
            .layer_count(1)
            .color_attachments(&color_attachments);
        self.device.cmd_begin_rendering(cmd, &rendering_info);
        self.device.cmd_end_rendering(cmd);

        let to_present = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_subresource)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_present],
        );
        self.device.end_command_buffer(cmd).unwrap();

        let wait_semaphores = [self.image_available[self.frame]];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [self.render_finished[image_index as usize]];
        let cmds = [cmd];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&cmds)
            .signal_semaphores(&signal_semaphores);
        self.device.queue_submit(self.queue, &[submit_info], fence).unwrap();

        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let present_info =
            vk::PresentInfoKHR::default().wait_semaphores(&signal_semaphores).swapchains(&swapchains).image_indices(&image_indices);
        match self.swapchain_loader.queue_present(self.queue, &present_info) {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                println!("[resize_probe] queue_present: OUT_OF_DATE, recreating");
                self.recreate_swapchain(self.extent.width, self.extent.height);
            }
            Err(e) => panic!("queue_present: {e:?}"),
        }

        self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
    }
}

/// TEST: cap the render loop instead of calling `render_frame` unconditionally on every `Poll`
/// iteration — see the module doc / ROADMAP.md's M6 status for why this is suspected to matter.
const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1000 / 60);

/// TEST: only actually recreate the swapchain after this long *without* a new `Resized` event —
/// i.e. once per resize gesture, not once per `WM_SIZE`. See ROADMAP.md's M6 status: the finer
/// instrumentation showed each recreation costs ~2s of externally-imposed blocking (RedrawRequested
/// essentially stops firing for ~2s after every recreate_swapchain call, not anything measurable
/// inside our own Vulkan calls), so cutting the number of recreations, not their individual cost,
/// is the only lever actually available.
const RESIZE_DEBOUNCE: std::time::Duration = std::time::Duration::from_millis(150);

struct App {
    window: Option<Window>,
    vk: Option<Vulkan>,
    width: u32,
    height: u32,
    last_applied_size: (u32, u32),
    last_frame: Instant,
    start: Instant,
    resized_count: u32,
    redraw_count: u32,
    last_log: Instant,
    last_resize_event: Instant,
    /// Scripted follow-up experiments (see `main`'s doc comment): after the window settles,
    /// force a same-size recreate (isolates "any recreate" vs "recreate with a real size change")
    /// and a no-`old_swapchain`-hint recreate (isolates whether the smooth-handoff request itself
    /// is what costs the ~2s, vs. any create/destroy cycle).
    ran_same_size_test: bool,
    ran_no_hint_test: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);
        let attrs = Window::default_attributes()
            .with_title("resize_probe")
            .with_inner_size(LogicalSize::new(self.width as f64, self.height as f64));
        let window = event_loop.create_window(attrs).expect("create window");
        let vk = unsafe { Vulkan::new(&window, self.width, self.height) };
        self.last_applied_size = (self.width, self.height);
        self.vk = Some(vk);
        self.window = Some(window);
        println!("[resize_probe] window up — drag its border to test resize latency");
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            // Same deferred-resize strategy as `fastgui-app`'s `MainResizePolicy::Debounced` `Resized`
            // handler: bookkeeping only, the actual recreation happens lazily inside the
            // self-perpetuating `RedrawRequested` loop below.
            WindowEvent::Resized(size) => {
                self.resized_count += 1;
                println!(
                    "[resize_probe] t={:?} Resized #{} -> {}x{}",
                    self.start.elapsed(),
                    self.resized_count,
                    size.width,
                    size.height
                );
                self.width = size.width;
                self.height = size.height;
                self.last_resize_event = Instant::now();
            }
            WindowEvent::RedrawRequested => {
                self.redraw_count += 1;
                if self.last_log.elapsed() >= std::time::Duration::from_millis(500) {
                    println!(
                        "[resize_probe] t={:?} RedrawRequested count so far: {}",
                        self.start.elapsed(),
                        self.redraw_count
                    );
                    self.last_log = Instant::now();
                }
                // Scripted follow-up experiments, run once each, 3s apart, well after any real
                // resize activity has settled — see `main`'s doc comment and the `App` fields'.
                if !self.ran_same_size_test && self.start.elapsed() >= std::time::Duration::from_secs(3) {
                    self.ran_same_size_test = true;
                    if let Some(vk) = &mut self.vk {
                        println!("[resize_probe] t={:?} running same-size-recreate test...", self.start.elapsed());
                        unsafe { vk.recreate_swapchain_ex(self.width, self.height, true, "same-size,hint") };
                    }
                } else if !self.ran_no_hint_test && self.start.elapsed() >= std::time::Duration::from_secs(6) {
                    self.ran_no_hint_test = true;
                    if let Some(vk) = &mut self.vk {
                        println!("[resize_probe] t={:?} running no-old-swapchain-hint test...", self.start.elapsed());
                        unsafe { vk.recreate_swapchain_ex(self.width, self.height, false, "same-size,no-hint") };
                    }
                }

                if let Some(vk) = &mut self.vk {
                    let settled = self.last_resize_event.elapsed() >= RESIZE_DEBOUNCE;
                    if self.last_applied_size != (self.width, self.height) && settled {
                        unsafe { vk.recreate_swapchain(self.width, self.height) };
                        self.last_applied_size = (self.width, self.height);
                    } else if self.last_applied_size == (self.width, self.height) && self.last_frame.elapsed() >= FRAME_INTERVAL {
                        unsafe { vk.render_frame() };
                        self.last_frame = Instant::now();
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().expect("create event loop");
    let now = Instant::now();
    let mut app = App {
        window: None,
        vk: None,
        width: 960,
        height: 600,
        last_applied_size: (0, 0),
        last_frame: now,
        start: now,
        resized_count: 0,
        redraw_count: 0,
        last_log: now,
        last_resize_event: now,
        ran_same_size_test: false,
        ran_no_hint_test: false,
    };
    event_loop.run_app(&mut app).expect("run event loop");
}
