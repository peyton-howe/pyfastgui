use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::{self, CStr};

use ash::{
    ext::debug_utils,
    khr::{self, surface, swapchain},
    vk, Device, Entry, Instance,
};
use fastgui_app::ViewportDraw;
use fastgui_core::{
    ChromeFrame, ChromeQuads, CpuFrame, D3d11Fence, D3d11Import, ImportedLayer, PixelRect, SharedFormat,
};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

use crate::cuda_texture::{CudaExportHandles, CudaExternal, CudaLayer};
use crate::d3d_import::{D3dSupport, ImportRing};
use crate::dmabuf::ImportedImage;
use crate::error::VkRendererError as Error;
use crate::pipeline::ViewportPipeline;
use crate::quad::{QuadChrome, QuadPipeline};
use crate::texture::ViewportTexture;

const FRAMES_IN_FLIGHT: usize = 2;

enum GpuImage {
    Cpu(ViewportTexture),
    Cuda(CudaLayer),
    Imported(ImportedImage),
}

struct SampledLayer {
    image: GpuImage,
    descriptor_set: vk::DescriptorSet,
}

/// Clears the swapchain to a solid color every frame and, once a viewport source has content,
/// draws it as a full-window textured quad on top — both via `VK_KHR_dynamic_rendering`, so
/// there's no render pass or per-swapchain-image framebuffer to manage.
pub struct VulkanRenderer {
    _entry: Entry,
    instance: Instance,
    debug: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    surface_loader: surface::Instance,
    surface: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    device: Device,
    queue: vk::Queue,
    swapchain_loader: swapchain::Device,
    swapchain: vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
    extent: vk::Extent2D,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available: Vec<vk::Semaphore>,
    render_finished: Vec<vk::Semaphore>,
    in_flight: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
    frame: usize,
    viewport_pipeline: ViewportPipeline,
    quad_pipeline: QuadPipeline,
    /// CPU-rasterized chrome (`set_chrome_frame`); cleared when quads are set, and vice versa.
    chrome: Option<SampledLayer>,
    /// Chrome as GPU quads (`set_chrome_quads`), the default path.
    quad_chrome: QuadChrome,
    /// GPU textures for in-tree `Viewport` widgets, keyed by the stable `viewport_id`.
    layers: HashMap<u64, SampledLayer>,
    /// D3D11 imports that are not on screen, per viewport, so resubmitting a texture the
    /// producer reuses costs a barrier instead of an import. See `set_imported_layer`.
    parked_imports: HashMap<u64, ImportRing<SampledLayer>>,
    // `None` when this device can't export memory CUDA can import (win32 on Windows, opaque fd
    // on Linux). `create_cuda_surface` reports that instead of failing device creation.
    cuda_interop: Option<CudaInteropLoaders>,
    /// `VK_EXT_external_memory_dma_buf` + `VK_KHR_external_memory_fd`, when the device has them.
    dmabuf: Option<khr::external_memory_fd::Device>,
    /// Tiling used to import a modifier-0 RGBA8 dma-buf. `None` when that image is not importable.
    dmabuf_tiling: Option<crate::dmabuf::ImportTiling>,
    /// What this device can import from Direct3D (Windows): textures, and shared fences.
    d3d11: Option<D3dSupport>,
    queue_family_index: u32,
    /// `VkPhysicalDeviceIDProperties::deviceUUID`, handed to CUDA to find the same GPU.
    device_uuid: [u8; 16],
}

enum CudaInteropLoaders {
    Win32 {
        external_memory_win32: khr::external_memory_win32::Device,
        external_semaphore_win32: khr::external_semaphore_win32::Device,
    },
    OpaqueFd {
        external_memory_fd: khr::external_memory_fd::Device,
        external_semaphore_fd: khr::external_semaphore_fd::Device,
    },
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    msg_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let message = if (*data).p_message.is_null() {
        std::borrow::Cow::from("")
    } else {
        ffi::CStr::from_ptr((*data).p_message).to_string_lossy()
    };
    eprintln!("[vulkan {severity:?}/{msg_type:?}] {message}");
    vk::FALSE
}

impl VulkanRenderer {
    pub fn new(
        window: &(impl HasWindowHandle + HasDisplayHandle),
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        unsafe {
            let entry = Entry::load()?;
            let display_handle = window.display_handle()?.as_raw();
            let window_handle = window.window_handle()?.as_raw();

            let available_layers = entry.enumerate_instance_layer_properties()?;
            let validation_name = c"VK_LAYER_KHRONOS_validation";
            let want_validation = available_layers.iter().any(|l| {
                CStr::from_ptr(l.layer_name.as_ptr()) == validation_name
            });
            let layer_names_raw: Vec<*const std::os::raw::c_char> = if want_validation {
                vec![validation_name.as_ptr()]
            } else {
                Vec::new()
            };

            let available_extensions = entry.enumerate_instance_extension_properties(None)?;
            let want_debug_utils = available_extensions.iter().any(|e| {
                CStr::from_ptr(e.extension_name.as_ptr()) == debug_utils::NAME
            });

            let mut extension_names = ash_window::enumerate_required_extensions(display_handle)?
                .to_vec();
            if want_debug_utils {
                extension_names.push(debug_utils::NAME.as_ptr());
            }
            // Portability drivers (MoltenVK on macOS) are only enumerated when asked for. Only
            // present on such systems, so this changes nothing on Windows/Linux drivers.
            let want_portability = available_extensions.iter().any(|e| {
                CStr::from_ptr(e.extension_name.as_ptr()) == khr::portability_enumeration::NAME
            });
            if want_portability {
                extension_names.push(khr::portability_enumeration::NAME.as_ptr());
            }

            let app_name = c"fastgui";
            let app_info = vk::ApplicationInfo::default()
                .application_name(app_name)
                .engine_name(app_name)
                .api_version(vk::make_api_version(0, 1, 3, 0));

            let instance_create_info = vk::InstanceCreateInfo::default()
                .application_info(&app_info)
                .enabled_layer_names(&layer_names_raw)
                .enabled_extension_names(&extension_names)
                .flags(if want_portability {
                    vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
                } else {
                    vk::InstanceCreateFlags::empty()
                });

            let instance = entry.create_instance(&instance_create_info, None)?;

            let debug = if want_debug_utils {
                let debug_utils_loader = debug_utils::Instance::new(&entry, &instance);
                let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(debug_callback));
                let messenger =
                    debug_utils_loader.create_debug_utils_messenger(&debug_info, None)?;
                Some((debug_utils_loader, messenger))
            } else {
                None
            };

            let surface =
                ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)?;
            let surface_loader = surface::Instance::new(&entry, &instance);

            // Requires Vulkan 1.3 (for core dynamic rendering) and the optional
            // `dynamicRendering` feature bit, on top of the usual graphics+present queue.
            let (pdevice, queue_family_index) = instance
                .enumerate_physical_devices()?
                .into_iter()
                .filter(|&pdevice| {
                    let props = instance.get_physical_device_properties(pdevice);
                    if props.api_version < vk::make_api_version(0, 1, 3, 0) {
                        return false;
                    }
                    let mut features13 = vk::PhysicalDeviceVulkan13Features::default();
                    let mut features2 =
                        vk::PhysicalDeviceFeatures2::default().push_next(&mut features13);
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
                                    .get_physical_device_surface_support(
                                        pdevice,
                                        *index as u32,
                                        surface,
                                    )
                                    .unwrap_or(false)
                        })
                        .map(|(index, _)| (pdevice, index as u32))
                })
                .ok_or(Error::NoSuitablePhysicalDevice)?;

            let memory_properties = instance.get_physical_device_memory_properties(pdevice);
            let mut id_properties = vk::PhysicalDeviceIDProperties::default();
            let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_properties);
            instance.get_physical_device_properties2(pdevice, &mut properties2);
            let device_uuid = id_properties.device_uuid;

            let available_device_extensions =
                instance.enumerate_device_extension_properties(pdevice)?;
            let has_device_extension = |name: &CStr| {
                available_device_extensions
                    .iter()
                    .any(|e| CStr::from_ptr(e.extension_name.as_ptr()) == name)
            };
            let cuda_win32 = cfg!(windows)
                && has_device_extension(khr::external_memory::NAME)
                && has_device_extension(khr::external_memory_win32::NAME)
                && has_device_extension(khr::external_semaphore::NAME)
                && has_device_extension(khr::external_semaphore_win32::NAME);
            let cuda_fd = cfg!(target_os = "linux")
                && has_device_extension(khr::external_memory::NAME)
                && has_device_extension(khr::external_memory_fd::NAME)
                && has_device_extension(khr::external_semaphore::NAME)
                && has_device_extension(khr::external_semaphore_fd::NAME);
            let dmabuf_import = has_device_extension(khr::external_memory::NAME)
                && has_device_extension(khr::external_memory_fd::NAME)
                && has_device_extension(ash::ext::external_memory_dma_buf::NAME);
            let d3d11_import = cfg!(windows)
                && has_device_extension(khr::external_memory::NAME)
                && has_device_extension(khr::external_memory_win32::NAME)
                && (crate::d3d_import::texture_importable(&instance, pdevice, SharedFormat::Rgba8)
                    || crate::d3d_import::texture_importable(&instance, pdevice, SharedFormat::Bgra8));
            let d3d_fence = d3d11_import
                && has_device_extension(khr::external_semaphore::NAME)
                && has_device_extension(khr::external_semaphore_win32::NAME)
                && crate::d3d_import::fence_importable(&instance, pdevice);
            let drm_modifier_ext = has_device_extension(ash::ext::image_drm_format_modifier::NAME);
            let dmabuf_tiling = if dmabuf_import {
                crate::dmabuf::rgba8_import_tiling(&instance, pdevice, drm_modifier_ext)
            } else {
                None
            };

            let mut device_extension_names_raw = vec![swapchain::NAME.as_ptr()];
            // Required on a portability driver whenever the device lists it.
            let portability_subset = c"VK_KHR_portability_subset";
            if has_device_extension(portability_subset) {
                device_extension_names_raw.push(portability_subset.as_ptr());
            }
            if cuda_win32 {
                device_extension_names_raw.extend([
                    khr::external_memory::NAME.as_ptr(),
                    khr::external_memory_win32::NAME.as_ptr(),
                    khr::external_semaphore::NAME.as_ptr(),
                    khr::external_semaphore_win32::NAME.as_ptr(),
                ]);
            } else if d3d11_import {
                device_extension_names_raw
                    .extend([khr::external_memory::NAME.as_ptr(), khr::external_memory_win32::NAME.as_ptr()]);
                if d3d_fence {
                    device_extension_names_raw.extend([
                        khr::external_semaphore::NAME.as_ptr(),
                        khr::external_semaphore_win32::NAME.as_ptr(),
                    ]);
                }
            }
            if cuda_fd || dmabuf_import {
                device_extension_names_raw.extend([
                    khr::external_memory::NAME.as_ptr(),
                    khr::external_memory_fd::NAME.as_ptr(),
                ]);
            }
            if cuda_fd {
                device_extension_names_raw.extend([
                    khr::external_semaphore::NAME.as_ptr(),
                    khr::external_semaphore_fd::NAME.as_ptr(),
                ]);
            }
            if dmabuf_import {
                device_extension_names_raw.push(ash::ext::external_memory_dma_buf::NAME.as_ptr());
            }
            if dmabuf_import && drm_modifier_ext {
                device_extension_names_raw.push(ash::ext::image_drm_format_modifier::NAME.as_ptr());
            }

            let queue_priorities = [1.0f32];
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&queue_priorities);
            let mut dynamic_rendering_features =
                vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
            // Timeline semaphores are core-1.2 functionality but still an opt-in feature bit —
            // needed for the CUDA-shared texture's export semaphore (see cuda_texture.rs),
            // enabled unconditionally since it's universally supported alongside 1.3.
            let mut timeline_semaphore_features =
                vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut dynamic_rendering_features)
                .push_next(&mut timeline_semaphore_features);

            let device = instance.create_device(pdevice, &device_create_info, None)?;
            let queue = device.get_device_queue(queue_family_index, 0);
            let swapchain_loader = swapchain::Device::new(&instance, &device);

            let cuda_interop = if cuda_win32 {
                Some(CudaInteropLoaders::Win32 {
                    external_memory_win32: khr::external_memory_win32::Device::new(&instance, &device),
                    external_semaphore_win32: khr::external_semaphore_win32::Device::new(&instance, &device),
                })
            } else if cuda_fd {
                Some(CudaInteropLoaders::OpaqueFd {
                    external_memory_fd: khr::external_memory_fd::Device::new(&instance, &device),
                    external_semaphore_fd: khr::external_semaphore_fd::Device::new(&instance, &device),
                })
            } else {
                None
            };
            let dmabuf = dmabuf_import.then(|| khr::external_memory_fd::Device::new(&instance, &device));
            let d3d11 = d3d11_import.then(|| D3dSupport {
                memory: khr::external_memory_win32::Device::new(&instance, &device),
                semaphore: d3d_fence.then(|| khr::external_semaphore_win32::Device::new(&instance, &device)),
                rgba8: crate::d3d_import::texture_importable(&instance, pdevice, SharedFormat::Rgba8),
                bgra8: crate::d3d_import::texture_importable(&instance, pdevice, SharedFormat::Bgra8),
                #[cfg(windows)]
                probe: crate::d3d_probe::for_device(&instance, pdevice),
            });

            // UNORM, deliberately not SRGB: an `_SRGB` surface format makes the GPU apply an
            // implicit linear->sRGB encoding curve to whatever the fragment shader outputs
            // before it lands in the swapchain image. Every color in this codebase (clear
            // colors, `CpuFrame` bytes from `Viewport.submit_frame`, `fastgui-chrome`'s
            // rasterized UI) is already meant to be the *final* displayed byte value, not a
            // linear light value — pairing that with an `_SRGB` target silently re-encodes
            // everything and washes out midtones (this is exactly what motivated this
            // comment: M4's UI backgrounds rendered visibly lighter than the colors specified
            // until this was UNORM instead of SRGB).
            let surface_formats =
                surface_loader.get_physical_device_surface_formats(pdevice, surface)?;
            let surface_format = surface_formats
                .iter()
                .find(|f| {
                    f.format == vk::Format::B8G8R8A8_UNORM
                        && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or(surface_formats.first())
                .copied()
                .ok_or(Error::NoSurfaceFormat)?;

            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            let command_pool = device.create_command_pool(&command_pool_create_info, None)?;

            let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(FRAMES_IN_FLIGHT as u32);
            let command_buffers = device.allocate_command_buffers(&command_buffer_alloc_info)?;

            let image_available = (0..FRAMES_IN_FLIGHT)
                .map(|_| device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<Vec<_>, _>>()?;
            let in_flight = (0..FRAMES_IN_FLIGHT)
                .map(|_| {
                    device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                        None,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;

            let viewport_pipeline = ViewportPipeline::new(&device, surface_format.format)?;
            let quad_pipeline =
                QuadPipeline::new(&device, surface_format.format, viewport_pipeline.descriptor_set_layout())?;

            let mut renderer = Self {
                _entry: entry,
                instance,
                debug,
                surface_loader,
                surface,
                pdevice,
                memory_properties,
                device,
                queue,
                swapchain_loader,
                swapchain: vk::SwapchainKHR::null(),
                surface_format,
                extent: vk::Extent2D { width, height },
                images: Vec::new(),
                image_views: Vec::new(),
                command_pool,
                command_buffers,
                image_available,
                render_finished: Vec::new(),
                in_flight,
                images_in_flight: Vec::new(),
                frame: 0,
                viewport_pipeline,
                quad_pipeline,
                quad_chrome: QuadChrome::new(FRAMES_IN_FLIGHT),
                chrome: None,
                layers: HashMap::new(),
                parked_imports: HashMap::new(),
                cuda_interop,
                dmabuf,
                dmabuf_tiling,
                d3d11,
                queue_family_index,
                device_uuid,
            };

            renderer.recreate_swapchain(width, height)?;
            Ok(renderer)
        }
    }

    fn recreate_swapchain(&mut self, width: u32, height: u32) -> Result<(), Error> {
        if width == 0 || height == 0 {
            // Window is minimized; keep the existing (possibly stale) swapchain and skip
            // rendering until a real resize arrives.
            return Ok(());
        }

        unsafe {
            // Must be `queue_wait_idle`/`device_wait_idle`, not a fence wait — tried the latter
            // during M6 development (see ROADMAP.md's M6 status) to avoid contending with DWM's
            // live-resize compositing on this shared integrated GPU, but it's actually *wrong*:
            // `in_flight` fences only track when a submitted command buffer's execution is done,
            // not when the subsequent `vkQueuePresentKHR` (a separate operation on the same
            // queue) has finished. That gap showed up as real validation errors —
            // `vkDestroySemaphore`/`vkDestroySwapchainKHR` "currently in use by VkQueue" — i.e.
            // genuine undefined behavior, destroying resources a still-in-flight present was
            // using. `queue_wait_idle` is the spec-correct primitive that actually covers
            // present completion, at the cost of being exactly as blocking as `device_wait_idle`
            // for this single-queue renderer (scoped to our one queue rather than "the device,"
            // which doesn't meaningfully differ here — this app never opens a second queue).
            self.device.queue_wait_idle(self.queue)?;

            let capabilities = self
                .surface_loader
                .get_physical_device_surface_capabilities(self.pdevice, self.surface)?;
            let extent = if capabilities.current_extent.width == u32::MAX {
                vk::Extent2D { width, height }
            } else {
                capabilities.current_extent
            };

            let mut image_count = capabilities.min_image_count + 1;
            if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
                image_count = capabilities.max_image_count;
            }

            // Present-mode choice turned out not to matter for the recreation-cost investigation
            // (see ROADMAP.md's M6 status) — back to preferring `MAILBOX` (lower input-to-photon
            // latency) when available, `FIFO` (universally supported) otherwise.
            let present_mode = self
                .surface_loader
                .get_physical_device_surface_present_modes(self.pdevice, self.surface)?
                .into_iter()
                .find(|&m| m == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            // Deliberately *not* passing `old_swapchain` — destroy the old one first, then create
            // fresh with `old_swapchain: null`, instead of the "normal"/recommended pattern (keep
            // the old one alive, pass it as a smooth-handoff hint, destroy it after). Root-caused
            // via `examples/resize_probe.rs` (see ROADMAP.md's M6 status, "avoiding the 2s
            // recreation" follow-up): requesting that handoff is specifically what costs
            // several seconds of externally-imposed blocking on this machine (almost certainly
            // DWM doing expensive synchronization to support it) — a probe test recreating at the
            // same size with the hint stalled `RedrawRequested` for ~2.5s afterward; the
            // *identical* recreate without the hint caused no measurable stall at all. Trade-off:
            // a brief window with no valid swapchain (a possible one-frame flicker) instead of a
            // multi-second freeze — and resize is already debounced to happen once per gesture,
            // not continuously, so that flicker is rare or invisible in practice.
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            }
            self.swapchain = vk::SwapchainKHR::null();

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
                .present_mode(present_mode)
                .clipped(true)
                .old_swapchain(vk::SwapchainKHR::null());

            let new_swapchain = self.swapchain_loader.create_swapchain(&create_info, None)?;
            self.swapchain = new_swapchain;
            self.extent = extent;
            self.images = self.swapchain_loader.get_swapchain_images(self.swapchain)?;

            for &view in &self.image_views {
                self.device.destroy_image_view(view, None);
            }
            self.image_views = self
                .images
                .iter()
                .map(|&image| {
                    device_create_image_view(&self.device, image, self.surface_format.format)
                })
                .collect::<Result<Vec<_>, _>>()?;

            for &s in &self.render_finished {
                self.device.destroy_semaphore(s, None);
            }
            self.render_finished = (0..self.images.len())
                .map(|_| self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None))
                .collect::<Result<Vec<_>, _>>()?;
            self.images_in_flight = vec![vk::Fence::null(); self.images.len()];
        }
        Ok(())
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.recreate_swapchain(width, height)
    }

    /// Copies only `frame.damage` when the existing texture already holds the previous frame
    /// at this size; see `SurfaceBackend::set_chrome_frame`.
    /// Take the chrome as quads + atlas uploads (see `fastgui_chrome::ChromeRenderer::build_quads`).
    pub fn set_chrome_quads(&mut self, frame: &ChromeQuads<'_>) -> Result<(), Error> {
        unsafe {
            if let Some(old) = self.chrome.take() {
                self.device.queue_wait_idle(self.queue)?;
                self.destroy_layer(old);
            }
            self.quad_chrome.update(&self.device, &self.memory_properties, &self.viewport_pipeline, frame)
        }
    }

    pub fn set_chrome_frame(&mut self, frame: &ChromeFrame<'_>) -> Result<(), Error> {
        if self.quad_chrome.is_active() {
            unsafe {
                self.device.device_wait_idle()?;
                self.quad_chrome.destroy(&self.device, &self.viewport_pipeline);
            }
        }
        let mut slot = self.chrome.take();
        let result = self.upsert_cpu_layer_slot(&mut slot, frame.width, frame.height, frame.data, frame.damage);
        self.chrome = slot;
        result
    }

    pub fn set_layer_frame(&mut self, viewport_id: u64, frame: CpuFrame) -> Result<(), Error> {
        let mut slot = self.layers.remove(&viewport_id);
        let result = self.upsert_cpu_layer_slot(&mut slot, frame.width, frame.height, &frame.data, None);
        if let Some(layer) = slot {
            self.layers.insert(viewport_id, layer);
        }
        // This viewport now shows CPU frames, so its parked D3D imports would only hold GPU memory.
        if let Some(mut ring) = self.parked_imports.remove(&viewport_id) {
            unsafe {
                let _ = self.device.queue_wait_idle(self.queue);
                for layer in ring.drain() {
                    self.destroy_layer(layer);
                }
            }
        }
        result
    }

    /// Write a tightly packed RGBA8 `width`x`height` frame into `slot`'s texture, (re)creating
    /// it on a size change. `damage`, when given, limits the copy to those rects — only valid
    /// when the texture already holds the previous frame, so it's ignored after a recreate.
    fn upsert_cpu_layer_slot(
        &mut self,
        slot: &mut Option<SampledLayer>,
        width: u32,
        height: u32,
        data: &[u8],
        damage: Option<&[PixelRect]>,
    ) -> Result<(), Error> {
        unsafe {
            let need_recreate = match slot {
                Some(SampledLayer { image: GpuImage::Cpu(texture), .. }) => {
                    texture.width != width || texture.height != height
                }
                Some(SampledLayer { image: GpuImage::Cuda(_) | GpuImage::Imported(_), .. }) | None => true,
            };

            if need_recreate {
                self.device.queue_wait_idle(self.queue)?;
                if let Some(old) = slot.take() {
                    self.destroy_layer(old);
                }
                let texture = ViewportTexture::new(&self.device, &self.memory_properties, width, height)?;
                let descriptor_set = self.viewport_pipeline.alloc_set(&self.device)?;
                self.viewport_pipeline.bind_texture(&self.device, descriptor_set, texture.view);
                *slot = Some(SampledLayer { image: GpuImage::Cpu(texture), descriptor_set });
            }

            if let Some(SampledLayer { image: GpuImage::Cpu(texture), .. }) = slot {
                match damage {
                    Some(rects) if !need_recreate => {
                        for rect in rects {
                            texture.upload_rect(data, *rect);
                        }
                    }
                    _ => texture.upload(data),
                }
            }
        }
        Ok(())
    }

    /// Create a fresh CUDA-importable texture for `viewport_id`, replacing whatever that
    /// layer was previously showing. Returns the raw Vulkan handles `fastgui-interop-cuda`
    /// needs to import it on the calling (Python) thread — this method only ever touches Vulkan.
    ///
    /// **Unverified** — see `cuda_texture` module docs.
    pub fn create_cuda_surface(
        &mut self,
        viewport_id: u64,
        width: u32,
        height: u32,
    ) -> Result<CudaExportHandles, Error> {
        if self.cuda_interop.is_none() {
            return Err(if cfg!(any(windows, target_os = "linux")) {
                Error::CudaInteropUnsupported
            } else {
                Error::CudaInteropNotImplemented
            });
        }
        unsafe {
            self.device.device_wait_idle()?;
            if let Some(old) = self.layers.remove(&viewport_id) {
                self.destroy_layer(old);
            }
            let external = match self.cuda_interop.as_ref().expect("checked above") {
                CudaInteropLoaders::Win32 { external_memory_win32, external_semaphore_win32 } => {
                    CudaExternal::Win32 { memory: external_memory_win32, semaphore: external_semaphore_win32 }
                }
                CudaInteropLoaders::OpaqueFd { external_memory_fd, external_semaphore_fd } => {
                    CudaExternal::OpaqueFd { memory: external_memory_fd, semaphore: external_semaphore_fd }
                }
            };
            let (texture, handles) = CudaLayer::new(
                &self.device,
                external,
                &self.memory_properties,
                self.device_uuid,
                width,
                height,
            )?;
            let descriptor_set = self.viewport_pipeline.alloc_set(&self.device)?;
            self.viewport_pipeline.bind_texture(&self.device, descriptor_set, texture.view);
            self.layers.insert(
                viewport_id,
                SampledLayer { image: GpuImage::Cuda(texture), descriptor_set },
            );
            Ok(handles)
        }
    }

    pub fn retain_layers(&mut self, live_ids: &[u64]) {
        let live: HashSet<u64> = live_ids.iter().copied().collect();
        let drop_ids: Vec<u64> = self.layers.keys().copied().filter(|id| !live.contains(id)).collect();
        let drop_parked: Vec<u64> = self.parked_imports.keys().copied().filter(|id| !live.contains(id)).collect();
        if drop_ids.is_empty() && drop_parked.is_empty() {
            return;
        }
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue);
            for id in drop_ids {
                if let Some(layer) = self.layers.remove(&id) {
                    self.destroy_layer(layer);
                }
            }
            for id in drop_parked {
                if let Some(mut ring) = self.parked_imports.remove(&id) {
                    for layer in ring.drain() {
                        self.destroy_layer(layer);
                    }
                }
            }
        }
    }

    pub fn set_imported_layer(&mut self, viewport_id: u64, layer: ImportedLayer) -> Result<(), Error> {
        let imported = match layer {
            ImportedLayer::DmaBuf(buf) => {
                let (Some(memory_fd), Some(tiling)) = (&self.dmabuf, self.dmabuf_tiling) else {
                    return Ok(());
                };
                let imported = unsafe {
                    crate::dmabuf::import(
                        &self.device,
                        memory_fd,
                        &self.memory_properties,
                        buf,
                        vk::Format::R8G8B8A8_UNORM,
                        tiling,
                    )
                };
                match imported {
                    Ok(imported) => imported,
                    // A bad fd or stride must not tear down the window. The previous frame stays.
                    Err(_) => return Ok(()),
                }
            }
            ImportedLayer::D3d11(texture) => {
                unsafe { self.submit_d3d11(viewport_id, texture) };
                return Ok(());
            }
            ImportedLayer::Metal { .. } => return Ok(()),
        };
        unsafe {
            let _ = self.install_import(viewport_id, imported);
        }
        Ok(())
    }

    /// Show `imported` for `viewport_id` and retire what was there. On `Err` the previous frame
    /// stays and `imported` is already destroyed.
    unsafe fn install_import(&mut self, viewport_id: u64, imported: ImportedImage) -> Result<(), String> {
        // Allocate before retiring the previous layer so a pool failure keeps the old frame
        // and does not fail the window.
        let descriptor_set = match self.viewport_pipeline.alloc_set(&self.device) {
            Ok(set) => set,
            Err(err) => {
                imported.destroy(&self.device);
                return Err(format!("no descriptor set for the import: {err}"));
            }
        };
        if let Err(err) = self.device.queue_wait_idle(self.queue) {
            imported.destroy(&self.device);
            self.viewport_pipeline.free_set(&self.device, descriptor_set);
            return Err(format!("waiting for the queue failed: {err}"));
        }
        self.viewport_pipeline.bind_texture(&self.device, descriptor_set, imported.view);
        let new = SampledLayer { image: GpuImage::Imported(imported), descriptor_set };
        if let Some(old) = self.layers.insert(viewport_id, new) {
            self.retire_layer(viewport_id, old);
        }
        Ok(())
    }

    /// Show a shared D3D11 texture. A texture this viewport already imported (on screen or parked)
    /// is re-acquired instead of imported again; the producer's fence, when it gave one, is waited
    /// on first. Every outcome lands in `texture.status`, and a failure keeps the previous frame.
    unsafe fn submit_d3d11(&mut self, viewport_id: u64, mut texture: D3d11Import) {
        let status = texture.status.clone();
        let Some(support) = self.d3d11.clone() else {
            status.set("unavailable: this GPU or driver cannot import D3D11 textures (VK_KHR_external_memory_win32)");
            return;
        };
        let fence = texture.fence.take();
        if fence.is_some() && support.semaphore.is_none() {
            status.set("unavailable: this GPU cannot import D3D fences; submit without a fence");
            return;
        }
        let device = self.device.clone();
        let queue = self.queue;

        let shown = matches!(
            self.layers.get(&viewport_id),
            Some(SampledLayer { image: GpuImage::Imported(image), .. }) if image.is_import_of(&texture)
        );
        if shown {
            let Some(SampledLayer { image: GpuImage::Imported(image), .. }) = self.layers.get_mut(&viewport_id) else {
                return;
            };
            match sync_import(&device, queue, &support, image, fence) {
                // The producer rewrote the texture since we last sampled it, so the next draw
                // acquires it from the external queue family again.
                Ok(()) => {
                    image.acquire_external = true;
                    status.set("ok: showing a texture that was already imported");
                }
                Err(message) => status.set(message),
            }
            return;
        }
        let parked = self
            .parked_imports
            .get_mut(&viewport_id)
            .and_then(|ring| ring.find_mut(|layer| matches!(&layer.image, GpuImage::Imported(i) if i.is_import_of(&texture))));
        if let Some(SampledLayer { image: GpuImage::Imported(image), .. }) = parked {
            if let Err(err) = device.queue_wait_idle(queue) {
                status.set(format!("waiting for the queue failed: {err}"));
                return;
            }
            if let Err(message) = sync_import(&device, queue, &support, image, fence) {
                status.set(message);
                return;
            }
            image.acquire_external = true;
            let found = self
                .parked_imports
                .get_mut(&viewport_id)
                .and_then(|ring| ring.take(|layer| matches!(&layer.image, GpuImage::Imported(i) if i.is_import_of(&texture))));
            if let Some(found) = found {
                if let Some(old) = self.layers.insert(viewport_id, found) {
                    self.retire_layer(viewport_id, old);
                }
            }
            status.set("ok: showing a texture that was already imported");
            return;
        }

        // Vulkan takes any size for an imported texture, so read the real one from D3D first.
        let format = match check_texture(&support, &texture) {
            Ok(format) => format,
            Err(message) => {
                status.set(message);
                return;
            }
        };
        let (width, height) = (texture.width, texture.height);
        let mut imported =
            match crate::d3d_import::import(&device, &support.memory, &self.memory_properties, texture, format) {
                Ok(imported) => imported,
                Err(err) => {
                    status.set(format!("import failed: {err} ({format:?}, {width}x{height})"));
                    return;
                }
            };
        if let Err(message) = sync_import(&device, queue, &support, &mut imported, fence) {
            imported.destroy(&device);
            status.set(message);
            return;
        }
        match self.install_import(viewport_id, imported) {
            Ok(()) => status.set("ok: imported a new texture"),
            Err(message) => status.set(message),
        }
    }

    /// Take a layer off screen. A D3D import that keeps its handle is parked for reuse; anything
    /// else is destroyed. Callers have already waited for the GPU to finish with it.
    unsafe fn retire_layer(&mut self, viewport_id: u64, layer: SampledLayer) {
        let cacheable = matches!(&layer.image, GpuImage::Imported(img) if img.shared_handle.is_some());
        if !cacheable {
            self.destroy_layer(layer);
            return;
        }
        let evicted = self.parked_imports.entry(viewport_id).or_default().park(layer);
        if !evicted.is_empty() {
            let _ = self.device.queue_wait_idle(self.queue);
            for layer in evicted {
                self.destroy_layer(layer);
            }
        }
    }

    unsafe fn destroy_layer(&mut self, layer: SampledLayer) {
        match layer.image {
            GpuImage::Cpu(texture) => texture.destroy(&self.device),
            GpuImage::Cuda(texture) => texture.destroy(&self.device),
            GpuImage::Imported(texture) => texture.destroy(&self.device),
        }
        self.viewport_pipeline.free_set(&self.device, layer.descriptor_set);
    }

    pub fn render_frame(
        &mut self,
        clear_color: [f32; 4],
        draw_chrome: bool,
        viewports: &[ViewportDraw],
    ) -> Result<(), Error> {
        if self.images.is_empty() {
            return Ok(());
        }

        unsafe {
            let fence = self.in_flight[self.frame];
            self.device.wait_for_fences(&[fence], true, u64::MAX)?;

            let image_index = match self.swapchain_loader.acquire_next_image(
                self.swapchain,
                u64::MAX,
                self.image_available[self.frame],
                vk::Fence::null(),
            ) {
                Ok((index, _suboptimal)) => index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return self.recreate_swapchain(self.extent.width, self.extent.height);
                }
                // Window destroyed out from under us (xkill, compositor teardown, …). Skip the
                // frame; `WindowEvent::Destroyed` in fastgui-app exits (main) or runs the floater
                // close callback. Recreating needs a live surface.
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => return Ok(()),
                Err(e) => return Err(e.into()),
            };

            let image_fence = self.images_in_flight[image_index as usize];
            if image_fence != vk::Fence::null() {
                self.device.wait_for_fences(&[image_fence], true, u64::MAX)?;
            }
            self.images_in_flight[image_index as usize] = fence;
            self.device.reset_fences(&[fence])?;

            if draw_chrome {
                // This slot's fence was just waited on, so its instance buffer is free to rewrite.
                self.quad_chrome.prepare(&self.device, &self.memory_properties, self.frame)?;
            }

            let cmd = self.command_buffers[self.frame];
            self.device
                .reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            let image = self.images[image_index as usize];
            let color_subresource = vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1);

            let mut barriers = vec![vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(color_subresource)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

            // CPU-uploaded texture: publish this frame's (or a past frame's still-mapped) host
            // writes to the shader stage that samples them. `old_layout` starts at
            // `PREINITIALIZED` on first use and is `GENERAL` (a no-op layout-wise, but still a
            // real synchronization barrier) every frame after.
            let mut wait_semaphores = vec![self.image_available[self.frame]];
            let mut wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let mut wait_values = vec![0u64];

            // CUDA layers: copy the newest published slot into the sampled image, once CUDA's
            // `ready` signal says it is written. Only for layers drawn this frame; a hidden
            // layer's frame waits (or is replaced by a newer one) until it is shown.
            let mut pickups = Vec::new();
            for draw in viewports {
                if let Some(SampledLayer { image: GpuImage::Cuda(layer), .. }) =
                    self.layers.get_mut(&draw.viewport_id)
                {
                    if let Some((semaphore, value)) =
                        layer.record_pickup(&self.device, cmd, self.queue_family_index)
                    {
                        wait_semaphores.push(semaphore);
                        wait_stages.push(vk::PipelineStageFlags::TRANSFER);
                        wait_values.push(value);
                        pickups.push(draw.viewport_id);
                    }
                }
            }

            if draw_chrome {
                barriers.extend(self.quad_chrome.atlas_barrier(color_subresource));
                if let Some(layer) = &mut self.chrome {
                    sync_sampled_image(
                        &mut layer.image,
                        color_subresource,
                        self.queue_family_index,
                        &mut barriers,
                    );
                }
            }
            let graphics_family = self.queue_family_index;
            for draw in viewports {
                if let Some(layer) = self.layers.get_mut(&draw.viewport_id) {
                    sync_sampled_image(
                        &mut layer.image,
                        color_subresource,
                        graphics_family,
                        &mut barriers,
                    );
                }
            }

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE | vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );

            let color_attachments = [vk::RenderingAttachmentInfo::default()
                .image_view(self.image_views[image_index as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: clear_color } })];
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent })
                .layer_count(1)
                .color_attachments(&color_attachments);

            self.device.cmd_begin_rendering(cmd, &rendering_info);

            if draw_chrome {
                self.quad_chrome.record(&self.device, cmd, &self.quad_pipeline, self.frame, self.extent, false);
            }

            let bind_pipeline = draw_chrome && self.chrome.is_some()
                || viewports.iter().any(|draw| self.layers.contains_key(&draw.viewport_id));
            if bind_pipeline {
                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.viewport_pipeline.pipeline(),
                );
            }
            if draw_chrome {
                if let Some(layer) = &self.chrome {
                    self.draw_sampled(
                        cmd,
                        layer,
                        vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: self.extent.width as f32,
                            height: self.extent.height as f32,
                            min_depth: 0.0,
                            max_depth: 1.0,
                        },
                        vk::Rect2D { offset: vk::Offset2D::default(), extent: self.extent },
                    );
                }
            }
            for draw in viewports {
                let Some(layer) = self.layers.get(&draw.viewport_id) else { continue };
                if matches!(&layer.image, GpuImage::Cuda(cuda) if !cuda.has_content() && !pickups.contains(&draw.viewport_id)) {
                    continue;
                }
                let (cw, ch) = match &layer.image {
                    GpuImage::Cpu(texture) => (texture.width, texture.height),
                    GpuImage::Cuda(cuda) => cuda.size(),
                    GpuImage::Imported(imported) => (imported.width, imported.height),
                };
                let draw = draw.letterboxed(cw, ch);
                let Some(([x, y, width, height], [sx, sy, sw, sh])) =
                    draw.viewport_and_scissor(self.extent.width, self.extent.height)
                else {
                    continue;
                };
                let vp = vk::Viewport { x, y, width, height, min_depth: 0.0, max_depth: 1.0 };
                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: sx as i32, y: sy as i32 },
                    extent: vk::Extent2D { width: sw, height: sh },
                };
                self.draw_sampled(cmd, layer, vp, scissor);
            }
            // Popups go over video too.
            if draw_chrome {
                self.quad_chrome.record(&self.device, cmd, &self.quad_pipeline, self.frame, self.extent, true);
            }

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

            // Every pickup's slot must be handed back however this ends, or its producer stalls.
            let mut signal_semaphores = vec![self.render_finished[image_index as usize]];
            let mut signal_values = vec![0u64];
            for id in &pickups {
                if let Some(SampledLayer { image: GpuImage::Cuda(layer), .. }) = self.layers.get(id) {
                    if let Some((semaphore, value)) = layer.release_signal() {
                        signal_semaphores.push(semaphore);
                        signal_values.push(value);
                    }
                }
            }
            // A D3D producer's fence: signaled when this frame has finished sampling its texture.
            let signal_result = viewports.iter().try_for_each(|draw| -> Result<(), vk::Result> {
                if let Some(SampledLayer { image: GpuImage::Imported(image), .. }) = self.layers.get_mut(&draw.viewport_id)
                {
                    if let (Some(value), Some(fence)) = (image.pending_signal.take(), &image.fence) {
                        let current = self.device.get_semaphore_counter_value(fence.semaphore)?;
                        if value > current {
                            signal_semaphores.push(fence.semaphore);
                            signal_values.push(value);
                        }
                    }
                }
                Ok(())
            });
            let cmds = [cmd];
            let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
                .wait_semaphore_values(&wait_values)
                .signal_semaphore_values(&signal_values);
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&cmds)
                .signal_semaphores(&signal_semaphores)
                .push_next(&mut timeline_info);
            let submitted = signal_result
                .and_then(|()| self.device.end_command_buffer(cmd))
                .and_then(|()| self.device.queue_submit(self.queue, &[submit_info], fence));
            for id in &pickups {
                if let Some(SampledLayer { image: GpuImage::Cuda(layer), .. }) = self.layers.get_mut(id) {
                    layer.finish_submit(submitted.is_ok());
                }
            }
            submitted?;
            let signal_semaphores = &signal_semaphores[..1];

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            match self.swapchain_loader.queue_present(self.queue, &present_info) {
                Ok(_suboptimal) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(self.extent.width, self.extent.height)?;
                }
                Err(vk::Result::ERROR_SURFACE_LOST_KHR) => return Ok(()),
                Err(e) => return Err(e.into()),
            }

            self.frame = (self.frame + 1) % FRAMES_IN_FLIGHT;
        }
        Ok(())
    }

    unsafe fn draw_sampled(
        &self,
        cmd: vk::CommandBuffer,
        layer: &SampledLayer,
        viewport: vk::Viewport,
        scissor: vk::Rect2D,
    ) {
        self.device.cmd_set_viewport(cmd, 0, &[viewport]);
        self.device.cmd_set_scissor(cmd, 0, &[scissor]);
        self.device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::GRAPHICS,
            self.viewport_pipeline.pipeline_layout(),
            0,
            &[layer.descriptor_set],
            &[],
        );
        self.device.cmd_draw(cmd, 3, 1, 0, 0);
    }
}


/// What `texture` really is, from D3D, checked against what the caller said and what this GPU can
/// import. `Err` is the status message; the submission is then dropped.
#[cfg(windows)]
pub(crate) fn check_texture(support: &D3dSupport, texture: &D3d11Import) -> Result<SharedFormat, String> {
    let probe = support
        .probe
        .as_ref()
        .map_err(|why| format!("cannot check the texture, so it is not imported: {why}"))?;
    let real = probe.describe(texture.handle())?;
    if (real.width, real.height) != (texture.width, texture.height) {
        return Err(format!(
            "the texture is {}x{} but submit_d3d11 was told {}x{}",
            real.width, real.height, texture.width, texture.height
        ));
    }
    if let Some(said) = texture.format {
        if said != real.format {
            return Err(format!("the texture is {:?} but submit_d3d11 was told {:?}", real.format, said));
        }
    }
    if !support.supports(real.format) {
        return Err(format!("this GPU cannot import {:?} D3D11 textures", real.format));
    }
    Ok(real.format)
}

#[cfg(not(windows))]
fn check_texture(_support: &D3dSupport, _texture: &D3d11Import) -> Result<SharedFormat, String> {
    Err("D3D11 import is Windows-only".to_string())
}

/// Attach `fence` to `image` (or keep the one it already has when it is the same fence) and wait
/// for its `wait_value`. `Err` carries the status message; the submission is then dropped.
pub(crate) unsafe fn sync_import(
    device: &Device,
    queue: vk::Queue,
    support: &D3dSupport,
    image: &mut ImportedImage,
    fence: Option<D3d11Fence>,
) -> Result<(), String> {
    let Some(mut fence) = fence else {
        image.pending_signal = None;
        return Ok(());
    };
    let loader = support.semaphore.as_ref().ok_or("this GPU cannot import D3D fences")?;
    if !image.fence.as_ref().is_some_and(|kept| kept.is_import_of(fence.handle())) {
        let imported = crate::d3d_import::import_fence(device, loader, &mut fence)
            .map_err(|err| format!("could not import the fence: {err} (it must be a shared ID3D11Fence or ID3D12Fence NT handle)"))?;
        if let Some(old) = image.fence.replace(imported) {
            let _ = device.queue_wait_idle(queue);
            old.destroy(device);
        }
    }
    let semaphore = image.fence.as_ref().map(|kept| kept.semaphore).ok_or("fence missing after import")?;
    if fence.wait_value > 0 {
        match crate::d3d_import::wait_fence(device, semaphore, fence.wait_value) {
            Ok(true) => {}
            Ok(false) => {
                return Err(format!(
                    "timed out after {} ms waiting for fence value {}; the previous frame stays",
                    crate::d3d_import::FENCE_WAIT.as_millis(),
                    fence.wait_value
                ))
            }
            Err(err) => return Err(format!("waiting for the fence failed: {err}")),
        }
    }
    image.pending_signal = (fence.signal_value > 0).then_some(fence.signal_value);
    Ok(())
}

fn sync_sampled_image(
    image: &mut GpuImage,
    color_subresource: vk::ImageSubresourceRange,
    graphics_family: u32,
    barriers: &mut Vec<vk::ImageMemoryBarrier<'_>>,
) {
    match image {
        GpuImage::Cpu(texture) => {
            barriers.push(
                vk::ImageMemoryBarrier::default()
                    .old_layout(texture.current_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .image(texture.image)
                    .subresource_range(color_subresource)
                    .src_access_mask(vk::AccessFlags::HOST_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            );
            texture.current_layout = vk::ImageLayout::GENERAL;
        }
        // Synchronized by its own pickup barriers (`CudaLayer::record_pickup`).
        GpuImage::Cuda(_) => {}
        GpuImage::Imported(texture) => {
            // A D3D texture is acquired from the external queue family on its first use. After
            // that it has been ours, and the barrier only keeps the layout GENERAL.
            let (src_family, dst_family) = if texture.acquire_external {
                (vk::QUEUE_FAMILY_EXTERNAL, graphics_family)
            } else {
                (vk::QUEUE_FAMILY_IGNORED, vk::QUEUE_FAMILY_IGNORED)
            };
            texture.acquire_external = false;
            barriers.push(
                vk::ImageMemoryBarrier::default()
                    .old_layout(texture.current_layout)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .src_queue_family_index(src_family)
                    .dst_queue_family_index(dst_family)
                    .image(texture.image)
                    .subresource_range(color_subresource)
                    .src_access_mask(vk::AccessFlags::empty())
                    .dst_access_mask(vk::AccessFlags::SHADER_READ),
            );
            texture.current_layout = vk::ImageLayout::GENERAL;
        }
    }
}

unsafe fn device_create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
) -> Result<vk::ImageView, Error> {
    let info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .level_count(1)
                .layer_count(1),
        );
    Ok(device.create_image_view(&info, None)?)
}

impl Drop for VulkanRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            if let Some(chrome) = self.chrome.take() {
                self.destroy_layer(chrome);
            }
            let ids: Vec<u64> = self.layers.keys().copied().collect();
            for id in ids {
                if let Some(layer) = self.layers.remove(&id) {
                    self.destroy_layer(layer);
                }
            }
            let parked: Vec<u64> = self.parked_imports.keys().copied().collect();
            for id in parked {
                if let Some(mut ring) = self.parked_imports.remove(&id) {
                    for layer in ring.drain() {
                        self.destroy_layer(layer);
                    }
                }
            }
            self.quad_chrome.destroy(&self.device, &self.viewport_pipeline);
            self.quad_pipeline.destroy(&self.device);
            self.viewport_pipeline.destroy(&self.device);
            for &s in &self.image_available {
                self.device.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished {
                self.device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight {
                self.device.destroy_fence(f, None);
            }
            for &view in &self.image_views {
                self.device.destroy_image_view(view, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            if self.swapchain != vk::SwapchainKHR::null() {
                self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            }
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            if let Some((loader, messenger)) = self.debug.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

impl fastgui_app::SurfaceBackend for VulkanRenderer {
    type Error = Error;

    fn new(
        window: &winit::window::Window,
        physical_width: u32,
        physical_height: u32,
    ) -> Result<Self, Self::Error> {
        VulkanRenderer::new(window, physical_width, physical_height)
    }

    fn resize(&mut self, physical_width: u32, physical_height: u32) -> Result<(), Self::Error> {
        VulkanRenderer::resize(self, physical_width, physical_height)
    }

    fn set_chrome_frame(&mut self, frame: &ChromeFrame<'_>) -> Result<(), Self::Error> {
        VulkanRenderer::set_chrome_frame(self, frame)
    }

    fn set_chrome_quads(&mut self, frame: &ChromeQuads<'_>) -> Result<(), Self::Error> {
        VulkanRenderer::set_chrome_quads(self, frame)
    }

    fn set_layer_frame(&mut self, viewport_id: u64, frame: CpuFrame) -> Result<(), Self::Error> {
        VulkanRenderer::set_layer_frame(self, viewport_id, frame)
    }

    fn set_imported_layer(&mut self, viewport_id: u64, layer: ImportedLayer) -> Result<(), Self::Error> {
        VulkanRenderer::set_imported_layer(self, viewport_id, layer)
    }

    fn retain_layers(&mut self, live_ids: &[u64]) {
        VulkanRenderer::retain_layers(self, live_ids)
    }

    fn surface_size(&self) -> (u32, u32) {
        (self.extent.width, self.extent.height)
    }

    fn render_frame(
        &mut self,
        clear: [f32; 4],
        draw_chrome: bool,
        draws: &[ViewportDraw],
    ) -> Result<(), Self::Error> {
        VulkanRenderer::render_frame(self, clear, draw_chrome, draws)
    }

    fn create_cuda_surface(
        &mut self,
        viewport_id: u64,
        width: u32,
        height: u32,
    ) -> Result<fastgui_app::CudaExportHandles, String> {
        VulkanRenderer::create_cuda_surface(self, viewport_id, width, height)
            .map_err(|e| e.to_string())
    }
}
