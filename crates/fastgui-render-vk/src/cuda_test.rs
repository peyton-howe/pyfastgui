//! GPU check of the Vulkan half of the CUDA path (`CudaLayer`), with Vulkan itself standing in
//! for CUDA: fill a slot of the exported buffer, signal `ready` from a queue submission, publish,
//! then run the renderer's pickup and read the sampled image back. Headless; Windows only (the
//! export is win32). Skips (passing) without a Vulkan 1.3 device that has the win32 external
//! memory/semaphore extensions. With the Khronos validation layer installed, any validation
//! error fails the test.
//!
//!   cargo test -p fastgui-render-vk cuda
//!
//! What this can't cover: whether CUDA accepts these exact handles, and the CUDA-side waits and
//! signals. That half needs an NVIDIA GPU.

use std::ffi::{c_char, CStr};
use std::sync::Mutex;
use std::time::Duration;

use ash::{ext::debug_utils, khr, vk, Device, Entry, Instance};
use fastgui_app::CudaExportHandles;

use crate::cuda_texture::CudaLayer;
use crate::texture::find_memory_type_index;

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

struct Gpu {
    _entry: Entry,
    instance: Instance,
    messenger: Option<(debug_utils::Instance, vk::DebugUtilsMessengerEXT)>,
    device: Device,
    family: u32,
    queue: vk::Queue,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    command_pool: vk::CommandPool,
    memory_win32: khr::external_memory_win32::Device,
    semaphore_win32: khr::external_semaphore_win32::Device,
    uuid: [u8; 16],
}

impl Gpu {
    unsafe fn new() -> Option<Self> {
        let entry = Entry::load().ok()?;
        let available_layers = entry.enumerate_instance_layer_properties().ok()?;
        let validation = c"VK_LAYER_KHRONOS_validation";
        let layers: Vec<*const c_char> = available_layers
            .iter()
            .any(|l| l.layer_name_as_c_str() == Ok(validation))
            .then(|| validation.as_ptr())
            .into_iter()
            .collect();
        let extensions = [debug_utils::NAME.as_ptr()];
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app)
                    .enabled_layer_names(&layers)
                    .enabled_extension_names(&extensions),
                None,
            )
            .ok()?;
        let messenger = (!layers.is_empty()).then(|| {
            let loader = debug_utils::Instance::new(&entry, &instance);
            let info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(on_validation_message));
            let messenger = loader.create_debug_utils_messenger(&info, None).expect("debug messenger");
            (loader, messenger)
        });
        if messenger.is_none() {
            eprintln!("note: Vulkan validation layer not found; running without it");
        }

        let wanted = [
            khr::external_memory::NAME,
            khr::external_memory_win32::NAME,
            khr::external_semaphore::NAME,
            khr::external_semaphore_win32::NAME,
        ];
        let (pdevice, family) = instance.enumerate_physical_devices().ok()?.into_iter().find_map(|pd| {
            if instance.get_physical_device_properties(pd).api_version < vk::API_VERSION_1_3 {
                return None;
            }
            let available = instance.enumerate_device_extension_properties(pd).ok()?;
            if !wanted.iter().all(|w| available.iter().any(|e| e.extension_name_as_c_str() == Ok(*w))) {
                return None;
            }
            let family = instance
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::GRAPHICS))?;
            Some((pd, family as u32))
        })?;
        let extension_names: Vec<*const c_char> = wanted.iter().map(|n| n.as_ptr()).collect();
        let priorities = [1.0];
        let queue_infos =
            [vk::DeviceQueueCreateInfo::default().queue_family_index(family).queue_priorities(&priorities)];
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let device = instance
            .create_device(
                pdevice,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queue_infos)
                    .enabled_extension_names(&extension_names)
                    .push_next(&mut features12),
                None,
            )
            .ok()?;
        let mut id_properties = vk::PhysicalDeviceIDProperties::default();
        let mut properties2 = vk::PhysicalDeviceProperties2::default().push_next(&mut id_properties);
        instance.get_physical_device_properties2(pdevice, &mut properties2);
        let uuid = id_properties.device_uuid;
        Some(Self {
            queue: device.get_device_queue(family, 0),
            memory_properties: instance.get_physical_device_memory_properties(pdevice),
            command_pool: device
                .create_command_pool(&vk::CommandPoolCreateInfo::default().queue_family_index(family), None)
                .ok()?,
            memory_win32: khr::external_memory_win32::Device::new(&instance, &device),
            semaphore_win32: khr::external_semaphore_win32::Device::new(&instance, &device),
            uuid,
            family,
            device,
            messenger,
            instance,
            _entry: entry,
        })
    }

    /// Record with `record`, submit with the given timeline waits/signals, and wait for idle.
    unsafe fn submit(
        &self,
        waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags)],
        signals: &[(vk::Semaphore, u64)],
        record: impl FnOnce(vk::CommandBuffer),
    ) {
        let device = &self.device;
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.command_pool).command_buffer_count(1),
            )
            .unwrap()[0];
        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        record(cmd);
        device.end_command_buffer(cmd).unwrap();
        let wait_semaphores: Vec<_> = waits.iter().map(|w| w.0).collect();
        let wait_values: Vec<_> = waits.iter().map(|w| w.1).collect();
        let wait_stages: Vec<_> = waits.iter().map(|w| w.2).collect();
        let signal_semaphores: Vec<_> = signals.iter().map(|s| s.0).collect();
        let signal_values: Vec<_> = signals.iter().map(|s| s.1).collect();
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .command_buffers(&cmds)
            .push_next(&mut timeline);
        device.queue_submit(self.queue, &[submit], vk::Fence::null()).unwrap();
        device.queue_wait_idle(self.queue).unwrap();
        device.free_command_buffers(self.command_pool, &cmds);
    }

    /// Play CUDA: write `pixel` over all of slot `index`, hand the bytes back to the external
    /// owner, and signal `ready = value`.
    unsafe fn produce(&self, layer: &CudaLayer, index: usize, pixel: [u8; 4], value: u64) {
        let (buffer, _, ready, _, stride) = layer.test_parts();
        self.submit(&[], &[(ready, value)], |cmd| {
            self.device.cmd_fill_buffer(cmd, buffer, index as u64 * stride, stride, u32::from_le_bytes(pixel));
            let to_external = vk::BufferMemoryBarrier::default()
                .src_queue_family_index(self.family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .buffer(buffer)
                .offset(index as u64 * stride)
                .size(stride);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[to_external],
                &[],
            );
        });
    }

    /// Run the renderer's pickup in a submission (waiting on `ready`, signalling `release`, as
    /// `render_frame` does) and read the sampled image back. `None` when nothing was published.
    unsafe fn pick_up(&self, layer: &mut CudaLayer, width: u32, height: u32) -> Option<Vec<u8>> {
        let device = &self.device;
        let (_, image, _, _, _) = layer.test_parts();
        let bytes = u64::from(width) * u64::from(height) * 4;
        let readback = device
            .create_buffer(&vk::BufferCreateInfo::default().size(bytes).usage(vk::BufferUsageFlags::TRANSFER_DST), None)
            .unwrap();
        let requirements = device.get_buffer_memory_requirements(readback);
        let index = find_memory_type_index(
            &self.memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .unwrap();
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default().allocation_size(requirements.size).memory_type_index(index),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(readback, memory, 0).unwrap();

        // Record first (it decides whether there is a wait), then submit.
        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default().command_pool(self.command_pool).command_buffer_count(1),
            )
            .unwrap()[0];
        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        let wait = layer.record_pickup(device, cmd, self.family);
        let picked = wait.is_some();
        if picked {
            let to_read = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[to_read],
            );
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::GENERAL,
                readback,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default().aspect_mask(vk::ImageAspectFlags::COLOR).layer_count(1),
                    )
                    .image_extent(vk::Extent3D { width, height, depth: 1 })],
            );
        }
        device.end_command_buffer(cmd).unwrap();
        let waits: Vec<_> = wait.into_iter().collect();
        let signals: Vec<_> = layer.release_signal().into_iter().collect();
        let wait_values: Vec<_> = waits.iter().map(|w| w.1).collect();
        let wait_semaphores: Vec<_> = waits.iter().map(|w| w.0).collect();
        let wait_stages = vec![vk::PipelineStageFlags::TRANSFER; waits.len()];
        let signal_values: Vec<_> = signals.iter().map(|s| s.1).collect();
        let signal_semaphores: Vec<_> = signals.iter().map(|s| s.0).collect();
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let cmds = [cmd];
        let submit = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .signal_semaphores(&signal_semaphores)
            .command_buffers(&cmds)
            .push_next(&mut timeline);
        device.queue_submit(self.queue, &[submit], vk::Fence::null()).unwrap();
        layer.finish_submit(true);
        device.queue_wait_idle(self.queue).unwrap();
        device.free_command_buffers(self.command_pool, &cmds);

        let pixels = picked.then(|| {
            let mapped = device.map_memory(memory, 0, bytes, vk::MemoryMapFlags::empty()).unwrap() as *const u8;
            let pixels = std::slice::from_raw_parts(mapped, bytes as usize).to_vec();
            device.unmap_memory(memory);
            pixels
        });
        device.destroy_buffer(readback, None);
        device.free_memory(memory, None);
        pixels
    }

    unsafe fn counter(&self, semaphore: vk::Semaphore) -> u64 {
        self.device.get_semaphore_counter_value(semaphore).unwrap()
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            if let Some((loader, messenger)) = self.messenger.take() {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn assert_solid(pixels: &[u8], pixel: [u8; 4]) {
    assert!(pixels.chunks_exact(4).all(|p| p == pixel), "expected solid {pixel:?}, got {:?}", &pixels[..8]);
}

#[test]
fn cuda_layer_round_trip_on_vulkan() {
    let Some(gpu) = (unsafe { Gpu::new() }) else {
        eprintln!("skipping: no Vulkan 1.3 device with win32 external memory/semaphore support");
        return;
    };
    // Odd size: rows aren't a multiple of anything convenient, slots get padded to 256 bytes.
    let (width, height) = (37, 11);
    unsafe {
        let (mut layer, handles): (CudaLayer, CudaExportHandles) = CudaLayer::new(
            &gpu.device,
            &gpu.memory_win32,
            &gpu.semaphore_win32,
            &gpu.memory_properties,
            gpu.uuid,
            width,
            height,
        )
        .expect("create CUDA layer");
        assert_ne!(handles.memory_win32_handle, 0);
        assert_ne!(handles.ready_win32_handle, 0);
        assert_ne!(handles.release_win32_handle, 0);
        assert_eq!(handles.slot_stride % 256, 0);
        assert!(handles.slot_stride >= u64::from(width * height * 4));
        assert!(handles.memory_size >= handles.slot_stride * fastgui_app::CUDA_SLOTS as u64);
        assert_eq!(handles.device_uuid, gpu.uuid);
        let shared = handles.shared.clone();
        let (_, _, _, release, _) = layer.test_parts();

        assert!(gpu.pick_up(&mut layer, width, height).is_none(), "nothing published yet");
        assert!(!layer.has_content());

        // Frame 1.
        let a = shared.acquire(Duration::ZERO).unwrap();
        assert_eq!(a.wait_release, 0);
        gpu.produce(&layer, a.index, [255, 0, 0, 255], 1);
        shared.publish(a.index, 1);
        let pixels = gpu.pick_up(&mut layer, width, height).expect("frame 1 picked up");
        assert_solid(&pixels, [255, 0, 0, 255]);
        assert!(layer.has_content());
        assert_eq!(gpu.counter(release), 1, "the copy signalled release");

        // Frame 2 lands in a different slot; frame 3 replaces it before any pickup.
        let b = shared.acquire(Duration::ZERO).unwrap();
        assert_ne!(b.index, a.index, "the just-copied slot is handed out last");
        gpu.produce(&layer, b.index, [0, 255, 0, 255], 2);
        shared.publish(b.index, 2);
        let c = shared.acquire(Duration::ZERO).unwrap();
        assert!(c.index != a.index && c.index != b.index);
        gpu.produce(&layer, c.index, [0, 0, 255, 128], 3);
        shared.publish(c.index, 3);
        let pixels = gpu.pick_up(&mut layer, width, height).expect("frame 3 picked up");
        assert_solid(&pixels, [0, 0, 255, 128]);
        assert_eq!(gpu.counter(release), 2);
        assert!(gpu.pick_up(&mut layer, width, height).is_none(), "frame 2 was dropped, not queued");

        // Reuse order follows release values: `b` (never copied), then `a`, which must wait
        // GPU-side for frame 1's release before being overwritten.
        let reuse = shared.acquire(Duration::ZERO).unwrap();
        assert_eq!((reuse.index, reuse.wait_release), (b.index, 0));
        let next = shared.acquire(Duration::ZERO).unwrap();
        assert_eq!((next.index, next.wait_release), (a.index, 1));

        drop(handles); // closes the exported NT handles
        layer.destroy(&gpu.device);
        assert!(shared.is_closed(), "producers learn the layer is gone");
    }
    let errors = std::mem::take(&mut *VALIDATION_ERRORS.lock().unwrap());
    assert!(errors.is_empty(), "validation errors:\n{}", errors.join("\n"));
}
