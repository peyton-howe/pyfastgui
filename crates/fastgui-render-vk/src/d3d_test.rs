//! A real Direct3D 11 producer on the same adapter as the Vulkan device: it fills a shared NT
//! texture, and the import path samples it. Skips when there is no Windows Vulkan 1.3 device with
//! the import extensions, or when no DXGI adapter matches its LUID. Included from `upload_bench`
//! so it can reuse the bench's sampling and readback helpers.

use super::*;
use crate::d3d_import;
use crate::dmabuf::ImportedImage;
use fastgui_core::{D3d11Fence, D3d11Import, SharedFormat};
use windows::core::{Interface, PCWSTR};
use windows::Win32::Foundation::{DuplicateHandle, DUPLICATE_SAME_ACCESS, HANDLE, HMODULE};
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::GetCurrentProcess;

/// A shape that catches row-pitch, channel-order, and flip mistakes, unlike a solid color.
fn pattern(width: u32, height: u32, shift: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            pixels.extend_from_slice(&[((x + shift) % 251) as u8, ((y + shift) % 241) as u8, ((x + y + shift) % 239) as u8, 255]);
        }
    }
    pixels
}

struct Producer {
    context: ID3D11DeviceContext,
    device: ID3D11Device,
    _texture: ID3D11Texture2D,
    handle: HANDLE,
}

impl Producer {
    unsafe fn new(luid: [u8; 8], width: u32, height: u32, format: DXGI_FORMAT, pixels: &[u8]) -> Option<Self> {
        let factory: IDXGIFactory1 = CreateDXGIFactory1().ok()?;
        let low = u32::from_le_bytes(luid[..4].try_into().unwrap());
        let high = i32::from_le_bytes(luid[4..].try_into().unwrap());
        let adapter = (0..)
            .map_while(|i| factory.EnumAdapters1(i).ok())
            .find(|a| a.GetDesc1().is_ok_and(|d| d.AdapterLuid.LowPart == low && d.AdapterLuid.HighPart == high))?;
        let adapter: IDXGIAdapter = adapter.cast().ok()?;
        let mut device = None;
        let mut context = None;
        D3D11CreateDevice(
            Some(&adapter),
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_FLAG(0),
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )
        .ok()?;
        let (device, context) = (device?, context?);
        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: (D3D11_RESOURCE_MISC_SHARED.0 | D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0) as u32,
        };
        let init = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr().cast(),
            SysMemPitch: width * 4,
            SysMemSlicePitch: 0,
        };
        let mut texture = None;
        device.CreateTexture2D(&desc, Some(&init), Some(&mut texture)).ok()?;
        let texture = texture?;
        let resource: IDXGIResource1 = texture.cast().ok()?;
        let handle = resource.CreateSharedHandle(None, 0x8000_0001, PCWSTR::null()).ok()?;
        let producer = Self { context, device, _texture: texture, handle };
        producer.finish();
        Some(producer)
    }

    /// Overwrite the whole texture, as a producer does between submits, then wait for the GPU.
    unsafe fn rewrite(&self, width: u32, pixels: &[u8]) {
        self.context
            .UpdateSubresource(&self._texture, 0, None, pixels.as_ptr().cast(), width * 4, 0);
        self.finish();
    }

    /// Flush and wait until the GPU has executed everything queued, as `submit_d3d11` expects.
    unsafe fn finish(&self) {
        let mut query = None;
        self.device
            .CreateQuery(&D3D11_QUERY_DESC { Query: D3D11_QUERY_EVENT, MiscFlags: 0 }, Some(&mut query))
            .unwrap();
        let query = query.unwrap();
        self.context.End(&query);
        self.context.Flush();
        let mut done = 0u32;
        // `S_FALSE` (not finished) and `S_OK` both come back as `Ok`; the BOOL says which.
        while done == 0 {
            self.context.GetData(&query, Some((&mut done as *mut u32).cast()), 4, 0).unwrap();
            std::thread::yield_now();
        }
    }

    /// A fresh handle for one import; `D3d11Import` closes it, like `Viewport.submit_d3d11`.
    unsafe fn import(&self, width: u32, height: u32) -> D3d11Import {
        D3d11Import::new(dup(self.handle), width, height)
    }

    /// A shared `ID3D11Fence`, as a producer would hand to `submit_d3d11(fence=...)`.
    unsafe fn make_fence(&self) -> (ID3D11Fence, HANDLE) {
        let device5: ID3D11Device5 = self.device.cast().expect("ID3D11Device5");
        let mut fence: Option<ID3D11Fence> = None;
        device5.CreateFence(0, D3D11_FENCE_FLAG_SHARED, &mut fence).expect("create a shared fence");
        let fence = fence.expect("fence");
        let handle = fence.CreateSharedHandle(None, 0x1000_0000, PCWSTR::null()).expect("share the fence");
        (fence, handle)
    }

    /// Signal `fence` to `value` from the GPU after everything queued so far.
    unsafe fn signal(&self, fence: &ID3D11Fence, value: u64) {
        let context4: ID3D11DeviceContext4 = self.context.cast().expect("ID3D11DeviceContext4");
        context4.Signal(fence, value).expect("signal the fence");
        self.context.Flush();
    }
}

/// Duplicate an NT handle into this process, as `submit_d3d11` does for the handle it is given.
unsafe fn dup(handle: HANDLE) -> isize {
    let mut out = HANDLE::default();
    DuplicateHandle(GetCurrentProcess(), handle, GetCurrentProcess(), &mut out, 0, false, DUPLICATE_SAME_ACCESS)
        .unwrap();
    out.0 as isize
}

impl Drop for Producer {
    fn drop(&mut self) {
        unsafe {
            let _ = windows::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

impl Bench {
    unsafe fn device_luid(&self) -> Option<[u8; 8]> {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        self.instance.get_physical_device_properties2(self.pdevice, &mut props);
        (id.device_luid_valid == vk::TRUE).then_some(id.device_luid)
    }

    /// The renderer's barrier for a D3D image: acquire from the external family. `UNDEFINED`
    /// on first use, `GENERAL` when a cached import is shown again.
    unsafe fn acquire_external(&self, image: vk::Image, old_layout: vk::ImageLayout) -> Result<(), Error> {
        self.device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty())?;
        self.device.begin_command_buffer(self.cmd, &vk::CommandBufferBeginInfo::default())?;
        let barrier = [vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::GENERAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
            .dst_queue_family_index(self.family)
            .image(image)
            .subresource_range(color_range())
            .dst_access_mask(vk::AccessFlags::SHADER_READ)];
        self.device.cmd_pipeline_barrier(
            self.cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &barrier,
        );
        self.device.end_command_buffer(self.cmd)?;
        let cmds = [self.cmd];
        self.device.queue_submit(
            self.queue,
            &[vk::SubmitInfo::default().command_buffers(&cmds)],
            vk::Fence::null(),
        )?;
        self.device.queue_wait_idle(self.queue)?;
        Ok(())
    }

    unsafe fn read_pixel(&self, image: vk::Image, x: u32, y: u32) -> [u8; 4] {
        let (buffer, _, mapped) = self.readback.expect("readback buffer");
        self.copy_pixel(image, buffer, x, y).expect("copy pixel");
        std::slice::from_raw_parts(mapped, 4).try_into().unwrap()
    }
}

#[test]
fn imported_d3d11_texture_samples_the_producers_pixels_and_times_the_import() {
    unsafe {
        let Some(bench) = Bench::open() else { return };
        let Some(support) = &bench.d3d else {
            eprintln!("d3d11 test skipped: no D3D11 import on {}", bench.name);
            return;
        };
        let Some(luid) = bench.device_luid() else {
            eprintln!("d3d11 test skipped: the Vulkan device reports no LUID");
            return;
        };
        for (width, height) in SIZES {
            let pixels = pattern(width, height, 0);
            let Some(producer) = Producer::new(luid, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, &pixels) else {
                eprintln!("d3d11 test skipped: no D3D11 device on the Vulkan adapter");
                return;
            };
            let target = OwnedTarget::new(&bench.device, &bench.memory_properties, width, height).unwrap();
            bench.transition_target(target.image).unwrap();
            let set = SetGuard::new(&bench.device, &bench.pipeline).unwrap();

            // One import, sampled once, and read back at corners, edges, and the middle.
            let imported = d3d_import::import(
                &bench.device,
                &support.memory,
                &bench.memory_properties,
                producer.import(width, height),
                SharedFormat::Rgba8,
            )
            .expect("import the shared D3D11 texture");
            bench.pipeline.bind_texture(&bench.device, set.set, imported.view);
            bench.acquire_external(imported.image, vk::ImageLayout::UNDEFINED).unwrap();
            let mut layout = vk::ImageLayout::GENERAL;
            bench
                .sample(imported.image, &mut layout, vk::AccessFlags::empty(), target.view, set.set, width, height)
                .unwrap();
            for (x, y) in [
                (0, 0),
                (width - 1, 0),
                (0, height - 1),
                (width - 1, height - 1),
                (width / 2, height / 2),
                (37, 211),
            ] {
                let i = (y as usize * width as usize + x as usize) * 4;
                assert_eq!(bench.read_pixel(target.image, x, y), pixels[i..i + 4], "{width}x{height} pixel ({x}, {y})");
            }
            eprintln!("{width}x{height} d3d11 sample: pixels match");

            // A cached import: the producer rewrites the texture, and the next draw acquires it
            // again without a new import. The same image must show the new pixels.
            let rewritten = pattern(width, height, 7);
            producer.rewrite(width, &rewritten);
            bench.acquire_external(imported.image, vk::ImageLayout::GENERAL).unwrap();
            let mut layout = vk::ImageLayout::GENERAL;
            bench
                .sample(imported.image, &mut layout, vk::AccessFlags::empty(), target.view, set.set, width, height)
                .unwrap();
            for (x, y) in [(0, 0), (width - 1, height - 1), (width / 2, height / 2), (37, 211)] {
                let i = (y as usize * width as usize + x as usize) * 4;
                assert_eq!(
                    bench.read_pixel(target.image, x, y),
                    rewritten[i..i + 4],
                    "{width}x{height} rewritten pixel ({x}, {y})"
                );
            }
            eprintln!("{width}x{height} d3d11 cached sample: rewritten pixels match without a re-import");

            // A cache hit costs this: one acquire barrier and the sample.
            let label = format!("{width}x{height} d3d11 cached sample");
            time(&label, || {
                bench.acquire_external(imported.image, vk::ImageLayout::GENERAL)?;
                let mut layout = vk::ImageLayout::GENERAL;
                bench.sample(imported.image, &mut layout, vk::AccessFlags::empty(), target.view, set.set, width, height)
            })
            .unwrap();
            imported.destroy(&bench.device);

            // What each submitted frame costs: re-import the handle, acquire, and sample.
            let label = format!("{width}x{height} d3d11 import+sample");
            time(&label, || {
                let imported = d3d_import::import(
                    &bench.device,
                    &support.memory,
                    &bench.memory_properties,
                    producer.import(width, height),
                    SharedFormat::Rgba8,
                )?;
                bench.pipeline.bind_texture(&bench.device, set.set, imported.view);
                bench.acquire_external(imported.image, vk::ImageLayout::UNDEFINED)?;
                let mut layout = vk::ImageLayout::GENERAL;
                let result = bench.sample(
                    imported.image,
                    &mut layout,
                    vk::AccessFlags::empty(),
                    target.view,
                    set.set,
                    width,
                    height,
                );
                imported.destroy(&bench.device);
                result
            })
            .unwrap();
        }
    }
}

/// Handle identity is what the renderer's import cache keys on: a duplicate of a handle is the same
/// object, and another texture's handle is not.
#[test]
fn duplicated_handles_compare_equal_and_other_textures_do_not() {
    unsafe {
        let Some(bench) = Bench::open() else { return };
        let Some(luid) = bench.device_luid() else { return };
        let (width, height) = (16, 16);
        let pixels = pattern(width, height, 0);
        let (Some(a), Some(b)) = (
            Producer::new(luid, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, &pixels),
            Producer::new(luid, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, &pixels),
        ) else {
            eprintln!("d3d11 identity test skipped: no D3D11 device on the Vulkan adapter");
            return;
        };
        let first = a.import(width, height);
        let second = a.import(width, height);
        assert_ne!(first.handle(), second.handle(), "each duplicate is a distinct handle value");
        assert!(d3d_import::same_object(first.handle(), second.handle()));
        let other = b.import(width, height);
        assert!(!d3d_import::same_object(first.handle(), other.handle()));
    }
}

fn to_bgra(rgba: &[u8]) -> Vec<u8> {
    rgba.chunks_exact(4).flat_map(|p| [p[2], p[1], p[0], p[3]]).collect()
}

/// Sample `imported` into a fresh target and compare corners, middle, and an interior point with
/// `expected` (RGBA bytes, row-major, tightly packed).
unsafe fn assert_samples(bench: &Bench, imported: &ImportedImage, width: u32, height: u32, expected: &[u8], what: &str) {
    let target = OwnedTarget::new(&bench.device, &bench.memory_properties, width, height).unwrap();
    bench.transition_target(target.image).unwrap();
    let set = SetGuard::new(&bench.device, &bench.pipeline).unwrap();
    bench.pipeline.bind_texture(&bench.device, set.set, imported.view);
    bench.acquire_external(imported.image, vk::ImageLayout::UNDEFINED).unwrap();
    let mut layout = vk::ImageLayout::GENERAL;
    bench
        .sample(imported.image, &mut layout, vk::AccessFlags::empty(), target.view, set.set, width, height)
        .unwrap();
    for (x, y) in [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1), (width / 2, height / 2), (37, 51)] {
        let i = (y as usize * width as usize + x as usize) * 4;
        assert_eq!(bench.read_pixel(target.image, x, y), expected[i..i + 4], "{what}: pixel ({x}, {y})");
    }
}

/// DXGI desktop duplication and most video output is BGRA. The shader must see the same RGBA the
/// producer meant, which only holds when the image is imported as `B8G8R8A8`.
#[test]
fn bgra_texture_samples_with_red_and_blue_in_place() {
    unsafe {
        let Some(bench) = Bench::open() else { return };
        let Some(support) = &bench.d3d else { return };
        let Some(luid) = bench.device_luid() else { return };
        if !support.bgra8 {
            eprintln!("bgra test skipped: {} cannot import B8G8R8A8", bench.name);
            return;
        }
        let (width, height) = (256, 128);
        let rgba = pattern(width, height, 3);
        let Some(producer) = Producer::new(luid, width, height, DXGI_FORMAT_B8G8R8A8_UNORM, &to_bgra(&rgba)) else {
            eprintln!("bgra test skipped: no D3D11 device on the Vulkan adapter");
            return;
        };
        let texture = producer.import(width, height).with_format(Some(SharedFormat::Bgra8));
        let imported = d3d_import::import(&bench.device, &support.memory, &bench.memory_properties, texture, SharedFormat::Bgra8)
            .expect("import the shared B8G8R8A8 texture");
        assert_eq!(imported.format, vk::Format::B8G8R8A8_UNORM);
        assert_samples(&bench, &imported, width, height, &rgba, "bgra");
        imported.destroy(&bench.device);
    }
}

/// Vulkan accepts any size for an imported texture (the driver does not look), so the renderer reads
/// the real size and format from D3D first. Every wrong claim must be caught there.
#[test]
fn check_texture_catches_wrong_claims_and_detects_the_format() {
    unsafe {
        let Some(bench) = Bench::open() else { return };
        let Some(support) = &bench.d3d else { return };
        let Some(luid) = bench.device_luid() else { return };
        let (width, height) = (256u32, 128u32);
        let (Some(rgba), Some(bgra)) = (
            Producer::new(luid, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, &pattern(width, height, 0)),
            Producer::new(luid, width, height, DXGI_FORMAT_B8G8R8A8_UNORM, &pattern(width, height, 0)),
        ) else {
            eprintln!("check_texture test skipped: no D3D11 device on the Vulkan adapter");
            return;
        };
        let check = |p: &Producer, w: u32, h: u32, format: Option<SharedFormat>| {
            crate::renderer::check_texture(support, &p.import(w, h).with_format(format))
        };

        // Right size: the format comes from the texture when the caller gave none.
        assert_eq!(check(&rgba, width, height, None), Ok(SharedFormat::Rgba8));
        assert_eq!(check(&bgra, width, height, None), Ok(SharedFormat::Bgra8));
        assert_eq!(check(&bgra, width, height, Some(SharedFormat::Bgra8)), Ok(SharedFormat::Bgra8));

        // Every wrong size is refused, in both directions.
        for (w, h) in [(width * 2, height), (width, height * 2), (width / 2, height / 2), (width + 1, height)] {
            let err = check(&rgba, w, h, None).expect_err("a wrong size must be refused");
            assert!(err.contains("was told"), "{err}");
        }
        // A wrong format is refused rather than sampled with red and blue swapped.
        let err = check(&rgba, width, height, Some(SharedFormat::Bgra8)).expect_err("wrong format");
        assert!(err.contains("was told"), "{err}");
        // A handle that is not a texture is refused with a message that points at the adapter.
        let (_fence, fence_handle) = rgba.make_fence();
        let not_a_texture = D3d11Import::new(dup(fence_handle), width, height);
        let err = crate::renderer::check_texture(support, &not_a_texture).expect_err("a fence is not a texture");
        assert!(err.contains("could not be opened") && err.contains("adapter LUID"), "{err}");
        let _ = windows::Win32::Foundation::CloseHandle(fence_handle);
    }
}

/// A D3D11 fence imported as a timeline semaphore: `sync_import` waits for the producer's value
/// (and times out, rather than hanging, when it never comes), and a value signaled from Vulkan is
/// visible to D3D.
#[test]
fn imported_d3d_fence_waits_for_the_producer_and_signals_back() {
    unsafe {
        let Some(bench) = Bench::open() else { return };
        let Some(support) = &bench.d3d else { return };
        if support.semaphore.is_none() {
            eprintln!("fence test skipped: {} cannot import D3D fences", bench.name);
            return;
        }
        let Some(luid) = bench.device_luid() else { return };
        let (width, height) = (64, 64);
        let Some(producer) =
            Producer::new(luid, width, height, DXGI_FORMAT_R8G8B8A8_UNORM, &pattern(width, height, 0))
        else {
            eprintln!("fence test skipped: no D3D11 device on the Vulkan adapter");
            return;
        };
        let (fence, fence_handle) = producer.make_fence();
        let mut imported =
            d3d_import::import(
                &bench.device,
                &support.memory,
                &bench.memory_properties,
                producer.import(width, height),
                SharedFormat::Rgba8,
            )
            .expect("import the texture");
        let ticket = |wait, signal| D3d11Fence::new(dup(fence_handle), wait, signal);

        // The producer has not signaled value 1, so the wait is bounded and reports it.
        let started = Instant::now();
        let late = crate::renderer::sync_import(&bench.device, bench.queue, support, &mut imported, Some(ticket(1, 2)))
            .expect_err("the fence is still at 0");
        assert!(late.contains("timed out"), "{late}");
        assert!(started.elapsed() >= d3d_import::FENCE_WAIT, "returned before the wait bound");
        assert!(imported.pending_signal.is_none(), "a dropped submission must not signal");
        let first_semaphore = imported.fence.as_ref().expect("fence imported on first use").semaphore;

        // Now the producer signals from its GPU queue; the same fence handle is recognized and reused.
        producer.signal(&fence, 1);
        crate::renderer::sync_import(&bench.device, bench.queue, support, &mut imported, Some(ticket(1, 2)))
            .expect("the fence reached 1");
        assert_eq!(imported.fence.as_ref().unwrap().semaphore, first_semaphore, "the fence import is cached");
        assert_eq!(imported.pending_signal, Some(2));

        // The render submit signals 2 once it has sampled the texture; D3D sees it.
        let semaphores = [first_semaphore];
        let values = [2u64];
        let mut timeline = vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&values);
        bench
            .device
            .queue_submit(
                bench.queue,
                &[vk::SubmitInfo::default().signal_semaphores(&semaphores).push_next(&mut timeline)],
                vk::Fence::null(),
            )
            .unwrap();
        bench.device.queue_wait_idle(bench.queue).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(1);
        while fence.GetCompletedValue() < 2 {
            assert!(Instant::now() < deadline, "D3D never saw the value Vulkan signaled");
            std::thread::yield_now();
        }

        // No fence in a submission clears a stale signal request.
        imported.pending_signal = Some(9);
        crate::renderer::sync_import(&bench.device, bench.queue, support, &mut imported, None).unwrap();
        assert!(imported.pending_signal.is_none());
        imported.destroy(&bench.device);
        let _ = windows::Win32::Foundation::CloseHandle(fence_handle);
    }
}
