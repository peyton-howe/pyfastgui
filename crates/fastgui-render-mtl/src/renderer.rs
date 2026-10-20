use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr::NonNull;

use fastgui_app::ViewportDraw;
use fastgui_core::{ChromeFrame, ChromeQuad, ChromeQuads, CpuFrame, ImportedLayer};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType;
use objc2_app_kit::NSView;
use objc2_foundation::CGSize;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLResourceOptions, MTLScissorRect, MTLStoreAction,
    MTLTexture, MTLViewport,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};

use crate::error::MtlRendererError as Error;
use crate::import::{OwnedMetalImport, PendingSync};
use crate::pipeline::{QuadPipeline, ViewportPipeline};
use crate::texture::ViewportTexture;

struct SampledLayer {
    texture: ViewportTexture,
}

enum LayerPixels {
    Owned(ViewportTexture),
    Imported(OwnedMetalImport),
}

struct ViewportLayer {
    pixels: LayerPixels,
}

impl ViewportLayer {
    fn size(&self) -> (u32, u32) {
        match &self.pixels {
            LayerPixels::Owned(texture) => (texture.width, texture.height),
            LayerPixels::Imported(texture) => (texture.width, texture.height),
        }
    }

    fn texture(&self) -> &ProtocolObject<dyn MTLTexture> {
        match &self.pixels {
            LayerPixels::Owned(texture) => &texture.texture,
            LayerPixels::Imported(texture) => &texture.texture,
        }
    }
}

/// Chrome as GPU draw data (`set_chrome_quads`): the instance buffer and the atlas it samples.
pub(crate) struct QuadChrome {
    /// `None` when there are no quads.
    instances: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    count: usize,
    /// `ChromeQuads::overlay_start`.
    overlay_start: usize,
    atlas: ViewportTexture,
    /// Physical size the quads were built for (their coordinate space).
    width: u32,
    height: u32,
}

impl QuadChrome {
    /// Apply `frame` on top of `previous` (reusing its atlas when the size matches, so earlier
    /// uploads stay valid).
    pub(crate) fn update(
        device: &ProtocolObject<dyn MTLDevice>,
        previous: Option<QuadChrome>,
        frame: &ChromeQuads<'_>,
    ) -> Result<Self, Error> {
        let atlas = match previous {
            Some(chrome) if chrome.atlas.width == frame.atlas_size => chrome.atlas,
            _ => ViewportTexture::new(device, frame.atlas_size, frame.atlas_size)?,
        };
        for upload in &frame.atlas_uploads {
            atlas.write_region(upload.rect, upload.pixels);
        }
        let instances = if frame.quads.is_empty() {
            None
        } else {
            let bytes = std::mem::size_of_val(frame.quads);
            // SAFETY: `frame.quads` is `bytes` long and only read during this call (the buffer is
            // a copy). `ChromeQuad` is `#[repr(C)]`, matching the shader's `Quad` layout.
            let ptr = NonNull::new(frame.quads.as_ptr() as *mut c_void).expect("slice pointers are never null");
            let buffer = unsafe {
                device.newBufferWithBytes_length_options(ptr, bytes, MTLResourceOptions::MTLResourceStorageModeShared)
            };
            Some(buffer.ok_or(Error::NoBuffer)?)
        };
        Ok(Self {
            instances,
            count: frame.quads.len(),
            overlay_start: frame.overlay_start.min(frame.quads.len()),
            atlas,
            width: frame.width,
            height: frame.height,
        })
    }

    /// One instanced draw of every quad (a 4-vertex strip per instance) into a `target_width` x
    /// `target_height` attachment, in the quads' own pixel space.
    /// Draw the quads before the overlay (`overlay: false`) or the overlay quads (`true`).
    pub(crate) fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        pipeline: &QuadPipeline,
        target_width: u32,
        target_height: u32,
        overlay: bool,
    ) {
        let Some(instances) = &self.instances else { return };
        let (first, count) = if overlay {
            (self.overlay_start, self.count - self.overlay_start)
        } else {
            (0, self.overlay_start)
        };
        if count == 0 {
            return;
        }
        encoder.setRenderPipelineState(pipeline.state());
        encoder.setViewport(MTLViewport {
            originX: 0.0,
            originY: 0.0,
            width: target_width as f64,
            height: target_height as f64,
            znear: 0.0,
            zfar: 1.0,
        });
        encoder.setScissorRect(MTLScissorRect { x: 0, y: 0, width: target_width as usize, height: target_height as usize });
        let size = [self.width as f32, self.height as f32];
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(Some(instances), first * std::mem::size_of::<ChromeQuad>(), 0);
            encoder.setVertexBytes_length_atIndex(NonNull::from(&size).cast(), std::mem::size_of_val(&size), 1);
            encoder.setFragmentTexture_atIndex(Some(&self.atlas.texture), 0);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                count,
            );
        }
    }
}

/// Clears the layer to a solid color every frame and, once a viewport source has content, draws
/// it as a full-window (or per-`Viewport`-widget-rect) textured quad on top.
pub struct MetalRenderer {
    // Kept alive for `resize()` (reads `.window().backingScaleFactor()`) -- otherwise unused
    // after `new()` sets `wantsLayer`/`layer` once.
    view: Retained<NSView>,
    layer: Retained<CAMetalLayer>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: ViewportPipeline,
    quad_pipeline: QuadPipeline,
    /// CPU-rasterized chrome (`set_chrome_frame`); cleared when quads are set, and vice versa.
    chrome: Option<SampledLayer>,
    quad_chrome: Option<QuadChrome>,
    /// The last submitted frame, waited on before a repacked atlas is overwritten.
    last_commands: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    /// GPU textures for in-tree `Viewport` / `Image` widgets, keyed by the stable layer id.
    layers: HashMap<u64, ViewportLayer>,
    width: u32,
    height: u32,
}

impl MetalRenderer {
    pub fn new(
        window: &(impl HasWindowHandle + HasDisplayHandle),
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let window_handle = window.window_handle()?.as_raw();
        let RawWindowHandle::AppKit(handle) = window_handle else {
            return Err(Error::NotAppKit);
        };
        // SAFETY: `handle.ns_view` is a valid, live `NSView*` for as long as `window` is --
        // guaranteed by `HasWindowHandle`'s contract. `Retained::retain` takes our own +1
        // reference; it doesn't consume or invalidate the borrowed pointer.
        let view: Retained<NSView> =
            unsafe { Retained::retain(handle.ns_view.as_ptr().cast()) }.ok_or(Error::NotAppKit)?;

        // Create-rule (+1). `from_raw` takes that ownership; `retain` would leak the device.
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            unsafe { Retained::from_raw(MTLCreateSystemDefaultDevice()) }.ok_or(Error::NoDevice)?;
        let queue = device.newCommandQueue().ok_or(Error::NoCommandQueue)?;

        // UNORM, deliberately not SRGB -- same reasoning as `fastgui-render-vk::VulkanRenderer`'s
        // choice of `B8G8R8A8_UNORM`: every color already meant as a final display byte value
        // (clear colors, `CpuFrame` bytes, `fastgui-chrome`'s rasterized UI) should reach the
        // drawable unmodified, not silently gamma-re-encoded.
        let layer = unsafe { CAMetalLayer::layer() };
        unsafe {
            layer.setDevice(Some(&device));
            layer.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
            layer.setFramebufferOnly(true);
        }

        // Layer-hosting (not layer-backed): AppKit does not keep a manually assigned `.layer`'s
        // frame/contentsScale in sync with the view on its own -- `resize()` below does that
        // explicitly on every call, seeded here with the initial size.
        view.setWantsLayer(true);
        unsafe {
            view.setLayer(Some(layer.as_super()));
        }

        let pipeline = ViewportPipeline::new(&device, MTLPixelFormat::BGRA8Unorm)?;
        let quad_pipeline = QuadPipeline::new(&device, MTLPixelFormat::BGRA8Unorm)?;

        let mut renderer = Self {
            view,
            layer,
            device,
            queue,
            pipeline,
            quad_pipeline,
            chrome: None,
            quad_chrome: None,
            last_commands: None,
            layers: HashMap::new(),
            width,
            height,
        };
        renderer.resize(width, height)?;
        Ok(renderer)
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), Error> {
        self.width = width;
        self.height = height;
        if width == 0 || height == 0 {
            // Window is minimized; keep the existing (possibly stale) layer size and skip
            // rendering until a real resize arrives -- same convention as the Vulkan backend's
            // `recreate_swapchain`.
            return Ok(());
        }

        let bounds = self.view.bounds();
        self.layer.setFrame(bounds);
        let scale = self.view.window().map(|w| w.backingScaleFactor()).unwrap_or(1.0);
        self.layer.setContentsScale(scale);
        unsafe {
            self.layer.setDrawableSize(CGSize { width: width as f64, height: height as f64 });
        }
        Ok(())
    }

    /// Take the chrome as quads + atlas uploads (see `fastgui_chrome::ChromeRenderer::build_quads`).
    pub fn set_chrome_quads(&mut self, frame: &ChromeQuads<'_>) -> Result<(), Error> {
        self.chrome = None;
        if frame.atlas_repacked {
            // Slots are being reused: let the frame that may still sample the old layout finish.
            if let Some(commands) = &self.last_commands {
                unsafe { commands.waitUntilCompleted() };
            }
        }
        self.quad_chrome = Some(QuadChrome::update(&self.device, self.quad_chrome.take(), frame)?);
        Ok(())
    }

    /// Copies only `frame.damage` when the existing texture already holds the previous frame
    /// at this size; see `SurfaceBackend::set_chrome_frame`.
    pub fn set_chrome_frame(&mut self, frame: &ChromeFrame<'_>) -> Result<(), Error> {
        self.quad_chrome = None;
        match (&self.chrome, frame.damage) {
            (Some(SampledLayer { texture }), Some(damage))
                if texture.width == frame.width && texture.height == frame.height =>
            {
                for rect in damage {
                    texture.upload_rect(frame.data, *rect);
                }
            }
            _ => {
                let texture = match self.chrome.take() {
                    Some(SampledLayer { texture })
                        if texture.width == frame.width && texture.height == frame.height =>
                    {
                        texture
                    }
                    _ => ViewportTexture::new(&self.device, frame.width, frame.height)?,
                };
                texture.upload(frame.data);
                self.chrome = Some(SampledLayer { texture });
            }
        }
        Ok(())
    }

    pub fn set_layer_frame(&mut self, viewport_id: u64, frame: CpuFrame) -> Result<(), Error> {
        let existing = self.layers.remove(&viewport_id);
        match existing {
            Some(ViewportLayer { pixels: LayerPixels::Owned(texture) })
                if texture.width == frame.width && texture.height == frame.height =>
            {
                texture.upload(&frame.data);
                self.layers.insert(viewport_id, ViewportLayer { pixels: LayerPixels::Owned(texture) });
            }
            other => {
                if let Some(old) = other {
                    self.retire_layer(old);
                }
                let texture = ViewportTexture::new(&self.device, frame.width, frame.height)?;
                texture.upload(&frame.data);
                self.layers.insert(viewport_id, ViewportLayer { pixels: LayerPixels::Owned(texture) });
            }
        }
        Ok(())
    }

    pub fn set_imported_layer(&mut self, viewport_id: u64, layer: ImportedLayer) -> Result<(), Error> {
        let ImportedLayer::Metal { retained, .. } = layer else {
            return Ok(());
        };
        let Ok(import) = retained.downcast::<OwnedMetalImport>() else {
            return Ok(());
        };
        if let Some(old) = self.layers.remove(&viewport_id) {
            self.retire_layer(old);
        }
        self.layers.insert(viewport_id, ViewportLayer { pixels: LayerPixels::Imported(*import) });
        Ok(())
    }

    /// An imported texture that a previous command buffer sampled has to finish before its
    /// retain is released. One that never reached a command buffer just drops (and signals).
    fn retire_layer(&mut self, layer: ViewportLayer) {
        let sampled = matches!(&layer.pixels, LayerPixels::Imported(import) if !import.sync_pending);
        if sampled {
            if let Some(commands) = &self.last_commands {
                unsafe { commands.waitUntilCompleted() };
            }
        }
        drop(layer);
    }

    pub fn retain_layers(&mut self, live_ids: &[u64]) {
        let live: HashSet<u64> = live_ids.iter().copied().collect();
        let stale: Vec<u64> = self.layers.keys().copied().filter(|id| !live.contains(id)).collect();
        for id in stale {
            if let Some(layer) = self.layers.remove(&id) {
                self.retire_layer(layer);
            }
        }
    }

    /// Waits and signals for imports this frame will actually sample. Hidden layers keep their
    /// sync until a later frame draws them, so a producer is not told the GPU is done early.
    fn take_pending_sync(&mut self, draws: &[ViewportDraw]) -> Vec<PendingSync> {
        let mut ids = Vec::new();
        for draw in draws {
            let Some(layer) = self.layers.get(&draw.viewport_id) else { continue };
            let (width, height) = layer.size();
            if draw.letterboxed(width, height).viewport_and_scissor(self.width, self.height).is_some() {
                ids.push(draw.viewport_id);
            }
        }
        let mut sync = Vec::new();
        for id in ids {
            let Some(ViewportLayer { pixels: LayerPixels::Imported(import) }) = self.layers.get_mut(&id) else {
                continue;
            };
            if !import.sync_pending {
                continue;
            }
            import.sync_pending = false;
            sync.push(PendingSync { wait: import.wait.clone(), signal: import.signal.clone() });
        }
        sync
    }

    pub fn render_frame(
        &mut self,
        clear_color: [f32; 4],
        draw_chrome: bool,
        viewports: &[ViewportDraw],
    ) -> Result<(), Error> {
        if self.width == 0 || self.height == 0 {
            return Ok(());
        }
        // `nextDrawable` legitimately returns `None` for a handful of frames around resize/
        // occlusion (e.g. the layer temporarily has no free drawable) -- skip this frame rather
        // than treating it as an error, same as `fastgui-render-vk`'s `ERROR_OUT_OF_DATE_KHR`
        // handling one layer down (a stale/recreating swapchain, not a real failure).
        let Some(drawable) = (unsafe { self.layer.nextDrawable() }) else {
            return Ok(());
        };
        let sync = self.take_pending_sync(viewports);

        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        let color_attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        let drawable_texture = unsafe { drawable.texture() };
        color_attachment.setTexture(Some(&drawable_texture));
        color_attachment.setLoadAction(MTLLoadAction::Clear);
        color_attachment.setStoreAction(MTLStoreAction::Store);
        color_attachment.setClearColor(MTLClearColor {
            red: clear_color[0] as f64,
            green: clear_color[1] as f64,
            blue: clear_color[2] as f64,
            alpha: clear_color[3] as f64,
        });

        let Some(command_buffer) = self.queue.commandBuffer() else {
            flush_sync(&self.queue, &sync);
            return Err(Error::NoCommandBuffer);
        };
        for op in &sync {
            op.encode_waits(&command_buffer);
        }
        let Some(encoder) = command_buffer.renderCommandEncoderWithDescriptor(&pass) else {
            for op in &sync {
                op.encode_signal(&command_buffer);
            }
            command_buffer.commit();
            return Err(Error::NoEncoder);
        };

        if draw_chrome {
            if let Some(chrome) = &self.quad_chrome {
                chrome.encode(&encoder, &self.quad_pipeline, self.width, self.height, false);
            }
            if let Some(chrome) = &self.chrome {
                encoder.setRenderPipelineState(self.pipeline.state());
                self.draw_sampled(
                    &encoder,
                    &chrome.texture.texture,
                    MTLViewport {
                        originX: 0.0,
                        originY: 0.0,
                        width: self.width as f64,
                        height: self.height as f64,
                        znear: 0.0,
                        zfar: 1.0,
                    },
                    MTLScissorRect {
                        x: 0,
                        y: 0,
                        width: self.width as usize,
                        height: self.height as usize,
                    },
                );
            }
        }
        if viewports.iter().any(|draw| self.layers.contains_key(&draw.viewport_id)) {
            encoder.setRenderPipelineState(self.pipeline.state());
        }
        for draw in viewports {
            let Some(layer) = self.layers.get(&draw.viewport_id) else { continue };
            let (content_w, content_h) = layer.size();
            let draw = draw.letterboxed(content_w, content_h);
            let Some(([x, y, w, h], [sx, sy, sw, sh])) = draw.viewport_and_scissor(self.width, self.height) else {
                continue;
            };
            let viewport = MTLViewport {
                originX: f64::from(x),
                originY: f64::from(y),
                width: f64::from(w),
                height: f64::from(h),
                znear: 0.0,
                zfar: 1.0,
            };
            let scissor = MTLScissorRect { x: sx as usize, y: sy as usize, width: sw as usize, height: sh as usize };
            self.draw_sampled(&encoder, layer.texture(), viewport, scissor);
        }
        // Popups go over video too.
        if draw_chrome {
            if let Some(chrome) = &self.quad_chrome {
                chrome.encode(&encoder, &self.quad_pipeline, self.width, self.height, true);
            }
        }

        encoder.endEncoding();
        for op in &sync {
            op.encode_signal(&command_buffer);
        }
        command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
        command_buffer.commit();
        self.last_commands = Some(command_buffer);
        Ok(())
    }

    fn draw_sampled(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        texture: &ProtocolObject<dyn MTLTexture>,
        viewport: MTLViewport,
        scissor: MTLScissorRect,
    ) {
        encoder.setViewport(viewport);
        encoder.setScissorRect(scissor);
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(texture), 0);
            encoder.setFragmentSamplerState_atIndex(Some(self.pipeline.sampler()), 0);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
    }
}

fn flush_sync(queue: &ProtocolObject<dyn MTLCommandQueue>, sync: &[PendingSync]) {
    if sync.is_empty() {
        return;
    }
    let Some(command_buffer) = queue.commandBuffer() else { return };
    for op in sync {
        op.encode_waits(&command_buffer);
        op.encode_signal(&command_buffer);
    }
    command_buffer.commit();
}


impl fastgui_app::SurfaceBackend for MetalRenderer {
    type Error = Error;

    fn new(
        window: &winit::window::Window,
        physical_width: u32,
        physical_height: u32,
    ) -> Result<Self, Self::Error> {
        MetalRenderer::new(window, physical_width, physical_height)
    }

    fn resize(&mut self, physical_width: u32, physical_height: u32) -> Result<(), Self::Error> {
        MetalRenderer::resize(self, physical_width, physical_height)
    }

    fn set_chrome_frame(&mut self, frame: &ChromeFrame<'_>) -> Result<(), Self::Error> {
        MetalRenderer::set_chrome_frame(self, frame)
    }

    fn set_chrome_quads(&mut self, frame: &ChromeQuads<'_>) -> Result<(), Self::Error> {
        MetalRenderer::set_chrome_quads(self, frame)
    }

    fn set_layer_frame(&mut self, viewport_id: u64, frame: CpuFrame) -> Result<(), Self::Error> {
        MetalRenderer::set_layer_frame(self, viewport_id, frame)
    }

    fn set_imported_layer(&mut self, viewport_id: u64, layer: ImportedLayer) -> Result<(), Self::Error> {
        MetalRenderer::set_imported_layer(self, viewport_id, layer)
    }

    fn retain_layers(&mut self, live_ids: &[u64]) {
        MetalRenderer::retain_layers(self, live_ids)
    }

    fn surface_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn render_frame(
        &mut self,
        clear: [f32; 4],
        draw_chrome: bool,
        draws: &[ViewportDraw],
    ) -> Result<(), Self::Error> {
        MetalRenderer::render_frame(self, clear, draw_chrome, draws)
    }
}

mod window_time {
    use std::ffi::c_void;
    use std::ptr::NonNull;
    use std::time::Instant;

    use fastgui_core::widget::{LayerFit, Rect};
    use fastgui_core::{CpuFrame, PixelFormat};
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCreateSystemDefaultDevice,
        MTLDevice, MTLPixelFormat, MTLResourceOptions, MTLSize, MTLStorageMode, MTLTexture, MTLTextureDescriptor,
        MTLTextureUsage,
    };
    use winit::application::ApplicationHandler;
    use winit::dpi::LogicalSize;
    use winit::event::WindowEvent;
    use winit::event_loop::ActiveEventLoop;
    use winit::window::{Window, WindowId};

    use super::MetalRenderer;
    use crate::import::prepare_metal_import;
    use fastgui_app::ViewportDraw;

    const WARMUP: usize = 8;
    const FRAMES: usize = 40;

    fn solid(width: u32, height: u32) -> Vec<u8> {
        let mut pixels = vec![0u8; width as usize * height as usize * 4];
        for px in pixels.chunks_exact_mut(4) {
            px.copy_from_slice(&[220, 40, 40, 255]);
        }
        pixels
    }

    fn private_texture(pixels: &[u8], width: u32, height: u32) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            unsafe { Retained::from_raw(MTLCreateSystemDefaultDevice()) }.expect("device");
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Private);
        let texture = device.newTextureWithDescriptor(&descriptor).expect("texture");
        let staging = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(pixels.as_ptr() as *mut c_void).unwrap(),
                pixels.len(),
                MTLResourceOptions::MTLResourceStorageModeShared,
            )
        }
        .expect("staging");
        let queue = device.newCommandQueue().expect("queue");
        let commands = queue.commandBuffer().expect("command buffer");
        let blit = commands.blitCommandEncoder().expect("blit");
        unsafe {
            blit.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &staging,
                0,
                (width as usize) * 4,
                pixels.len(),
                MTLSize { width: width as usize, height: height as usize, depth: 1 },
                &texture,
                0,
                0,
                objc2_metal::MTLOrigin { x: 0, y: 0, z: 0 },
            );
        }
        blit.endEncoding();
        commands.commit();
        unsafe { commands.waitUntilCompleted() };
        texture
    }

    fn summarize(label: &str, mut samples: Vec<f64>) {
        assert!(!samples.is_empty(), "{label} produced no frames");
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

    pub(super) struct Bench {
        pub(super) window: Option<Window>,
        pub(super) done: bool,
    }

    fn time_offscreen(pixels: &[u8], texture: &ProtocolObject<dyn MTLTexture>, width: u32, height: u32) {
        use objc2_metal::{
            MTLCommandQueue, MTLLoadAction, MTLPrimitiveType, MTLRenderCommandEncoder,
            MTLRenderPassDescriptor, MTLResource, MTLScissorRect, MTLStoreAction, MTLViewport,
        };

        use crate::pipeline::ViewportPipeline;
        use crate::texture::ViewportTexture;

        let device = texture.device();
        let pipeline = ViewportPipeline::new(&device, MTLPixelFormat::BGRA8Unorm).expect("pipeline");
        let target_desc = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::BGRA8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        target_desc.setUsage(MTLTextureUsage::RenderTarget);
        target_desc.setStorageMode(MTLStorageMode::Private);
        let target = device.newTextureWithDescriptor(&target_desc).expect("target");
        let queue = device.newCommandQueue().expect("queue");
        let owned = ViewportTexture::new(&device, width, height).expect("owned");

        let mut sample = Vec::new();
        let mut upload = Vec::new();
        for i in 0..WARMUP + FRAMES {
            let started = Instant::now();
            let commands = queue.commandBuffer().expect("command buffer");
            let pass = MTLRenderPassDescriptor::renderPassDescriptor();
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(&target));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setStoreAction(MTLStoreAction::Store);
            let encoder = commands.renderCommandEncoderWithDescriptor(&pass).expect("encoder");
            encoder.setRenderPipelineState(pipeline.state());
            encoder.setViewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: width as f64,
                height: height as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            encoder.setScissorRect(MTLScissorRect { x: 0, y: 0, width: width as usize, height: height as usize });
            unsafe {
                encoder.setFragmentTexture_atIndex(Some(texture), 0);
                encoder.setFragmentSamplerState_atIndex(Some(pipeline.sampler()), 0);
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            encoder.endEncoding();
            commands.commit();
            unsafe { commands.waitUntilCompleted() };
            if i >= WARMUP {
                sample.push(started.elapsed().as_secs_f64() * 1e3);
            }

            let started = Instant::now();
            owned.upload(pixels);
            let commands = queue.commandBuffer().expect("command buffer");
            let pass = MTLRenderPassDescriptor::renderPassDescriptor();
            let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
            attachment.setTexture(Some(&target));
            attachment.setLoadAction(MTLLoadAction::Clear);
            attachment.setStoreAction(MTLStoreAction::Store);
            let encoder = commands.renderCommandEncoderWithDescriptor(&pass).expect("encoder");
            encoder.setRenderPipelineState(pipeline.state());
            encoder.setViewport(MTLViewport {
                originX: 0.0,
                originY: 0.0,
                width: width as f64,
                height: height as f64,
                znear: 0.0,
                zfar: 1.0,
            });
            encoder.setScissorRect(MTLScissorRect { x: 0, y: 0, width: width as usize, height: height as usize });
            unsafe {
                encoder.setFragmentTexture_atIndex(Some(&owned.texture), 0);
                encoder.setFragmentSamplerState_atIndex(Some(pipeline.sampler()), 0);
                encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
            }
            encoder.endEncoding();
            commands.commit();
            unsafe { commands.waitUntilCompleted() };
            if i >= WARMUP {
                upload.push(started.elapsed().as_secs_f64() * 1e3);
            }
        }
        summarize(&format!("offscreen sample {width}x{height}"), sample);
        summarize(&format!("offscreen upload+sample {width}x{height}"), upload);
    }

    impl Bench {
        fn present(renderer: &mut MetalRenderer, draw: &ViewportDraw) -> Option<f64> {
            let before = renderer.last_commands.as_ref().map(|commands| Retained::as_ptr(commands) as usize);
            let started = Instant::now();
            renderer.render_frame([0.05, 0.05, 0.05, 1.0], false, std::slice::from_ref(draw)).expect("render");
            let Some(commands) = &renderer.last_commands else { return None };
            if Some(Retained::as_ptr(commands) as usize) == before {
                return None;
            }
            unsafe { commands.waitUntilCompleted() };
            Some(started.elapsed().as_secs_f64() * 1e3)
        }

        fn time_import(renderer: &mut MetalRenderer, draw: &ViewportDraw, texture: &ProtocolObject<dyn MTLTexture>) {
            let layer = prepare_metal_import(texture as *const _ as usize, 0, 0, 0, 0).expect("import");
            renderer.set_imported_layer(draw.viewport_id, layer).expect("install");
            let mut present = Vec::new();
            while present.len() < WARMUP + FRAMES {
                if let Some(ms) = Self::present(renderer, draw) {
                    present.push(ms);
                }
            }
            summarize(
                &format!("import present {}x{}", texture.width(), texture.height()),
                present.split_off(WARMUP),
            );
        }

        fn time_upload(renderer: &mut MetalRenderer, draw: &ViewportDraw, pixels: &[u8], width: u32, height: u32) {
            let mut upload = Vec::new();
            let mut present = Vec::new();
            while present.len() < WARMUP + FRAMES {
                let started = Instant::now();
                let frame = CpuFrame {
                    width,
                    height,
                    format: PixelFormat::Rgba8,
                    data: pixels.to_vec(),
                };
                renderer.set_layer_frame(draw.viewport_id, frame).expect("upload");
                let upload_ms = started.elapsed().as_secs_f64() * 1e3;
                if let Some(ms) = Self::present(renderer, draw) {
                    upload.push(upload_ms);
                    present.push(ms);
                }
            }
            summarize(
                &format!("cpu copy+upload {width}x{height}"),
                upload.split_off(WARMUP),
            );
            summarize(
                &format!("cpu present {width}x{height}"),
                present.split_off(WARMUP),
            );
        }
    }

    impl ApplicationHandler for Bench {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            let window = event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("fastgui — import timing")
                        .with_inner_size(LogicalSize::new(960.0, 540.0)),
                )
                .expect("window");
            window.request_redraw();
            self.window = Some(window);
        }

        fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
            if self.done || !matches!(event, WindowEvent::RedrawRequested) {
                return;
            }
            self.done = true;
            let window = self.window.as_ref().expect("window");
            let size = window.inner_size();
            let mut renderer = MetalRenderer::new(window, size.width, size.height).expect("renderer");
            let draw = ViewportDraw {
                viewport_id: 1,
                rect: Rect { x: 0.0, y: 0.0, width: renderer.width as f32, height: renderer.height as f32 },
                visible: Rect { x: 0.0, y: 0.0, width: renderer.width as f32, height: renderer.height as f32 },
                fit: LayerFit::Stretch,
            };
            for (width, height) in [(1920, 1080), (3840, 2160)] {
                let pixels = solid(width, height);
                let texture = private_texture(&pixels, width, height);
                time_offscreen(&pixels, &texture, width, height);
                Self::time_import(&mut renderer, &draw, &texture);
                Self::time_upload(&mut renderer, &draw, &pixels, width, height);
            }
            event_loop.exit();
        }
    }

}

pub fn run_import_window_bench() {
    use winit::event_loop::EventLoop;
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.run_app(&mut window_time::Bench { window: None, done: false }).expect("run");
}
