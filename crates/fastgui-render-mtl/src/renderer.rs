use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::ptr::NonNull;

use fastgui_core::widget::Rect;
use fastgui_core::{ChromeFrame, ChromeQuads, CpuFrame};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType;
use objc2_app_kit::NSView;
use objc2_foundation::CGSize;
use objc2_metal::{
    MTLBuffer, MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLResourceOptions, MTLScissorRect, MTLStoreAction,
    MTLViewport,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};

use crate::error::MtlRendererError as Error;
use crate::pipeline::{QuadPipeline, ViewportPipeline};
use crate::texture::ViewportTexture;

struct SampledLayer {
    texture: ViewportTexture,
}

/// Chrome as GPU draw data (`set_chrome_quads`): the instance buffer and the atlas it samples.
pub(crate) struct QuadChrome {
    /// `None` when there are no quads.
    instances: Option<Retained<ProtocolObject<dyn MTLBuffer>>>,
    count: usize,
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
        Ok(Self { instances, count: frame.quads.len(), atlas, width: frame.width, height: frame.height })
    }

    /// One instanced draw of every quad (a 4-vertex strip per instance) into a `target_width` x
    /// `target_height` attachment, in the quads' own pixel space.
    pub(crate) fn encode(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        pipeline: &QuadPipeline,
        target_width: u32,
        target_height: u32,
    ) {
        let Some(instances) = &self.instances else { return };
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
            encoder.setVertexBuffer_offset_atIndex(Some(instances), 0, 0);
            encoder.setVertexBytes_length_atIndex(NonNull::from(&size).cast(), std::mem::size_of_val(&size), 1);
            encoder.setFragmentTexture_atIndex(Some(&self.atlas.texture), 0);
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                self.count,
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
    /// GPU textures for in-tree `Viewport` widgets, keyed by the stable `viewport_id`.
    layers: HashMap<u64, SampledLayer>,
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
        let layer = self.upsert_cpu_layer(existing, frame)?;
        self.layers.insert(viewport_id, layer);
        Ok(())
    }

    fn upsert_cpu_layer(
        &self,
        existing: Option<SampledLayer>,
        frame: CpuFrame,
    ) -> Result<SampledLayer, Error> {
        let texture = match existing {
            Some(SampledLayer { texture })
                if texture.width == frame.width && texture.height == frame.height =>
            {
                texture
            }
            _ => ViewportTexture::new(&self.device, frame.width, frame.height)?,
        };
        texture.upload(&frame.data);
        Ok(SampledLayer { texture })
    }

    pub fn retain_layers(&mut self, live_ids: &[u64]) {
        let live: HashSet<u64> = live_ids.iter().copied().collect();
        self.layers.retain(|id, _| live.contains(id));
    }

    pub fn render_frame(
        &mut self,
        clear_color: [f32; 4],
        draw_chrome: bool,
        viewports: &[(u64, Rect)],
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

        let command_buffer = self.queue.commandBuffer().ok_or(Error::NoCommandBuffer)?;
        let encoder = command_buffer
            .renderCommandEncoderWithDescriptor(&pass)
            .ok_or(Error::NoEncoder)?;

        if draw_chrome {
            if let Some(chrome) = &self.quad_chrome {
                chrome.encode(&encoder, &self.quad_pipeline, self.width, self.height);
            }
            if let Some(chrome) = &self.chrome {
                encoder.setRenderPipelineState(self.pipeline.state());
                self.draw_sampled(
                    &encoder,
                    chrome,
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
        if viewports.iter().any(|(id, _)| self.layers.contains_key(id)) {
            encoder.setRenderPipelineState(self.pipeline.state());
        }
        for (viewport_id, rect) in viewports {
            let Some(layer) = self.layers.get(viewport_id) else { continue };
            let Some((viewport, scissor)) = widget_rect_to_mtl(*rect, self.width, self.height)
            else {
                continue;
            };
            self.draw_sampled(&encoder, layer, viewport, scissor);
        }

        encoder.endEncoding();
        command_buffer.presentDrawable(ProtocolObject::from_ref(&*drawable));
        command_buffer.commit();
        self.last_commands = Some(command_buffer);
        Ok(())
    }

    fn draw_sampled(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        layer: &SampledLayer,
        viewport: MTLViewport,
        scissor: MTLScissorRect,
    ) {
        encoder.setViewport(viewport);
        encoder.setScissorRect(scissor);
        unsafe {
            encoder.setFragmentTexture_atIndex(Some(&layer.texture.texture), 0);
            encoder.setFragmentSamplerState_atIndex(Some(self.pipeline.sampler()), 0);
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::Triangle, 0, 3);
        }
    }
}

fn widget_rect_to_mtl(
    rect: Rect,
    width: u32,
    height: u32,
) -> Option<(MTLViewport, MTLScissorRect)> {
    let x = rect.x.round().max(0.0);
    let y = rect.y.round().max(0.0);
    let w = rect.width.round().max(0.0);
    let h = rect.height.round().max(0.0);
    if w < 1.0 || h < 1.0 {
        return None;
    }
    let x = (x as u32).min(width);
    let y = (y as u32).min(height);
    let w = (w as u32).min(width.saturating_sub(x));
    let h = (h as u32).min(height.saturating_sub(y));
    if w == 0 || h == 0 {
        return None;
    }
    Some((
        MTLViewport {
            originX: x as f64,
            originY: y as f64,
            width: w as f64,
            height: h as f64,
            znear: 0.0,
            zfar: 1.0,
        },
        MTLScissorRect { x: x as usize, y: y as usize, width: w as usize, height: h as usize },
    ))
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

    fn retain_layers(&mut self, live_ids: &[u64]) {
        MetalRenderer::retain_layers(self, live_ids)
    }

    fn render_frame(
        &mut self,
        clear: [f32; 4],
        draw_chrome: bool,
        draws: &[(u64, Rect)],
    ) -> Result<(), Self::Error> {
        MetalRenderer::render_frame(self, clear, draw_chrome, draws)
    }
}
