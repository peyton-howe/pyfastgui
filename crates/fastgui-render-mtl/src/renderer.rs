use std::collections::{HashMap, HashSet};

use fastgui_core::widget::Rect;
use fastgui_core::CpuFrame;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::ClassType;
use objc2_app_kit::NSView;
use objc2_foundation::{CGPoint, CGRect, CGSize};
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLoadAction, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPassDescriptor, MTLScissorRect, MTLStoreAction, MTLViewport,
};
use objc2_quartz_core::{CAMetalDrawable, CAMetalLayer};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawWindowHandle};

use crate::error::MtlRendererError as Error;
use crate::pipeline::ViewportPipeline;
use crate::texture::ViewportTexture;

struct SampledLayer {
    texture: ViewportTexture,
}

/// Clears the layer to a solid color every frame and, once a viewport source has content, draws
/// it as a full-window (or per-`Viewport`-widget-rect) textured quad on top -- the Metal
/// counterpart to `fastgui-render-vk::VulkanRenderer`. No CUDA interop path: see
/// `crate::command::Command`'s doc comment for why that's out of scope on macOS.
pub struct MetalRenderer {
    // Kept alive for `resize()` (reads `.window().backingScaleFactor()`) -- otherwise unused
    // after `new()` sets `wantsLayer`/`layer` once.
    view: Retained<NSView>,
    layer: Retained<CAMetalLayer>,
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: ViewportPipeline,
    chrome: Option<SampledLayer>,
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

        let device = unsafe { MTLCreateSystemDefaultDevice() };
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            unsafe { Retained::retain(device) }.ok_or(Error::NoDevice)?;
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

        let mut renderer = Self {
            view,
            layer,
            device,
            queue,
            pipeline,
            chrome: None,
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

        let scale = self.view.window().map(|w| w.backingScaleFactor()).unwrap_or(1.0);
        let point_size =
            CGSize { width: width as f64 / scale, height: height as f64 / scale };
        self.layer.setFrame(CGRect { origin: CGPoint { x: 0.0, y: 0.0 }, size: point_size });
        self.layer.setContentsScale(scale);
        unsafe {
            self.layer.setDrawableSize(CGSize { width: width as f64, height: height as f64 });
        }
        Ok(())
    }

    pub fn set_chrome_frame(&mut self, frame: CpuFrame) -> Result<(), Error> {
        let existing = self.chrome.take();
        let layer = self.upsert_cpu_layer(existing, frame)?;
        self.chrome = Some(layer);
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

        let bind_pipeline = (draw_chrome && self.chrome.is_some())
            || viewports.iter().any(|(id, _)| self.layers.contains_key(id));
        if bind_pipeline {
            encoder.setRenderPipelineState(self.pipeline.state());
        }

        if draw_chrome {
            if let Some(chrome) = &self.chrome {
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

impl fastgui_render::Renderer for MetalRenderer {
    type Error = Error;

    fn new(
        window: &(impl HasWindowHandle + HasDisplayHandle),
        width: u32,
        height: u32,
    ) -> Result<Self, Self::Error> {
        MetalRenderer::new(window, width, height)
    }

    fn resize(&mut self, width: u32, height: u32) -> Result<(), Self::Error> {
        MetalRenderer::resize(self, width, height)
    }

    fn render_frame(&mut self, color: [f32; 4]) -> Result<(), Self::Error> {
        MetalRenderer::render_frame(self, color, false, &[])
    }
}
