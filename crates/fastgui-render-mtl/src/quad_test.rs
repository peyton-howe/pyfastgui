//! GPU check of the chrome quad pipeline: draw `build_quads` output with the real Metal shaders
//! into an offscreen texture, read it back, and compare with the CPU painter
//! (`fastgui_chrome::testing::check_gpu_backend`).
//! Needs a Metal device, so it runs on any Mac — no window or screen recording involved.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLoadAction, MTLOrigin, MTLPixelFormat, MTLRegion, MTLRenderPassDescriptor, MTLSize, MTLStorageMode,
    MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};

use crate::pipeline::QuadPipeline;
use crate::renderer::QuadChrome;

struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: QuadPipeline,
    target: Retained<ProtocolObject<dyn MTLTexture>>,
    chrome: Option<QuadChrome>,
    width: u32,
    height: u32,
}

impl Gpu {
    fn new(width: u32, height: u32) -> Self {
        let device: Retained<ProtocolObject<dyn MTLDevice>> =
            unsafe { Retained::from_raw(MTLCreateSystemDefaultDevice()) }.expect("a Metal device");
        let queue = device.newCommandQueue().unwrap();
        let pipeline = QuadPipeline::new(&device, MTLPixelFormat::BGRA8Unorm).unwrap();
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::BGRA8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);
        let target = device.newTextureWithDescriptor(&descriptor).unwrap();
        Self { device, queue, pipeline, target, chrome: None, width, height }
    }

    /// Apply a `build_quads` result (if any) and draw the current quads; returns RGBA8 pixels.
    fn draw(&mut self, frame: Option<fastgui_core::ChromeQuads<'_>>) -> Vec<u8> {
        if let Some(frame) = frame {
            self.chrome = Some(QuadChrome::update(&self.device, self.chrome.take(), &frame).unwrap());
        }
        let pass = MTLRenderPassDescriptor::renderPassDescriptor();
        let attachment = unsafe { pass.colorAttachments().objectAtIndexedSubscript(0) };
        attachment.setTexture(Some(&self.target));
        attachment.setLoadAction(MTLLoadAction::Clear);
        attachment.setStoreAction(MTLStoreAction::Store);
        attachment.setClearColor(MTLClearColor { red: 0.0, green: 0.0, blue: 0.0, alpha: 0.0 });
        let commands = self.queue.commandBuffer().unwrap();
        let encoder = commands.renderCommandEncoderWithDescriptor(&pass).unwrap();
        self.chrome.as_ref().unwrap().encode(&encoder, &self.pipeline, self.width, self.height);
        encoder.endEncoding();
        commands.commit();
        unsafe { commands.waitUntilCompleted() };

        let mut bgra = vec![0u8; (self.width * self.height * 4) as usize];
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: self.width as usize, height: self.height as usize, depth: 1 },
        };
        unsafe {
            self.target.getBytes_bytesPerRow_fromRegion_mipmapLevel(
                NonNull::new(bgra.as_mut_ptr() as *mut c_void).unwrap(),
                self.width as usize * 4,
                region,
                0,
            );
        }
        bgra.as_chunks::<4>().0.iter().flat_map(|p| [p[2], p[1], p[0], p[3]]).collect()
    }
}

#[test]
fn metal_quads_match_cpu_painter() {
    fastgui_chrome::testing::check_gpu_backend(Gpu::new, |gpu, frame| gpu.draw(frame));
}
