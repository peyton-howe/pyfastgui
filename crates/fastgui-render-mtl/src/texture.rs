use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};

use fastgui_core::PixelRect;

use crate::error::MtlRendererError as Error;

/// CPU-writable RGBA8 texture sampled each frame. `StorageModeShared` is the right default on
/// unified memory; `replaceRegion` copies tightly packed `CpuFrame` bytes (no row-pitch).
pub struct ViewportTexture {
    pub texture: Retained<ProtocolObject<dyn MTLTexture>>,
    pub width: u32,
    pub height: u32,
}

impl ViewportTexture {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        width: u32,
        height: u32,
    ) -> Result<Self, Error> {
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                MTLPixelFormat::RGBA8Unorm,
                width as usize,
                height as usize,
                false,
            )
        };
        descriptor.setUsage(MTLTextureUsage::ShaderRead);
        descriptor.setStorageMode(MTLStorageMode::Shared);

        let texture = device.newTextureWithDescriptor(&descriptor).ok_or(Error::NoTexture)?;

        Ok(Self { texture, width, height })
    }

    /// Copy `data`'s tightly-packed RGBA8 bytes into the texture. `data` must be exactly
    /// `width * height * 4` bytes, row-major, no padding -- same contract as
    /// `fastgui_core::CpuFrame`.
    pub fn upload(&self, data: &[u8]) {
        self.upload_rect(data, PixelRect { x: 0, y: 0, width: self.width, height: self.height });
    }

    /// Copy just `rect` of `data` (a whole tightly packed `width`x`height` frame, same contract
    /// as `upload`) into the same rect of the texture. `rect` is clipped to the texture.
    pub fn upload_rect(&self, data: &[u8], rect: PixelRect) {
        debug_assert_eq!(data.len(), self.width as usize * self.height as usize * 4);
        let x1 = rect.right().min(self.width);
        let y1 = rect.bottom().min(self.height);
        if rect.x >= x1 || rect.y >= y1 {
            return;
        }
        let region = MTLRegion {
            origin: MTLOrigin { x: rect.x as usize, y: rect.y as usize, z: 0 },
            size: MTLSize { width: (x1 - rect.x) as usize, height: (y1 - rect.y) as usize, depth: 1 },
        };
        let bytes_per_row = self.width as usize * 4;
        let offset = rect.y as usize * bytes_per_row + rect.x as usize * 4;
        // SAFETY: `data` holds the whole frame, so from `offset` it covers `region`'s rows at
        // `bytes_per_row` stride for the duration of this call -- `replaceRegion` copies out of it
        // synchronously and retains no reference to it afterward.
        let ptr = NonNull::new(data[offset..].as_ptr() as *mut c_void).expect("frame data is never null");
        unsafe {
            self.texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
                region,
                0,
                ptr,
                bytes_per_row,
            );
        }
    }
}
