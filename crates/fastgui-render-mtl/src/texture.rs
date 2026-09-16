use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLDevice, MTLOrigin, MTLPixelFormat, MTLRegion, MTLSize, MTLStorageMode, MTLTexture,
    MTLTextureDescriptor, MTLTextureUsage,
};

use crate::error::MtlRendererError as Error;

/// A GPU-sampled RGBA8 texture the CPU writes into every frame -- the Metal counterpart to
/// `fastgui-render-vk::ViewportTexture`, which instead persistently maps a host-visible Vulkan
/// image and memcpys into it respecting the driver-reported row pitch. Metal has no equivalent
/// "map once, write forever" API for a sampled texture; `replaceRegion` is the sanctioned way to
/// get CPU bytes into one, and it handles any internal row padding/tiling itself -- there's no
/// row-pitch bookkeeping to do by hand here the way `ViewportTexture::upload` needs.
///
/// `StorageModeShared` unconditionally: on Apple Silicon (unified memory) this is already the
/// fastest option, and this project's primary target hardware is Apple Silicon. A discrete-GPU
/// Intel Mac would do better with `Managed` storage plus an explicit `didModifyRange` sync after
/// each `replaceRegion` -- not implemented here, mirroring the Vulkan backend's own carried-
/// forward simplifications (see `ViewportTexture`'s single-buffered-upload caveat there).
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
        debug_assert_eq!(data.len(), self.width as usize * self.height as usize * 4);
        let region = MTLRegion {
            origin: MTLOrigin { x: 0, y: 0, z: 0 },
            size: MTLSize { width: self.width as usize, height: self.height as usize, depth: 1 },
        };
        let bytes_per_row = self.width as usize * 4;
        // SAFETY: `data` is a valid, non-null, `bytes_per_row * height`-byte region for the
        // duration of this call -- `replaceRegion` copies out of it synchronously and retains no
        // reference to it afterward.
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).expect("frame data is never null");
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
