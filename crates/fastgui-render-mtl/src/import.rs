//! Retain an `MTLTexture` submitted from Python and sample it in the viewport pass.
//!
//! Validation runs on the calling thread, before the frame is queued, so a bad pointer raises
//! in Python instead of failing the window. The retained object lives in [`OwnedMetalImport`];
//! dropping it before a command buffer has encoded its signal unblocks the producer.

use objc2::rc::Retained;
use objc2::runtime::{Bool, ProtocolObject};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLCreateSystemDefaultDevice, MTLDevice, MTLEvent, MTLPixelFormat,
    MTLResource, MTLTexture, MTLTextureType, MTLTextureUsage,
};

use fastgui_core::ImportedLayer;

/// GPU object behind [`ImportedLayer::Metal`]. The renderer downcasts this out of the mailbox.
pub(crate) struct OwnedMetalImport {
    pub(crate) texture: Retained<ProtocolObject<dyn MTLTexture>>,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) wait: Option<(Retained<ProtocolObject<dyn MTLEvent>>, u64)>,
    pub(crate) signal: Option<(Retained<ProtocolObject<dyn MTLEvent>>, u64)>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Wait/signal still need to be encoded. Cleared once a command buffer has them.
    pub(crate) sync_pending: bool,
}

// Metal textures, events, and command queues may be retained and signaled from any thread.
// `Retained<ProtocolObject<dyn _>>` does not implement `Send` (the protocol object is a trait
// object), but these three objects are thread-safe and the mailbox crosses threads.
unsafe impl Send for OwnedMetalImport {}

impl Drop for OwnedMetalImport {
    fn drop(&mut self) {
        if !self.sync_pending {
            return;
        }
        self.sync_pending = false;
        let Some((event, value)) = self.signal.take() else { return };
        // The texture was never sampled, so the producer only needs to be unblocked. A shared
        // event can be signaled on the CPU; anything else gets a command-buffer signal.
        if signal_shared_event(&event, value) {
            return;
        }
        let Some(command_buffer) = self.queue.commandBuffer() else { return };
        command_buffer.encodeSignalEvent_value(&event, value);
        command_buffer.commit();
    }
}

fn signal_shared_event(event: &ProtocolObject<dyn MTLEvent>, value: u64) -> bool {
    let selector = objc2::runtime::Sel::register("setSignaledValue:");
    let responds: Bool = unsafe { objc2::msg_send![event, respondsToSelector: selector] };
    if !responds.as_bool() {
        return false;
    }
    let () = unsafe { objc2::msg_send![event, setSignaledValue: value] };
    true
}

/// Check `texture` (an `id<MTLTexture>`) and retain it, plus any wait/signal events.
/// `0` for an event pointer omits that side. Dimensions come from the texture.
pub fn prepare_metal_import(
    texture: usize,
    wait_event: usize,
    wait_value: u64,
    signal_event: usize,
    signal_value: u64,
) -> Result<ImportedLayer, String> {
    if texture == 0 {
        return Err("texture pointer is null".into());
    }
    let device: Retained<ProtocolObject<dyn MTLDevice>> =
        unsafe { Retained::from_raw(MTLCreateSystemDefaultDevice()) }.ok_or("no Metal device")?;
    let retained_texture = retain_texture(texture, &device)?;
    let width = u32::try_from(retained_texture.width()).map_err(|_| "texture width does not fit in u32")?;
    let height = u32::try_from(retained_texture.height()).map_err(|_| "texture height does not fit in u32")?;
    if width == 0 || height == 0 {
        return Err("texture must be at least 1x1".into());
    }
    let wait = retain_event(wait_event, &device, "wait_event")?.map(|event| (event, wait_value));
    let signal = retain_event(signal_event, &device, "signal_event")?.map(|event| (event, signal_value));
    let queue = device.newCommandQueue().ok_or("failed to create a Metal command queue")?;
    let retained = OwnedMetalImport {
        texture: retained_texture,
        width,
        height,
        wait,
        signal,
        queue,
        sync_pending: true,
    };
    Ok(ImportedLayer::Metal {
        texture,
        width,
        height,
        wait_event,
        wait_value,
        signal_event,
        signal_value,
        retained: Box::new(retained),
    })
}

fn retain_texture(
    ptr: usize,
    default_device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLTexture>>, String> {
    let texture: Retained<ProtocolObject<dyn MTLTexture>> =
        unsafe { Retained::retain(ptr as *mut ProtocolObject<dyn MTLTexture>) }
            .ok_or("texture pointer is null")?;
    if !same_device(&texture.device(), default_device) {
        return Err("texture is not on the system default Metal device".into());
    }
    if texture.textureType() != MTLTextureType::MTLTextureType2D {
        return Err("texture must be a 2D MTLTexture".into());
    }
    if texture.sampleCount() != 1 {
        return Err("multisampled textures cannot be sampled as a viewport".into());
    }
    if !texture.usage().contains(MTLTextureUsage::ShaderRead) {
        return Err("texture needs ShaderRead usage (mtlpy readable=True)".into());
    }
    if is_framebuffer_only(&texture) {
        return Err("framebufferOnly textures cannot be sampled".into());
    }
    if !sampleable_as_float(texture.pixelFormat()) {
        return Err(
            "texture format must be float or unorm (the viewport shader samples texture2d<float>); integer, depth, and YUV formats are rejected"
                .into(),
        );
    }
    Ok(texture)
}

fn retain_event(
    ptr: usize,
    default_device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
) -> Result<Option<Retained<ProtocolObject<dyn MTLEvent>>>, String> {
    if ptr == 0 {
        return Ok(None);
    }
    let event: Retained<ProtocolObject<dyn MTLEvent>> =
        unsafe { Retained::retain(ptr as *mut ProtocolObject<dyn MTLEvent>) }
            .ok_or_else(|| format!("{label} pointer is null"))?;
    if let Some(device) = event.device() {
        if !same_device(&device, default_device) {
            return Err(format!("{label} is not on the system default Metal device"));
        }
    }
    Ok(Some(event))
}

fn same_device(a: &ProtocolObject<dyn MTLDevice>, b: &ProtocolObject<dyn MTLDevice>) -> bool {
    std::ptr::eq(a, b)
}

fn is_framebuffer_only(texture: &ProtocolObject<dyn MTLTexture>) -> bool {
    // Ordinary `MTLTexture`s don't implement this. Drawable textures do; the getter name
    // differs by OS, so only message a selector the object actually has.
    if responds(texture, "isFramebufferOnly") {
        let value: Bool = unsafe { objc2::msg_send![texture, isFramebufferOnly] };
        return value.as_bool();
    }
    if responds(texture, "framebufferOnly") {
        let value: Bool = unsafe { objc2::msg_send![texture, framebufferOnly] };
        return value.as_bool();
    }
    false
}

fn responds(texture: &ProtocolObject<dyn MTLTexture>, selector: &str) -> bool {
    let selector = objc2::runtime::Sel::register(selector);
    let responds: Bool = unsafe { objc2::msg_send![texture, respondsToSelector: selector] };
    responds.as_bool()
}

fn sampleable_as_float(format: MTLPixelFormat) -> bool {
    !matches!(
        format,
        MTLPixelFormat::Invalid
            | MTLPixelFormat::R8Uint
            | MTLPixelFormat::R8Sint
            | MTLPixelFormat::R16Uint
            | MTLPixelFormat::R16Sint
            | MTLPixelFormat::R32Uint
            | MTLPixelFormat::R32Sint
            | MTLPixelFormat::RG8Uint
            | MTLPixelFormat::RG8Sint
            | MTLPixelFormat::RG16Uint
            | MTLPixelFormat::RG16Sint
            | MTLPixelFormat::RG32Uint
            | MTLPixelFormat::RG32Sint
            | MTLPixelFormat::RGBA8Uint
            | MTLPixelFormat::RGBA8Sint
            | MTLPixelFormat::RGB10A2Uint
            | MTLPixelFormat::RGBA16Uint
            | MTLPixelFormat::RGBA16Sint
            | MTLPixelFormat::RGBA32Uint
            | MTLPixelFormat::RGBA32Sint
            | MTLPixelFormat::GBGR422
            | MTLPixelFormat::BGRG422
            | MTLPixelFormat::Depth16Unorm
            | MTLPixelFormat::Depth32Float
            | MTLPixelFormat::Stencil8
            | MTLPixelFormat::Depth24Unorm_Stencil8
            | MTLPixelFormat::Depth32Float_Stencil8
            | MTLPixelFormat::X32_Stencil8
            | MTLPixelFormat::X24_Stencil8
    )
}

/// Encode this submission's wait at the start of a command buffer and its signal at the end.
pub(crate) struct PendingSync {
    pub(crate) wait: Option<(Retained<ProtocolObject<dyn MTLEvent>>, u64)>,
    pub(crate) signal: Option<(Retained<ProtocolObject<dyn MTLEvent>>, u64)>,
}

impl PendingSync {
    pub(crate) fn encode_waits(&self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
        if let Some((event, value)) = &self.wait {
            command_buffer.encodeWaitForEvent_value(event, *value);
        }
    }

    pub(crate) fn encode_signal(&self, command_buffer: &ProtocolObject<dyn MTLCommandBuffer>) {
        if let Some((event, value)) = &self.signal {
            command_buffer.encodeSignalEvent_value(event, *value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLDevice, MTLSharedEvent, MTLTextureDescriptor};

    fn device() -> Retained<ProtocolObject<dyn MTLDevice>> {
        unsafe { Retained::from_raw(MTLCreateSystemDefaultDevice()) }.expect("metal device")
    }

    fn texture(device: &ProtocolObject<dyn MTLDevice>, format: MTLPixelFormat, usage: MTLTextureUsage) -> Retained<ProtocolObject<dyn MTLTexture>> {
        let descriptor = unsafe {
            MTLTextureDescriptor::texture2DDescriptorWithPixelFormat_width_height_mipmapped(
                format,
                4,
                2,
                false,
            )
        };
        descriptor.setUsage(usage);
        device.newTextureWithDescriptor(&descriptor).expect("texture")
    }

    fn ptr(texture: &ProtocolObject<dyn MTLTexture>) -> usize {
        texture as *const ProtocolObject<dyn MTLTexture> as usize
    }

    #[test]
    fn null_texture_is_rejected() {
        let err = prepare_metal_import(0, 0, 0, 0, 0).unwrap_err();
        assert!(err.contains("null"), "{err}");
    }

    #[test]
    fn rgba8_unorm_is_retained_and_sized_from_the_texture() {
        let device = device();
        let texture = texture(&device, MTLPixelFormat::RGBA8Unorm, MTLTextureUsage::ShaderRead);
        let layer = prepare_metal_import(ptr(&texture), 0, 0, 0, 0).expect("import");
        match layer {
            ImportedLayer::Metal { width, height, retained, .. } => {
                assert_eq!((width, height), (4, 2));
                let owned = retained.downcast::<OwnedMetalImport>().expect("owned import");
                assert!(std::ptr::eq(&*owned.texture, &*texture));
            }
            ImportedLayer::DmaBuf(_) => panic!("metal import"),
        }
    }

    #[test]
    fn integer_and_unreadable_textures_are_rejected() {
        let device = device();
        let integer = texture(&device, MTLPixelFormat::RGBA8Uint, MTLTextureUsage::ShaderRead);
        let err = prepare_metal_import(ptr(&integer), 0, 0, 0, 0).unwrap_err();
        assert!(err.contains("integer"), "{err}");

        let target = texture(&device, MTLPixelFormat::RGBA8Unorm, MTLTextureUsage::RenderTarget);
        let err = prepare_metal_import(ptr(&target), 0, 0, 0, 0).unwrap_err();
        assert!(err.contains("ShaderRead"), "{err}");
    }

    #[test]
    fn dropping_an_unsampled_import_signals_the_shared_event() {
        let device = device();
        let texture = texture(&device, MTLPixelFormat::BGRA8Unorm, MTLTextureUsage::ShaderRead);
        let event = device.newSharedEvent().expect("shared event");
        let event_ptr = Retained::as_ptr(&event) as usize;
        let layer = prepare_metal_import(ptr(&texture), event_ptr, 3, event_ptr, 4).expect("import");
        drop(layer);
        assert_eq!(unsafe { event.signaledValue() }, 4);
    }
}
