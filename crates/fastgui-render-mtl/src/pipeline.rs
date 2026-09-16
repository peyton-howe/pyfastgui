use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLDevice, MTLLibrary, MTLPixelFormat, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
    MTLSamplerAddressMode, MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState,
};

use crate::error::MtlRendererError as Error;

/// A vertex function that generates a full-screen triangle from `vertex_id` alone (no vertex
/// buffer -- same trick `fastgui-render-vk/shaders/viewport.vert` uses in GLSL), and a fragment
/// function that samples the viewport texture. The vertex/UV table here is Metal's own (NDC
/// y-up, unlike Vulkan's y-down) worked out so the result matches: `uv=(0,0)` lands at the
/// window's top-left, `uv=(1,1)` at bottom-right -- top-left-origin, v-down, the same texture
/// addressing convention `CpuFrame`'s row-major byte layout already assumes.
const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut viewport_vertex(uint vertex_id [[vertex_id]]) {
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 uvs[3]       = { float2(0.0, 1.0),   float2(2.0, 1.0),  float2(0.0, -1.0) };
    VertexOut out;
    out.position = float4(positions[vertex_id], 0.0, 1.0);
    out.uv = uvs[vertex_id];
    return out;
}

fragment float4 viewport_fragment(VertexOut in [[stage_in]],
                                   texture2d<float> tex [[texture(0)]],
                                   sampler smp [[sampler(0)]]) {
    return tex.sample(smp, in.uv);
}
"#;

/// Draws a GPU texture as a full-window (or per-rect) textured quad. One pipeline state and
/// sampler are shared across chrome and every viewport layer.
pub struct ViewportPipeline {
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
}

impl ViewportPipeline {
    pub fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        color_format: MTLPixelFormat,
    ) -> Result<Self, Error> {
        let source = ns_string!(SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(source, None)
            .map_err(|err| Error::ShaderCompile(err.localizedDescription().to_string()))?;

        let vertex_function = library
            .newFunctionWithName(ns_string!("viewport_vertex"))
            .ok_or(Error::MissingFunction("viewport_vertex"))?;
        let fragment_function = library
            .newFunctionWithName(ns_string!("viewport_fragment"))
            .ok_or(Error::MissingFunction("viewport_fragment"))?;

        let descriptor = MTLRenderPipelineDescriptor::new();
        unsafe {
            descriptor.setVertexFunction(Some(&vertex_function));
            descriptor.setFragmentFunction(Some(&fragment_function));
            descriptor
                .colorAttachments()
                .objectAtIndexedSubscript(0)
                .setPixelFormat(color_format);
        }

        let state = device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|err| Error::PipelineState(err.localizedDescription().to_string()))?;

        let sampler_descriptor = MTLSamplerDescriptor::new();
        sampler_descriptor.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sampler_descriptor.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        let sampler = device
            .newSamplerStateWithDescriptor(&sampler_descriptor)
            .ok_or(Error::NoSampler)?;

        Ok(Self { state, sampler })
    }

    pub fn state(&self) -> &ProtocolObject<dyn MTLRenderPipelineState> {
        &self.state
    }

    pub fn sampler(&self) -> &ProtocolObject<dyn MTLSamplerState> {
        &self.sampler
    }
}
