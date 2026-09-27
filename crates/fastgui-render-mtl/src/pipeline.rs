use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::ns_string;
use objc2_metal::{
    MTLBlendFactor, MTLDevice, MTLLibrary, MTLPixelFormat, MTLRenderPipelineDescriptor, MTLRenderPipelineState,
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

/// Chrome as instanced quads (`fastgui_core::ChromeQuad`, 64 bytes each, read straight from the
/// instance buffer). Kinds and their coverage rules match `fastgui-chrome`'s `gpu` module and
/// `fastgui-render-vk/shaders/quad.{vert,frag}`: solid rects get box-filter coverage at
/// fractional edges, circles analytic edge coverage, sprites a 1:1 atlas copy. Output is
/// premultiplied, blended ONE / ONE_MINUS_SRC_ALPHA.
const QUAD_SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Quad {
    float4 rect;    // left, top, right, bottom (physical px, top-left origin)
    float4 color;   // straight RGBA
    float4 params;  // circle: cx, cy, radius; sprite: atlas x, y
    uint kind;      // 0 solid, 1 circle, 2 sprite
    uint pad0;
    uint pad1;
    uint pad2;
};

struct QuadOut {
    float4 position [[position]];
    float4 rect [[flat]];
    float4 color [[flat]];
    float4 params [[flat]];
    uint kind [[flat]];
};

vertex QuadOut quad_vertex(uint vertex_id [[vertex_id]],
                           uint instance_id [[instance_id]],
                           const device Quad* quads [[buffer(0)]],
                           constant float2& viewport [[buffer(1)]]) {
    Quad q = quads[instance_id];
    // Anti-aliased kinds reach one pixel past their edges so partly covered pixels get drawn.
    float pad = q.kind == 2 ? 0.0 : 1.0;
    float2 corner = float2(vertex_id & 1, vertex_id >> 1);
    float2 p = mix(q.rect.xy - pad, q.rect.zw + pad, corner);
    QuadOut out;
    // Metal NDC is y-up; pixel coordinates are y-down.
    out.position = float4(p.x / viewport.x * 2.0 - 1.0, 1.0 - p.y / viewport.y * 2.0, 0.0, 1.0);
    out.rect = q.rect;
    out.color = q.color;
    out.params = q.params;
    out.kind = q.kind;
    return out;
}

fragment float4 quad_fragment(QuadOut in [[stage_in]], texture2d<float> atlas [[texture(0)]]) {
    // `position` is the pixel centre, top-left origin.
    float2 px = floor(in.position.xy);
    if (in.kind == 2) {
        uint2 texel = uint2(in.params.xy + (px - in.rect.xy));
        return atlas.read(texel);
    }
    float coverage;
    if (in.kind == 1) {
        coverage = clamp(in.params.z + 0.5 - distance(in.position.xy, in.params.xy), 0.0, 1.0);
    } else {
        float2 lo = max(px, in.rect.xy);
        float2 hi = min(px + 1.0, in.rect.zw);
        float2 c = clamp(hi - lo, 0.0, 1.0);
        coverage = c.x * c.y;
    }
    float a = in.color.a * coverage;
    return float4(in.color.rgb * a, a);
}
"#;

/// Pipeline state for `QUAD_SHADER_SOURCE`, with premultiplied source-over blending.
pub struct QuadPipeline {
    state: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
}

impl QuadPipeline {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, color_format: MTLPixelFormat) -> Result<Self, Error> {
        let library = device
            .newLibraryWithSource_options_error(ns_string!(QUAD_SHADER_SOURCE), None)
            .map_err(|err| Error::ShaderCompile(err.localizedDescription().to_string()))?;
        let vertex_function =
            library.newFunctionWithName(ns_string!("quad_vertex")).ok_or(Error::MissingFunction("quad_vertex"))?;
        let fragment_function = library
            .newFunctionWithName(ns_string!("quad_fragment"))
            .ok_or(Error::MissingFunction("quad_fragment"))?;

        let descriptor = MTLRenderPipelineDescriptor::new();
        unsafe {
            descriptor.setVertexFunction(Some(&vertex_function));
            descriptor.setFragmentFunction(Some(&fragment_function));
            let attachment = descriptor.colorAttachments().objectAtIndexedSubscript(0);
            attachment.setPixelFormat(color_format);
            attachment.setBlendingEnabled(true);
            attachment.setSourceRGBBlendFactor(MTLBlendFactor::One);
            attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
            attachment.setSourceAlphaBlendFactor(MTLBlendFactor::One);
            attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        }
        let state = device
            .newRenderPipelineStateWithDescriptor_error(&descriptor)
            .map_err(|err| Error::PipelineState(err.localizedDescription().to_string()))?;
        Ok(Self { state })
    }

    pub fn state(&self) -> &ProtocolObject<dyn MTLRenderPipelineState> {
        &self.state
    }
}
