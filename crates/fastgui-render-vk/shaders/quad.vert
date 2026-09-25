#version 450

// Chrome as instanced quads: one `fastgui_core::ChromeQuad` (64 bytes) per instance, expanded to
// a 4-vertex triangle strip from `gl_VertexIndex`. Kinds and coverage rules match
// `fastgui-chrome`'s `gpu` module and the Metal backend's `QUAD_SHADER_SOURCE`.
layout(location = 0) in vec4 in_rect;    // left, top, right, bottom (physical px, top-left origin)
layout(location = 1) in vec4 in_color;   // straight RGBA
layout(location = 2) in vec4 in_params;  // circle: cx, cy, radius; sprite: atlas x, y
layout(location = 3) in uint in_kind;    // 0 solid, 1 circle, 2 sprite

layout(push_constant) uniform Push {
    vec2 viewport;  // size of the quads' pixel space
} push;

layout(location = 0) flat out vec4 rect;
layout(location = 1) flat out vec4 color;
layout(location = 2) flat out vec4 params;
layout(location = 3) flat out uint kind;

void main() {
    // Anti-aliased kinds reach one pixel past their edges so partly covered pixels get drawn.
    float pad = in_kind == 2u ? 0.0 : 1.0;
    vec2 corner = vec2(gl_VertexIndex & 1, gl_VertexIndex >> 1);
    vec2 p = mix(in_rect.xy - pad, in_rect.zw + pad, corner);
    // Vulkan NDC is y-down, like pixel coordinates.
    gl_Position = vec4(p / push.viewport * 2.0 - 1.0, 0.0, 1.0);
    rect = in_rect;
    color = in_color;
    params = in_params;
    kind = in_kind;
}
