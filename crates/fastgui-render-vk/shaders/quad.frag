#version 450

layout(location = 0) flat in vec4 rect;
layout(location = 1) flat in vec4 color;
layout(location = 2) flat in vec4 params;
layout(location = 3) flat in uint kind;

layout(location = 0) out vec4 out_color;
layout(binding = 0) uniform sampler2D atlas;

layout(push_constant) uniform Push {
    vec2 viewport;  // size of the quads' pixel space
    vec2 target;    // size of the render target (the swapchain extent)
} push;

// Output is premultiplied; the pipeline blends ONE / ONE_MINUS_SRC_ALPHA.
void main() {
    // `gl_FragCoord` is the pixel centre, top-left origin, in render-target pixels. The quads
    // are in `push.viewport` pixels; the two differ while a debounced main-window resize hasn't
    // recreated the swapchain yet (the vertex shader then stretches the quads over the target,
    // like the CPU chrome texture). Map back into quad space so coverage and atlas fetches use
    // one space; the scale is exactly 1 otherwise.
    vec2 frag = gl_FragCoord.xy * (push.viewport / push.target);
    vec2 px = floor(frag);
    if (kind == 2u) {
        px = clamp(px, rect.xy, rect.zw - 1.0);
        out_color = texelFetch(atlas, ivec2(params.xy + (px - rect.xy)), 0);
        return;
    }
    float coverage;
    if (kind == 1u) {
        coverage = clamp(params.z + 0.5 - distance(frag, params.xy), 0.0, 1.0);
    } else {
        vec2 c = clamp(min(px + 1.0, rect.zw) - max(px, rect.xy), 0.0, 1.0);
        coverage = c.x * c.y;
    }
    float a = color.a * coverage;
    out_color = vec4(color.rgb * a, a);
}
