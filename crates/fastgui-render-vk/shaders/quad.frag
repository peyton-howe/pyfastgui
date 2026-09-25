#version 450

layout(location = 0) flat in vec4 rect;
layout(location = 1) flat in vec4 color;
layout(location = 2) flat in vec4 params;
layout(location = 3) flat in uint kind;

layout(location = 0) out vec4 out_color;
layout(binding = 0) uniform sampler2D atlas;

// Output is premultiplied; the pipeline blends ONE / ONE_MINUS_SRC_ALPHA.
void main() {
    // `gl_FragCoord` is the pixel centre, top-left origin.
    vec2 px = floor(gl_FragCoord.xy);
    if (kind == 2u) {
        out_color = texelFetch(atlas, ivec2(params.xy + (px - rect.xy)), 0);
        return;
    }
    float coverage;
    if (kind == 1u) {
        coverage = clamp(params.z + 0.5 - distance(gl_FragCoord.xy, params.xy), 0.0, 1.0);
    } else {
        vec2 c = clamp(min(px + 1.0, rect.zw) - max(px, rect.xy), 0.0, 1.0);
        coverage = c.x * c.y;
    }
    float a = color.a * coverage;
    out_color = vec4(color.rgb * a, a);
}
