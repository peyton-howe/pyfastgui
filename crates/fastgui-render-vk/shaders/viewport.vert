#version 450

// Full-screen triangle from just the vertex index -- no vertex buffer needed. Covers the
// whole clip-space square; the rasterizer clips the excess outside the viewport.
layout(location = 0) out vec2 uv;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
