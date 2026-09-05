#version 450

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 out_color;
layout(binding = 0) uniform sampler2D viewport_texture;

void main() {
    out_color = texture(viewport_texture, uv);
}
