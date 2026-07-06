#version 460 core
layout(location = 0) in vec3 in_position;
uniform mat4 u_model;
uniform mat4 u_light_vp;
void main() {
    gl_Position = u_light_vp * u_model * vec4(in_position, 1.0);
}
