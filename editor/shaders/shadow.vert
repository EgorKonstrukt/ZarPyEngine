#version 460 core
layout(location = 0) in vec3 in_position;
layout(location = 3) in vec4 in_model0;
layout(location = 4) in vec4 in_model1;
layout(location = 5) in vec4 in_model2;
layout(location = 6) in vec4 in_model3;
uniform mat4 u_model;
uniform mat4 u_light_vp;
uniform int u_use_instancing;
void main() {
    mat4 model = u_model;
    if (u_use_instancing == 1) {
        model = mat4(in_model0, in_model1, in_model2, in_model3);
    }
    gl_Position = u_light_vp * model * vec4(in_position, 1.0);
}
