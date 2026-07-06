#version 460 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_model0;
layout(location = 4) in vec4 in_model1;
layout(location = 5) in vec4 in_model2;
layout(location = 6) in vec4 in_model3;
layout(std430, binding = 4) readonly buffer InstanceModels {
    mat4 models[];
};
layout(std430, binding = 5) readonly buffer InstanceIndices {
    int indices[];
};
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_matrix;
uniform int u_use_instancing;
out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_view_pos;
void main() {
    mat4 model = u_model;
    mat3 nm = u_normal_matrix;
    if (u_use_instancing == 1) {
        mat4 inst_model = mat4(in_model0, in_model1, in_model2, in_model3);
        model = inst_model;
        nm = transpose(inverse(mat3(model)));
    } else if (u_use_instancing == 2) {
        int idx = indices[gl_InstanceID];
        model = models[idx];
        nm = transpose(inverse(mat3(model)));
    }
    vec4 world_pos = model * vec4(in_position, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(nm * in_normal);
    v_uv = in_uv;
    vec4 view_pos = u_view * world_pos;
    v_view_pos = view_pos.xyz;
    gl_Position = u_proj * u_view * world_pos;
}
