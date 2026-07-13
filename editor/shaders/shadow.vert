#version 460 core
layout(location = 0) in vec3 in_position;
layout(location = 3) in vec4 in_model0;
layout(location = 4) in vec4 in_model1;
layout(location = 5) in vec4 in_model2;
layout(location = 6) in vec4 in_model3;
layout(location = 7) in vec4 in_bone_indices;
layout(location = 8) in vec4 in_bone_weights;
layout(std430, binding = 6) readonly buffer BoneMatrices {
    mat4 u_bone_matrices[];
};
uniform mat4 u_model;
uniform mat4 u_light_vp;
uniform int u_use_instancing;
uniform int u_use_skinning;
uniform int u_bone_count;
void main() {
    mat4 model = u_model;
    if (u_use_instancing == 1) {
        model = mat4(in_model0, in_model1, in_model2, in_model3);
    }
    vec3 local_pos = in_position;
    if (u_use_skinning == 1) {
        mat4 skin = mat4(0.0);
        for (int i = 0; i < 4; i++) {
            int bi = int(in_bone_indices[i] + 0.5);
            float bw = in_bone_weights[i];
            if (bi >= 0 && bi < u_bone_count && bw > 0.0) {
                skin += bw * u_bone_matrices[bi];
            }
        }
        local_pos = (skin * vec4(in_position, 1.0)).xyz;
    }
    gl_Position = u_light_vp * model * vec4(local_pos, 1.0);
}
