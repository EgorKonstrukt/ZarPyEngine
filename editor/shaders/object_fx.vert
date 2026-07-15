// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_model0;
layout(location = 4) in vec4 in_model1;
layout(location = 5) in vec4 in_model2;
layout(location = 6) in vec4 in_model3;
layout(location = 7) in vec4 in_bone_indices;
layout(location = 8) in vec4 in_bone_weights;
layout(std430, binding = 4) readonly buffer InstanceModels {
    mat4 models[];
};
layout(std430, binding = 5) readonly buffer InstanceIndices {
    int indices[];
};
layout(std430, binding = 6) readonly buffer BoneMatrices {
    mat4 u_bone_matrices[];
};
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_matrix;
uniform int u_use_instancing;
uniform int u_use_skinning;
uniform int u_bone_count;
uniform float u_time;
uniform float u_disint_amount;
uniform vec3 u_disint_dir;
uniform float u_disint_cell;
uniform float u_disint_noise_scale;
uniform float u_disint_speed;
uniform float u_disint_drag;
uniform float u_disint_gravity;
uniform float u_disint_rot;
uniform float u_wind_amount;
uniform vec3 u_wind_dir;
uniform float u_wind_speed;
uniform float u_wind_strength;
uniform float u_glitch_amount;
uniform float u_glitch_intensity;
uniform float u_glitch_speed;
uniform float u_glitch_block;
uniform vec3 u_obj_center;
uniform float u_obj_scale;
out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_view_pos;
out vec3 v_local_pos;

float hash31(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float noise3(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    float n000 = hash31(i + vec3(0.0, 0.0, 0.0));
    float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
    float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
    float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
    float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
    float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
    float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
    float n111 = hash31(i + vec3(1.0, 1.0, 1.0));
    float nx00 = mix(n000, n100, f.x);
    float nx10 = mix(n010, n110, f.x);
    float nx01 = mix(n001, n101, f.x);
    float nx11 = mix(n011, n111, f.x);
    float nxy0 = mix(nx00, nx10, f.y);
    float nxy1 = mix(nx01, nx11, f.y);
    return mix(nxy0, nxy1, f.z);
}

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
    vec3 local_pos = in_position;
    vec3 local_nrm = in_normal;
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
        local_nrm = mat3(skin) * in_normal;
    }
    if (u_disint_amount > 0.0) {
        float cs = max(0.001, u_disint_cell);
        vec3 cellId = floor((in_position - u_obj_center) / cs);
        float h = hash31(cellId);
        float h2 = hash31(cellId + 7.3);
        float h3 = hash31(cellId + 19.1);
        float t = u_disint_amount;
        float drag = max(0.01, u_disint_drag);
        float s = (1.0 - exp(-drag * t)) / drag;

        vec3 cellCenter = (cellId + 0.5) * cs + u_obj_center;
        vec3 rel = in_position - cellCenter;

        if (u_disint_rot > 0.0) {
            float ang = h * 6.2831853 + t * u_disint_rot * (h2 - 0.5) * 4.0;
            vec3 axis = normalize(vec3(h, h2, h3) * 2.0 - 1.0 + vec3(1e-4));
            float c = cos(ang);
            float si = sin(ang);
            vec3 cr = cross(axis, rel);
            rel = rel * c + cr * si + axis * dot(axis, rel) * (1.0 - c);
            vec3 nr = cross(axis, local_nrm);
            local_nrm = local_nrm * c + nr * si + axis * dot(axis, local_nrm) * (1.0 - c);
        }

        vec3 outward = normalize(in_position - u_obj_center + vec3(1e-4));
        vec3 rnd = vec3(h, h2, h3) * 2.0 - 1.0;
        vec3 vel = normalize(u_disint_dir * 1.5 + outward * 1.3 + rnd * 0.6);
        float speed = (0.3 + h * 1.2) * u_obj_scale * u_disint_speed;
        vec3 disp = vel * (speed * s);

        disp += vec3(0.0, -1.0, 0.0) * u_disint_gravity * u_obj_scale * (s * s);

        float jitter = (noise3(in_position * max(0.1, u_disint_noise_scale)) - 0.5);
        disp += outward * jitter * t * u_obj_scale * 0.6;

        local_pos = cellCenter + rel + disp;
    }
    if (u_wind_amount > 0.0) {
        float phase = u_time * u_wind_speed;
        float h = clamp(in_position.y / max(0.001, u_obj_scale) * 0.5 + 0.5, 0.0, 1.0);
        float sway = sin(phase + (in_position.x + in_position.z) * 0.5) * u_wind_strength * h * h * u_wind_amount;
        vec3 wdir = normalize(u_wind_dir + vec3(1e-4, 0.0, 1e-4));
        local_pos += wdir * sway;
    }
    if (u_glitch_amount > 0.0) {
        float blk = floor(in_position.y / max(0.01, u_glitch_block) + u_time * u_glitch_speed);
        float r = hash31(vec3(blk, 1.7, 3.3));
        if (r < u_glitch_amount) {
            float off = (hash31(vec3(blk, 9.1, 2.2)) - 0.5) * u_glitch_intensity * u_obj_scale;
            local_pos.x += off;
        }
    }
    vec4 world_pos = model * vec4(local_pos, 1.0);
    v_world_pos = world_pos.xyz;
    v_normal = normalize(nm * local_nrm);
    v_uv = in_uv;
    v_local_pos = in_position;
    vec4 view_pos = u_view * world_pos;
    v_view_pos = view_pos.xyz;
    gl_Position = u_proj * u_view * world_pos;
}
