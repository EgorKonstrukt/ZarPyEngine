// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

in vec3 gs_world_pos[3];
in vec3 gs_normal[3];
in vec2 gs_uv[3];
in vec3 gs_view_pos[3];
in vec3 gs_local_pos[3];

out vec3 v_world_pos;
out vec3 v_normal;
out vec2 v_uv;
out vec3 v_view_pos;
out vec3 v_local_pos;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat3 u_normal_matrix;
uniform float u_disint_amount;
uniform vec3 u_disint_dir;
uniform float u_disint_cell;
uniform float u_disint_noise_scale;
uniform float u_disint_speed;
uniform float u_disint_drag;
uniform float u_disint_gravity;
uniform float u_disint_rot;
uniform float u_disint_twist;
uniform float u_disint_sp_variance;
uniform float u_disint_outward;
uniform float u_disint_scatter;
uniform float u_disint_jitter;
uniform vec3 u_obj_center;
uniform float u_obj_scale;

float hash31(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}
float value_noise3(vec3 x) {
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
    if (u_disint_amount <= 0.0) {
        for (int i = 0; i < 3; i++) {
            gl_Position = gl_in[i].gl_Position;
            v_world_pos = gs_world_pos[i];
            v_normal = gs_normal[i];
            v_uv = gs_uv[i];
            v_view_pos = gs_view_pos[i];
            v_local_pos = gs_local_pos[i];
            EmitVertex();
        }
        EndPrimitive();
        return;
    }

    vec3 triCenter = (gs_local_pos[0] + gs_local_pos[1] + gs_local_pos[2]) / 3.0;
    float cs = max(0.001, u_disint_cell);
    vec3 cellId = floor((triCenter - u_obj_center) / cs);
    float h = hash31(cellId);
    float h2 = hash31(cellId + 7.3);
    float h3 = hash31(cellId + 19.1);
    float t = u_disint_amount;
    float drag = max(0.01, u_disint_drag);
    float s = (1.0 - exp(-drag * t)) / drag;

    vec3 cellCenter = (cellId + 0.5) * cs + u_obj_center;
    vec3 outward = normalize(cellCenter - u_obj_center + vec3(1e-4));
    vec3 rnd = vec3(h, h2, h3) * 2.0 - 1.0;
    vec3 vel = normalize(u_disint_dir * 1.5 + outward * u_disint_outward + rnd * u_disint_scatter);
    float speed = (0.3 + h * u_disint_sp_variance) * u_obj_scale * u_disint_speed;
    vec3 disp = vel * (speed * s);
    disp += vec3(0.0, -1.0, 0.0) * u_disint_gravity * u_obj_scale * (s * s);

    float jitter = (value_noise3(cellCenter * max(0.1, u_disint_noise_scale)) - 0.5);
    disp += outward * jitter * t * u_obj_scale * u_disint_jitter;

    float rot_angle = t * (h * 6.2831853 + u_disint_rot * (h2 - 0.5) * 4.0 + u_disint_twist);
    vec3 rot_axis = normalize(vec3(h, h2, h3) * 2.0 - 1.0 + vec3(1e-4));
    bool do_rot = u_disint_rot > 0.0 || u_disint_twist > 0.0;

    for (int i = 0; i < 3; i++) {
        vec3 rel = gs_local_pos[i] - cellCenter;
        vec3 nrm = gs_normal[i];

        if (do_rot) {
            float c = cos(rot_angle);
            float si = sin(rot_angle);
            vec3 cr = cross(rot_axis, rel);
            rel = rel * c + cr * si + rot_axis * dot(rot_axis, rel) * (1.0 - c);
            vec3 nr = cross(rot_axis, nrm);
            nrm = nrm * c + nr * si + rot_axis * dot(rot_axis, nrm) * (1.0 - c);
        }

        vec3 new_local_pos = cellCenter + rel + disp;
        vec4 wp = u_model * vec4(new_local_pos, 1.0);
        v_world_pos = wp.xyz;
        v_normal = normalize(u_normal_matrix * nrm);
        v_uv = gs_uv[i];
        vec4 vp = u_view * wp;
        v_view_pos = vp.xyz;
        v_local_pos = gs_local_pos[i];
        gl_Position = u_proj * vp;
        EmitVertex();
    }
    EndPrimitive();
}
