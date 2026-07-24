// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;

in vec3 v_world_pos[3];
in vec3 v_normal[3];
in vec2 v_uv[3];
in vec3 v_view_pos[3];
in vec4 v_color[3];
in float v_branch_level[3];
in float v_stiffness[3];
in float v_phase_offset[3];
in float v_is_leaf[3];

out vec3 gs_world_pos;
out vec3 gs_normal;
out vec2 gs_uv;
out vec3 gs_view_pos;
out vec4 gs_color;
out float gs_branch_level;
out float gs_stiffness;
out float gs_phase_offset;
out float gs_is_leaf;

uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform float u_time;
uniform vec3 u_camera_pos;
uniform float _LeafFlutterAmount;
uniform float _LeafFlutterSpeed;
uniform float _WindStrength;

float hash21(vec2 p) {
    p = fract(p * vec2(234.34, 435.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void emit_vertex(vec3 pos, vec3 nrm, vec2 uv, vec3 vp, vec4 col, float bl, float st, float ph, float il) {
    gs_world_pos = pos;
    gs_normal = nrm;
    gs_uv = uv;
    gs_view_pos = vp;
    gs_color = col;
    gs_branch_level = bl;
    gs_stiffness = st;
    gs_phase_offset = ph;
    gs_is_leaf = il;
    gl_Position = u_proj * u_view * vec4(pos, 1.0);
    EmitVertex();
}

void main() {
    float is_leaf = v_is_leaf[0];

    if (is_leaf < 0.5) {
        for (int i = 0; i < 3; i++) {
            emit_vertex(v_world_pos[i], v_normal[i], v_uv[i], v_view_pos[i],
                        v_color[i], v_branch_level[i], v_stiffness[i],
                        v_phase_offset[i], v_is_leaf[i]);
        }
        EndPrimitive();
        return;
    }

    vec3 center = (v_world_pos[0] + v_world_pos[1] + v_world_pos[2]) / 3.0;
    vec3 eye_dir = normalize(u_camera_pos - center);
    float phase = v_phase_offset[0];
    float rnd = hash21(center.xz + center.y);

    vec3 up = normalize(cross(eye_dir, vec3(0.0, 1.0, 0.0)));
    if (length(up) < 0.001) up = vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(eye_dir, up));
    up = normalize(cross(right, eye_dir));

    float wing_angle = mix(0.3, 1.2, rnd);
    vec3 axis1 = normalize(mix(right, up, 0.3));
    vec3 axis2 = normalize(mix(-right, up, 0.3));
    float size = mix(0.15, 0.35, rnd);

    float flutter = sin(u_time * _LeafFlutterSpeed + phase * 12.566) * _LeafFlutterAmount * 0.5;
    float curl = sin(u_time * 0.5 + phase * 3.0) * 0.1;

    vec3 corners[4];
    corners[0] = center - axis1 * size - up * size * 0.5;
    corners[1] = center + axis1 * size - up * size * 0.5;
    corners[2] = center + axis2 * size + up * size * 0.5;
    corners[3] = center - axis2 * size + up * size * 0.5;

    vec3 nrm = normalize(eye_dir + vec3(flutter, abs(flutter) * 0.3, flutter * 0.5));
    vec2 uvs[4] = vec2[](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));

    emit_vertex(corners[0], nrm, uvs[0], v_view_pos[0], v_color[0], v_branch_level[0], v_stiffness[0], phase, 1.0);
    emit_vertex(corners[1], nrm, uvs[1], v_view_pos[1], v_color[1], v_branch_level[1], v_stiffness[1], phase, 1.0);
    emit_vertex(corners[2], nrm, uvs[2], v_view_pos[2], v_color[2], v_branch_level[2], v_stiffness[2], phase, 1.0);
    EndPrimitive();

    emit_vertex(corners[0], nrm, uvs[0], v_view_pos[0], v_color[0], v_branch_level[0], v_stiffness[0], phase, 1.0);
    emit_vertex(corners[2], nrm, uvs[2], v_view_pos[2], v_color[2], v_branch_level[2], v_stiffness[2], phase, 1.0);
    emit_vertex(corners[3], nrm, uvs[3], v_view_pos[3], v_color[3], v_branch_level[3], v_stiffness[3], phase, 1.0);
    EndPrimitive();
}
