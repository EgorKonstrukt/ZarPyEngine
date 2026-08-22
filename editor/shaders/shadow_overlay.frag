// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core
#define CASCADE_COUNT 4
#define MAX_POINT_SHADOWS 4
#define MAX_SPOT_SHADOWS 4
uniform sampler2D u_scene_color;
uniform sampler2D u_depth_tex;
uniform mat4 u_inv_vp;
uniform sampler2D u_shadow_map_0;
uniform sampler2D u_shadow_map_1;
uniform sampler2D u_shadow_map_2;
uniform sampler2D u_shadow_map_3;
uniform mat4 u_light_space_matrices[CASCADE_COUNT];
uniform float u_cascade_splits[CASCADE_COUNT];
uniform int u_cascade_count;
uniform float u_shadow_bias;
uniform mat4 u_view;
uniform sampler2D u_point_shadow_maps[MAX_POINT_SHADOWS * 6];
uniform mat4 u_point_shadow_vps[MAX_POINT_SHADOWS * 6];
uniform vec3 u_point_shadow_light_positions[MAX_POINT_SHADOWS];
uniform float u_point_shadow_light_ranges[MAX_POINT_SHADOWS];
uniform int u_point_shadow_count;
uniform sampler2D u_spot_shadow_maps[MAX_SPOT_SHADOWS];
uniform mat4 u_spot_shadow_vps[MAX_SPOT_SHADOWS];
uniform int u_spot_shadow_count;
in vec2 v_uv;
out vec4 frag_color;
float sample_shadow(sampler2D shadow_map, vec3 proj_coords) {
    float current_depth = proj_coords.z - u_shadow_bias;
    float result = 0.0;
    vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
    float radius = 0.75;
    float weight_sum = 0.0;
    for (int x = -1; x <= 1; x++) {
        for (int y = -1; y <= 1; y++) {
            float weight = 1.0;
            if (x == 0)
            {
                weight += 1.0;
            }
            if (y == 0)
            {
                weight += 1.0;
            }
            float pcf_depth = texture(shadow_map, proj_coords.xy + vec2(x, y) * texel_size * radius).r;
            result += (current_depth > pcf_depth ? 1.0 : 0.0) * weight;
            weight_sum += weight;
        }
    }
    float lit = 1.0 - result / weight_sum;
    return smoothstep(0.12, 0.88, lit);
}
float compute_directional_shadow(vec3 world_pos) {
    if (u_cascade_count <= 0) return 1.0;
    int cascade_idx = 0;
    vec4 view_pos = u_view * vec4(world_pos, 1.0);
    float frag_depth = abs(view_pos.z);
    for (int i = 0; i < CASCADE_COUNT - 1; i++) {
        if (frag_depth > u_cascade_splits[i]) cascade_idx = i + 1;
    }
    vec4 light_space_pos = u_light_space_matrices[cascade_idx] * vec4(world_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
    if (cascade_idx == 0) return sample_shadow(u_shadow_map_0, proj_coords);
    else if (cascade_idx == 1) return sample_shadow(u_shadow_map_1, proj_coords);
    else if (cascade_idx == 2) return sample_shadow(u_shadow_map_2, proj_coords);
    return sample_shadow(u_shadow_map_3, proj_coords);
}
float compute_point_shadow_pass(vec3 world_pos) {
    if (u_point_shadow_count <= 0) return 1.0;
    float result = 1.0;
    for (int li = 0; li < u_point_shadow_count; li++) {
        vec3 light_pos = u_point_shadow_light_positions[li];
        vec3 dir = world_pos - light_pos;
        vec3 abs_dir = abs(dir);
        int face = 0;
        if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) face = dir.x >= 0 ? 0 : 1;
        else if (abs_dir.y >= abs_dir.z) face = dir.y >= 0 ? 2 : 3;
        else face = dir.z >= 0 ? 4 : 5;
        int base = li * 6;
        vec4 light_space_pos = u_point_shadow_vps[base + face] * vec4(world_pos, 1.0);
        vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
        proj_coords = proj_coords * 0.5 + 0.5;
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) continue;
        float d = length(world_pos - light_pos);
        float range = u_point_shadow_light_ranges[li];
        if (d > range) continue;
        result = min(result, sample_shadow(u_point_shadow_maps[base + face], proj_coords));
    }
    return result;
}
float compute_spot_shadow_pass(vec3 world_pos) {
    if (u_spot_shadow_count <= 0) return 1.0;
    float result = 1.0;
    for (int li = 0; li < u_spot_shadow_count; li++) {
        vec4 light_space_pos = u_spot_shadow_vps[li] * vec4(world_pos, 1.0);
        vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
        proj_coords = proj_coords * 0.5 + 0.5;
        if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) continue;
        result = min(result, sample_shadow(u_spot_shadow_maps[li], proj_coords));
    }
    return result;
}
void main() {
    vec3 scene_color = texture(u_scene_color, v_uv).rgb;
    float depth = texture(u_depth_tex, v_uv).r;
    if (depth >= 1.0) {
        frag_color = vec4(scene_color, 1.0);
        return;
    }
    vec4 clip_pos = vec4(v_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 world_pos4 = u_inv_vp * clip_pos;
    vec3 world_pos = world_pos4.xyz / world_pos4.w;
    float shadow = min(min(compute_directional_shadow(world_pos), compute_point_shadow_pass(world_pos)), compute_spot_shadow_pass(world_pos));
    float visibility = mix(0.46, 1.0, shadow);
    frag_color = vec4(scene_color * visibility, 1.0);
}
