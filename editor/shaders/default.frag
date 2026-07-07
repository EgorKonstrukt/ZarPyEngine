// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core
#define MAX_LIGHTS 8
#define CASCADE_COUNT 3
#define PI 3.14159265359
in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;
in vec3 v_view_pos;
out vec4 frag_color;
struct Light {
    int type;
    vec3 position;
    vec3 direction;
    vec3 color;
    float intensity;
    float range;
    float spot_angle;
    float spot_inner_angle;
    vec3 right;
    vec3 up;
    float area_width;
    float area_height;
    int area_type;
    int area_samples;
    float area_double_sided;
};
uniform vec4 u_albedo_color;
uniform float u_metallic;
uniform float u_smoothness;
uniform vec3 u_emission;
uniform vec3 u_camera_pos;
uniform Light u_lights[MAX_LIGHTS];
uniform int u_light_count;
uniform vec3 u_ambient;
uniform int u_shadow_light_index;
uniform sampler2D u_albedo_tex;
uniform int u_use_albedo_tex;
uniform sampler2D u_normal_tex;
uniform int u_use_normal_tex;
uniform sampler2D u_roughness_tex;
uniform int u_use_roughness_tex;
uniform sampler2D u_shadow_map_0;
uniform sampler2D u_shadow_map_1;
uniform sampler2D u_shadow_map_2;
uniform mat4 u_light_space_matrices[CASCADE_COUNT];
uniform float u_cascade_splits[CASCADE_COUNT];
uniform float u_shadow_bias;
uniform int u_cascade_count;
uniform sampler2D u_point_shadow_map_0;
uniform sampler2D u_point_shadow_map_1;
uniform sampler2D u_point_shadow_map_2;
uniform sampler2D u_point_shadow_map_3;
uniform sampler2D u_point_shadow_map_4;
uniform sampler2D u_point_shadow_map_5;
uniform mat4 u_point_light_vps[6];
uniform vec3 u_point_light_pos;
uniform float u_point_light_range;
uniform int u_point_shadow_count;
uniform int u_point_shadow_light_index;
uniform sampler2D u_spot_shadow_map;
uniform mat4 u_spot_light_vp;
uniform int u_spot_shadow_count;
uniform int u_spot_shadow_light_index;
uniform sampler2D u_area_shadow_map;
uniform mat4 u_area_light_vp;
uniform float u_area_light_size;
uniform float u_area_light_fov_scale;
uniform vec2 u_area_light_near_far;
uniform int u_area_shadow_light_index;
uniform float u_area_shadow_bias;
float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
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
float compute_shadow() {
    if (u_cascade_count <= 0) return 1.0;
    int cascade_idx = 0;
    float frag_depth = abs(v_view_pos.z);
    for (int i = 0; i < CASCADE_COUNT - 1; i++) {
        if (frag_depth > u_cascade_splits[i]) cascade_idx = i + 1;
    }
    vec4 light_space_pos = u_light_space_matrices[cascade_idx] * vec4(v_world_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
    vec2 border = 1.0 - abs(proj_coords.xy - 0.5) * 2.0;
    float fade = clamp(border.x * border.y * 20.0, 0.0, 1.0);
    float shadow;
    if (cascade_idx == 0) shadow = sample_shadow(u_shadow_map_0, proj_coords);
    else if (cascade_idx == 1) shadow = sample_shadow(u_shadow_map_1, proj_coords);
    else shadow = sample_shadow(u_shadow_map_2, proj_coords);
    return mix(1.0, shadow, fade);
}
float compute_point_shadow() {
    if (u_point_shadow_count <= 0) return 1.0;
    vec3 dir = v_world_pos - u_point_light_pos;
    vec3 abs_dir = abs(dir);
    int face = 0;
    if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) {
        face = dir.x >= 0 ? 0 : 1;
    } else if (abs_dir.y >= abs_dir.z) {
        face = dir.y >= 0 ? 2 : 3;
    } else {
        face = dir.z >= 0 ? 4 : 5;
    }
    vec4 light_space_pos = u_point_light_vps[face] * vec4(v_world_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
    if (face == 0) return sample_shadow(u_point_shadow_map_0, proj_coords);
    else if (face == 1) return sample_shadow(u_point_shadow_map_1, proj_coords);
    else if (face == 2) return sample_shadow(u_point_shadow_map_2, proj_coords);
    else if (face == 3) return sample_shadow(u_point_shadow_map_3, proj_coords);
    else if (face == 4) return sample_shadow(u_point_shadow_map_4, proj_coords);
    return sample_shadow(u_point_shadow_map_5, proj_coords);
}
float compute_spot_shadow() {
    if (u_spot_shadow_count <= 0) return 1.0;
    vec4 light_space_pos = u_spot_light_vp * vec4(v_world_pos, 1.0);
    vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
    proj_coords = proj_coords * 0.5 + 0.5;
    if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
    return sample_shadow(u_spot_shadow_map, proj_coords);
}
// @SHADOW_INCLUDE
vec3 calc_area_light(Light light, vec3 normal, vec3 view_dir, vec3 albedo) {
    vec3 right = light.right;
    vec3 up = light.up;
    float hw = light.area_width * 0.5;
    float hh = light.area_height * 0.5;
    vec3 c = light.position;
    int S = max(1, light.area_samples);
    bool ds = light.area_double_sided > 0.5;
    float inv_n = 1.0 / float(S * S);
    vec3 diff = vec3(0.0);
    vec3 spec = vec3(0.0);
    float r1 = 1.0 / float(S);
    float r2 = 1.0 / float(S);
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) {
            float jx = hash(gl_FragCoord.xy + vec2(float(i), float(j))) - 0.5;
            float jy = hash(gl_FragCoord.xy + vec2(float(j), float(i))) - 0.5;
            float u = (float(i) + 0.5 + jx) * r2 * 2.0 - 1.0;
            float v = (float(j) + 0.5 + jy) * r1 * 2.0 - 1.0;
            if (light.area_type == 1) {
                float a = u;
                float b = v;
                float phi_val, r;
                if (abs(a) > abs(b)) {
                    r = a;
                    phi_val = (PI / 4.0) * (b / max(a, 1e-6));
                } else {
                    r = b;
                    phi_val = (PI / 2.0) - (PI / 4.0) * (a / max(b, 1e-6));
                }
                u = r * cos(phi_val);
                v = r * sin(phi_val);
            }
            vec3 sp = c + right * u * hw + up * v * hh;
            vec3 to_sp = sp - v_world_pos;
            float dist = length(to_sp);
            vec3 ld = to_sp / dist;
            float NdL = dot(normal, ld);
            if (!ds) {
                NdL = max(NdL, 0.0);
                if (NdL <= 0.0) continue;
            } else {
                NdL = abs(NdL);
            }
            float att = clamp(1.0 - dist / light.range, 0.0, 1.0);
            att *= att;
            vec3 contrib = light.color * light.intensity * att * inv_n;
            diff += contrib * NdL;
            vec3 h = normalize(ld + view_dir);
            float NdH = max(dot(normal, h), 0.0);
            spec += contrib * pow(NdH, max(1.0, u_smoothness * 128.0));
        }
    }
    return diff * albedo + spec * u_metallic;
}
vec3 calc_light(Light light, vec3 normal, vec3 view_dir, vec3 albedo, float shadow_factor) {
    vec3 light_dir;
    float attenuation = 1.0;
    if (light.type == 0) {
        light_dir = normalize(-light.direction);
    } else {
        vec3 to_light = light.position - v_world_pos;
        float dist = length(to_light);
        light_dir = normalize(to_light);
        attenuation = clamp(1.0 - dist / light.range, 0.0, 1.0);
        attenuation *= attenuation;
        if (light.type == 2) {
            float theta = dot(light_dir, normalize(-light.direction));
            float inner = cos(radians(light.spot_inner_angle));
            float outer = cos(radians(light.spot_angle));
            float eps = inner - outer;
            attenuation *= clamp((theta - outer) / eps, 0.0, 1.0);
        }
    }
    float diff = max(dot(normal, light_dir), 0.0);
    vec3 diffuse = diff * albedo * light.color * light.intensity * shadow_factor;
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec_pow = max(1.0, u_smoothness * 128.0);
    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), spec_pow);
    vec3 specular = spec * light.color * light.intensity * u_metallic * shadow_factor;
    return (diffuse + specular) * attenuation;
}
void main() {
    vec3 albedo = u_albedo_color.rgb;
    if (u_use_albedo_tex == 1) {
        vec4 texColor = texture(u_albedo_tex, v_uv);
        albedo *= texColor.rgb;
    }
    vec3 normal = normalize(v_normal);
    if (u_use_normal_tex == 1) {
        vec3 tangentNormal = texture(u_normal_tex, v_uv).rgb * 2.0 - 1.0;
        normal = normalize(normal + tangentNormal * 0.5);
    }
    float roughness = u_smoothness;
    if (u_use_roughness_tex == 1) {
        float r = texture(u_roughness_tex, v_uv).r;
        roughness = mix(roughness, r, 0.5);
    }
    vec3 view_dir = normalize(u_camera_pos - v_world_pos);
    vec3 result = u_ambient * albedo;
    float shadow_factor = compute_shadow();
    float point_shadow_factor = compute_point_shadow();
    float spot_shadow_factor = compute_spot_shadow();
    for (int i = 0; i < u_light_count && i < MAX_LIGHTS; i++) {
        float sf = 1.0;
        if (i == u_shadow_light_index) sf = min(sf, shadow_factor);
        if (i == u_point_shadow_light_index) sf = min(sf, point_shadow_factor);
        if (i == u_spot_shadow_light_index) sf = min(sf, spot_shadow_factor);
        if (u_lights[i].type == 3) {
            float area_sf = 1.0;
            if (i == u_area_shadow_light_index) area_sf = compute_area_shadow();
            result += calc_area_light(u_lights[i], normal, view_dir, albedo) * area_sf;
        } else {
            result += calc_light(u_lights[i], normal, view_dir, albedo, sf);
        }
    }
    result += u_emission;
    frag_color = vec4(result, u_albedo_color.a);
}
