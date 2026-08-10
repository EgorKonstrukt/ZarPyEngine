// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

Shader "Zarin/PBR"
{
    Properties
    {
        [MainColor] _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        [MainTexture] _BaseMap("Base Map", 2D) = "white" {}
        _Metallic("Metallic", Range(0, 1)) = 0
        _Smoothness("Smoothness", Range(0, 1)) = 0.5
        _EmissionColor("Emission Color", Color) = (0, 0, 0, 0)
        _EmissionIntensity("Emission Intensity", Float) = 0
        _EmissionMap("Emission Map", 2D) = "white" {}
        _NormalMap("Normal Map", 2D) = "bump" {}
        _NormalStrength("Normal Strength", Range(0, 2)) = 1
        _OcclusionMap("Occlusion Map", 2D) = "white" {}
        _OcclusionStrength("Occlusion Strength", Range(0, 1)) = 1
        _HeightMap("Height Map", 2D) = "white" {}
        _HeightScale("Height Scale", Range(0, 0.2)) = 0.05
        _Cutoff("Alpha Cutoff", Range(0, 1)) = 0.5
        _ClearCoat("Clear Coat", Range(0, 1)) = 0
        _ClearCoatRoughness("Clear Coat Roughness", Range(0, 1)) = 0.1
        _Anisotropy("Anisotropy", Range(-1, 1)) = 0
        _AnisotropyDirection("Anisotropy Direction", Range(0, 360)) = 0
        _DetailAlbedoMap("Detail Albedo Map", 2D) = "white" {}
        _DetailNormalMap("Detail Normal Map", 2D) = "bump" {}
        _DetailNormalStrength("Detail Normal Strength", Range(0, 2)) = 1
        _DetailUVScale("Detail UV Scale", Float) = 4
        _SubsurfaceColor("Subsurface Color", Color) = (1, 1, 1, 1)
        _SubsurfaceAmount("Subsurface Amount", Range(0, 1)) = 0
        _ThinFilmThickness("Thin Film Thickness", Range(0, 600)) = 0
        _ThinFilmIntensity("Thin Film Intensity", Range(0, 1)) = 0
        _Transmission("Transmission", Range(0, 1)) = 0
        _IOR("IOR", Range(1.0, 3.0)) = 1.5
        _double_sided("Double Sided", Range(0, 1)) = 0
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" }

        Pass
        {
            GLSLPROGRAM
            #version 460 core
            layout(location = 0) in vec3 in_position;
            layout(location = 1) in vec3 in_normal;
            layout(location = 2) in vec2 in_uv;
            layout(location = 3) in vec4 in_model0;
            layout(location = 4) in vec4 in_model1;
            layout(location = 5) in vec4 in_model2;
            layout(location = 6) in vec4 in_model3;
            layout(location = 7) in vec4 in_color;
            layout(std430, binding = 4) readonly buffer InstanceModels { mat4 _ssbo_models[]; };
            layout(std430, binding = 5) readonly buffer InstanceIndices { int _ssbo_indices[]; };
            uniform int u_use_instancing;
            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform mat3 u_normal_matrix;
            out vec3 v_world_pos;
            out vec3 v_normal;
            out vec2 v_uv;
            out vec3 v_view_pos;
            out vec4 v_color;
            void main() {
                mat4 inst_model = mat4(in_model0, in_model1, in_model2, in_model3);
                mat4 _model = (u_use_instancing == 1) ? inst_model : ((u_use_instancing == 2) ? _ssbo_models[_ssbo_indices[gl_InstanceID]] : u_model);
                mat3 _normal_matrix = (u_use_instancing >= 1) ? transpose(inverse(mat3(_model))) : u_normal_matrix;
                vec4 world_pos = _model * vec4(in_position, 1.0);
                v_world_pos = world_pos.xyz;
                v_normal = normalize(_normal_matrix * in_normal);
                v_uv = in_uv;
                v_color = in_color;
                vec4 view_pos = u_view * world_pos;
                v_view_pos = view_pos.xyz;
                gl_Position = u_proj * u_view * world_pos;
            }

            // @FRAGMENT

            #version 460 core
            #define MAX_LIGHTS 8
            #define CASCADE_COUNT 3
            #define PI 3.14159265359
            #define PI_4 0.78539816339
            #define EPS 0.0001
            #define POISSON_SAMPLES 16
            in vec3 v_world_pos;
            in vec3 v_normal;
            in vec2 v_uv;
            in vec3 v_view_pos;
            in vec4 v_color;
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

            uniform vec4 _BaseColor;
            uniform float _Metallic;
            uniform float _Smoothness;
            uniform vec3 _EmissionColor;
            uniform float _EmissionIntensity;
            uniform sampler2D _BaseMap;
            uniform int _BaseMap_Active;
            uniform sampler2D _NormalMap;
            uniform int _NormalMap_Active;
            uniform float _NormalStrength;
            uniform sampler2D _OcclusionMap;
            uniform int _OcclusionMap_Active;
            uniform float _OcclusionStrength;
            uniform sampler2D _HeightMap;
            uniform int _HeightMap_Active;
            uniform float _HeightScale;
            uniform sampler2D _EmissionMap;
            uniform int _EmissionMap_Active;
            uniform float _Cutoff;
            uniform float _ClearCoat;
            uniform float _ClearCoatRoughness;
            uniform float _Anisotropy;
            uniform float _AnisotropyDirection;
            uniform sampler2D _DetailAlbedoMap;
            uniform int _DetailAlbedoMap_Active;
            uniform sampler2D _DetailNormalMap;
            uniform int _DetailNormalMap_Active;
            uniform float _DetailNormalStrength;
            uniform float _DetailUVScale;
            uniform vec3 _SubsurfaceColor;
            uniform float _SubsurfaceAmount;
            uniform float _ThinFilmThickness;
            uniform float _ThinFilmIntensity;
            uniform float _Transmission;
            uniform float _IOR;
            uniform int u_double_sided;

            uniform vec3 u_camera_pos;
            uniform Light u_lights[MAX_LIGHTS];
            uniform int u_light_count;
            uniform vec3 u_ambient;
            uniform int u_shadow_light_index;
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

            uniform samplerCube u_irradiance_map;
            uniform int u_irradiance_map_Active;
            uniform samplerCube u_prefilter_map;
            uniform int u_prefilter_map_Active;
            uniform sampler2D u_brdf_lut;
            uniform int u_brdf_lut_Active;
            uniform float u_env_map_rotation;

            vec3 get_tangent_frame(out vec3 T, out vec3 B);
            vec3 get_normal_from_map(sampler2D normal_map, vec2 uv, float strength);

            const vec2 poisson_disk[POISSON_SAMPLES] = vec2[](
                vec2(0.079575, 0.578682), vec2(-0.882423, 0.073115),
                vec2(-0.362586, 0.584344), vec2(0.509705, -0.620710),
                vec2(-0.123670, -0.953501), vec2(0.860160, 0.299207),
                vec2(0.691263, -0.507288), vec2(-0.453053, -0.725332),
                vec2(-0.725407, 0.390592), vec2(0.251858, 0.158784),
                vec2(0.744786, -0.187990), vec2(-0.293813, -0.178377),
                vec2(-0.659467, -0.024936), vec2(0.192856, 0.880197),
                vec2(-0.087864, -0.142065), vec2(0.218582, -0.643362)
            );

            vec2 parallax_uv(vec2 uv, vec3 V, vec3 N, out float height) {
                height = 0.0;
                if (_HeightMap_Active == 0) return uv;
                vec3 T, B;
                get_tangent_frame(T, B);
                vec3 TBN_V = normalize(vec3(dot(V, T), dot(V, B), dot(V, N)));
                float num_layers = mix(8.0, 32.0, abs(TBN_V.z));
                float layer_depth = 1.0 / num_layers;
                float cur_depth = 1.0;
                vec2 delta = TBN_V.xy * _HeightScale / num_layers;
                vec2 cur_uv = uv;
                float h = 1.0 - texture(_HeightMap, cur_uv).r;
                while (cur_depth > h) {
                    cur_uv -= delta;
                    h = 1.0 - texture(_HeightMap, cur_uv).r;
                    cur_depth -= layer_depth;
                    if (cur_uv.x < 0.0 || cur_uv.x > 1.0 || cur_uv.y < 0.0 || cur_uv.y > 1.0) break;
                }
                vec2 prev_uv = cur_uv + delta;
                float after = h - cur_depth;
                float before = 1.0 - texture(_HeightMap, prev_uv).r - (cur_depth + layer_depth);
                height = h;
                return mix(cur_uv, prev_uv, after / (after + before + EPS));
            }

            float sample_shadow_improved(sampler2D shadow_map, vec3 proj_coords) {
                float current_depth = proj_coords.z - u_shadow_bias;
                float result = 0.0;
                vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
                float radius = 1.5;
                for (int i = 0; i < POISSON_SAMPLES; i++) {
                    float pcf_depth = texture(shadow_map, proj_coords.xy + poisson_disk[i] * texel_size * radius).r;
                    result += (current_depth > pcf_depth + u_shadow_bias) ? 1.0 : 0.0;
                }
                float lit = 1.0 - result / float(POISSON_SAMPLES);
                return smoothstep(0.08, 0.92, lit);
            }

            float compute_shadow_improved() {
                if (u_cascade_count <= 0) return 1.0;
                int cascade_idx = 0;
                float frag_depth = abs(v_view_pos.z);
                for (int i = 0; i < CASCADE_COUNT - 1; i++) {
                    if (frag_depth > u_cascade_splits[i]) cascade_idx = i + 1;
                }
                vec4 light_space_pos = u_light_space_matrices[cascade_idx] * vec4(v_world_pos, 1.0);
                vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
                proj_coords = proj_coords * 0.5 + 0.5;
                if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
                if (cascade_idx == 0) return sample_shadow_improved(u_shadow_map_0, proj_coords);
                else if (cascade_idx == 1) return sample_shadow_improved(u_shadow_map_1, proj_coords);
                return sample_shadow_improved(u_shadow_map_2, proj_coords);
            }

            float compute_point_shadow_improved() {
                if (u_point_shadow_count <= 0) return 1.0;
                vec3 dir = v_world_pos - u_point_light_pos;
                vec3 abs_dir = abs(dir);
                int face = 0;
                if (abs_dir.x >= abs_dir.y && abs_dir.x >= abs_dir.z) face = dir.x >= 0 ? 0 : 1;
                else if (abs_dir.y >= abs_dir.z) face = dir.y >= 0 ? 2 : 3;
                else face = dir.z >= 0 ? 4 : 5;
                vec4 light_space_pos = u_point_light_vps[face] * vec4(v_world_pos, 1.0);
                vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
                proj_coords = proj_coords * 0.5 + 0.5;
                if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
                if (face == 0) return sample_shadow_improved(u_point_shadow_map_0, proj_coords);
                else if (face == 1) return sample_shadow_improved(u_point_shadow_map_1, proj_coords);
                else if (face == 2) return sample_shadow_improved(u_point_shadow_map_2, proj_coords);
                else if (face == 3) return sample_shadow_improved(u_point_shadow_map_3, proj_coords);
                else if (face == 4) return sample_shadow_improved(u_point_shadow_map_4, proj_coords);
                return sample_shadow_improved(u_point_shadow_map_5, proj_coords);
            }

            float compute_spot_shadow_improved() {
                if (u_spot_shadow_count <= 0) return 1.0;
                vec4 light_space_pos = u_spot_light_vp * vec4(v_world_pos, 1.0);
                vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
                proj_coords = proj_coords * 0.5 + 0.5;
                if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
                return sample_shadow_improved(u_spot_shadow_map, proj_coords);
            }

            #define PBR_AREA_SHADOWS
            // @SHADOW_INCLUDE

            vec3 fresnel_schlick(float cos_theta, vec3 F0) {
                return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
            }

            vec3 fresnel_schlick_roughness(float cos_theta, vec3 F0, float roughness) {
                return F0 + (max(vec3(1.0 - roughness), F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
            }

            float distribution_ggx(vec3 N, vec3 H, float roughness) {
                float a = roughness * roughness;
                float a2 = a * a;
                float NdotH = max(dot(N, H), 0.0);
                float NdotH2 = NdotH * NdotH;
                float denom = NdotH2 * (a2 - 1.0) + 1.0;
                return a2 / (PI * denom * denom);
            }

            float distribution_ggx_aniso(vec3 N, vec3 H, vec3 T, vec3 B, float ax, float ay) {
                float NdotH = max(dot(N, H), 0.0);
                float HdotT = dot(H, T);
                float HdotB = dot(H, B);
                float denom = HdotT * HdotT / (ax * ax) + HdotB * HdotB / (ay * ay) + NdotH * NdotH;
                return 1.0 / (PI * ax * ay * denom * denom);
            }

            float geometry_schlick_ggx(float NdotV, float roughness) {
                float r = roughness + 1.0;
                float k = r * r / 8.0;
                return NdotV / (NdotV * (1.0 - k) + k);
            }

            float geometry_smith(vec3 N, vec3 V, vec3 L, float roughness) {
                float NdotV = max(dot(N, V), 0.0);
                float NdotL = max(dot(N, L), 0.0);
                return geometry_schlick_ggx(NdotV, roughness) * geometry_schlick_ggx(NdotL, roughness);
            }

            float geometry_smith_aniso(vec3 N, vec3 V, vec3 L, vec3 T, vec3 B, float ax, float ay) {
                float NdotV = max(dot(N, V), 0.0);
                float NdotL = max(dot(N, L), 0.0);
                float VdotT = dot(V, T);
                float VdotB = dot(V, B);
                float LdotT = dot(L, T);
                float LdotB = dot(L, B);
                float kx = ax + 1.0;
                float ky = ay + 1.0;
                float k = kx * ky / 8.0;
                float gv = NdotV / (NdotV * (1.0 - k) + k);
                float gl = NdotL / (NdotL * (1.0 - k) + k);
                return gv * gl;
            }

            vec3 fresnel_schlick_iridescence(float cos_theta, vec3 F0, vec3 iridescence_fresnel, float iridescence_intensity) {
                vec3 mixed = mix(F0, iridescence_fresnel, iridescence_intensity);
                return mixed + (1.0 - mixed) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
            }

            float iridescence_factor(float thickness, float cos_theta) {
                float delta = 2.0 * thickness * cos_theta;
                float phase_r = delta / 0.650;
                float phase_g = delta / 0.532;
                float phase_b = delta / 0.473;
                vec3 iri;
                iri.r = 0.5 + 0.5 * cos(phase_r * 2.0 * PI);
                iri.g = 0.5 + 0.5 * cos(phase_g * 2.0 * PI + 0.33);
                iri.b = 0.5 + 0.5 * cos(phase_b * 2.0 * PI + 0.67);
                return (iri.r + iri.g + iri.b) / 3.0;
            }

            vec3 get_normal_from_map(sampler2D normal_map, vec2 uv, float strength) {
                vec3 tangent_normal = texture(normal_map, uv).rgb * 2.0 - 1.0;
                tangent_normal.xy *= strength;
                tangent_normal.z = sqrt(max(1.0 - dot(tangent_normal.xy, tangent_normal.xy), 0.0));
                vec3 N = normalize(v_normal);
                vec3 dp1 = dFdx(v_world_pos);
                vec3 dp2 = dFdy(v_world_pos);
                vec2 duv1 = dFdx(v_uv);
                vec2 duv2 = dFdy(v_uv);
                vec3 T = normalize(dp1 * duv2.y - dp2 * duv1.y);
                vec3 B = normalize(cross(N, T));
                return normalize(T * tangent_normal.x + B * tangent_normal.y + N * tangent_normal.z);
            }

            vec3 get_tangent_frame(out vec3 T, out vec3 B) {
                vec3 N = normalize(v_normal);
                vec3 dp1 = dFdx(v_world_pos);
                vec3 dp2 = dFdy(v_world_pos);
                vec2 duv1 = dFdx(v_uv);
                vec2 duv2 = dFdy(v_uv);
                float det = duv1.x * duv2.y - duv2.x * duv1.y;
                if (abs(det) < EPS) {
                    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
                    T = normalize(cross(up, N));
                    B = normalize(cross(N, T));
                    return N;
                }
                float r = 1.0 / det;
                T = normalize((dp1 * duv2.y - dp2 * duv1.y) * r);
                B = normalize(cross(N, T));
                return N;
            }

            vec3 get_anisotropic_tangent(vec3 T, vec3 N, float aniso_dir_rad) {
                vec3 bitangent = normalize(cross(N, T));
                float cos_a = cos(aniso_dir_rad);
                float sin_a = sin(aniso_dir_rad);
                return normalize(T * cos_a + bitangent * sin_a);
            }

            vec3 calc_light_pbr(Light light, vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0, float shadow_factor, vec3 T, vec3 B, float anisotropy, vec3 subsurface_color, float subsurface_amount) {
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
                vec3 L = light_dir;
                float NdotL = max(dot(N, L), 0.0);
                vec3 H = normalize(V + L);
                vec3 radiance = light.color * light.intensity * attenuation * shadow_factor;
                vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
                vec3 specular;
                vec3 kS;
                vec3 kD;
                if (abs(anisotropy) > EPS) {
                    float ax = max(roughness * roughness * (1.0 + anisotropy), 0.001);
                    float ay = max(roughness * roughness * (1.0 - anisotropy), 0.001);
                    float NDF = distribution_ggx_aniso(N, H, T, B, ax, ay);
                    float G = geometry_smith_aniso(N, V, L, T, B, ax, ay);
                    kS = F;
                    kD = (vec3(1.0) - kS) * (1.0 - metallic);
                    vec3 numerator = NDF * G * F;
                    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                    specular = numerator / denominator;
                } else {
                    float NDF = distribution_ggx(N, H, roughness);
                    float G = geometry_smith(N, V, L, roughness);
                    kS = F;
                    kD = (vec3(1.0) - kS) * (1.0 - metallic);
                    vec3 numerator = NDF * G * F;
                    float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                    specular = numerator / denominator;
                }
                vec3 diffuse = kD * albedo / PI;
                vec3 lo = (diffuse + specular) * radiance * NdotL;
                if (subsurface_amount > EPS) {
                    float NdotL_ss = max(dot(N, -L), 0.0);
                    float ss = pow(NdotL_ss, 3.0) * subsurface_amount;
                    lo += subsurface_color * radiance * ss * 0.5;
                }
                return lo;
            }

            vec3 calc_light_clearcoat(vec3 N, vec3 V, vec3 L, float clear_coat, float cc_roughness) {
                vec3 H = normalize(V + L);
                float NDF = distribution_ggx(N, H, cc_roughness);
                float G = geometry_smith(N, V, L, cc_roughness);
                vec3 F = fresnel_schlick(max(dot(H, V), 0.0), vec3(0.04));
                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                return numerator / denominator * clear_coat * 0.25;
            }

            float hash2(vec2 p) {
                p = fract(p * vec2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return fract(p.x * p.y);
            }

            vec3 calc_area_light_pbr(Light light, vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0, vec3 T, vec3 B, float anisotropy, vec3 subsurface_color, float subsurface_amount) {
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
                        float jx = hash2(gl_FragCoord.xy + vec2(float(i), float(j))) - 0.5;
                        float jy = hash2(gl_FragCoord.xy + vec2(float(j), float(i))) - 0.5;
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
                        float NdL = dot(N, ld);
                        if (!ds) {
                            NdL = max(NdL, 0.0);
                            if (NdL <= 0.0) continue;
                        } else {
                            NdL = abs(NdL);
                        }
                        float att = clamp(1.0 - dist / light.range, 0.0, 1.0);
                        att *= att;
                        vec3 radiance = light.color * light.intensity * att * inv_n;
                        vec3 H = normalize(V + ld);
                        vec3 F = fresnel_schlick(max(dot(H, V), 0.0), F0);
                        float NDF;
                        float G;
                        if (abs(anisotropy) > EPS) {
                            float ax = max(roughness * roughness * (1.0 + anisotropy), 0.001);
                            float ay = max(roughness * roughness * (1.0 - anisotropy), 0.001);
                            NDF = distribution_ggx_aniso(N, H, T, B, ax, ay);
                            G = geometry_smith_aniso(N, V, ld, T, B, ax, ay);
                        } else {
                            NDF = distribution_ggx(N, H, roughness);
                            G = geometry_smith(N, V, ld, roughness);
                        }
                        vec3 kS = F;
                        vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
                        vec3 specular = NDF * G * F / max(4.0 * max(dot(N, V), 0.0) * max(dot(N, ld), 0.0) + 0.0001, 0.0001);
                        vec3 diffuse = kD * albedo / PI;
                        vec3 lo = (diffuse + specular) * radiance * NdL;
                        if (subsurface_amount > EPS) {
                            float NdotL_ss = max(dot(N, -ld), 0.0);
                            float ss = pow(NdotL_ss, 3.0) * subsurface_amount;
                            lo += subsurface_color * radiance * ss * 0.5;
                        }
                        diff += lo;
                    }
                }
                return diff;
            }

            vec3 ibl_contribution(vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0, float occlusion, vec3 T, vec3 B, float anisotropy, float clear_coat, float cc_roughness) {
                vec3 irradiance = u_ambient * albedo;
                vec3 specular_ibl = vec3(0.0);
                if (u_irradiance_map_Active == 1) {
                    vec3 rot_dir = N;
                    if (u_env_map_rotation != 0.0) {
                        float s = sin(u_env_map_rotation);
                        float c = cos(u_env_map_rotation);
                        rot_dir.x = N.x * c + N.z * s;
                        rot_dir.z = -N.x * s + N.z * c;
                    }
                    irradiance = texture(u_irradiance_map, rot_dir).rgb;
                }
                vec3 kS;
                if (u_prefilter_map_Active == 1 && u_brdf_lut_Active == 1) {
                    float NdotV = max(dot(N, V), 0.0);
                    vec3 R = reflect(-V, N);
                    if (u_env_map_rotation != 0.0) {
                        float s = sin(u_env_map_rotation);
                        float c = cos(u_env_map_rotation);
                        float rx = R.x * c + R.z * s;
                        float rz = -R.x * s + R.z * c;
                        R.x = rx;
                        R.z = rz;
                    }
                    float prefilter_lod = roughness * 4.0;
                    R = normalize(R);
                    vec3 prefiltered = textureLod(u_prefilter_map, R, prefilter_lod).rgb;
                    vec2 env_brdf = texture(u_brdf_lut, vec2(NdotV, roughness)).rg;
                    kS = fresnel_schlick_roughness(NdotV, F0, roughness);
                    specular_ibl = prefiltered * (kS * env_brdf.x + env_brdf.y);
                } else {
                    kS = vec3(0.0);
                }
                vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
                vec3 diffuse_ibl = kD * irradiance * albedo;
                vec3 ambient = (diffuse_ibl + specular_ibl) * occlusion;
                if (clear_coat > EPS && u_prefilter_map_Active == 1) {
                    float NdotV = max(dot(N, V), 0.0);
                    vec3 R = reflect(-V, N);
                    float prefilter_lod = cc_roughness * 4.0;
                    R = normalize(R);
                    if (u_env_map_rotation != 0.0) {
                        float s = sin(u_env_map_rotation);
                        float c = cos(u_env_map_rotation);
                        float rx = R.x * c + R.z * s;
                        float rz = -R.x * s + R.z * c;
                        R.x = rx;
                        R.z = rz;
                    }
                    vec3 prefiltered = textureLod(u_prefilter_map, R, prefilter_lod).rgb;
                    vec2 env_brdf = texture(u_brdf_lut, vec2(NdotV, cc_roughness)).rg;
                    vec3 cc_fresnel = fresnel_schlick_roughness(NdotV, vec3(0.04), cc_roughness);
                    vec3 cc_specular = prefiltered * (cc_fresnel * env_brdf.x + env_brdf.y);
                    ambient += cc_specular * clear_coat * 0.25;
                }
                return ambient;
            }

            void main() {
                vec2 uv = v_uv;
                float height = 0.0;
                {
                    vec3 V = normalize(u_camera_pos - v_world_pos);
                    vec3 N_tmp = normalize(v_normal);
                    uv = parallax_uv(v_uv, V, N_tmp, height);
                }
                vec3 albedo = _BaseColor.rgb;
                if (_BaseMap_Active == 1) {
                    vec4 texColor = texture(_BaseMap, uv);
                    albedo *= texColor.rgb;
                }
                if (_DetailAlbedoMap_Active == 1) {
                    vec4 detailColor = texture(_DetailAlbedoMap, uv * _DetailUVScale);
                    albedo = mix(albedo, detailColor.rgb, detailColor.a);
                }
                albedo *= v_color.rgb;
                float alpha = _BaseColor.a;
                if (_BaseMap_Active == 1) {
                    alpha *= texture(_BaseMap, uv).a;
                }
                if (alpha < _Cutoff) discard;
                vec3 T, B;
                vec3 N = get_tangent_frame(T, B);
                if (_NormalMap_Active == 1) {
                    N = get_normal_from_map(_NormalMap, uv, _NormalStrength);
                }
                if (_DetailNormalMap_Active == 1) {
                    vec3 detail_N = get_normal_from_map(_DetailNormalMap, uv * _DetailUVScale, _DetailNormalStrength);
                    N = normalize(N + detail_N);
                }
                if (u_double_sided == 1 && !gl_FrontFacing) N = -N;
                float roughness = 1.0 - _Smoothness;
                roughness = max(roughness, 0.001);
                float metallic = _Metallic;
                vec3 F0 = mix(vec3(0.04), albedo, metallic);
                vec3 V = normalize(u_camera_pos - v_world_pos);
                float NdotV = max(dot(N, V), 0.0);
                float occlusion = 1.0;
                if (_OcclusionMap_Active == 1) {
                    occlusion = texture(_OcclusionMap, uv).r;
                    occlusion = mix(1.0, occlusion, _OcclusionStrength);
                }
                vec3 result = vec3(0.0);
                float aniso_rad = radians(_AnisotropyDirection);
                vec3 aniso_T = get_anisotropic_tangent(T, N, aniso_rad);
                vec3 aniso_B = normalize(cross(N, aniso_T));
                float cc_roughness = max(_ClearCoatRoughness, 0.001);
                result += ibl_contribution(N, V, albedo, roughness, metallic, F0, occlusion, aniso_T, aniso_B, _Anisotropy, _ClearCoat, cc_roughness);
                float shadow_factor = compute_shadow_improved();
                float point_shadow_factor = compute_point_shadow_improved();
                float spot_shadow_factor = compute_spot_shadow_improved();
                float area_shadow_factor = compute_area_shadow();
                for (int i = 0; i < u_light_count && i < MAX_LIGHTS; i++) {
                    float sf = 1.0;
                    if (i == u_shadow_light_index) sf = min(sf, shadow_factor);
                    if (i == u_point_shadow_light_index) sf = min(sf, point_shadow_factor);
                    if (i == u_spot_shadow_light_index) sf = min(sf, spot_shadow_factor);
                    if (i == u_area_shadow_light_index) sf = min(sf, area_shadow_factor);
                    if (u_lights[i].type == 3) {
                    result += calc_area_light_pbr(u_lights[i], N, V, albedo, roughness, metallic, F0, aniso_T, aniso_B, _Anisotropy, _SubsurfaceColor, _SubsurfaceAmount) * sf;
                    } else {
                        result += calc_light_pbr(u_lights[i], N, V, albedo, roughness, metallic, F0, sf, aniso_T, aniso_B, _Anisotropy, _SubsurfaceColor, _SubsurfaceAmount);
                    }
                }
                if (_ClearCoat > EPS) {
                    float cc_spec = 0.0;
                    for (int i = 0; i < u_light_count && i < MAX_LIGHTS; i++) {
                        vec3 light_dir;
                        float attenuation = 1.0;
                        if (u_lights[i].type == 0) {
                            light_dir = normalize(-u_lights[i].direction);
                        } else {
                            vec3 to_light = u_lights[i].position - v_world_pos;
                            float dist = length(to_light);
                            light_dir = normalize(to_light);
                            attenuation = clamp(1.0 - dist / u_lights[i].range, 0.0, 1.0);
                            attenuation *= attenuation;
                            if (u_lights[i].type == 2) {
                                float theta = dot(light_dir, normalize(-u_lights[i].direction));
                                float inner = cos(radians(u_lights[i].spot_inner_angle));
                                float outer = cos(radians(u_lights[i].spot_angle));
                                float eps = inner - outer;
                                attenuation *= clamp((theta - outer) / eps, 0.0, 1.0);
                            }
                        }
                        cc_spec += calc_light_clearcoat(N, V, light_dir, _ClearCoat, cc_roughness).r * attenuation;
                    }
                    result += vec3(cc_spec);
                }
                vec3 emission = _EmissionColor * _EmissionIntensity;
                if (_EmissionMap_Active == 1) {
                    emission *= texture(_EmissionMap, uv).rgb;
                }
                result += emission;
                float spec_occlusion = clamp(pow(NdotV, 2.0), 0.0, 1.0);
                result = mix(result, result * spec_occlusion, occlusion * (1.0 - roughness));
                result = result / (result + vec3(1.0));
                result = pow(result, vec3(1.0 / 2.2));
                frag_color = vec4(result, alpha);
            }
            ENDGLSL
        }
    }
}