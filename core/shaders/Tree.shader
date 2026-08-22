// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

Shader "Zarin/Tree"
{
    Properties
    {
        [MainColor] _BarkColor("Bark Color", Color) = (0.45, 0.25, 0.12, 1)
        _BarkRoughness("Bark Roughness", Range(0, 1)) = 0.85
        _BarkMetallic("Bark Metallic", Range(0, 1)) = 0.0
        [MainColor] _LeafColor("Leaf Color", Color) = (0.2, 0.55, 0.12, 1)
        _LeafRoughness("Leaf Roughness", Range(0, 1)) = 0.6
        _LeafMetallic("Leaf Metallic", Range(0, 1)) = 0.0
        _LeafSubsurface("Leaf Subsurface", Range(0, 1)) = 0.6
        _LeafSubsurfaceColor("Leaf Subsurface Color", Color) = (0.4, 0.8, 0.2, 1)
        _LeafFlutterSpeed("Leaf Flutter Speed", Float) = 8.0
        _LeafFlutterAmount("Leaf Flutter Amount", Float) = 0.04
        _WindInfluence("Wind Influence", Range(0, 1)) = 1.0
        _WindDir("Wind Direction", Vector) = (1, 0, 0.5, 0)
        _WindSpeed("Wind Speed", Float) = 2.0
        _WindStrength("Wind Strength", Float) = 0.3
        _TurbulenceScale("Turbulence Scale", Float) = 0.5
        _TurbulenceAmount("Turbulence Amount", Range(0, 1)) = 0.3
        _Cutoff("Alpha Cutoff", Range(0, 1)) = 0.3
        _double_sided("Double Sided", Range(0, 1)) = 1
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
            uniform float u_time;
            uniform float _WindInfluence;
            uniform vec3 _WindDir;
            uniform float _WindSpeed;
            uniform float _WindStrength;
            uniform float _LeafFlutterSpeed;
            uniform float _LeafFlutterAmount;
            uniform float _TurbulenceScale;
            uniform float _TurbulenceAmount;
            out vec3 v_world_pos;
            out vec3 v_normal;
            out vec2 v_uv;
            out vec3 v_view_pos;
            out vec4 v_color;
            out float v_branch_level;
            out float v_stiffness;
            out float v_phase_offset;
            out float v_is_leaf;

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

            float fbm_noise(vec3 p, int octaves) {
                float value = 0.0;
                float amplitude = 0.5;
                float frequency = 1.0;
                for (int i = 0; i < 6; i++) {
                    if (i >= octaves) break;
                    value += amplitude * value_noise3(p * frequency);
                    amplitude *= 0.5;
                    frequency *= 2.0;
                }
                return value;
            }

            void main() {
                mat4 inst_model = mat4(in_model0, in_model1, in_model2, in_model3);
                mat4 _model = (u_use_instancing == 1) ? inst_model : ((u_use_instancing == 2 || u_use_instancing == 3) ? _ssbo_models[_ssbo_indices[gl_InstanceID]] : u_model);
                mat3 _normal_matrix = (u_use_instancing >= 1) ? transpose(inverse(mat3(_model))) : u_normal_matrix;
                vec3 local_pos = in_position;
                vec3 local_nrm = in_normal;

                float branch_level = in_color.r;
                float stiffness = in_color.g;
                float phase_off = in_color.b;
                float is_leaf = in_color.a;
                float obj_scale = length(vec3(_model[0][0], _model[1][0], _model[2][0]));
                float height_factor = clamp(local_pos.y / max(0.001, obj_scale) * 0.5 + 0.5, 0.0, 1.0);

                if (_WindInfluence > 0.0 && stiffness < 0.999) {
                    vec3 wind_dir = normalize(_WindDir + vec3(1e-6));
                    float t = u_time * _WindSpeed;
                    float phase = t + phase_off * 6.2831853 + local_pos.x * 0.3 + local_pos.z * 0.3;

                    float sway = sin(phase) * _WindStrength * (1.0 - stiffness) * height_factor * height_factor * _WindInfluence;

                    float turbulence = fbm_noise(local_pos * _TurbulenceScale + u_time * 0.3, 3) - 0.5;
                    sway += turbulence * _TurbulenceAmount * (1.0 - stiffness) * 0.15 * _WindInfluence;

                    local_pos += wind_dir * sway;

                    if (is_leaf > 0.5) {
                        float flutter_phase = u_time * _LeafFlutterSpeed + phase_off * 12.566;
                        float flutter = sin(flutter_phase) * _LeafFlutterAmount * (1.0 - stiffness) * _WindInfluence;
                        vec3 flutter_dir = normalize(vec3(
                            sin(phase_off * 13.37),
                            cos(phase_off * 7.31) * 0.3,
                            cos(phase_off * 11.23)
                        ));
                        local_pos += flutter_dir * flutter;
                        local_pos += wind_dir * flutter * 0.5;
                    }

                    float noise_phase = value_noise3(local_pos * 0.5 + u_time * 0.2) - 0.5;
                    local_pos += wind_dir * noise_phase * _WindStrength * 0.05 * (1.0 - stiffness) * _WindInfluence;
                }

                vec4 world_pos = _model * vec4(local_pos, 1.0);
                v_world_pos = world_pos.xyz;
                v_normal = normalize(_normal_matrix * local_nrm);
                v_uv = in_uv;
                v_color = in_color;
                v_branch_level = branch_level;
                v_stiffness = stiffness;
                v_phase_offset = phase_off;
                v_is_leaf = is_leaf;
                vec4 view_pos = u_view * world_pos;
                v_view_pos = view_pos.xyz;
                gl_Position = u_proj * u_view * world_pos;
            }

            // @FRAGMENT

            #version 460 core
            #define MAX_LIGHTS 8
            #define MAX_POINT_SHADOWS 4
            #define MAX_SPOT_SHADOWS 4
            #define CASCADE_COUNT 4
            #define PI 3.14159265359
            #define EPS 0.0001
            in vec3 v_world_pos;
            in vec3 v_normal;
            in vec2 v_uv;
            in vec3 v_view_pos;
            in vec4 v_color;
            in float v_branch_level;
            in float v_stiffness;
            in float v_phase_offset;
            in float v_is_leaf;
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

            uniform vec4 _BarkColor;
            uniform float _BarkRoughness;
            uniform float _BarkMetallic;
            uniform vec4 _LeafColor;
            uniform float _LeafRoughness;
            uniform float _LeafMetallic;
            uniform float _LeafSubsurface;
            uniform vec3 _LeafSubsurfaceColor;
            uniform float _Cutoff;
            uniform int u_double_sided;
            uniform vec3 u_camera_pos;
            uniform Light u_lights[MAX_LIGHTS];
            uniform int u_light_count;
            uniform vec3 u_ambient;
            uniform int u_shadow_light_index;
            uniform sampler2D u_shadow_map_0;
            uniform sampler2D u_shadow_map_1;
            uniform sampler2D u_shadow_map_2;
uniform sampler2D u_shadow_map_3;
            uniform mat4 u_light_space_matrices[CASCADE_COUNT];
            uniform float u_cascade_splits[CASCADE_COUNT];
            uniform float u_shadow_bias;
            uniform int u_cascade_count;
            uniform sampler2D u_point_shadow_maps[MAX_POINT_SHADOWS * 6];
            uniform mat4 u_point_shadow_vps[MAX_POINT_SHADOWS * 6];
            uniform vec3 u_point_shadow_light_positions[MAX_POINT_SHADOWS];
            uniform float u_point_shadow_light_ranges[MAX_POINT_SHADOWS];
            uniform int u_point_shadow_count;
            uniform int u_point_shadow_light_indices[MAX_POINT_SHADOWS];
            uniform sampler2D u_spot_shadow_maps[MAX_SPOT_SHADOWS];
            uniform mat4 u_spot_shadow_vps[MAX_SPOT_SHADOWS];
            uniform int u_spot_shadow_count;
            uniform int u_spot_shadow_light_indices[MAX_SPOT_SHADOWS];
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

            vec3 fresnel_schlick(float cos_theta, vec3 F0) {
                return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
            }

            float distribution_ggx(vec3 N, vec3 H, float roughness) {
                float a = roughness * roughness;
                float a2 = a * a;
                float NdotH = max(dot(N, H), 0.0);
                float NdotH2 = NdotH * NdotH;
                float denom = NdotH2 * (a2 - 1.0) + 1.0;
                return a2 / (PI * denom * denom);
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

            vec3 calc_light_pbr(Light light, vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0, float shadow_factor) {
                vec3 light_dir;
                float attenuation = 1.0;
                if (light.type == 0) {
                    light_dir = normalize(-light.direction);
                } else {
                    vec3 to_light = light.position - v_world_pos;
                    float dist = length(to_light);
                    light_dir = to_light / dist;
                    float range_fade = clamp(1.0 - pow(dist / max(light.range, 1e-4), 4.0), 0.0, 1.0);
                    attenuation = range_fade * range_fade / (dist * dist + 1.0);
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
                float NDF = distribution_ggx(N, H, roughness);
                float G = geometry_smith(N, V, L, roughness);
                vec3 kS = F;
                vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                vec3 specular = numerator / denominator;
                vec3 diffuse = kD * albedo / PI;
                return (diffuse + specular) * radiance * NdotL;
            }

            vec3 calc_leaf_light(Light light, vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0, float shadow_factor) {
                vec3 light_dir;
                float attenuation = 1.0;
                if (light.type == 0) {
                    light_dir = normalize(-light.direction);
                } else {
                    vec3 to_light = light.position - v_world_pos;
                    float dist = length(to_light);
                    light_dir = to_light / dist;
                    float range_fade = clamp(1.0 - pow(dist / max(light.range, 1e-4), 4.0), 0.0, 1.0);
                    attenuation = range_fade * range_fade / (dist * dist + 1.0);
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
                float NDF = distribution_ggx(N, H, roughness);
                float G = geometry_smith(N, V, L, roughness);
                vec3 kS = F;
                vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
                vec3 numerator = NDF * G * F;
                float denominator = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001;
                vec3 specular = numerator / denominator;
                vec3 diffuse = kD * albedo / PI;
                vec3 lo = (diffuse + specular) * radiance * NdotL;
                float NdotL_ss = max(dot(N, -L), 0.0);
                float ss = pow(NdotL_ss, 3.0) * _LeafSubsurface;
                lo += _LeafSubsurfaceColor * radiance * ss * 0.5;
                return lo;
            }

            vec3 ibl_contribution(vec3 N, vec3 V, vec3 albedo, float roughness, float metallic, vec3 F0, float occlusion) {
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
                    vec3 kS = fresnel_schlick(NdotV, F0);
                    specular_ibl = prefiltered * (kS * env_brdf.x + env_brdf.y);
                }
                vec3 kS = fresnel_schlick(max(dot(N, V), 0.0), F0);
                vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
                return (kD * irradiance * albedo + specular_ibl) * occlusion;
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
                if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0) return 1.0;
                float current_depth = proj_coords.z - u_shadow_bias;
                float result = 0.0;
                vec2 texel_size = 1.0 / vec2(textureSize(u_shadow_map_0, 0));
                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        vec2 offset = vec2(float(i), float(j)) * texel_size;
                        float pcf_depth = 0.0;
                        if (cascade_idx == 0) pcf_depth = texture(u_shadow_map_0, proj_coords.xy + offset).r;
                        else if (cascade_idx == 1) pcf_depth = texture(u_shadow_map_1, proj_coords.xy + offset).r;
                        else if (cascade_idx == 2) pcf_depth = texture(u_shadow_map_2, proj_coords.xy + offset).r;
                        else pcf_depth = texture(u_shadow_map_3, proj_coords.xy + offset).r;
                        result += (current_depth > pcf_depth + u_shadow_bias) ? 1.0 : 0.0;
                    }
                }
                return 1.0 - result / 9.0;
            }

            void main() {
                float is_leaf = v_is_leaf;
                vec3 albedo;
                float roughness;
                float metallic;
                float alpha = 1.0;

                if (is_leaf > 0.5) {
                    albedo = mix(_LeafColor.rgb, _BarkColor.rgb, 0.1);
                    roughness = _LeafRoughness;
                    metallic = _LeafMetallic;
                    alpha = _LeafColor.a;
                    if (alpha < _Cutoff) discard;
                } else {
                    albedo = _BarkColor.rgb;
                    roughness = _BarkRoughness;
                    metallic = _BarkMetallic;
                }

                vec3 N = normalize(v_normal);
                if (!gl_FrontFacing) {
                    if (is_leaf > 0.5 || u_double_sided == 1) N = -N;
                }
                vec3 V = normalize(u_camera_pos - v_world_pos);
                vec3 F0 = mix(vec3(0.04), albedo, metallic);
                roughness = max(roughness, 0.001);

                vec3 result = vec3(0.0);
                result += u_ambient * albedo;
                float shadow_factor = compute_shadow();

                if (is_leaf > 0.5) {
                    for (int i = 0; i < u_light_count && i < MAX_LIGHTS; i++) {
                        result += calc_leaf_light(u_lights[i], N, V, albedo, roughness, metallic, F0, shadow_factor);
                    }
                } else {
                    for (int i = 0; i < u_light_count && i < MAX_LIGHTS; i++) {
                        result += calc_light_pbr(u_lights[i], N, V, albedo, roughness, metallic, F0, shadow_factor);
                    }
                }

                result += ibl_contribution(N, V, albedo, roughness, metallic, F0, 1.0);
                result = result / (result + vec3(1.0));
                result = pow(result, vec3(1.0 / 2.2));

                if (is_leaf > 0.5) {
                    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);
                    result += vec3(0.1) * fresnel * _LeafSubsurface * 0.3;
                }

                frag_color = vec4(result, alpha);
            }
            ENDGLSL
        }
    }
}
