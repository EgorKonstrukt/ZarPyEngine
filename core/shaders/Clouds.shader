// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

Shader "Zarin/Clouds"
{
    Properties
    {
        _Coverage("Cloud Coverage", Range(0, 1)) = 0.5
        _Density("Cloud Density", Float) = 1.0
        _Speed("Wind Speed", Float) = 0.1
        _WindDirection("Wind Direction", Float) = 18.0
        _Height("Cloud Height", Float) = 50.0
        _Thickness("Cloud Thickness", Float) = 20.0
        _Scale("Noise Scale", Float) = 0.02
        _Opacity("Opacity", Range(0, 1)) = 0.9
        _ShadowStrength("Shadow Strength", Range(0, 1)) = 0.55
        _FogDensity("Fog Density", Range(0, 1)) = 0.12
        _HeightFalloff("Height Falloff", Float) = 0.18
        _Scattering("Scattering", Range(0, 2)) = 0.85
        _Absorption("Absorption", Range(0, 4)) = 1.0
        _Softness("Softness", Range(0.02, 0.8)) = 0.28
        _DetailStrength("Detail Strength", Range(0, 1)) = 0.45
        _Steps("Volume Steps", Float) = 14
        _Evolution("Evolution", Range(0, 2)) = 0.8
        _Contrast("Contrast", Range(0.2, 2.5)) = 1.35
        _SilverLining("Silver Lining", Range(0, 2)) = 1.0
        _Seed("Seed", Float) = 11.3
        _Tint("Tint", Color) = (0.82, 0.90, 1.0, 1.0)
    }
    SubShader
    {
        Pass
        {
            GLSLPROGRAM
            #version 460 core
            layout(location = 0) in vec3 in_position;
            layout(location = 2) in vec2 in_uv;
            out vec2 v_uv;
            void main()
            {
                v_uv = in_uv;
                gl_Position = vec4(in_position.xy, 0.0, 1.0);
            }

            // @FRAGMENT

            #version 460 core
            #define CASCADE_COUNT 4
            in vec2 v_uv;
            out vec4 frag_color;
            uniform float u_time;
            uniform vec3 _SunDirection;
            uniform vec3 _SunColor;
            uniform float _SunIntensity;
            uniform float _Coverage;
            uniform float _Density;
            uniform float _Speed;
            uniform float _WindDirection;
            uniform float _Height;
            uniform float _Thickness;
            uniform float _Scale;
            uniform float _Opacity;
            uniform float _ShadowStrength;
            uniform float _FogDensity;
            uniform float _HeightFalloff;
            uniform float _Scattering;
            uniform float _Absorption;
            uniform float _Softness;
            uniform float _DetailStrength;
            uniform int _Steps;
            uniform float _Evolution;
            uniform float _Contrast;
            uniform float _SilverLining;
            uniform float _Seed;
            uniform vec3 _Tint;
            uniform vec3 u_cam_pos;
            uniform mat4 u_inv_view_proj;
            uniform mat4 u_view;
            uniform vec2 u_viewport_size;
            uniform sampler2D u_depth_tex;
            uniform int u_has_depth;
            uniform sampler2D u_shadow_map_0;
            uniform sampler2D u_shadow_map_1;
            uniform sampler2D u_shadow_map_2;
uniform sampler2D u_shadow_map_3;
            uniform mat4 u_light_space_matrices[CASCADE_COUNT];
            uniform float u_cascade_splits[CASCADE_COUNT];
            uniform float u_shadow_bias;
            uniform int u_cascade_count;
            float hash(vec2 p)
            {
                return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
            }
            float hash(vec3 p)
            {
                p = fract(p * 0.3183099 + 0.1);
                p *= 17.0;
                return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
            }
            float noise2d(vec2 p)
            {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));
                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }
            float noise3d(vec3 p)
            {
                vec3 i = floor(p);
                vec3 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float n000 = hash(i + vec3(0.0, 0.0, 0.0));
                float n100 = hash(i + vec3(1.0, 0.0, 0.0));
                float n010 = hash(i + vec3(0.0, 1.0, 0.0));
                float n110 = hash(i + vec3(1.0, 1.0, 0.0));
                float n001 = hash(i + vec3(0.0, 0.0, 1.0));
                float n101 = hash(i + vec3(1.0, 0.0, 1.0));
                float n011 = hash(i + vec3(0.0, 1.0, 1.0));
                float n111 = hash(i + vec3(1.0, 1.0, 1.0));
                float x00 = mix(n000, n100, f.x);
                float x10 = mix(n010, n110, f.x);
                float x01 = mix(n001, n101, f.x);
                float x11 = mix(n011, n111, f.x);
                float y0 = mix(x00, x10, f.y);
                float y1 = mix(x01, x11, f.y);
                return mix(y0, y1, f.z);
            }
            float fbm3d(vec3 p, int octaves)
            {
                float v = 0.0;
                float amp = 0.5;
                float freq = 1.0;
                for (int i = 0; i < 8; i++)
                {
                    if (i >= octaves)
                    {
                        break;
                    }
                    v += amp * noise3d(p * freq);
                    freq *= 2.03;
                    amp *= 0.52;
                }
                return v;
            }
            float cloud_shape(vec3 p, float coverage, float detail_strength)
            {
                vec3 q = p + vec3(_Seed * 1.37, _Seed * 0.53, _Seed * 2.11);
                float broad = fbm3d(q * 0.38, 5);
                float base = fbm3d(q, 5);
                float detail = fbm3d(q * 3.4 + vec3(15.0, 3.0, 9.0), 4);
                float curl = fbm3d(vec3(q.x * 0.72 + q.z * 0.16, q.y * 0.7, q.z * 1.28 - q.x * 0.08), 4);
                float cells = noise2d(q.xz * 0.42 + vec2(_Seed * 0.21, _Seed * 0.13));
                float c = clamp(coverage, 0.0, 1.0);
                float threshold = mix(0.56, 0.20, c);
                float edge = mix(0.075, 0.055, c) + _Softness * mix(0.025, 0.060, c);
                float shape = broad * 0.48 + base * 0.45 + curl * 0.20 + (detail - 0.5) * detail_strength * 0.54;
                shape += (cells - 0.5) * 0.18;
                shape = clamp((shape - 0.5) * _Contrast + 0.5, 0.0, 1.0);
                float cloud = smoothstep(threshold, threshold + edge, shape);
                float cluster = smoothstep(mix(0.66, 0.38, c), mix(0.86, 0.68, c), broad * 0.72 + base * 0.20 + cells * 0.22);
                float broken = smoothstep(mix(0.38, 0.30, c), mix(0.78, 0.66, c), base * 0.62 + curl * 0.28 + cells * 0.18);
                float erosion = 1.0 - smoothstep(0.58, 0.94, detail + curl * 0.18);
                float puffs = smoothstep(0.34, 0.78, broad + cells * 0.18);
                cloud = clamp(cloud * mix(cluster * broken, 1.0, smoothstep(0.34, 0.76, c)), 0.0, 1.0);
                cloud = clamp(cloud * mix(0.66, 1.20, erosion) * mix(0.82, 1.18, puffs), 0.0, 1.0);
                return pow(cloud, mix(1.35, 0.72, c));
            }
            vec3 reconstruct_world(float depth)
            {
                vec4 clip_pos = vec4(v_uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
                vec4 world_pos = u_inv_view_proj * clip_pos;
                return world_pos.xyz / world_pos.w;
            }
            float sample_shadow(sampler2D shadow_map, vec3 proj_coords)
            {
                float current_depth = proj_coords.z - u_shadow_bias;
                float result = 0.0;
                vec2 texel_size = 1.0 / vec2(textureSize(shadow_map, 0));
                float radius = 1.25;
                float weight_sum = 0.0;
                for (int x = -1; x <= 1; x++)
                {
                    for (int y = -1; y <= 1; y++)
                    {
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
                return 1.0 - result / weight_sum;
            }
            float compute_directional_shadow(vec3 world_pos)
            {
                if (u_cascade_count <= 0)
                {
                    return 1.0;
                }
                int cascade_idx = 0;
                vec4 view_pos = u_view * vec4(world_pos, 1.0);
                float frag_depth = abs(view_pos.z);
                for (int i = 0; i < CASCADE_COUNT - 1; i++)
                {
                    if (frag_depth > u_cascade_splits[i])
                    {
                        cascade_idx = i + 1;
                    }
                }
                vec4 light_space_pos = u_light_space_matrices[cascade_idx] * vec4(world_pos, 1.0);
                vec3 proj_coords = light_space_pos.xyz / light_space_pos.w;
                proj_coords = proj_coords * 0.5 + 0.5;
                if (proj_coords.x < 0.0 || proj_coords.x > 1.0 || proj_coords.y < 0.0 || proj_coords.y > 1.0 || proj_coords.z < 0.0 || proj_coords.z > 1.0)
                {
                    return 1.0;
                }
                if (cascade_idx == 0)
                {
                    return sample_shadow(u_shadow_map_0, proj_coords);
                }
                if (cascade_idx == 1)
                {
                    return sample_shadow(u_shadow_map_1, proj_coords);
                }
                if (cascade_idx == 2)
                {
                    return sample_shadow(u_shadow_map_2, proj_coords);
                }
                return sample_shadow(u_shadow_map_3, proj_coords);
            }
            vec3 cloud_color(float density, float phase, float shadow, float top_light)
            {
                vec3 tint = mix(_Tint, vec3(1.0), 0.72);
                vec3 shade_col = tint * mix(vec3(0.58, 0.66, 0.78), vec3(0.72, 0.80, 0.90), top_light);
                vec3 mid_col = tint * vec3(0.88, 0.93, 0.98);
                vec3 lit_col = mix(tint * vec3(1.04, 1.05, 1.04), _SunColor * (0.88 + _SunIntensity * 0.16), clamp(phase, 0.0, 1.0));
                float core = smoothstep(0.18, 0.72, density);
                vec3 col = mix(shade_col, mid_col, core);
                col = mix(col, lit_col, clamp(phase * 0.76 + shadow * 0.24, 0.0, 1.0));
                col *= mix(0.68, 1.0, shadow);
                return col;
            }
            void main()
            {
                float depth = u_has_depth == 1 ? texture(u_depth_tex, v_uv).r : 1.0;
                bool has_surface = depth < 0.9999;
                vec3 surface_pos = reconstruct_world(depth);
                vec3 far_pos = reconstruct_world(1.0);
                vec3 ray_dir = normalize(far_pos - u_cam_pos);
                vec3 sun_dir = normalize(_SunDirection);
                float max_t = has_surface ? length(surface_pos - u_cam_pos) : 1400.0;
                max_t = clamp(max_t, 0.0, 1400.0);
                float layer_min = _Height;
                float layer_max = _Height + max(_Thickness, 0.05);
                float layer_mid = mix(layer_min, layer_max, 0.55);
                float high_layer = smoothstep(8.0, 32.0, layer_mid - u_cam_pos.y);
                float t_enter = 0.0;
                float t_exit = max_t;
                bool hit_layer = true;
                if (abs(ray_dir.y) > 0.0005)
                {
                    float a = (layer_min - u_cam_pos.y) / ray_dir.y;
                    float b = (layer_max - u_cam_pos.y) / ray_dir.y;
                    t_enter = max(min(a, b), 0.0);
                    t_exit = min(max(a, b), max_t);
                    hit_layer = t_exit > t_enter;
                }
                else
                {
                    hit_layer = u_cam_pos.y >= layer_min && u_cam_pos.y <= layer_max;
                }
                int steps = clamp(_Steps, 4, 32);
                float transmittance = 1.0;
                vec3 volume_premul = vec3(0.0);
                float horizon = pow(clamp(1.0 - abs(ray_dir.y), 0.0, 1.0), 2.2);
                float forward = pow(max(dot(ray_dir, sun_dir), 0.0), 5.0);
                float sun_height = clamp(sun_dir.y * 0.5 + 0.5, 0.0, 1.0);
                float wind_angle = radians(_WindDirection);
                vec2 wind_dir = vec2(cos(wind_angle), sin(wind_angle));
                vec2 wind_side = vec2(-wind_dir.y, wind_dir.x);
                float wind_speed = max(_Speed, 0.0);
                vec2 wind = wind_dir * u_time * wind_speed * 18.0 + wind_side * sin(u_time * wind_speed * 0.7 + _Seed) * _Evolution * 9.0;
                float evolve = u_time * wind_speed * _Evolution;
                float coverage = clamp(_Coverage, 0.0, 1.0);
                float sky_alpha = 0.0;
                vec3 sky_col = vec3(0.0);
                float dome_alpha = 0.0;
                vec3 dome_col = vec3(0.0);
                if (high_layer > 0.01 && _Opacity > 0.001)
                {
                    float dome_fade = smoothstep(-0.04, 0.20, ray_dir.y);
                    float sky_t = abs(ray_dir.y) > 0.0005 ? (layer_mid - u_cam_pos.y) / ray_dir.y : -1.0;
                    float dome_t = mix(520.0, 1350.0, clamp(1.0 - ray_dir.y, 0.0, 1.0));
                    float use_plane = sky_t > 0.0 && sky_t < 1800.0 ? smoothstep(0.08, 0.24, abs(ray_dir.y)) : 0.0;
                    float sample_t = mix(dome_t, sky_t, use_plane);
                    if (sample_t > 0.0)
                    {
                        vec3 p = u_cam_pos + ray_dir * sample_t;
                        vec3 sample_pos = vec3((p.x + wind.x) * _Scale, layer_mid * _Scale * 0.12 + evolve * 0.055, (p.z + wind.y) * _Scale);
                        float cloud = cloud_shape(sample_pos, coverage, _DetailStrength);
                        float distance_fade = mix(0.82, 1.0 - smoothstep(1250.0, 2100.0, sample_t), use_plane);
                        float angle_fade = max(dome_fade, smoothstep(0.012, 0.13, abs(ray_dir.y)) * use_plane);
                        float surface_fade = has_surface ? smoothstep(120.0, 900.0, max_t) : 1.0;
                        cloud *= distance_fade * angle_fade * surface_fade;
                        float sky_gain = mix(3.20, 4.70, smoothstep(0.22, 0.82, coverage));
                        sky_alpha = clamp(pow(cloud, 0.72) * _Density * _Opacity * sky_gain, 0.0, mix(0.78, 0.93, coverage));
                        float shadow = compute_directional_shadow(p - sun_dir * max(_Thickness, 1.0) * 1.15);
                        float phase = 0.48 + forward * _Scattering * 0.55 + sun_height * 0.24;
                        sky_col = cloud_color(cloud, phase, shadow, 0.86);
                        sky_col += _SunColor * forward * _Scattering * _SilverLining * 0.22;
                    }
                }
                if (high_layer > 0.01 && _Opacity > 0.001)
                {
                    float dome_y = max(ray_dir.y, 0.035);
                    float dome_t = clamp((layer_mid - u_cam_pos.y) / dome_y, 260.0, 2200.0);
                    vec3 p = u_cam_pos + ray_dir * dome_t;
                    p.y = layer_mid + ray_dir.y * _Thickness * 0.28;
                    vec3 dome_pos = vec3((p.x + wind.x) * _Scale, p.y * _Scale * 0.16 + evolve * 0.06, (p.z + wind.y) * _Scale);
                    float broad = cloud_shape(dome_pos, coverage, _DetailStrength);
                    float wisps = fbm3d(dome_pos * 5.2 + vec3(7.0, _Seed * 0.09, 13.0), 4);
                    float cloud = clamp(broad + (wisps - 0.5) * _DetailStrength * 0.24, 0.0, 1.0);
                    cloud = smoothstep(mix(0.36, 0.08, coverage), mix(0.86, 0.66, coverage), cloud);
                    float sky_mask = smoothstep(-0.02, 0.20, ray_dir.y);
                    float depth_mask = has_surface ? smoothstep(90.0, 420.0, max_t) : 1.0;
                    float distance_fade = 1.0 - smoothstep(1450.0, 2300.0, dome_t);
                    dome_alpha = clamp(pow(cloud, 0.76) * sky_mask * depth_mask * distance_fade * _Density * _Opacity * mix(2.20, 3.15, coverage), 0.0, mix(0.72, 0.90, coverage));
                    float dome_shadow = mix(0.64, 1.0, smoothstep(0.18, 0.82, cloud));
                    float dome_phase = clamp(sun_height * 0.42 + forward * 0.48 + cloud * 0.16, 0.0, 1.0);
                    dome_col = cloud_color(cloud, dome_phase, dome_shadow, 0.95);
                    dome_col += _SunColor * forward * _Scattering * _SilverLining * 0.18;
                }
                bool march_volume = hit_layer && max_t > 0.01 && _Opacity > 0.001 && (has_surface || high_layer < 0.45 || ray_dir.y > 0.025);
                if (march_volume)
                {
                    float segment = max(t_exit - t_enter, 0.01);
                    float segment_limit = mix(340.0, 920.0, smoothstep(0.04, 0.30, abs(ray_dir.y)));
                    segment = min(segment, segment_limit);
                    float step_len = segment / float(steps);
                    float jitter = noise2d(v_uv * u_viewport_size * 0.12 + u_time * 0.03);
                    for (int i = 0; i < 32; i++)
                    {
                        if (i >= steps)
                        {
                            break;
                        }
                        float t = t_enter + (float(i) + jitter) * step_len;
                        if (t > max_t)
                        {
                            break;
                        }
                        vec3 p = u_cam_pos + ray_dir * t;
                        float h = clamp((p.y - layer_min) / max(layer_max - layer_min, 0.01), 0.0, 1.0);
                        float edge = max(_Softness, 0.06);
                        float height_mask = smoothstep(0.0, edge, h) * (1.0 - smoothstep(1.0 - edge, 1.0, h));
                        vec3 sample_pos = vec3((p.x + wind.x) * _Scale, p.y * _Scale * 0.45 + evolve * 0.08, (p.z + wind.y) * _Scale);
                        float density = cloud_shape(sample_pos, coverage, _DetailStrength);
                        density *= _Density * height_mask;
                        density *= exp(-max(p.y - layer_min, 0.0) * _HeightFalloff * 0.035);
                        float shadow_probe_t = max(_Thickness, 1.0) * 1.25;
                        float shadow = compute_directional_shadow(p - sun_dir * shadow_probe_t);
                        float phase = 0.48 + forward * _Scattering * 0.68 + sun_height * 0.20;
                        vec3 sample_col = cloud_color(density, phase, shadow, h);
                        sample_col += _SunColor * forward * _Scattering * _SilverLining * 0.22;
                        float sample_alpha = 1.0 - exp(-density * step_len * max(_Absorption, 0.001) * 0.038);
                        sample_alpha *= _Opacity;
                        volume_premul += transmittance * sample_alpha * sample_col;
                        transmittance *= 1.0 - sample_alpha;
                        if (transmittance < 0.02)
                        {
                            break;
                        }
                    }
                }
                float volume_alpha = 1.0 - transmittance;
                float fog_alpha = 0.0;
                if (_FogDensity > 0.001 && max_t > 0.01)
                {
                    float fog_density = _FogDensity * mix(1.0, 0.12, high_layer);
                    float fog_path = has_surface ? max_t : max_t * horizon * mix(0.45, 0.12, high_layer);
                    float surface_height = has_surface ? surface_pos.y : _Height;
                    float height_term = exp(-max(min(u_cam_pos.y, surface_height) - _Height, 0.0) * _HeightFalloff);
                    float distance_term = 1.0 - exp(-fog_path * fog_density * 0.010);
                    fog_alpha = distance_term * height_term;
                    fog_alpha += horizon * fog_density * mix(0.22, 0.04, high_layer);
                    fog_alpha = clamp(fog_alpha * _Opacity, 0.0, mix(0.78, 0.18, high_layer));
                }
                vec3 fog_col = mix(_Tint * vec3(0.78, 0.86, 1.0), _SunColor, clamp(forward * 0.32 + sun_height * 0.16, 0.0, 1.0));
                fog_col += _SunColor * forward * _Scattering * _SilverLining * 0.18;
                float inside_alpha = 0.0;
                vec3 inside_col = vec3(0.0);
                float inside_lower = smoothstep(layer_min - 10.0, layer_min + 8.0, u_cam_pos.y);
                float inside_upper = 1.0 - smoothstep(layer_max - 8.0, layer_max + 18.0, u_cam_pos.y);
                float inside_layer = inside_lower * inside_upper;
                if (inside_layer > 0.001 && _Opacity > 0.001)
                {
                    int inside_steps = clamp(_Steps / 2, 4, 12);
                    float inside_path = max(min(max_t, 220.0), 45.0);
                    float inside_step_len = inside_path / float(inside_steps);
                    float inside_jitter = noise2d(v_uv * u_viewport_size * 0.16 + u_time * 0.05 + _Seed);
                    float density_total = 0.0;
                    vec3 color_total = vec3(0.0);
                    for (int i = 0; i < 12; i++)
                    {
                        if (i >= inside_steps)
                        {
                            break;
                        }
                        float t = (float(i) + inside_jitter) * inside_step_len;
                        vec3 p = u_cam_pos + ray_dir * t;
                        float h = (p.y - layer_min) / max(layer_max - layer_min, 0.01);
                        float height_mask = smoothstep(-0.24, 0.16, h) * (1.0 - smoothstep(0.84, 1.26, h));
                        vec3 sample_pos = vec3((p.x + wind.x) * _Scale, p.y * _Scale * 0.45 + evolve * 0.08, (p.z + wind.y) * _Scale);
                        float shape = cloud_shape(sample_pos, mix(coverage, clamp(coverage + 0.12, 0.0, 1.0), 0.45), _DetailStrength);
                        float density = mix(0.22, 1.0, shape) * height_mask;
                        float local_forward = pow(max(dot(ray_dir, sun_dir), 0.0), 4.0);
                        float phase = clamp(0.34 + local_forward * _Scattering * 0.46 + sun_height * 0.24 + shape * 0.20, 0.0, 1.0);
                        color_total += cloud_color(shape, phase, 1.0, clamp(h, 0.0, 1.0)) * density;
                        density_total += density;
                    }
                    float average_density = density_total / float(inside_steps);
                    inside_alpha = 1.0 - exp(-average_density * inside_path * _Density * _Opacity * max(_Absorption, 0.001) * 0.018);
                    inside_alpha = clamp(inside_alpha * inside_layer, 0.0, 0.74);
                    inside_col = density_total > 0.001 ? color_total / density_total : _Tint * vec3(0.82, 0.90, 1.0);
                    inside_col += _SunColor * forward * _Scattering * _SilverLining * 0.14;
                }
                float sky_total = clamp(sky_alpha + dome_alpha * (1.0 - sky_alpha), 0.0, 0.88);
                vec3 sky_premul = sky_col * sky_alpha + dome_col * dome_alpha * (1.0 - sky_alpha);
                float base_alpha = clamp(fog_alpha + volume_alpha * (1.0 - fog_alpha) + sky_total * (1.0 - fog_alpha) * (1.0 - volume_alpha), 0.0, 0.9);
                vec3 base_premul = fog_col * fog_alpha * (1.0 - volume_alpha) * (1.0 - sky_total) + volume_premul + sky_premul * (1.0 - fog_alpha) * (1.0 - volume_alpha);
                float alpha = clamp(base_alpha + inside_alpha * (1.0 - base_alpha), 0.0, 0.94);
                vec3 premul = base_premul + inside_col * inside_alpha * (1.0 - base_alpha);
                if (alpha <= 0.001)
                {
                    discard;
                }
                frag_color = vec4(premul / max(alpha, 0.001), alpha);
            }
            ENDGLSL
        }
    }
}
