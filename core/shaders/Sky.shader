// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

Shader "Zarin/Sky"
{
    Properties
    {
        _SunDirection("Sun Direction", Vector) = (0, -0.3, -1, 0)
        _SunColor("Sun Color", Color) = (1, 0.95, 0.85, 1)
        _SunIntensity("Sun Intensity", Float) = 1
        _SunAngularRadius("Sun Angular Radius (deg)", Float) = 0.27
        _SunLimbDarkening("Sun Limb Darkening", Range(0, 1)) = 0.7
        _SunConvergence("Sun Edge Softness", Range(0, 1)) = 0.5
        _NightSkyEnabled("Night Sky", Float) = 1
        _StarEnabled("Stars", Float) = 1
        _StarDensity("Star Density", Range(0, 1)) = 0.45
        _StarIntensity("Star Intensity", Float) = 1
        _StarScale("Star Density Scale", Float) = 80
        _StarTwinkle("Star Twinkle", Range(0, 1)) = 0.5
        _StarSeed("Star Seed", Float) = 1
        _StarColor("Star Tint", Color) = (0.9, 0.93, 1, 1)
        _MilkyWayEnabled("Milky Way", Float) = 1
        _MilkyWayIntensity("Milky Way Intensity", Float) = 0.6
        _MilkyWayPole("Milky Way Pole", Vector) = (0.4, 0.3, 0.85, 0)
        _MoonEnabled("Moon", Float) = 1
        _MoonDirection("Moon Direction", Vector) = (0.25, 0.6, 0.75, 0)
        _MoonSize("Moon Angular Radius (deg)", Float) = 0.27
        _MoonIntensity("Moon Intensity", Float) = 1
        _MoonPhase("Moon Phase", Range(0, 1)) = 1
        _MoonOrbitSpeed("Moon Orbit Speed (deg/s)", Float) = 2
        _NightExposure("Night Exposure", Float) = 1
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "Queue" = "Background" }

        Pass
        {
            GLSLPROGRAM
            #version 460 core
            layout(location = 0) in vec3 in_position;
            uniform mat4 u_mvp;
            out vec3 v_uv;
            void main() {
                vec4 pos = u_mvp * vec4(in_position, 1.0);
                gl_Position = pos.xyww;
                v_uv = in_position;
            }

            // @FRAGMENT

            #version 460 core
            in vec3 v_uv;
            out vec4 frag_color;
            uniform vec3 _SunDirection;
            uniform vec3 _SunColor;
            uniform float _SunIntensity;
            uniform float _SunAngularRadius;
            uniform float _SunLimbDarkening;
            uniform float _SunConvergence;
            uniform sampler2D u_env_tex;
            uniform float u_use_env;
            uniform sampler2D u_transmittance_lut;
            uniform sampler2D u_sky_lut;
            uniform float u_use_atmosphere;
            uniform float u_atmosphere_intensity;

            uniform float _NightSkyEnabled;
            uniform float _StarEnabled;
            uniform float _StarDensity;
            uniform float _StarIntensity;
            uniform float _StarScale;
            uniform float _StarTwinkle;
            uniform float _StarSeed;
            uniform vec3 _StarColor;
            uniform float _MilkyWayEnabled;
            uniform float _MilkyWayIntensity;
            uniform vec3 _MilkyWayPole;
            uniform float _MoonEnabled;
            uniform vec3 _MoonDirection;
            uniform float _MoonSize;
            uniform float _MoonIntensity;
            uniform float _MoonPhase;
            uniform float _MoonOrbitSpeed;
            uniform sampler2D u_moon_tex;
            uniform float u_use_moon_tex;
            uniform float _NightExposure;
            uniform float u_time;

            const float PI = 3.14159265359;
            const float Rg = 6360.0;
            const float Rt = 6420.0;
            const float CAM_HEIGHT_KM = 0.02;

            float hash12(vec2 p) {
                vec3 p3 = fract(vec3(p.xyx) * 0.1031);
                p3 += dot(p3, p3.yzx + 33.33);
                return fract((p3.x + p3.y) * p3.z);
            }

            float hash13(vec3 p3) {
                p3 = fract(p3 * 0.1031);
                p3 += dot(p3, p3.zyx + 31.32);
                return fract((p3.x + p3.y) * p3.z);
            }

            vec3 hash33(vec3 p3) {
                p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
                p3 += dot(p3, p3.yxz + 33.33);
                return fract((p3.xxy + p3.yxx) * p3.zyx);
            }

            float vnoise3(vec3 p) {
                vec3 i = floor(p);
                vec3 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float n000 = hash13(i);
                float n100 = hash13(i + vec3(1.0, 0.0, 0.0));
                float n010 = hash13(i + vec3(0.0, 1.0, 0.0));
                float n110 = hash13(i + vec3(1.0, 1.0, 0.0));
                float n001 = hash13(i + vec3(0.0, 0.0, 1.0));
                float n101 = hash13(i + vec3(1.0, 0.0, 1.0));
                float n011 = hash13(i + vec3(0.0, 1.0, 1.0));
                float n111 = hash13(i + vec3(1.0, 1.0, 1.0));
                return mix(
                    mix(mix(n000, n100, f.x), mix(n010, n110, f.x), f.y),
                    mix(mix(n001, n101, f.x), mix(n011, n111, f.x), f.y),
                    f.z);
            }

            float fbm3(vec3 p) {
                float v = 0.0;
                float a = 0.5;
                for (int i = 0; i < 4; i++) {
                    v += a * vnoise3(p);
                    p = p * 2.03 + vec3(17.3, 9.1, 4.7);
                    a *= 0.5;
                }
                return v;
            }

            float vnoise2(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                f = f * f * (3.0 - 2.0 * f);
                float a = hash12(i);
                float b = hash12(i + vec2(1.0, 0.0));
                float c = hash12(i + vec2(0.0, 1.0));
                float d = hash12(i + vec2(1.0, 1.0));
                return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
            }

            float fbm2(vec2 p) {
                float v = 0.0;
                float a = 0.5;
                for (int i = 0; i < 5; i++) {
                    v += a * vnoise2(p);
                    p = p * 2.03 + vec2(9.2, 3.7);
                    a *= 0.5;
                }
                return v;
            }

            vec3 sample_transmittance_dir(vec3 dir) {
                float mu = clamp(dot(normalize(dir), vec3(0.0, 1.0, 0.0)), -1.0, 1.0);
                float u = (Rg + CAM_HEIGHT_KM - Rg) / (Rt - Rg);
                float v = mu * 0.5 + 0.5;
                return texture(u_transmittance_lut, vec2(u, v)).rgb;
            }

            float sun_disk_factor(vec3 dir, vec3 sun_dir) {
                float radius = radians(max(_SunAngularRadius, 0.01));
                float ry = radius * mix(1.0, 0.35, smoothstep(0.0, -0.08, sun_dir.y));
                vec3 u = sun_dir;
                vec3 v = cross(vec3(0.0, 1.0, 0.0), u);
                float vl = length(v);
                v = (vl > 1e-5) ? (v / vl) : vec3(1.0, 0.0, 0.0);
                vec3 w = cross(v, u);
                float elev = asin(clamp(dot(dir, w), -1.0, 1.0));
                float az = asin(clamp(dot(dir, v), -1.0, 1.0));
                float dist = length(vec2(elev / ry, az / radius));
                float soft = _SunConvergence * 0.35;
                float disk = 1.0 - smoothstep(1.0 - soft, 1.0 + soft * 1.5, dist);
                float rn = clamp(dist, 0.0, 1.0);
                float limb = sqrt(max(1.0 - rn * rn, 0.0));
                disk = disk * mix(1.0 - _SunLimbDarkening, 1.0, limb);
                return disk;
            }

            vec3 procedural_stars(vec3 dir) {
                vec3 g = dir * _StarScale + vec3(_StarSeed * 137.31);
                vec3 id = floor(g);
                vec3 f = fract(g);
                vec3 acc = vec3(0.0);
                for (int x = -1; x <= 1; x++)
                for (int y = -1; y <= 1; y++)
                for (int z = -1; z <= 1; z++) {
                    vec3 o = vec3(float(x), float(y), float(z));
                    vec3 cell = id + o;
                    float h = hash13(cell);
                    if (h >= _StarDensity) continue;
                    vec3 star_pos = o + hash33(cell + vec3(7.31, 11.7, 13.13));
                    vec3 d = f - star_pos;
                    float r2 = dot(d, d);
                    float radius = 0.06 + 0.14 * hash13(cell + 1.71);
                    if (r2 > radius * radius) continue;
                    float brightness = hash13(cell + 2.23);
                    float glow = exp(-r2 / (0.02 + 0.20 * brightness));
                    float twinkle = 1.0;
                    if (_StarTwinkle > 0.01) {
                        twinkle = 0.7 + 0.3 * sin(u_time * (0.6 + brightness * 3.0) + hash13(cell + 3.37) * 6.28318);
                    }
                    float tint_mix = hash13(cell + 4.17);
                    vec3 tint = mix(vec3(1.0, 0.96, 0.88), vec3(0.58, 0.70, 1.0), smoothstep(0.35, 0.85, tint_mix));
                    acc += _StarColor * tint * glow * (0.25 + 0.75 * brightness) * (1.0 + twinkle * _StarTwinkle);
                }
                return acc * _StarIntensity;
            }

            vec3 milky_way(vec3 dir) {
                vec3 pole = normalize(_MilkyWayPole);
                float band = pow(1.0 - abs(dot(dir, pole)), 3.0);
                vec3 p = dir * 6.0 + vec3(_StarSeed * 13.7);
                float n = pow(fbm3(p), 2.5);
                float streak = smoothstep(0.25, 0.7, fbm3(p * vec3(3.0, 9.0, 3.0)));
                vec3 tint = mix(vec3(0.75, 0.82, 0.95), vec3(0.9, 0.72, 0.88), 0.35);
                return tint * band * n * (0.4 + 0.6 * streak) * _MilkyWayIntensity;
            }

    vec3 procedural_moon(vec3 dir, vec3 sun_dir) {
        vec3 u0 = normalize(_MoonDirection);
        float orbit = radians(_MoonOrbitSpeed) * u_time;
        float ca = cos(orbit);
        float sa = sin(orbit);
        vec3 u = normalize(vec3(u0.x * ca + u0.z * sa, u0.y, -u0.x * sa + u0.z * ca));
        vec3 v = cross(vec3(0.0, 1.0, 0.0), u);
        float vl = length(v);
        v = (vl > 1e-5) ? (v / vl) : vec3(1.0, 0.0, 0.0);
        vec3 w = cross(v, u);
        vec3 lp = vec3(dot(dir, v), dot(dir, w), dot(dir, u));
        float radius = radians(max(_MoonSize, 0.05));
        float dist = length(lp.xy);
        float disk = 1.0 - smoothstep(radius * 0.92, radius * 1.06, dist);
        if (disk <= 0.0) return vec3(0.0);
        float az = asin(clamp(lp.x, -1.0, 1.0));
        float elev = asin(clamp(lp.y, -1.0, 1.0));
        float maria = pow(fbm2(vec2(az, elev) * 8.0 + _StarSeed * 13.0), 1.5);
        float craters = fbm2(vec2(az, elev) * 44.0 + _StarSeed * 7.0);
        vec3 albedo = mix(vec3(0.33, 0.33, 0.35), vec3(0.55, 0.55, 0.58), maria);
        float shade = 0.42 + 0.58 * pow(craters, 1.5);
        float limb = sqrt(max(1.0 - clamp(dist / radius, 0.0, 1.0) * clamp(dist / radius, 0.0, 1.0), 0.0));
        vec3 surface = albedo * shade * (0.5 + 0.5 * limb);
        vec2 tex_uv = vec2(0.5 + 0.5 * az / radius, 0.5 - 0.5 * elev / radius);
        surface = mix(surface, texture(u_moon_tex, tex_uv).rgb * (0.5 + 0.5 * limb), u_use_moon_tex);
        vec3 sun_local = normalize(vec3(dot(sun_dir, v), dot(sun_dir, w), dot(sun_dir, u)));
        vec3 phase_axis = vec3(sun_local.x, sun_local.y, 0.0);
        float pa_len = length(phase_axis);
        phase_axis = (pa_len > 1e-4) ? (phase_axis / pa_len) : vec3(1.0, 0.0, 0.0);
        float pix_z = sqrt(max(1.0 - dot(lp.xy, lp.xy), 0.0));
        vec3 pixel_local = normalize(vec3(lp.xy, pix_z));
        float term = mix(1.5, -1.5, clamp(_MoonPhase, 0.0, 1.0));
        float lit = smoothstep(term - 0.25, term + 0.25, dot(pixel_local, phase_axis));
        return surface * disk * lit * _MoonIntensity;
    }

            vec3 night_sky(vec3 dir, vec3 sun_dir, float night) {
                if (_NightSkyEnabled < 0.5) return vec3(0.0);
                vec3 c = vec3(0.0);
                if (_StarEnabled > 0.5 && night > 0.001) {
                    c += procedural_stars(dir) * night;
                }
                if (_MilkyWayEnabled > 0.5 && night > 0.001) {
                    c += milky_way(dir) * night;
                }
                if (_MoonEnabled > 0.5 && night > 0.001) {
                    c += procedural_moon(dir, sun_dir) * night;
                }
                return c * _NightExposure;
            }

            void main() {
                vec3 dir = normalize(v_uv);
                vec3 sun_dir = normalize(_SunDirection);
                float cos_gamma = dot(dir, sun_dir);
                float sun_height = sun_dir.y;
                float night = smoothstep(0.05, -0.4, sun_height);
                vec3 color;
                if (u_use_atmosphere > 0.5) {
                    float theta = acos(clamp(dir.y, 0.0, 1.0));
                    float phi = atan(dir.z, dir.x);
                    vec2 sky_uv = vec2(phi / 6.28318530718 + 0.5, theta / 1.57079632679);
                    color = texture(u_sky_lut, sky_uv).rgb * u_atmosphere_intensity;
                    color = max(color, vec3(0.0));
                    vec3 sun_trans = sample_transmittance_dir(sun_dir);
                    color += _SunColor * _SunIntensity * sun_trans
                           * sun_disk_factor(dir, sun_dir)
                           * u_atmosphere_intensity * 2.0;
                    color *= (1.0 - night * 0.72);
                } else if (u_use_env > 0.5) {
                    vec2 uv = vec2(0.5 + atan(dir.z, dir.x) / 6.28318530718, acos(clamp(dir.y, -1.0, 1.0)) / 3.14159265359);
                    color = texture(u_env_tex, uv).rgb;
                } else {
                    float cos_theta = max(dir.y, 0.0);
                    float optical_depth = 1.0 / max(cos_theta, 0.005);
                    float rayleigh_phase = 0.75 * (1.0 + cos_gamma * cos_gamma);
                    vec3 rayleigh = vec3(0.55, 0.65, 0.90) * rayleigh_phase;
                    float g = 0.76;
                    float gg = g * g;
                    float mie_phase = (1.0 - gg) / max(pow(1.0 + gg - 2.0 * g * cos_gamma, 1.5), 0.001);
                    vec3 mie = vec3(1.0, 0.80, 0.50) * mie_phase;
                    color = (rayleigh * 0.10 + mie * 0.05) * (1.0 - exp(-optical_depth * 0.4));
                    vec3 sky_top = vec3(0.20, 0.42, 0.90);
                    vec3 sky_horizon = vec3(0.78, 0.88, 1.0);
                    vec3 sky_sunset = vec3(1.0, 0.58, 0.25);
                    vec3 night_top = vec3(0.012, 0.022, 0.06);
                    vec3 night_horizon = vec3(0.045, 0.055, 0.11);
                    vec3 horizon_color = mix(mix(sky_sunset, sky_horizon, smoothstep(0.0, 0.3, sun_height)), night_horizon, night);
                    float height_gradient = 1.0 - pow(1.0 - cos_theta, 4.0);
                    vec3 base_sky = mix(horizon_color, mix(sky_top, night_top, night), height_gradient);
                    color = max(color + base_sky * 0.85 + vec3(0.015, 0.025, 0.04), 0.0);
                    float below_horizon = smoothstep(0.0, -0.12, sun_height);
                    color += _SunColor * _SunIntensity
                           * sun_disk_factor(dir, sun_dir) * (1.0 - below_horizon);
                    color *= (1.0 - night * 0.72);
                }
                color += night_sky(dir, sun_dir, night);
                frag_color = vec4(color, 1.0);
            }
            ENDGLSL
        }
    }
}
