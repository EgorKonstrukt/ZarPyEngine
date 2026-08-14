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

            const float PI = 3.14159265359;
            const float Rg = 6360.0;
            const float Rt = 6420.0;
            const float CAM_HEIGHT_KM = 0.02;

            // Atmospheric transmittance from the camera toward a world
            // direction, sampled from the transmittance LUT (see
            // Atmosphere.compute for the texel layout).
            vec3 sample_transmittance_dir(vec3 dir) {
                float mu = clamp(dot(normalize(dir), vec3(0.0, 1.0, 0.0)), -1.0, 1.0);
                float u = (Rg + CAM_HEIGHT_KM - Rg) / (Rt - Rg);
                float v = mu * 0.5 + 0.5;
                return texture(u_transmittance_lut, vec2(u, v)).rgb;
            }

            // Visible solar disc: physical angular radius, limb darkening and
            // vertical flattening near the horizon (atmospheric refraction).
            // Angles are measured in the sun's own frame (vertical plane axis w
            // and perpendicular axis v) so near-zenith suns stay well defined.
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

            void main() {
                vec3 dir = normalize(v_uv);
                vec3 sun_dir = normalize(_SunDirection);
                float cos_gamma = dot(dir, sun_dir);
                float sun_height = sun_dir.y;
                float night = smoothstep(0.05, -0.4, sun_height);
                vec3 color;
                if (u_use_atmosphere > 0.5) {
                    // Sample the precomputed sky-view radiance LUT. The LUT was
                    // generated for the full sky dome (zenith -> horizon), so
                    // directions below the horizon clamp to the horizon colour.
                    float theta = acos(clamp(dir.y, 0.0, 1.0));
                    float phi = atan(dir.z, dir.x);
                    vec2 sky_uv = vec2(phi / 6.28318530718 + 0.5, theta / 1.57079632679);
                    color = texture(u_sky_lut, sky_uv).rgb * u_atmosphere_intensity;
                    color = max(color, vec3(0.0));
                    // Direct solar disc, attenuated by the atmosphere along the
                    // view ray so it reddens and dims as it nears the horizon.
                    // The direct sun is far brighter than the scattered halo,
                    // hence the extra boost over the sky LUT scale.
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
                    vec3 horizon_color = mix(sky_sunset, sky_horizon, smoothstep(0.0, 0.3, sun_height));
                    float height_gradient = 1.0 - pow(1.0 - cos_theta, 4.0);
                    vec3 base_sky = mix(horizon_color, sky_top, height_gradient);
                    color = max(color + base_sky * 0.85 + vec3(0.015, 0.025, 0.04), 0.0);
                    float below_horizon = smoothstep(0.0, -0.12, sun_height);
                    color += _SunColor * _SunIntensity
                           * sun_disk_factor(dir, sun_dir) * (1.0 - below_horizon);
                    color *= (1.0 - night * 0.72);
                }
                frag_color = vec4(color, 1.0);
            }
            ENDGLSL
        }
    }
}
