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
        _SunSize("Sun Size", Float) = 0.0008
        _SunConvergence("Sun Convergence", Range(0, 1)) = 0.5
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
            uniform float _SunSize;
            uniform float _SunConvergence;
            uniform sampler2D u_env_tex;
            uniform float u_use_env;
            void main() {
                vec3 dir = normalize(v_uv);
                vec3 sun_dir = normalize(_SunDirection);
                float cos_gamma = dot(dir, sun_dir);
                vec3 color;
                if (u_use_env > 0.5) {
                    vec2 uv = vec2(0.5 + atan(dir.z, dir.x) / 6.28318530718, acos(clamp(dir.y, -1.0, 1.0)) / 3.14159265359);
                    color = texture(u_env_tex, uv).rgb;
                } else {
                    float cos_theta = max(dir.y, 0.0);
                    float sun_height = sun_dir.y;
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
                    float sun_start = 1.0 - _SunSize * 3.0;
                    float sun_end = 1.0 - _SunSize * (1.0 - _SunConvergence * 0.8);
                    float sun_disk = smoothstep(sun_start, sun_end, cos_gamma);
                    color += _SunColor * _SunIntensity * sun_disk;
                    float night = smoothstep(0.05, -0.4, sun_height);
                    color *= (1.0 - night * 0.72);
                }
                frag_color = vec4(color, 1.0);
            }
            ENDGLSL
        }
    }
}
