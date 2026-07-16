// This Source Code Form is subject to the terms of Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun
//
// GPU height-field water simulation. Runs as a ping-pong fragment pass on an
// RG32F texture: R = water height (relative to rest), G = vertical velocity.
// Physically integrates a damped 2D wave equation and injects disturbances
// from every scene collider that dips into / moves through the surface, so
// the ripples are generated entirely on the GPU and are not authored by hand.
//
// MAX_INTERACTORS must match the Renderer's interaction source limit (64).

Shader "Zarin/WaterSim"
{
    SubShader
    {
        Pass
        {
            GLSLPROGRAM
            #version 460 core
            layout(location = 0) in vec2 in_position;
            out vec2 v_uv;
            void main() {
                v_uv = in_position * 0.5 + 0.5;
                gl_Position = vec4(in_position, 0.0, 1.0);
            }

            // @FRAGMENT

            #version 460 core
            #define MAX_INTERACTORS 64
            in vec2 v_uv;
            out vec4 frag_color;

            uniform sampler2D _PrevState;
            uniform vec2 _Texel;
            uniform float _Dt;
            uniform float _Damping;
            uniform float _Propagation;
            uniform float _Saturation;
            uniform float _GridSize;
            uniform vec2 _GridCenter;
            uniform float _InteractionStrength;
            uniform int _InteractorCount;
            uniform vec4 _Interactors[MAX_INTERACTORS];
            uniform vec4 _InteractorVel[MAX_INTERACTORS];

            float hash(vec2 p) {
                p = fract(p * vec2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return fract(p.x * p.y);
            }

            void main() {
                vec2 uv = v_uv;
                vec2 world = _GridCenter + (uv - 0.5) * _GridSize;

                vec4 c = texture(_PrevState, uv);
                vec4 x1 = texture(_PrevState, uv + vec2(_Texel.x, 0.0));
                vec4 x2 = texture(_PrevState, uv - vec2(_Texel.x, 0.0));
                vec4 y1 = texture(_PrevState, uv + vec2(0.0, _Texel.y));
                vec4 y2 = texture(_PrevState, uv - vec2(0.0, _Texel.y));

                float h = c.r;
                float v = c.g;

                float lap = (x1.r + x2.r + y1.r + y2.r) - 4.0 * h;
                float k = _Propagation * _Dt * _Dt;
                k = min(k, 0.20);
                float accel = k * lap - _Damping * v;
                v += accel * _Dt;
                h += v * _Dt;

                float disturb = 0.0;
                float cell = _GridSize / 512.0;
                for (int i = 0; i < MAX_INTERACTORS; i++) {
                    if (i >= _InteractorCount) break;
                    vec4 it = _Interactors[i];
                    vec4 iv = _InteractorVel[i];
                    float reach = max(it.z, cell * 1.5);
                    vec2 d = world - it.xy;
                    float dd = length(d);
                    float fall = exp(-dd * dd / (reach * reach));
                    float vd = it.w;
                    float vrad = max(iv.w, 0.05);
                    float slab = max(cell * 1.5, 0.5) + vrad;
                    float surface_dist = abs(vd) - vrad;
                    float vgate = 1.0 - smoothstep(0.0, slab, surface_dist);
                    float vvel = iv.y;
                    float hvel = length(iv.xz);
                    float speed = length(iv.xyz);
                    float norm = speed / (speed + 3.0);
                    float push = _InteractionStrength * norm * fall * vgate;
                    disturb += push;
                }

                h += disturb * _Dt;
                if (_InteractorCount <= 0) {
                    h *= 0.6;
                    v *= 0.6;
                } else {
                    h *= 0.992;
                    v *= 0.992;
                }
                h = clamp(h, -_Saturation, _Saturation);
                v = clamp(v, -_Saturation, _Saturation);

                frag_color = vec4(h, v, 0.0, 1.0);
            }
            ENDGLSL
        }
    }
}
