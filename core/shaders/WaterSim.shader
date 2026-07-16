// This Source Code Form is subject to the terms of Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun
//
// GPU height-field water simulation. Runs as a ping-pong fragment pass on an
// RG16F texture: R = water height (relative to rest), G = vertical velocity.
// Physically integrates a damped 2D wave equation and injects a signed
// velocity impulse from every scene collider that crosses the surface, so
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

                // lap is an *unnormalized* discrete Laplacian (no /dx^2), so _Propagation
                // already plays the role of c^2/dx^2 for this grid. It must NOT be
                // multiplied by dt^2 again -- the Verlet-style integration below
                // (v += accel*dt, h += v*dt) already contributes the dt^2 factor to h.
                // We only clamp k when a frame hitch makes the explicit integrator
                // unstable (CFL condition for the 5-point stencil), so ripple speed on
                // a normal frame is the real configured speed, not a fraction of it.
                float lap = (x1.r + x2.r + y1.r + y2.r) - 4.0 * h;
                float dt2 = max(_Dt * _Dt, 1e-8);
                float maxK = 0.45 / dt2;
                float k = min(_Propagation, maxK);
                float accel = k * lap - _Damping * v;
                v += accel * _Dt;

                // Interaction forcing
                // Injected into velocity (not height) so the wave equation itself
                // carries it outward as an expanding ring, and signed by what the
                // collider is actually doing at the surface instead of always
                // lifting the water on contact.
                float cell = _GridSize * _Texel.x;
                float impulse = 0.0;
                for (int i = 0; i < MAX_INTERACTORS; i++) {
                    if (i >= _InteractorCount) break;
                    vec4 it = _Interactors[i];   // xy = world xz, z = radius, w = center.y - rest_y
                    vec4 iv = _InteractorVel[i]; // xyz = velocity, w = vertical extent

                    vec2 d = world - it.xy;
                    float dd2 = dot(d, d);
                    float reach = max(it.z, cell * 1.5);
                    float fall = exp(-dd2 / (reach * reach));
                    if (fall < 0.001) continue;

                    float vrad = max(iv.w, 0.05);
                    float surface_dist = abs(it.w) - vrad;
                    float slab = max(cell * 1.5, 0.5) + vrad;
                    float gate = 1.0 - smoothstep(0.0, slab, surface_dist);
                    if (gate <= 0.0) continue;

                    float vy = iv.y;
                    // Entering (vy<0) presses the surface into a trough; leaving
                    // (vy>0) drags it back up, weaker, since a free surface resists
                    // being pulled up far less cleanly than being pushed down.
                    float dirScale = vy < 0.0 ? 1.0 : 0.6;
                    impulse += vy * dirScale * fall * gate;

                    // Horizontal motion through the surface (wading, boat hulls)
                    // still displaces water, but only ever pushes down -- an
                    // upward "lift" from pure sideways motion isn't physical.
                    float hvel = length(iv.xz);
                    impulse -= 0.25 * hvel * fall * gate;
                }
                v += impulse * _InteractionStrength * _Dt;

                h += v * _Dt;

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
