// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun
//
// Water shader (Unity 6-style realistic water surface).
//
// This is the DEFAULT water material and also serves as the reference for
// writing CUSTOM water shaders. Any .shader file assigned to the Water
// component's "Water Shader" field will be compiled and rendered the same
// way; just declare the uniforms you need (all are optional and guarded).
//
// Standard uniforms provided by the engine / Water component:
//   u_view, u_proj           mat4   camera view & projection
//   u_model                 mat4   water plane world transform
//   u_camera_pos            vec3   camera world position
//   _Time                   float  seconds since water creation
//   _SunDirection           vec3   normalized direction TOWARD the sun
//   _SunColor               vec3
//   _SunIntensity           float
//   _WaveCount              int    number of active Gerstner waves
//   _WaveDirection[MAX]     vec2   unit direction (x,z) of each wave
//   _WaveParams[MAX]        vec4   (amplitude, wavelength, speed, steepness)
//   _DeepColor              vec3
//   _ShallowColor           vec3
//   _FoamColor              vec3
//   _SSSColor               vec3    subsurface scattering tint
//   _HorizonColor           vec3    ocean distance fade color
//   _Smoothness             float  specular sharpness
//   _Distortion             float  refraction distortion amount
//   _NormalStrength         float  micro-detail normal strength
//   _WaveTiling             float  detail normal tiling frequency
//   _RefractStrength        float  how strongly deep color tints refraction
//   _FresnelPower           float  fresnel exponent
//   _FoamStrength           float  crest + shore foam amount
//   _Specular               float  sun specular intensity
//   _ShoreFade              float  shoreline foam falloff distance
//   _SceneColor             sampler2D  opaque scene color (for refraction/reflection)
//   _SceneDepth             sampler2D  opaque scene depth
//   _HasScene               int   1 if scene textures are available
//   _ViewportSize           vec2   render target size in pixels
//   _CamNear, _CamFar       float  camera clip planes (for shore depth)
//
// MAX_WAVES must stay in sync with the Water component (8).

Shader "Zarin/Water"
{
    Properties
    {
        _DeepColor("Deep Color", Color) = (0.02, 0.18, 0.28, 1)
        _ShallowColor("Shallow Color", Color) = (0.10, 0.42, 0.52, 1)
        _FoamColor("Foam Color", Color) = (0.92, 0.97, 1.0, 1)
        _SSSColor("SSS Color", Color) = (0.0, 0.55, 0.45, 1)
        _HorizonColor("Horizon Color", Color) = (0.62, 0.78, 0.86, 1)
        _Smoothness("Smoothness", Float) = 0.96
        _Distortion("Distortion", Float) = 0.04
        _NormalStrength("Normal Strength", Float) = 0.35
        _WaveTiling("Wave Tiling", Float) = 0.25
        _RefractStrength("Refraction Tint", Float) = 0.55
        _FresnelPower("Fresnel Power", Float) = 4.0
        _FoamStrength("Foam Strength", Float) = 1.0
        _Specular("Specular", Float) = 1.0
        _ShoreFade("Shore Fade", Float) = 3.0
    }

    SubShader
    {
        Tags { "RenderType" = "Transparent" "Queue" = "Transparent" }

        Pass
        {
            GLSLPROGRAM
            #version 460 core
            #define MAX_WAVES 8
            layout(location = 0) in vec3 in_position;
            layout(location = 2) in vec2 in_uv;
            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform vec3 u_camera_pos;
            uniform float _Time;
            uniform int _WaveCount;
            uniform vec2 _WaveDirection[MAX_WAVES];
            uniform vec4 _WaveParams[MAX_WAVES];

            out vec3 v_world_pos;
            out vec3 v_normal;
            out vec2 v_screen_uv;
            out float v_crest;

            const float G = 9.81;

            void main() {
                vec3 world = (u_model * vec4(in_position, 1.0)).xyz;
                vec2 p = world.xz;

                vec3 disp = vec3(0.0);
                vec3 tangent = vec3(1.0, 0.0, 0.0);
                vec3 binormal = vec3(0.0, 0.0, 1.0);

                for (int i = 0; i < MAX_WAVES; i++) {
                    if (i >= _WaveCount) break;
                    vec2 d = _WaveDirection[i];
                    float amp = _WaveParams[i].x;
                    float wlen = max(_WaveParams[i].y, 0.0001);
                    float speed = _WaveParams[i].z;
                    float steep = _WaveParams[i].w;
                    float k = 6.2831853 / wlen;
                    float c = sqrt(G / k);
                    float f = k * (dot(d, p) - c * speed * _Time);
                    float a = amp;
                    float q = steep / (k * a * float(_WaveCount) + 1e-4);
                    float cosf = cos(f);
                    float sinf = sin(f);
                    disp.x += q * a * d.x * cosf;
                    disp.z += q * a * d.y * cosf;
                    disp.y += a * sinf;
                    float wa = k * a;
                    tangent += vec3(-q * d.x * d.x * wa * sinf, d.x * wa * cosf, -q * d.x * d.y * wa * sinf);
                    binormal += vec3(-q * d.x * d.y * wa * sinf, d.y * wa * cosf, -q * d.y * d.y * wa * sinf);
                }

                world += disp;
                v_world_pos = world;
                v_crest = disp.y;
                vec3 n = normalize(cross(binormal, tangent));
                v_normal = n;

                vec4 clip = u_proj * u_view * vec4(world, 1.0);
                v_screen_uv = clip.xy / clip.w * 0.5 + 0.5;
                gl_Position = clip;
            }

            // @FRAGMENT

            #version 460 core
            #define MAX_WAVES 8
            in vec3 v_world_pos;
            in vec3 v_normal;
            in vec2 v_screen_uv;
            in float v_crest;
            out vec4 frag_color;

            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform vec3 u_camera_pos;
            uniform float _Time;
            uniform vec3 _SunDirection;
            uniform vec3 _SunColor;
            uniform float _SunIntensity;
            uniform vec3 _DeepColor;
            uniform vec3 _ShallowColor;
            uniform vec3 _FoamColor;
            uniform vec3 _SSSColor;
            uniform vec3 _HorizonColor;
            uniform float _Smoothness;
            uniform float _Distortion;
            uniform float _NormalStrength;
            uniform float _WaveTiling;
            uniform float _RefractStrength;
            uniform float _FresnelPower;
            uniform float _FoamStrength;
            uniform float _Specular;
            uniform float _ShoreFade;
            uniform float _CamNear;
            uniform float _CamFar;
            uniform vec2 _ViewportSize;
            uniform int _HasScene;
            uniform sampler2D _SceneColor;
            uniform sampler2D _SceneDepth;

            float hash(vec2 p) {
                p = fract(p * vec2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return fract(p.x * p.y);
            }
            float vnoise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                vec2 u = f * f * (3.0 - 2.0 * f);
                float a = hash(i);
                float b = hash(i + vec2(1.0, 0.0));
                float c = hash(i + vec2(0.0, 1.0));
                float d = hash(i + vec2(1.0, 1.0));
                return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
            }
            vec3 detail_normal(vec2 uv, float t) {
                float e = 0.05;
                float n1 = vnoise(uv * 6.0 + vec2(t * 0.6, 0.0));
                float n2 = vnoise(uv * 13.0 - vec2(0.0, t * 0.5));
                float nx = vnoise(uv * 8.0 + vec2(t * 0.3, 0.0) + vec2(e, 0.0)) - vnoise(uv * 8.0 + vec2(t * 0.3, 0.0) - vec2(e, 0.0));
                float nz = vnoise(uv * 8.0 + vec2(0.0, t * 0.3) + vec2(0.0, e)) - vnoise(uv * 8.0 + vec2(0.0, t * 0.3) - vec2(0.0, e));
                vec3 nrm = normalize(vec3(-nx, 1.0, -nz));
                nrm.xy *= _NormalStrength;
                return nrm;
            }
            float linearize_depth(float d) {
                float z_n = 2.0 * d - 1.0;
                return 2.0 * _CamNear * _CamFar / (_CamFar + _CamNear - z_n * (_CamFar - _CamNear));
            }
            vec3 sky_tint(vec3 dir) {
                float h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
                vec3 horizon = _HorizonColor;
                vec3 zenith = vec3(0.18, 0.40, 0.85);
                return mix(horizon, zenith, pow(h, 0.6));
            }

            void main() {
                vec3 N = normalize(v_normal);
                vec3 dn = detail_normal(v_world_pos.xz * _WaveTiling, _Time);
                N = normalize(N + vec3(dn.x, 0.0, dn.z));

                vec3 V = normalize(u_camera_pos - v_world_pos);
                float fres = pow(clamp(1.0 - max(dot(N, V), 0.0), 0.0, 1.0), _FresnelPower);
                fres = mix(0.02, 1.0, fres);

                vec3 color;
                if (_HasScene == 1) {
                    vec2 refr_uv = clamp(v_screen_uv + N.xz * _Distortion, 0.001, 0.999);
                    vec3 refr = texture(_SceneColor, refr_uv).rgb;
                    refr = mix(refr, _DeepColor, _RefractStrength);

                    vec3 R = reflect(-V, N);
                    vec4 rclip = u_proj * u_view * vec4(v_world_pos + R * 60.0, 1.0);
                    vec2 ruv = rclip.xy / rclip.w * 0.5 + 0.5;
                    vec3 refl;
                    if (ruv.x > 0.0 && ruv.x < 1.0 && ruv.y > 0.0 && ruv.y < 1.0) {
                        refl = texture(_SceneColor, ruv).rgb;
                    } else {
                        refl = sky_tint(R);
                    }
                    color = mix(refr, refl, fres);
                } else {
                    vec3 R = reflect(-V, N);
                    color = mix(_DeepColor, sky_tint(R), fres);
                }

                // Subsurface scattering approximation
                float back = max(dot(V, -_SunDirection), 0.0);
                vec3 sss = _SSSColor * clamp(v_crest * 1.5 + 0.2, 0.0, 1.0) * back;
                color += sss * 0.6;

                // Sun specular highlight
                vec3 H = normalize(V + _SunDirection);
                float spec = pow(max(dot(N, H), 0.0), mix(8.0, 1024.0, _Smoothness));
                color += _SunColor * _SunIntensity * spec * _Specular;

                // Foam: crest + shoreline
                float crest_foam = smoothstep(0.35, 0.85, v_crest) * 0.6
                                + smoothstep(0.6, 1.0, abs(dn.x) + abs(dn.z)) * 0.4;
                float shore_foam = 0.0;
                if (_HasScene == 1) {
                    float scene_d = linearize_depth(texture(_SceneDepth, v_screen_uv).r);
                    float water_d = linearize_depth(gl_FragCoord.z);
                    float diff = scene_d - water_d;
                    shore_foam = (1.0 - clamp(diff / max(_ShoreFade, 0.001), 0.0, 1.0));
                }
                float foam = clamp((crest_foam + shore_foam) * _FoamStrength, 0.0, 1.0);
                color = mix(color, _FoamColor, foam);

                // Distance fade to horizon (for infinite ocean)
                float dist = length(u_camera_pos.xz - v_world_pos.xz);
                float fade = clamp(dist / 900.0, 0.0, 1.0);
                color = mix(color, _HorizonColor, fade * fade);
                float alpha = mix(0.92, 0.98, fres);
                alpha = mix(alpha, 1.0, foam);

                frag_color = vec4(color, alpha);
            }
            ENDGLSL
        }
    }
}
