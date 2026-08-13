// This Source Code Form is subject to the terms of Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun
//
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
//   _WindDir                vec2   wind heading unit vector (x,z)
//   _WindSpeed              float  wind speed (m/s, 0..60)
//   _WindGust               float  current gust factor (0..1)
//   _WindTurbulence         float  wind direction scatter (0..1)
//   _DeepColor              vec3
//   _ShallowColor           vec3
//   _FoamColor              vec3
//   _SSSColor               vec3    subsurface scattering tint
//   _HorizonColor           vec3    ocean distance fade color
//   _Smoothness             float  specular sharpness (0..1)
//   _Distortion             float  refraction distortion amount
//   _NormalStrength         float  micro-detail normal strength
//   _WaveTiling             float  detail normal tiling frequency
//   _WarpAmount             float  domain-warp strength (breaks repetition)
//   _DetailSpeed            float  detail ripple scroll speed
//   _Choppiness             float  overall wave steepness multiplier
//   _Caustics               float  caustic intensity on shallow water
//   _RefractStrength        float  how strongly deep/shallow color tints refraction
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
// Detail / resolution uniforms (added for max realism):
//   _MacroWave              float  large-scale swell amplitude (kills the
//                                  regular Gerstner grid at distance)
//   _Chaos                  float  randomises wave directions + adds micro
//                                  sparkle so the surface never repeats
//   _DetailScale            float  multiplier on the detail normal frequency
//   _DetailOctaves          float  number of detail noise octaves (1..12)
//   _DetailFade             float  distance at which fine detail fades out
//   _IsBox                  int   1 when rendering a Pond XYZ cube (aquarium);
//                                  disables scene refraction and draws the
//                                  side walls as a translucent volume
//
// MAX_WAVES must stay in sync with the Water component (8).

Shader "Zarin/Water"
{
    Properties
    {
        _DeepColor("Deep Color", Color) = (0.015, 0.13, 0.22, 1)
        _ShallowColor("Shallow Color", Color) = (0.13, 0.46, 0.52, 1)
        _FoamColor("Foam Color", Color) = (0.92, 0.97, 1.0, 1)
        _SSSColor("SSS Color", Color) = (0.0, 0.55, 0.45, 1)
        _HorizonColor("Horizon Color", Color) = (0.62, 0.78, 0.86, 1)
        _Smoothness("Smoothness", Float) = 0.92
        _Distortion("Distortion", Float) = 0.035
        _NormalStrength("Normal Strength", Float) = 0.55
        _WaveTiling("Wave Tiling", Float) = 0.6
        _WarpAmount("Domain Warp", Float) = 1.6
        _DetailSpeed("Detail Speed", Float) = 1.0
        _Choppiness("Choppiness", Float) = 1.0
        _Caustics("Caustics", Float) = 0.6
        _RefractStrength("Refraction Tint", Float) = 0.6
        _FresnelPower("Fresnel Power", Float) = 5.0
        _FoamStrength("Foam Strength", Float) = 0.9
        _Specular("Specular", Float) = 1.0
        _ShoreFade("Shore Fade", Float) = 3.0
        _MacroWave("Macro Waves", Float) = 0.5
        _Chaos("Chaos", Float) = 0.5
        _DetailScale("Detail Scale", Float) = 1.0
        _DetailOctaves("Detail Octaves", Float) = 6.0
        _DetailFade("Detail Fade Dist", Float) = 350.0
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
            layout(location = 1) in vec3 in_normal;
            layout(location = 2) in vec2 in_uv;
            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform vec3 u_camera_pos;
            uniform float _Time;
            uniform int _WaveCount;
            uniform vec2 _WaveDirection[MAX_WAVES];
            uniform vec4 _WaveParams[MAX_WAVES];
            uniform vec2 _WindDir;
            uniform float _WindSpeed;
            uniform float _WindGust;
            uniform float _WindTurbulence;
            uniform float _WindAlign;
            uniform float _WarpAmount;
            uniform float _Choppiness;
            uniform float _MacroWave;
            uniform float _Chaos;

            uniform sampler2D _SimTex;
            uniform float _HasSim;
            uniform vec2 _SimGridCenter;
            uniform float _SimGridSize;
            uniform float _SimDispScale;
            uniform float _SimNormalScale;

            out vec3 v_world_pos;
            out vec3 v_normal;
            out vec2 v_screen_uv;
            out vec2 v_detail_coord;
            out float v_crest;
            out float v_chop;
            out float v_foamMask;
            out float v_face;
            out float v_local_y;

            const float G = 9.81;

            float vhash(vec2 p) {
                p = fract(p * vec2(123.34, 456.21));
                p += dot(p, p + 45.32);
                return fract(p.x * p.y);
            }
            float vnoise(vec2 p) {
                vec2 i = floor(p);
                vec2 f = fract(p);
                vec2 u = f * f * (3.0 - 2.0 * f);
                float a = vhash(i);
                float b = vhash(i + vec2(1.0, 0.0));
                float c = vhash(i + vec2(0.0, 1.0));
                float d = vhash(i + vec2(1.0, 1.0));
                return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
            }

            // Multi-octave, large-scale domain warp.
            vec2 domain_warp(vec2 p, float t, float chaos) {
                const float TAU = 6.2831853;
                float s = 1.0 + chaos;
                vec2 w;
                w.x = sin(p.x * 0.0123 + t * 0.070) + 0.5 * sin(p.y * 0.0171 - t * 0.053) + 0.26 * sin((p.x + p.y) * 0.0093 + t * 0.031);
                w.y = cos(p.y * 0.0137 + t * 0.061) + 0.5 * cos(p.x * 0.0213 + t * 0.043) + 0.26 * cos((p.x - p.y) * 0.0151 - t * 0.037);
                return w * (_WarpAmount * s);
            }

            // Low-frequency swell unique across the whole ocean. Because it is
            // driven by world coordinates (not a fixed grid) it never repeats.
            float swell(vec2 p, float t) {
                float a = sin(p.x * 0.030 + t * 0.25);
                float b = sin(p.y * 0.024 - t * 0.21);
                float c = 0.5 * sin((p.x + p.y) * 0.015 + t * 0.17);
                float d = 0.4 * sin((p.x - p.y) * 0.019 - t * 0.13);
                return (a + b + c + d);
            }

            float sim_height(vec2 world) {
                if (_HasSim < 0.5) return 0.0;
                vec2 uv = (world - _SimGridCenter) / _SimGridSize + 0.5;
                if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return 0.0;
                float h = texture(_SimTex, uv).r * _SimDispScale;
                if (!(h == h)) h = 0.0;
                return clamp(h, -8.0, 8.0);
            }

            void main() {
                vec3 world = (u_model * vec4(in_position, 1.0)).xyz;
                v_local_y = in_position.y;
                bool isTop = in_normal.y > 0.5;
                bool isBottom = in_normal.y < -0.5;
                v_face = isTop ? 0.0 : (isBottom ? 2.0 : 1.0);

                // Compute Gerstner wave displacement for all faces.
                vec2 p = world.xz;
                vec2 wp = p + domain_warp(p, _Time, _Chaos);

                vec3 disp = vec3(0.0);
                vec3 tangent = vec3(1.0, 0.0, 0.0);
                vec3 binormal = vec3(0.0, 0.0, 1.0);

                float windNorm = clamp(_WindSpeed / 60.0, 0.0, 1.0);
                float storm = pow(windNorm, 1.3);
                float gust = 1.0 + _WindGust * _WindTurbulence * (0.4 + 0.8 * windNorm);
                float ampScale = mix(0.8, 3.6, storm);
                float wlenScale = mix(1.0, 2.6, storm);
                float chop = _Choppiness * mix(0.6, 2.2, windNorm);
                vec2 wdir = normalize(_WindDir + vec2(1e-5));

                float Jxx = 1.0;
                float Jzz = 1.0;
                float Jxz = 0.0;

                for (int i = 0; i < MAX_WAVES; i++) {
                    if (i >= _WaveCount) break;
                    vec2 d = _WaveDirection[i];
                    // Steer each wave's heading toward the wind as it
                    // strengthens, so the sea becomes a coherent, wind-driven
                    // system instead of randomly crossed swells.
                    if (_WindAlign > 0.001) {
                        float a0 = atan(d.y, d.x);
                        float aw = atan(wdir.y, wdir.x);
                        float da = mod(a0 - aw + 3.14159265, 6.2831853) - 3.14159265;
                        float a = a0 - da * _WindAlign;
                        d = vec2(cos(a), sin(a));
                    }
                    float amp = _WaveParams[i].x * gust * ampScale;
                    float wlenGrow = mix(1.0, wlenScale, clamp(1.0 - float(i) / max(float(_WaveCount - 1), 1.0), 0.0, 1.0));
                    float wlen = max(_WaveParams[i].y, 0.0001) * wlenGrow;
                    float speed = _WaveParams[i].z;
                    float steep = _WaveParams[i].w * chop;
                    float k = 6.2831853 / wlen;
                    float c = sqrt(G / k);

                    if (_Chaos > 0.0) {
                        float n1 = vnoise(wp * 0.05 + vec2(float(i) * 7.3, 1.7));
                        float ang = (n1 - 0.5) * _Chaos * 1.3;
                        float cs = cos(ang);
                        float sn = sin(ang);
                        d = vec2(d.x * cs - d.y * sn, d.x * sn + d.y * cs);
                    }

                    float f = k * (dot(d, wp) - c * speed * _Time);
                    f = mod(f, 6.2831853);
                    float a = amp;
                    float wa = k * a;
                    float q = steep / (wa * float(_WaveCount) + 1e-4);
                    // Safety clamp on q*k*a (the actual dimensionless
                    // steepness that folds the surface), not on q alone --
                    // q alone is scale-dependent on k and a, so clamping it
                    // directly collapses to near-flat once wavelength/
                    // amplitude grow with wind instead of guarding anything.
                    float qwa = q * wa;
                    const float QWA_MAX = 0.32;
                    if (qwa > QWA_MAX) {
                        q *= QWA_MAX / qwa;
                        qwa = QWA_MAX;
                    }
                    float cosf = cos(f);
                    float sinf = sin(f);
                    disp.x += q * a * d.x * cosf;
                    disp.z += q * a * d.y * cosf;
                    disp.y += a * sinf;
                    tangent += vec3(-q * d.x * d.x * wa * sinf, d.x * wa * cosf, -q * d.x * d.y * wa * sinf);
                    binormal += vec3(-q * d.x * d.y * wa * sinf, d.y * wa * cosf, -q * d.y * d.y * wa * sinf);
                    // Jacobian of the horizontal displacement field: where it
                    // drops toward/below zero the surface is folding in on
                    // itself (a breaking crest), which is the physically
                    // correct place for foam -- not just "wherever it's tall".
                    Jxx -= qwa * d.x * d.x * sinf;
                    Jzz -= qwa * d.y * d.y * sinf;
                    Jxz -= qwa * d.x * d.y * sinf;
                }

                float jacobian = Jxx * Jzz - Jxz * Jxz;
                float foamMask = clamp(1.0 - jacobian, 0.0, 3.0);

                disp.y += _MacroWave * swell(p * mix(1.0, 0.4, storm), _Time) * (0.3 + 1.4 * windNorm);

                // Wind-driven chop: short, fast, wind-aligned ripples that only
                // build up in a blow. This is the broken, agitated skin of a
                // stormy sea (its high-frequency normal is covered by the
                // fragment detail normal).
                {
                    float chopAmp = storm * 0.16;
                    float baseWL = mix(2.6, 0.9, windNorm);
                    for (int c = 0; c < 4; c++) {
                        float ph = float(c) * 2.39996323;
                        vec2 cd = normalize(mix(wdir, vec2(cos(ph), sin(ph)), 0.45));
                        float wl = baseWL * (0.7 + 0.3 * float(c));
                        float k = 6.2831853 / wl;
                        float sp = 1.0 + 0.6 * float(c);
                        float f = k * (dot(cd, wp) - sqrt(G / k) * sp * _Time);
                        f = mod(f, 6.2831853);
                        float a = chopAmp / (1.0 + float(c));
                        disp.x += cd.x * a * cos(f);
                        disp.z += cd.y * a * cos(f);
                        disp.y += a * sin(f);
                    }
                }

                if (isTop) {
                    // Top surface: full Gerstner displacement.
                    float sh = sim_height(p);
                    world += disp;
                    world.y += sh;
                    v_normal = normalize(cross(binormal, tangent));
                    v_world_pos = world;
                    v_crest = disp.y + sh;
                    v_chop = chop;
                    v_foamMask = foamMask + clamp(abs(sh) * 3.0, 0.0, 1.5);
                    v_detail_coord = wp;
                } else if (isBottom) {
                    // Bottom: no displacement.
                    v_world_pos = world;
                    v_normal = in_normal;
                    v_crest = 0.0;
                    v_chop = 0.0;
                    v_foamMask = 0.0;
                    v_detail_coord = world.xz;
                } else {
                    // Side walls: follow the full Gerstner displacement so
                    // the waterline stays connected. Blend factor is 0 at
                    // the bottom (floor stays rigid) and 1 at the top rim.
                    float blend = clamp(v_local_y + 0.5, 0.0, 1.0);
                    float sh = sim_height(p);
                    world.x += disp.x * blend;
                    world.z += disp.z * blend;
                    world.y += disp.y * blend;
                    world.y += sh * blend;
                    v_world_pos = world;
                    v_normal = in_normal;
                    v_crest = disp.y * blend + sh * blend;
                    v_chop = 0.0;
                    v_foamMask = foamMask * blend + clamp(abs(sh) * 3.0, 0.0, 1.5) * blend;
                    v_detail_coord = wp;
                }

                vec4 clip = u_proj * u_view * vec4(world, 1.0);
                v_screen_uv = clip.xy / clip.w * 0.5 + 0.5;
                gl_Position = clip;
            }

            // @FRAGMENT

            #version 460 core
            #define MAX_WAVES 8
            #define MAX_LIGHTS 16
            in vec3 v_world_pos;
            in vec3 v_normal;
            in vec2 v_screen_uv;
            in vec2 v_detail_coord;
            in float v_crest;
            in float v_chop;
            in float v_foamMask;
            in float v_face;
            in float v_local_y;
            out vec4 frag_color;

            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform vec3 u_camera_pos;
            uniform float _Time;
            uniform vec3 _SunDirection;
            uniform vec3 _SunColor;
            uniform float _SunIntensity;
            uniform int _LightCount;
            uniform vec3 _LightPos[MAX_LIGHTS];
            uniform vec3 _LightColor[MAX_LIGHTS];
            uniform float _LightIntensity[MAX_LIGHTS];
            uniform float _LightRange[MAX_LIGHTS];
            uniform vec3 _LightDir[MAX_LIGHTS];
            uniform float _LightSpotCos[MAX_LIGHTS];
            uniform vec2 _WindDir;
            uniform float _WindSpeed;
            uniform float _WindGust;
            uniform float _WindTurbulence;
            uniform vec3 _DeepColor;
            uniform vec3 _ShallowColor;
            uniform vec3 _FoamColor;
            uniform vec3 _SSSColor;
            uniform vec3 _HorizonColor;
            uniform float _Smoothness;
            uniform float _Distortion;
            uniform float _NormalStrength;
            uniform float _WaveTiling;
            uniform float _DetailSpeed;
            uniform float _Choppiness;
            uniform float _Caustics;
            uniform float _RefractStrength;
            uniform float _FresnelPower;
            uniform float _FoamStrength;
            uniform float _Specular;
            uniform float _ShoreFade;
            uniform float _MacroWave;
            uniform float _Chaos;
            uniform float _DetailScale;
            uniform float _DetailOctaves;
            uniform float _DetailFade;
            uniform int _IsBox;
            uniform sampler2D _SimTex;
            uniform float _HasSim;
            uniform vec2 _SimGridCenter;
            uniform float _SimGridSize;
            uniform float _SimDispScale;
            uniform float _SimNormalScale;
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
            float fbm(vec2 p, int octaves) {
                float v = 0.0;
                float amp = 0.5;
                mat2 rot = mat2(0.83, -0.56, 0.56, 0.83);
                for (int i = 0; i < 6; i++) {
                    if (i >= octaves) break;
                    v += amp * vnoise(p);
                    p = rot * p * 2.03 + 7.3 + float(i) * 1.7;
                    amp *= 0.5;
                }
                return v;
            }
            vec2 hash2(vec2 p) {
                p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
                return fract(sin(p) * 43758.5453);
            }

            float caustic_web(vec2 uv, float t) {
                vec2 n = floor(uv);
                vec2 f = fract(uv);
                float f1 = 8.0;
                float f2 = 8.0;
                for (int j = -1; j <= 1; j++) {
                    for (int i = -1; i <= 1; i++) {
                        vec2 g = vec2(float(i), float(j));
                        vec2 o = hash2(n + g);
                        o = 0.5 + 0.5 * sin(t * 0.9 + 6.2831853 * o);
                        vec2 r = g + o - f;
                        float d = dot(r, r);
                        if (d < f1) { f2 = f1; f1 = d; }
                        else if (d < f2) { f2 = d; }
                    }
                }
                float edge = sqrt(f2) - sqrt(f1);
                return 1.0 - smoothstep(0.0, 0.07, edge);
            }
            // Multi-octave detail normal. Each octave uses an incommensurate
            // frequency (1.93x) and an independent time scroll + phase, so the
            // micro-surface never tiles into a recognisable pattern even when
            // viewed from far away or high above.
            vec3 detail_normal(vec2 coord, float t, int octaves, float chaos) {
                float e = 0.12;
                vec2 nrm = vec2(0.0);
                float totalW = 0.0;
                float amp = 1.0;
                float freq = 1.3;
                float phase = 0.0;
                for (int i = 0; i < 12; i++) {
                    if (i >= octaves) break;
                    float f = freq * pow(1.93, float(i));
                    vec2 q = coord * f + vec2(t * (0.11 + 0.017 * float(i)) + phase,
                                              -t * (0.08 + 0.013 * float(i)) - phase);
                    float n0 = vnoise(q);
                    float hx = (vnoise(q + vec2(e, 0.0)) - n0) * 2.0;
                    float hz = (vnoise(q + vec2(0.0, e)) - n0) * 2.0;
                    nrm += vec2(hx, hz) * amp;
                    totalW += amp;
                    phase += 19.1;
                    amp *= 0.6;
                }
                nrm /= max(totalW, 0.001);
                vec3 n = vec3(-nrm.x, 1.0, -nrm.y);
                // Wind-aligned capillary streak.
                vec2 wdir = normalize(_WindDir + vec2(1e-5));
                vec2 perpv = vec2(-wdir.y, wdir.x);
                float streakFreq = 9.0 + _WindSpeed * 0.25;
                float streak = sin(dot(coord, wdir) * streakFreq + t * (1.5 + _WindSpeed * 0.1));
                streak += 0.5 * sin(dot(coord, perpv) * streakFreq * 1.7 - t * 0.9);
                n.x += wdir.x * streak * 0.05 * (0.4 + _WindTurbulence);
                n.z += wdir.y * streak * 0.05 * (0.4 + _WindTurbulence);
                n.xy *= _NormalStrength;
                // Chaos micro-sparkle.
                if (chaos > 0.0) {
                    float m1 = vnoise(coord * 41.0 + t * 0.7) - 0.5;
                    float m2 = vnoise(coord * 73.0 - t * 0.5) - 0.5;
                    float m3 = vnoise(coord * 53.0 + 11.0 - t * 0.6) - 0.5;
                    n.x += (m1 + 0.5 * m2) * chaos * 0.15;
                    n.z += (m3 + 0.5 * m2) * chaos * 0.15;
                }
                return normalize(n);
            }
            float linearize_depth(float d) {
                float z_n = 2.0 * d - 1.0;
                return 2.0 * _CamNear * _CamFar / (_CamFar + _CamNear - z_n * (_CamFar - _CamNear));
            }

            // Screen-space reflection ray march with occlusion testing and a
            // binary-search refine step.
            vec4 trace_ssr(vec3 ro, vec3 rd) {
                const int STEPS = 20;
                const int REFINE = 6;
                const float MAX_DIST = 240.0;
                float t = 0.35;
                float stepLen = MAX_DIST / float(STEPS);
                bool hit = false;
                float tHit = 0.0;
                for (int i = 0; i < STEPS; i++) {
                    vec3 p = ro + rd * t;
                    vec4 clip = u_proj * u_view * vec4(p, 1.0);
                    if (clip.w <= 0.001) break;
                    vec2 uv = clip.xy / clip.w * 0.5 + 0.5;
                    if (uv.x <= 0.0 || uv.x >= 1.0 || uv.y <= 0.0 || uv.y >= 1.0) break;
                    float sceneD = linearize_depth(texture(_SceneDepth, uv).r);
                    float rayD = linearize_depth(clip.z / clip.w * 0.5 + 0.5);
                    if (rayD > sceneD && rayD - sceneD < 6.0) {
                        hit = true;
                        tHit = t;
                        break;
                    }
                    t += stepLen * (1.0 + float(i) * 0.08);
                }
                if (!hit) return vec4(0.0);
                float lo = max(tHit - stepLen, 0.01);
                float hi = tHit;
                vec2 hitUV = vec2(0.5);
                for (int j = 0; j < REFINE; j++) {
                    float mid = (lo + hi) * 0.5;
                    vec3 pm = ro + rd * mid;
                    vec4 cm = u_proj * u_view * vec4(pm, 1.0);
                    vec2 um = cm.xy / cm.w * 0.5 + 0.5;
                    float sceneD = linearize_depth(texture(_SceneDepth, um).r);
                    float rayD = linearize_depth(cm.z / cm.w * 0.5 + 0.5);
                    hitUV = um;
                    if (rayD > sceneD) hi = mid; else lo = mid;
                }
                vec2 edge = min(hitUV, 1.0 - hitUV);
                float edgeFade = smoothstep(0.0, 0.10, min(edge.x, edge.y));
                float distFade = 1.0 - smoothstep(MAX_DIST * 0.6, MAX_DIST, tHit);
                return vec4(texture(_SceneColor, hitUV).rgb, edgeFade * distFade);
            }
            vec3 sky_tint(vec3 dir) {
                float h = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
                vec3 horizon = _HorizonColor;
                vec3 zenith = vec3(0.12, 0.30, 0.74);
                vec3 ground = vec3(0.20, 0.22, 0.26);
                vec3 sky = mix(horizon, zenith, pow(h, 0.55));
                sky = mix(ground, sky, smoothstep(-0.05, 0.05, dir.y));
                float sun = max(dot(normalize(dir), normalize(_SunDirection)), 0.0);
                sky += _SunColor * _SunIntensity * pow(sun, 8.0) * 0.35 * horizon;
                sky += _SunColor * _SunIntensity * smoothstep(0.9995, 0.99995, sun) * 6.0;
                sky += _SunColor * _SunIntensity * pow(sun, 2000.0) * 3.0;
                return sky;
            }

            void main() {
                // Wrap the detail coordinate to keep noise precise far from the
                // origin while staying fixed in world space.
                vec2 dc = mod(v_detail_coord + 2048.0, 4096.0) - 2048.0;
                float dist = length(u_camera_pos.xz - v_world_pos.xz);
                float detailFade = 1.0 - smoothstep(_DetailFade * 0.3, _DetailFade, dist);

                if (v_face > 0.5) {
                    vec3 Nv = normalize(v_normal);
                    vec3 V = normalize(u_camera_pos - v_world_pos);
                    bool backface = dot(V, Nv) < 0.0;
                    if (backface) Nv = -Nv;
                    float ndv = max(dot(Nv, V), 0.0);
                    float F0 = 0.02;
                    float fr = F0 + (1.0 - F0) * pow(1.0 - ndv, _FresnelPower);
                    // Water body color: deeper at the bottom, shallow at top.
                    float h = clamp(v_local_y + 0.5, 0.0, 1.0);
                    vec3 waterBody = mix(_DeepColor, _ShallowColor, h * h);
                    // SSS: light passing through the volume from the sun.
                    vec3 L = normalize(_SunDirection);
                    float sssAmt = pow(clamp(dot(V, -L) * 0.5 + 0.5, 0.0, 1.0), 3.0);
                    vec3 sss = _SSSColor * sssAmt * (0.3 + 0.6 * h) * _SunIntensity;
                    // Caustics on the submerged walls/floor are now projected in
                    // world space (see caustics.glsl) by the underwater pass when
                    // the camera is below the surface, so the glass itself only
                    // shows its body colour + SSS + sky reflection here.
                    // Sky/environment reflection.
                    vec3 R = reflect(-V, Nv);
                    vec3 sky = sky_tint(R);
                    vec3 col = waterBody + sss * 0.5;
                    col = mix(col, sky, fr * 0.45);
                    // Sun specular on the glass surface.
                    vec3 Hh = normalize(V + L);
                    float sp = pow(max(dot(Nv, Hh), 0.0), mix(64.0, 4096.0, pow(_Smoothness, 1.5)));
                    col += _SunColor * _SunIntensity * sp * _Specular * 0.6;
                    float alpha = mix(0.55, 0.88, h);
                    alpha = mix(alpha, 0.95, fr * 0.5);
                    frag_color = vec4(col, alpha);
                    return;
                }

                // ---------- Surface (ocean / pond top) ----------
                int oct = int(clamp(_DetailOctaves, 1.0, 12.0));
                oct = int(clamp(float(oct) * mix(0.35, 1.0, detailFade), 1.0, 12.0));
                vec3 dn = detail_normal(dc * _WaveTiling * _DetailScale, _Time * _DetailSpeed, oct, _Chaos);
                vec3 N = normalize(v_normal);
                if (_HasSim > 0.5) {
                    vec2 suv = (v_world_pos.xz - _SimGridCenter) / _SimGridSize + 0.5;
                    if (suv.x >= 0.0 && suv.x <= 1.0 && suv.y >= 0.0 && suv.y <= 1.0) {
                        float e = _SimGridSize / 512.0;
                        float hx = (texture(_SimTex, suv + vec2(e / _SimGridSize, 0.0)).r - texture(_SimTex, suv - vec2(e / _SimGridSize, 0.0)).r) * _SimDispScale;
                        float hz = (texture(_SimTex, suv + vec2(0.0, e / _SimGridSize)).r - texture(_SimTex, suv - vec2(0.0, e / _SimGridSize)).r) * _SimDispScale;
                        if (!(hx == hx)) hx = 0.0;
                        if (!(hz == hz)) hz = 0.0;
                        vec3 simN = normalize(vec3(-hx * _SimNormalScale, 2.0 * e, -hz * _SimNormalScale));
                        N = normalize(mix(N, simN, clamp(_SimNormalScale, 0.0, 1.0) * 0.5));
                    }
                    N = normalize(N + vec3(dn.x, 0.0, dn.z) * 0.5);
                } else {
                    N = normalize(N + vec3(dn.x, 0.0, dn.z));
                }

                vec3 V = normalize(u_camera_pos - v_world_pos);
                bool backface = dot(V, N) < 0.0;
                if (backface) N = -N;

                float ndv = max(dot(N, V), 0.0);
                float F0 = 0.02;
                float fres;
                if (backface) {
                    fres = F0 + (1.0 - F0) * pow(1.0 - ndv, _FresnelPower) * 0.2;
                } else {
                    fres = F0 + (1.0 - F0) * pow(1.0 - ndv, _FresnelPower);
                }

                // Water body color from depth. Real light absorption through
                // water is exponential (Beer-Lambert), not linear, and red
                // wavelengths are absorbed several times faster than blue/
                // green -- that's what actually produces the shallow-turquoise
                // to deep-navy gradient instead of a flat color lerp.
                float depthDist = 0.0;
                bool useScene = (_HasScene == 1) && (v_face < 0.5);
                if (useScene) {
                    float scene_d = linearize_depth(texture(_SceneDepth, v_screen_uv).r);
                    float water_d = linearize_depth(gl_FragCoord.z);
                    depthDist = max(scene_d - water_d, 0.0);
                }
                float depthT = 1.0 - exp(-depthDist * 0.22);
                vec3 waterBody = mix(_ShallowColor, _DeepColor, depthT);
                vec3 absorb = exp(-depthDist * vec3(0.45, 0.16, 0.10));

                vec3 color;
                if (useScene) {
                    // Drive the refraction warp from the ripple detail normal
                    // (dn), which always varies, instead of the wave normal N
                    // that is near-flat for calm water and produced no warp.
                    vec2 refr_uv = clamp(v_screen_uv + (N.xz * 0.4 + dn.xz) * _Distortion, 0.001, 0.999);
                    vec3 refr = texture(_SceneColor, refr_uv).rgb * absorb;
                    refr = mix(refr, waterBody, _RefractStrength * (0.4 + 0.6 * depthT));

                    vec3 R = reflect(-V, N);
                    vec3 sky = sky_tint(R);
                    vec3 refl = sky;
                    if (fres > 0.025) {
                        vec4 ssr = trace_ssr(v_world_pos + N * 0.05, R);
                        refl = mix(sky, ssr.rgb, ssr.a);
                    }
                    color = mix(refr, refl, fres);
                } else {
                    vec3 R = reflect(-V, N);
                    color = mix(waterBody, sky_tint(R), fres);
                }
                // Faint ambient occlusion in wave troughs for volume.
                color *= mix(0.88, 1.0, smoothstep(-0.35, 0.15, v_crest));

                // Subsurface scattering: light transmitted through thin wave
                // crests, modulated by how directly we look toward the sun.
                float crest = clamp(v_crest * 0.6 + 0.4, 0.0, 1.0);
                vec3 L = normalize(_SunDirection);
                float sssAmount = pow(clamp(dot(V, -L) * 0.5 + 0.5, 0.0, 1.0), 3.0);
                vec3 sss = _SSSColor * sssAmount * (0.35 + 0.85 * crest) * _SunIntensity;
                if (backface) sss *= 1.8;
                color += sss * 0.6;

                // Sun specular highlight (Blinn-Phong), perturbed by the detail
                // normal for a moving sun glitter.
                vec3 H = normalize(V + L);
                // Specular anti-aliasing: a fixed 9000-power sun-glitter term
                // has no filtering, so distant or grazing-angle pixels alias
                // into hard, flickering noise instead of a soft glitter path.
                // Widen (soften) the highlight as distance / grazing angle
                // increase, matching the loss of resolvable micro-detail.
                float aa = clamp(1.0 - dist / 260.0, 0.0, 1.0) * clamp(ndv * 1.4, 0.1, 1.0);
                float shininess = mix(32.0, 4096.0, pow(_Smoothness, 1.5)) * mix(0.18, 1.0, aa);
                float spec = pow(max(dot(N, H), 0.0), shininess);
                float sparkleExp = mix(700.0, 9000.0, aa);
                float sparkle = pow(max(dot(N, H), 0.0), sparkleExp) * smoothstep(0.2, 1.0, dn.y) * mix(0.35, 1.0, aa);
                color += _SunColor * _SunIntensity * (spec * _Specular + sparkle * _Specular * 1.5);

                // Interaction with non-directional light sources.
                for (int i = 0; i < MAX_LIGHTS; i++) {
                    if (i >= _LightCount) break;
                    vec3 Lv = _LightPos[i] - v_world_pos;
                    float d = length(Lv);
                    if (d < 1e-3) continue;
                    vec3 Ll = Lv / d;
                    float range = max(_LightRange[i], 0.001);
                    float range_fade = clamp(1.0 - pow(d / range, 4.0), 0.0, 1.0);
                    float att = range_fade * range_fade / (d * d + 1.0);
                    float spot = 1.0;
                    if (_LightSpotCos[i] > -0.999) {
                        spot = smoothstep(_LightSpotCos[i], mix(_LightSpotCos[i], 1.0, 0.15), dot(-Ll, normalize(_LightDir[i])));
                        if (spot <= 0.0) continue;
                    }
                    float lspec = pow(max(dot(N, normalize(V + Ll)), 0.0), shininess);
                    color += _LightColor[i] * _LightIntensity[i] * att * spot * (lspec * _Specular * 1.2 + 0.04);
                }

                // ---- Foam ----
                float ft = _Time * _DetailSpeed;
                // Turbulent, scrolling foam texture (two fbm octaves) plus a
                // finer octave used to erode the edges into churned wisps.
                float foamTex = fbm(dc * 1.6 + vec2(ft * 0.10, -ft * 0.07), 4);
                foamTex = foamTex * 0.6 + 0.4 * fbm(dc * 4.2 - vec2(ft * 0.06, ft * 0.11), 4);
                float fine = fbm(dc * 7.0 + vec2(-ft * 0.2, ft * 0.15), 3);
                foamTex = clamp(foamTex, 0.0, 1.0);
                fine = clamp(fine, 0.0, 1.0);

                // Crest foam: driven by actual wave-crest folding (the
                // Jacobian mask from the vertex stage), not just raw wave
                // height -- so foam sits where the surface genuinely breaks
                // instead of coating every tall-but-smooth swell.
                // Whitecap coverage follows a Monahan-style curve (roughly
                // wind^3): negligible below ~7 m/s, growing fast only once
                // the sea is genuinely stormy, instead of blanketing the
                // ocean at any moderate breeze.
                float crestAmt = smoothstep(0.25, 1.1, v_foamMask);
                float coverage = pow(clamp((_WindSpeed - 7.0) / 40.0, 0.0, 1.0), 2.4);
                float whitecap = smoothstep(0.6, 1.6, v_foamMask) * coverage;
                float crest_foam = crestAmt * smoothstep(0.35, 0.75, foamTex) + whitecap * (0.7 + 0.3 * foamTex);

                // Shoreline foam from scene depth.
                float shore_foam = 0.0;
                if (useScene) {
                    float shoreMask = 1.0 - clamp(depthDist / max(_ShoreFade, 0.001), 0.0, 1.0);
                    shore_foam = shoreMask * smoothstep(0.35, 0.75, foamTex);
                }

                // Erode the density with fine turbulence so the foam reads as
                // churned sea-foam with fingered boundaries, not a solid blob.
                float dens = clamp((crest_foam + shore_foam) * _FoamStrength, 0.0, 1.0);
                dens *= smoothstep(0.2, 0.9, dens * 1.3 + fine * 0.6 - 0.35);
                float foam = clamp(dens, 0.0, 1.0);

                if (foam > 0.001) {
                    vec3 foamCol = _FoamColor;
                    // Thick foam is bright white; thin rims stay cool/translucent.
                    foamCol *= 0.82 + 0.25 * foamTex;
                    // Glossy sun glint on the broken surface.
                    float foamSpec = pow(max(dot(N, H), 0.0), mix(140.0, 600.0, _Smoothness) * mix(0.3, 1.0, aa)) * _Specular;
                    foamCol += _SunColor * _SunIntensity * foamSpec * 0.6;
                    // Trapped-air bubble sparkle.
                    float bubbles = smoothstep(0.6, 0.95, vnoise(dc * 11.0 + ft * 0.6));
                    foamCol += _SunColor * _SunIntensity * bubbles * foam * 0.5;
                    color = mix(color, foamCol, foam);
                }

                // Distance fade to horizon (for infinite ocean only).
                if (_IsBox != 1) {
                    float fade = clamp(dist / 1100.0, 0.0, 1.0);
                    color = mix(color, _HorizonColor, fade * fade);
                }
                // The water surface is essentially opaque: refraction and
                // reflection are resolved in the COLOR above, so low alpha
                // here only lets the raw scene bleed through as patchy
                // transparency. Keep a high base alpha and only let true
                // undersides (backfaces) and foam modulate it.
                float alpha;
                if (backface) {
                    alpha = mix(0.75, 1.0, fres);
                    alpha = mix(alpha, 1.0, foam);
                } else {
                    alpha = mix(0.96, 1.0, fres);
                    alpha = mix(alpha, 1.0, foam);
                }

                frag_color = vec4(color, alpha);
            }
            ENDGLSL
        }
    }
}
