// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun
//
// Physically motivated underwater caustics, shared by Water.shader (caustics
// seen THROUGH the surface, on the submerged floor) and underwater.frag
// (caustics on any submerged geometry when the camera is below the surface).
//
// The caustic pattern is NOT a texture: it is the light focusing produced by
// the very same Gerstner wave field that displaces the water surface. For a
// floor point we march the sun ray up to the wavy surface, refract it through
// the surface normal (Snell, n = 1.33) and project it back down to the floor;
// where neighbouring refracted rays converge the light is concentrated into the
// bright caustic filaments, where they diverge it goes dark. That Jacobian of
// the surface->floor mapping is exactly the caustic intensity.
//
// The HOST shader must declare these uniforms before including this file
// (they are already provided by the Water component / renderer):
//   #define MAX_WAVES 8
//   uniform int   _WaveCount;
//   uniform vec2  _WaveDirection[MAX_WAVES];
//   uniform vec4  _WaveParams[MAX_WAVES];   // (amplitude, wavelength, speed, steepness)
//   uniform vec2  _WindDir;
//   uniform float _WindSpeed;
//   uniform float _WindGust;
//   uniform float _WindTurbulence;
//   uniform float _WindAlign;
//   uniform float _Choppiness;
//   uniform float _MacroWave;
//   uniform float _Chaos;
// All functions take the animation time `t` explicitly (seconds).

#ifndef CAUSTICS_GLSL
#define CAUSTICS_GLSL

const float CAUSTIC_G = 9.81;

float caus_hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

// Vertical wave displacement AND its planar gradient at world (x,z). Matches
// the surface Gerstner field closely enough to drive caustics (the horizontal
// Gerstner displacement is ignored -- standard and negligible for focusing).
// The gradient is computed analytically from the phase of each wave.
void caustic_wave(vec2 world_xz, float t, out float h, out vec2 grad) {
    float windNorm = clamp(_WindSpeed / 60.0, 0.0, 1.0);
    float storm = pow(windNorm, 1.3);
    float gust = 1.0 + _WindGust * _WindTurbulence * (0.4 + 0.8 * windNorm);
    float ampScale = mix(0.8, 3.6, storm);
    float wlenScale = mix(1.0, 2.6, storm);
    float chop = _Choppiness * mix(0.6, 2.2, windNorm);
    vec2 wdir = normalize(_WindDir + vec2(1e-5));

    h = 0.0;
    grad = vec2(0.0);
    float totalW = max(float(_WaveCount), 1.0);
    for (int i = 0; i < MAX_WAVES; i++) {
        if (i >= _WaveCount) break;
        vec2 d = _WaveDirection[i];
        // Steer toward the wind as it strengthens (mirrors the surface).
        if (_WindAlign > 0.001) {
            float a0 = atan(d.y, d.x);
            float aw = atan(wdir.y, wdir.x);
            float da = mod(a0 - aw + 3.14159265, 6.2831853) - 3.14159265;
            float a = a0 - da * _WindAlign;
            d = vec2(cos(a), sin(a));
        }
        // Chaos: per-wave heading jitter (treated as locally constant).
        if (_Chaos > 0.0) {
            float n1 = caus_hash(floor(world_xz * 0.05) + vec2(float(i) * 7.3, 1.7));
            float ang = (n1 - 0.5) * _Chaos * 1.3;
            float cs = cos(ang);
            float sn = sin(ang);
            d = vec2(d.x * cs - d.y * sn, d.x * sn + d.y * cs);
        }
        float amp = _WaveParams[i].x * gust * ampScale;
        float wlenGrow = mix(1.0, wlenScale, clamp(1.0 - float(i) / max(totalW - 1.0, 1.0), 0.0, 1.0));
        float wlen = max(_WaveParams[i].y, 0.0001) * wlenGrow;
        float speed = _WaveParams[i].z;
        float k = 6.2831853 / wlen;
        float c = sqrt(CAUSTIC_G / k);
        float phase = k * (dot(d, world_xz) - c * speed * t);
        float s = sin(phase);
        float co = cos(phase);
        h += amp * s;
        // dh/d(world) = amp * cos(phase) * d(phase)/d(world)
        //             = amp * cos(phase) * k * d
        grad += amp * co * k * d;
    }
    // Low-frequency swell + analytic gradient.
    float sg = 0.3 + 1.4 * windNorm;
    h += _MacroWave * sg * (sin(world_xz.x * 0.030 + t * 0.25)
                          + sin(world_xz.y * 0.024 - t * 0.21)
                          + 0.5 * sin((world_xz.x + world_xz.y) * 0.015 + t * 0.17)
                          + 0.4 * sin((world_xz.x - world_xz.y) * 0.019 - t * 0.13));
    grad += _MacroWave * sg * vec2(
        0.030 * cos(world_xz.x * 0.030 + t * 0.25)
      + 0.015 * cos((world_xz.x + world_xz.y) * 0.015 + t * 0.17)
      + 0.019 * cos((world_xz.x - world_xz.y) * 0.019 - t * 0.13),
        0.024 * cos(world_xz.y * 0.024 - t * 0.21)
      + 0.015 * cos((world_xz.x + world_xz.y) * 0.015 + t * 0.17)
      - 0.019 * cos((world_xz.x - world_xz.y) * 0.019 - t * 0.13));
}

// Where a refracted sun ray leaving the wavy surface at surf_xz lands on a
// flat floor at floorY. Returns the floor hit xz.
vec2 caustic_floor_hit(vec2 surf_xz, float surfaceY, float floorY, vec3 L, float t) {
    float h; vec2 g;
    caustic_wave(surf_xz, t, h, g);
    vec3 N = normalize(vec3(-g.x, 1.0, -g.y));
    vec3 T = refract(-L, N, 1.0 / 1.33);
    if (T.y > -1e-3) return surf_xz;   // ray would not descend
    float tf = (floorY - (surfaceY + h)) / T.y;
    vec3 S = vec3(surf_xz.x, surfaceY + h, surf_xz.y);
    return S.xz + T.xz * tf;
}

// Caustic intensity at a floor point: 1 / |Jacobian| of the surface->floor
// mapping, i.e. where refracted rays converge (bright) or diverge (dark).
// L is the direction TOWARD the sun (normalized).
float caustic_intensity(vec2 floor_xz, float surfaceY, float floorY, vec3 L, float t) {
    if (L.y <= 0.02) return 0.0;       // sun below the horizon -> no caustics
    float e = 0.6;
    vec2 h0 = caustic_floor_hit(floor_xz, surfaceY, floorY, L, t);
    vec2 hx = caustic_floor_hit(floor_xz + vec2(e, 0.0), surfaceY, floorY, L, t);
    vec2 hz = caustic_floor_hit(floor_xz + vec2(0.0, e), surfaceY, floorY, L, t);
    float jxx = (hx.x - h0.x) / e;
    float jxz = (hz.x - h0.x) / e;
    float jzx = (hx.y - h0.y) / e;
    float jzz = (hz.y - h0.y) / e;
    float det = jxx * jzz - jxz * jzx;
    float focus = 1.0 / max(abs(det), 0.08);
    return clamp(focus, 0.0, 5.0);
}

// Convenience: sun-tinted caustic contribution for a submerged world point.
// Centered on the focusing deviation (c == 1 means a flat surface, i.e. no
// extra light), so only converging rays add bright filaments on top of the
// normally lit floor -- diverging regions stay at the floor's base lighting.
vec3 caustic_light(vec3 world_pos, float surfaceY, vec3 sun_dir,
                   vec3 sun_color, float sun_intensity, float strength, float t) {
    vec3 L = normalize(sun_dir);
    float c = caustic_intensity(world_pos.xz, surfaceY, world_pos.y, L, t);
    float fil = pow(max(c - 1.0, 0.0), 1.3) * 0.6;
    return sun_color * sun_intensity * fil * strength;
}

#endif // CAUSTICS_GLSL
