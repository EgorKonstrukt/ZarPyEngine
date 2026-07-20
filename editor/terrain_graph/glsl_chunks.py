# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

UTILITY_GLSL = """
mat2 rot2(float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c);
}
"""

NOISE_GLSL = """
vec2 hash2(vec2 p, float seed) {
    p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
    p = p + seed * 0.137;
    return fract(sin(p) * 43758.5453123) * 2.0 - 1.0;
}

float hash(vec2 p, float seed) {
    return fract(sin(dot(p, vec2(127.1, 311.7)) + seed * 0.137) * 43758.5453123) * 2.0 - 1.0;
}

float gradient_noise(vec2 p, float seed) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    vec2 g00 = hash2(i, seed);
    vec2 g10 = hash2(i + vec2(1.0, 0.0), seed);
    vec2 g01 = hash2(i + vec2(0.0, 1.0), seed);
    vec2 g11 = hash2(i + vec2(1.0, 1.0), seed);
    float n00 = dot(g00, f);
    float n10 = dot(g10, f - vec2(1.0, 0.0));
    float n01 = dot(g01, f - vec2(0.0, 1.0));
    float n11 = dot(g11, f - vec2(1.0, 1.0));
    return mix(mix(n00, n10, u.x), mix(n01, n11, u.x), u.y);
}

float fbm(vec2 p, float seed, int octaves, float lacunarity, float persistence,
          float ridgeMix, float ridgePower, float ridgeSharpness,
          float billowMix, float billowPower, float twist) {
    float sum = 0.0;
    float amp = 0.5;
    float freq = 1.0;
    float norm = 0.0;
    for (int o = 0; o < 16; o++) {
        if (o >= octaves) break;
        vec2 q = p * freq;
        if (twist > 0.0) {
            float angle = twist * float(o);
            q = rot2(angle) * q;
        }
        float n = gradient_noise(q, seed + float(o) * 17.3);
        float ridged = 1.0 - abs(n);
        ridged = pow(clamp(ridged, 0.0, 1.0), ridgePower);
        ridged = mix(ridged, 1.0 - ridged, 1.0 - ridgeSharpness);
        float billow_val = abs(n) * 2.0 - 1.0;
        billow_val = pow(abs(billow_val) + 1e-10, billowPower) * sign(n);
        n = n * (1.0 - ridgeMix - billowMix) + ridged * ridgeMix - billow_val * billowMix;
        sum += n * amp;
        norm += amp;
        amp *= persistence;
        freq *= lacunarity;
    }
    return norm > 0.0 ? sum / norm : 0.0;
}

float voronoi(vec2 p, float freq, float seed) {
    vec2 sp = p * freq;
    vec2 i = floor(sp);
    vec2 f = fract(sp);
    float md = 1e10;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            vec2 n = vec2(float(dx), float(dy));
            vec2 pt = vec2(hash(i + n, seed) * 0.5 + 0.5,
                           hash(i + n + vec2(7.3, 1.7), seed) * 0.5 + 0.5);
            vec2 diff = n + pt - f;
            float d = dot(diff, diff);
            md = min(md, d);
        }
    }
    return sqrt(md);
}
"""

SHAPE_GLSL = """
float terrace_func(float h, float steps) {
    steps = max(2.0, steps);
    float t = h * steps;
    float fl = floor(t);
    float fr = fract(t);
    float sm = smoothstep(0.35, 0.65, fr);
    return (fl + sm) / steps;
}

float plateau_func(float h, float level, float sharpness) {
    if (sharpness <= 0.0) return h;
    float d = abs(h - level);
    float mask = clamp(1.0 - pow(d, sharpness), 0.0, 1.0);
    return h * (1.0 - mask) + level * mask;
}

float strata_func(float h, float scale) {
    return sin(h * scale * 6.28318) * 0.5;
}

float dune_func(vec2 p, float freq, float direction) {
    vec2 dir = vec2(cos(direction), sin(direction));
    float proj = dot(p, dir);
    return sin(proj * freq * 6.28318) * 0.5;
}

float continent_func(vec2 p, float seed, float scale, float falloff) {
    float n = fbm(p * scale, seed + 90.0, 6, 2.0, 0.5, 0.0, 4.0, 1.0, 0.0, 2.0, 0.0);
    n = n * 0.5 + 0.5;
    return smoothstep(0.5 - 0.5 * falloff, 0.5 + 0.5 * falloff, n);
}

float river_carve(vec2 uv, float seed) {
    float n = fbm(uv * 3.0 + 50.0, seed + 300.0, 6, 2.0, 0.5, 1.0, 6.0, 1.0, 0.0, 2.0, 0.0);
    float ridge = 1.0 - abs(n);
    ridge = pow(clamp(ridge, 0.0, 1.0), 6.0);
    return ridge;
}
"""

EROSION_GLSL = """
float sample_h(ivec2 ip) {
    return heights[ip.y * u_resolution + ip.x];
}
"""

ALL_GLSL_FUNCTIONS = UTILITY_GLSL + NOISE_GLSL + SHAPE_GLSL + EROSION_GLSL
