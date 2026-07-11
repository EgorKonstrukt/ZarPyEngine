// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun
//
// Screen-space underwater post effect. Applied when the camera is below a
// water surface. Uses the opaque scene color + depth to build a physically
// motivated murk (exponential depth fog), and projects animated caustics
// onto the submerged geometry -- the caustic pattern is computed from the
// actual Gerstner wave field refracted through the surface toward the sun
// (see caustics.glsl), so it lands on the seafloor and objects under water
// instead of being a flat texture. Also adds volumetric sunlight shafts,
// chromatic lens wobble at the edges and a teal color grade.

#version 460 core
#define MAX_WAVES 8

// Wave-field uniforms consumed by caustics.glsl -- must be declared BEFORE
// the include because the included functions reference them.
uniform int   _WaveCount;
uniform vec2  _WaveDirection[MAX_WAVES];
uniform vec4  _WaveParams[MAX_WAVES];
uniform vec2  _WindDir;
uniform float _WindSpeed;
uniform float _WindGust;
uniform float _WindTurbulence;
uniform float _WindAlign;
uniform float _Choppiness;
uniform float _MacroWave;
uniform float _Chaos;

// @CAUSTICS_INCLUDE
in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_scene;
uniform sampler2D u_depth;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec3 u_cam_pos;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform float u_sun_intensity;
uniform vec3 u_fog_color;
uniform vec3 u_caustic_color;
uniform float u_depth_below;
uniform float u_cam_near;
uniform float u_cam_far;
uniform mat4 u_view;
uniform mat4 u_proj;
uniform mat4 u_inv_view_proj;
uniform float u_fog_density;
uniform float u_surface_y;
uniform float u_caustics_strength;

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
float fbm(vec2 p) {
    float v = 0.0;
    float amp = 0.5;
    mat2 rot = mat2(0.8, -0.6, 0.6, 0.8);
    for (int i = 0; i < 5; i++) {
        v += amp * vnoise(p);
        p = rot * p * 2.02 + 7.3;
        amp *= 0.5;
    }
    return v;
}
float linearize_depth(float d) {
    float z_n = 2.0 * d - 1.0;
    return 2.0 * u_cam_near * u_cam_far / (u_cam_far + u_cam_near - z_n * (u_cam_far - u_cam_near));
}

// Reconstruct the world position of the submerged geometry at this pixel.
bool reconstruct_world(vec2 uv, out vec3 world_pos, out float scene_dist) {
    float depth = texture(u_depth, uv).r;
    if (depth >= 0.99999) return false;
    vec4 clip = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 wp = u_inv_view_proj * clip;
    if (abs(wp.w) < 1e-5) return false;
    world_pos = wp.xyz / wp.w;
    scene_dist = linearize_depth(depth);
    return true;
}

void main() {
    vec2 uv = v_uv;

    // Edge lens wobble: the surface above acts like a moving lens, strongest
    // toward the frame edges where refraction through the wave troughs is wildest.
    vec2 c = uv - 0.5;
    float r = length(c);
    float ang = atan(c.y, c.x);
    float wob = sin(ang * 6.0 + u_time * 1.5) * 0.012 + sin(ang * 13.0 - u_time * 0.9) * 0.006;
    vec2 duv = clamp(uv + normalize(c + 1e-5) * wob * smoothstep(0.15, 0.75, r), 0.001, 0.999);
    vec3 scene = texture(u_scene, duv).rgb;

    float depth = texture(u_depth, uv).r;
    float dist = linearize_depth(depth);

    // Exponential depth fog: water absorbs red first, so the murk reads blue-green.
    float fog = 1.0 - exp(-dist * u_fog_density);
    fog = clamp(fog, 0.0, 1.0);
    float deep = clamp(u_depth_below / 40.0, 0.0, 1.0);
    vec3 fogCol = mix(u_fog_color, u_fog_color * 0.35, deep);
    vec3 col = mix(scene, fogCol, fog);

    // Volumetric sunlight shafts radiating from the sun's screen position.
    vec3 sd = normalize(u_sun_dir);
    vec4 sunClip = u_proj * u_view * vec4(u_cam_pos + sd * 10.0, 1.0);
    vec2 sunUV = sunClip.xy / sunClip.w * 0.5 + 0.5;
    vec2 d2 = uv - sunUV;
    float rays = 0.0;
    if (sd.y > 0.0) {
        float ra = atan(d2.y, d2.x);
        float rr = length(d2);
        float shaft = (sin(ra * 30.0 + u_time * 0.5) * 0.5 + 0.5) * (sin(ra * 17.0 - u_time * 0.3) * 0.5 + 0.5);
        rays = shaft * exp(-rr * 3.0) * smoothstep(0.0, 0.3, sd.y);
    }
    float causVisible = (1.0 - fog);
    col += u_sun_color * u_sun_intensity * rays * causVisible * 0.18;

    // Color grade: push toward teal and suppress red (selective absorption).
    col.g = mix(col.g, (col.g + col.b) * 0.5, 0.15);
    col.r *= 0.7;

    // Soft vignette to focus the eye and sell depth.
    col *= 1.0 - 0.35 * smoothstep(0.4, 1.0, r);

    frag_color = vec4(col, 1.0);
}
