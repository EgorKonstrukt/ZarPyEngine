// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun
//
// Material-independent underwater caustics projection pass.
//
// This pass runs for ANY scene (any materials / shaders on the geometry
// below the water) and projects physically based caustics onto every
// submerged pixel. It reads the opaque scene colour + depth, reconstructs
// the world position of the submerged geometry and lights it with the
// caustic pattern produced by the real Gerstner wave field refracted
// through the surface toward the sun (see caustics.glsl). Because it
// operates entirely in screen space on the final image, it works with
// every material and from above or below the surface.

#version 460 core
#define MAX_WAVES 8

// Wave-field uniforms consumed by caustics.glsl -- declared BEFORE the
// include because the included functions reference them.
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
uniform mat4 u_inv_view_proj;
uniform float u_surface_y;
uniform float u_caustics_strength;
uniform float u_cam_near;
uniform float u_cam_far;
uniform vec3 u_caustic_tint;

float linearize_depth(float d) {
    float z_n = 2.0 * d - 1.0;
    return 2.0 * u_cam_near * u_cam_far / (u_cam_far + u_cam_near - z_n * (u_cam_far - u_cam_near));
}

vec3 reconstruct_world(vec2 uv, out float dist) {
    float depth = texture(u_depth, uv).r;
    vec4 clip = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 wp = u_inv_view_proj * clip;
    vec3 p = wp.xyz / wp.w;
    dist = linearize_depth(depth);
    return p;
}

void main() {
    vec2 uv = v_uv;
    vec3 scene = texture(u_scene, uv).rgb;

    float depth = texture(u_depth, uv).r;
    if (depth >= 0.99999) {
        frag_color = vec4(scene, 1.0);
        return;
    }

    float dist;
    vec3 wpos = reconstruct_world(uv, dist);

    // Only submerged geometry is lit by caustics.
    if (wpos.y >= u_surface_y) {
        frag_color = vec4(scene, 1.0);
        return;
    }

    float strength = u_caustics_strength;
    if (strength <= 0.001) {
        frag_color = vec4(scene, 1.0);
        return;
    }

    // March the sun ray through the wavy surface and focus the light on this
    // submerged point (caustics.glsl). Driven by the real waves + sun.
    vec3 L = normalize(u_sun_dir);
    vec3 caus = caustic_light(wpos, u_surface_y, L,
                              u_sun_color, u_sun_intensity, strength, u_time);

    // Caustics fade with how far the light travels through water (depth below
    // the surface) and with camera distance so distant murk stays calm.
    float below = u_surface_y - wpos.y;
    float depthFade = exp(-max(below, 0.0) * 0.035);
    float distFade = exp(-dist * 0.0035);
    caus *= depthFade * distFade;

    // Keep the artistic caustic tint but let the sun colour dominate.
    caus = mix(caus, caus * (u_caustic_tint + 0.4), 0.5);

    frag_color = vec4(scene + caus, 1.0);
}
