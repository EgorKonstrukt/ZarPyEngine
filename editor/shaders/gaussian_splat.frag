// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core

in vec3 v_color;
in vec2 v_uv;
in float v_alpha;

out vec4 frag_color;

void main() {
    vec2 d = v_uv - vec2(0.5);
    float power = -0.5 * dot(d * 2.0, d * 2.0);
    if (power < -4.0) discard;

    float alpha = v_alpha * exp(power);
    if (alpha < 0.005) discard;

    vec3 col = max(v_color, vec3(0.0));
    frag_color = vec4(col * alpha, alpha);
}
