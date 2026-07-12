// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core
in vec2 v_uv;
in float v_alpha;
uniform sampler2D u_texture;
uniform vec4 u_color;
uniform float u_alpha;
out vec4 frag_color;
void main() {
    vec4 tex = texture(u_texture, v_uv);
    float a = tex.a * u_alpha * v_alpha;
    frag_color = vec4(tex.rgb * u_color.rgb, a);
}
