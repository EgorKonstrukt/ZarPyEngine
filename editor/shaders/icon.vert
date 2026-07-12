// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.
//
// Copyright (c) 2026 Zarrakun

#version 460 core
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in float in_alpha;
uniform mat4 u_mvp;
out vec2 v_uv;
out float v_alpha;
void main() {
    v_uv = in_uv;
    v_alpha = in_alpha;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
