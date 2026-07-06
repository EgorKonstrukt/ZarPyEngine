# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from typing import Optional


def generate_shader_code(graph, shader_name: str = "Custom/GeneratedShader") -> str:
    nodes = list(graph.all_nodes())
    if not nodes:
        return _empty_shader(shader_name)

    vertex_output = None
    fragment_output = None
    uid_map = {}
    uid_counter = [0]

    def make_uid(node):
        name = getattr(node, 'NODE_NAME', 'node').lower().replace(' ', '_')
        uid = f"{name}_{uid_counter[0]}"
        uid_counter[0] += 1
        return uid

    for node in nodes:
        uid_map[node] = make_uid(node)

    for node in nodes:
        class_name = type(node).__name__
        if class_name == '_VertexOutputNode':
            vertex_output = node
        elif class_name == '_FragmentOutputNode':
            fragment_output = node

    vertex_code = _generate_vertex_section(graph, nodes, uid_map, vertex_output)
    fragment_code = _generate_fragment_section(graph, nodes, uid_map, fragment_output)

    return f"""Shader "{shader_name}"
{{
    Properties
    {{
        _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        _BaseMap("Base Map", 2D) = "white" {{}}
        _Metallic("Metallic", Range(0, 1)) = 0
        _Smoothness("Smoothness", Range(0, 1)) = 0.5
    }}

    SubShader
    {{
        Tags {{ "RenderType" = "Opaque" }}

        Pass
        {{
            GLSLPROGRAM
            #version 460 core
            layout(location = 0) in vec3 in_position;
            layout(location = 1) in vec3 in_normal;
            layout(location = 2) in vec2 in_uv;
            layout(location = 3) in vec4 in_model0;
            layout(location = 4) in vec4 in_model1;
            layout(location = 5) in vec4 in_model2;
            layout(location = 6) in vec4 in_model3;
            layout(std430, binding = 4) readonly buffer InstanceModels {{ mat4 _ssbo_models[]; }};
            layout(std430, binding = 5) readonly buffer InstanceIndices {{ int _ssbo_indices[]; }};
            uniform int u_use_instancing;
            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform mat3 u_normal_matrix;
            uniform float u_time;
            out vec3 v_world_pos;
            out vec3 v_normal;
            out vec2 v_uv;
            out vec3 v_view_pos;

            void main() {{
                mat4 inst_model = mat4(in_model0, in_model1, in_model2, in_model3);
                mat4 _model = (u_use_instancing == 1) ? inst_model : ((u_use_instancing == 2) ? _ssbo_models[_ssbo_indices[gl_InstanceID]] : u_model);
                mat3 _normal_matrix = (u_use_instancing >= 1) ? transpose(inverse(mat3(_model))) : u_normal_matrix;
{vertex_code}
            }}

            // @FRAGMENT

            #version 460 core
            in vec3 v_world_pos;
            in vec3 v_normal;
            in vec2 v_uv;
            in vec3 v_view_pos;
            out vec4 frag_color;
            uniform float u_time;
            uniform vec4 _BaseColor;
            uniform float _Metallic;
            uniform float _Smoothness;
            uniform sampler2D _BaseMap;
            uniform int _BaseMap_Active;
            uniform vec3 u_camera_pos;

            void main() {{
                vec3 albedo = _BaseColor.rgb;
                if (_BaseMap_Active == 1) {{
                    vec4 texColor = texture(_BaseMap, v_uv);
                    albedo *= texColor.rgb;
                }}
{fragment_code}
            }}
            ENDGLSL
        }}
    }}

    Fallback "Unlit"
}}"""


def _generate_vertex_section(graph, nodes, uid_map, vertex_output) -> str:
    lines = []
    generated = set()

    for node in nodes:
        if node == vertex_output:
            continue
        if hasattr(node, 'generate_code'):
            code = node.generate_code(uid_map)
            if code:
                lines.append(f"                {code.replace(chr(10), chr(10) + '                ')}")
                generated.add(node)

    if vertex_output and hasattr(vertex_output, 'generate_code'):
        code = vertex_output.generate_code(uid_map)
        for line in code.split('\n'):
            lines.append(f"                {line}")

    if not lines:
        lines.append("                vec4 world_pos = _model * vec4(in_position, 1.0);")
        lines.append("                v_world_pos = world_pos.xyz;")
        lines.append("                v_normal = normalize(_normal_matrix * in_normal);")
        lines.append("                v_uv = in_uv;")
        lines.append("                vec4 view_pos = u_view * world_pos;")
        lines.append("                v_view_pos = view_pos.xyz;")
        lines.append("                gl_Position = u_proj * u_view * world_pos;")

    return '\n'.join(lines)


def _generate_fragment_section(graph, nodes, uid_map, fragment_output) -> str:
    lines = []
    generated = set()

    for node in nodes:
        if node == fragment_output:
            continue
        if hasattr(node, 'generate_code'):
            code = node.generate_code(uid_map)
            if code:
                lines.append(f"                {code.replace(chr(10), chr(10) + '                ')}")
                generated.add(node)

    if fragment_output and hasattr(fragment_output, 'generate_code'):
        code = fragment_output.generate_code(uid_map)
        for line in code.split('\n'):
            lines.append(f"                {line}")

    if not lines:
        lines.append("                vec3 result = albedo;")
        lines.append("                frag_color = vec4(result, 1.0);")

    return '\n'.join(lines)


def _empty_shader(name):
    return f"""Shader "{name}"
{{
    Properties
    {{
        _BaseColor("Base Color", Color) = (1, 1, 1, 1)
    }}

    SubShader
    {{
        Tags {{ "RenderType" = "Opaque" }}

        Pass
        {{
            GLSLPROGRAM
            #version 460 core
            layout(location = 0) in vec3 in_position;
            layout(location = 1) in vec3 in_normal;
            layout(location = 2) in vec2 in_uv;
            layout(location = 3) in vec4 in_model0;
            layout(location = 4) in vec4 in_model1;
            layout(location = 5) in vec4 in_model2;
            layout(location = 6) in vec4 in_model3;
            layout(std430, binding = 4) readonly buffer InstanceModels {{ mat4 _ssbo_models[]; }};
            layout(std430, binding = 5) readonly buffer InstanceIndices {{ int _ssbo_indices[]; }};
            uniform int u_use_instancing;
            uniform mat4 u_model;
            uniform mat4 u_view;
            uniform mat4 u_proj;
            uniform mat3 u_normal_matrix;
            out vec3 v_world_pos;
            out vec3 v_normal;
            out vec2 v_uv;

            void main() {{
                mat4 inst_model = mat4(in_model0, in_model1, in_model2, in_model3);
                mat4 _model = (u_use_instancing == 1) ? inst_model : ((u_use_instancing == 2) ? _ssbo_models[_ssbo_indices[gl_InstanceID]] : u_model);
                mat3 _normal_matrix = (u_use_instancing >= 1) ? transpose(inverse(mat3(_model))) : u_normal_matrix;
                vec4 world_pos = _model * vec4(in_position, 1.0);
                v_world_pos = world_pos.xyz;
                v_normal = normalize(_normal_matrix * in_normal);
                v_uv = in_uv;
                gl_Position = u_proj * u_view * world_pos;
            }}

            // @FRAGMENT

            #version 460 core
            in vec3 v_world_pos;
            in vec3 v_normal;
            in vec2 v_uv;
            out vec4 frag_color;
            uniform vec4 _BaseColor;

            void main() {{
                frag_color = _BaseColor;
            }}
            ENDGLSL
        }}
    }}

    Fallback "Unlit"
}}"""
