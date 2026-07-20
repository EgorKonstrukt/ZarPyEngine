# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import List, Tuple, Optional
from editor.terrain_graph.nodes import HeightOutputNode
from editor.terrain_graph.glsl_chunks import ALL_GLSL_FUNCTIONS


def _topo_sort(nodes) -> List:
    visited = set()
    result = []
    node_map = {}

    for n in nodes:
        nid = id(n)
        node_map[nid] = n

    def visit(nid):
        if nid in visited:
            return
        visited.add(nid)
        for port in node_map[nid].input_ports():
            for connected in port.connected_ports():
                cn = connected.node()
                if id(cn) in node_map:
                    visit(id(cn))
        result.append(node_map[nid])

    for nid in node_map:
        visit(nid)

    return result


def generate_shader(graph, resolution: int) -> Tuple[str, dict, float]:
    nodes = list(graph.all_nodes())

    if not nodes:
        return "", {}, 0.0

    output_nodes = [n for n in nodes if isinstance(n, HeightOutputNode)]
    if not output_nodes:
        return "", {}, 0.0

    sorted_nodes = _topo_sort(nodes)

    var_map = {}
    var_counter = [0]

    def make_var():
        v = "n{}".format(var_counter[0])
        var_counter[0] += 1
        return v

    for n in sorted_nodes:
        var_name = make_var()
        var_map[id(n)] = var_name

    uniform_lines = []
    all_uniforms = {}
    for n in sorted_nodes:
        vn = var_map[id(n)]
        for line in n.get_uniforms(vn):
            uniform_lines.append("    " + line)
        all_uniforms.update(n.get_uniform_values(vn))

    code_blocks = []
    for n in sorted_nodes:
        vn = var_map[id(n)]
        block = n.get_glsl(vn, var_map)
        code_blocks.append(block)

    all_uniforms["u_resolution"] = resolution

    out_var = var_map[id(output_nodes[0])]

    code_str = "\n".join("    " + c.replace("\n", "\n    ") for c in code_blocks)

    source = (
        "#version 430\n"
        "layout(local_size_x=16, local_size_y=16, local_size_z=1) in;\n"
        "layout(std430, binding=0) buffer HeightBuffer {\n"
        "    float heights[];\n"
        "};\n"
        "\n"
        "uniform int u_resolution;\n"
        "\n"
        + "\n".join(uniform_lines) + "\n"
        "\n"
        + ALL_GLSL_FUNCTIONS + "\n"
        "\n"
        "void main() {\n"
        "    int x = int(gl_GlobalInvocationID.x);\n"
        "    int y = int(gl_GlobalInvocationID.y);\n"
        "    int res = u_resolution;\n"
        "    if (x >= res || y >= res) return;\n"
        "\n"
        "    vec2 uv = (vec2(x, y) + 0.5) / float(res);\n"
        "    vec2 p = uv;\n"
        "\n"
        + code_str + "\n"
        "    heights[y * res + x] = " + out_var + ";\n"
        "}\n"
    )

    return source, all_uniforms, output_nodes[0]._get_param("heightScale")
