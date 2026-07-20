# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from typing import Optional, Set
from editor.terrain_graph.glsl_chunks import ALL_GLSL_FUNCTIONS
from core.foundation.logger import Logger


def _collect_ancestors(target, graph_nodes) -> Set[int]:
    visited = set()
    queue = [target]
    while queue:
        node = queue.pop()
        nid = id(node)
        if nid in visited:
            continue
        visited.add(nid)
        for port in node.input_ports():
            for connected in port.connected_ports():
                cn = connected.node()
                if id(cn) in {id(n) for n in graph_nodes}:
                    queue.append(cn)
    return visited


def _topo_sort_nodes(nodes):
    visited = set()
    result = []
    node_map = {id(n): n for n in nodes}

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


def compute_node_preview(target_node, graph, resolution: int = 64) -> Optional[np.ndarray]:
    graph_nodes = list(graph.all_nodes())
    if not graph_nodes:
        return None

    ancestor_ids = _collect_ancestors(target_node, graph_nodes)
    relevant = [n for n in graph_nodes if id(n) in ancestor_ids]

    sorted_nodes = _topo_sort_nodes(relevant)

    var_map = {}
    var_counter = [0]

    def make_var():
        v = "n{}".format(var_counter[0])
        var_counter[0] += 1
        return v

    for n in sorted_nodes:
        var_map[id(n)] = make_var()

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

    target_var = var_map.get(id(target_node))
    if target_var is None:
        return None

    all_uniforms["u_resolution"] = resolution

    code_str = "\n".join("    " + c.replace("\n", "\n    ") for c in code_blocks)

    source = (
        "#version 430\n"
        "// node_preview:{}\n".format(id(target_node)) +
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
        "    heights[y * res + x] = " + target_var + ";\n"
        "}\n"
    )

    try:
        from editor.terrain_graph.gpu_runner import run_preview_shader
        hf = run_preview_shader(source, resolution, all_uniforms)
        return hf
    except Exception as e:
        Logger.warning(f"TerrainNodePreview: failed to compute preview: {e}")
        return None


def update_all_previews(graph, resolution: int = 64):
    graph_nodes = list(graph.all_nodes())
    for node in graph_nodes:
        pw = getattr(node, '_preview_widget', None)
        if pw is None:
            continue
        hf = compute_node_preview(node, graph, resolution)
        pw.set_preview(hf)
