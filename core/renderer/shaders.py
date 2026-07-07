# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import re
import moderngl
from typing import Optional
from core.logger import Logger
from core.renderer.mesh_data import SHADER_DIR

_ENGINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_shader_path(shader_path: str) -> str:
    if os.path.isabs(shader_path) or os.path.exists(shader_path):
        return shader_path
    candidates = [
        os.path.join(_ENGINE_ROOT, shader_path),
        os.path.join(_ENGINE_ROOT, "core", "shaders", shader_path),
    ]
    shader_name = os.path.basename(shader_path)
    if shader_name != shader_path:
        candidates.append(os.path.join(_ENGINE_ROOT, "core", "shaders", shader_name))
    for c in candidates:
        if os.path.exists(c):
            return c
    return shader_path


class ShaderManager:
    """Compiles, caches and retrieves shader programs."""

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._cache: dict[str, Optional[moderngl.Program]] = {}

    def get_or_compile(self, shader_path: str) -> Optional[moderngl.Program]:
        if not shader_path:
            return None
        if shader_path in self._cache:
            return self._cache[shader_path]
        ext = os.path.splitext(shader_path)[1].lower()
        if ext == ".shader":
            return self._compile_shader_file(shader_path)
        return self._compile_vert_frag(shader_path)

    def _compile_vert_frag(self, shader_path: str) -> Optional[moderngl.Program]:
        try:
            base = os.path.join(SHADER_DIR, shader_path)
            vert_file = f"{base}.vert"
            frag_file = f"{base}.frag"
            if not os.path.exists(vert_file):
                vert_file = f"{base}_vert.glsl"
            if not os.path.exists(frag_file):
                frag_file = f"{base}_frag.glsl"
            with open(vert_file, "r") as f:
                vert_src = f.read()
            with open(frag_file, "r") as f:
                frag_src = f.read()
            vert_src = self._inject_instancing_vertex(vert_src)
            frag_src = self._inject_area_shadows(frag_src)
            prog = self._ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            self._cache[shader_path] = prog
            return prog
        except Exception as e:
            Logger.error(f"Failed to compile shader '{shader_path}': {e}", e)
            self._cache[shader_path] = None
            return None

    def _compile_shader_file(self, shader_path: str) -> Optional[moderngl.Program]:
        """Compile a .shader file containing GLSLPROGRAM...ENDGLSL blocks."""
        from core.material import _extract_glsl_from_shader
        resolved = _resolve_shader_path(shader_path)
        try:
            with open(resolved, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            Logger.error(f"Failed to read '{shader_path}': {e}", e)
            self._cache[shader_path] = None
            return None
        result = _extract_glsl_from_shader(text)
        if not result:
            Logger.error(f"No valid GLSL found in '{shader_path}'")
            self._cache[shader_path] = None
            return None
        vert_src, frag_src = result
        vert_src = self._inject_instancing_vertex(vert_src)
        frag_src = self._inject_area_shadows(frag_src)
        try:
            prog = self._ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
            self._cache[shader_path] = prog
            return prog
        except Exception as e:
            Logger.error(f"Failed to compile shader '{shader_path}': {e}", e)
            self._cache[shader_path] = None
            return None

    @staticmethod
    def _inject_instancing_vertex(src: str) -> str:
        if "in_model0" in src:
            return src
        src = src.replace("uniform mat4 u_model;", "")
        src = src.replace("uniform mat3 u_normal_matrix;", "")
        src = re.sub(r'\bu_model\b', '_resolve_model()', src)
        src = re.sub(r'\bu_normal_matrix\b', '_resolve_normal_matrix()', src)
        injection = """layout(location = 3) in vec4 in_model0;
layout(location = 4) in vec4 in_model1;
layout(location = 5) in vec4 in_model2;
layout(location = 6) in vec4 in_model3;
layout(std430, binding = 4) readonly buffer InstanceModels { mat4 _ssbo_models[]; };
layout(std430, binding = 5) readonly buffer InstanceIndices { int _ssbo_indices[]; };
uniform int u_use_instancing;
uniform mat4 u_model;
uniform mat3 u_normal_matrix;
mat4 _resolve_model() {
    if (u_use_instancing == 1) return mat4(in_model0, in_model1, in_model2, in_model3);
    if (u_use_instancing == 2) return _ssbo_models[_ssbo_indices[gl_InstanceID]];
    return u_model;
}
mat3 _resolve_normal_matrix() {
    if (u_use_instancing >= 1) return transpose(inverse(mat3(_resolve_model())));
    return u_normal_matrix;
}

"""
        idx = src.find("\n")
        if idx >= 0:
            return src[:idx+1] + injection + src[idx+1:]
        return injection + src

    @staticmethod
    def _inject_area_shadows(src: str) -> str:
        marker = "// @SHADOW_INCLUDE"
        if marker not in src:
            return src
        include_path = os.path.join(_ENGINE_ROOT, "core", "shaders", "area_shadows.glsl")
        try:
            with open(include_path, "r", encoding="utf-8") as f:
                include_src = f.read()
        except Exception as e:
            Logger.warning(f"Failed to read area_shadows.glsl: {e}")
            return src.replace(marker, "// area shadows include failed to load")
        return src.replace(marker, include_src)

    def store(self, key: str, prog: moderngl.Program):
        self._cache[key] = prog

    def release(self):
        for prog in self._cache.values():
            if prog:
                try:
                    prog.release()
                except Exception:
                    pass
        self._cache.clear()
