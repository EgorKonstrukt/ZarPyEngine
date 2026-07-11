# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import sys
import numpy as np
import moderngl
from typing import Optional, Any
from core.foundation.logger import Logger

# Compute shader directory relative to project root or executable.
# In a Nuitka build, __file__ may point to the source path (development)
# or inside the dist. We try both: source path first (dev), then exe path (dist).
_SHADER_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "editor", "shaders"),
    os.path.join(os.path.dirname(sys.executable), "editor", "shaders"),
]
SHADER_DIR = next((p for p in _SHADER_CANDIDATES if os.path.isdir(p)), _SHADER_CANDIDATES[0])


def read_shader(name: str) -> str:
    path = os.path.join(SHADER_DIR, name)
    with open(path, "r") as f:
        return f.read()


class MeshData:
    """GPU-ready mesh with vertex buffers, index buffers and VAO cache."""

    def __init__(self):
        self.vertices: np.ndarray = np.array([], dtype=np.float32)
        self.indices: np.ndarray = np.array([], dtype=np.uint32)
        self.normals: np.ndarray = np.array([], dtype=np.float32)
        self.uvs: np.ndarray = np.array([], dtype=np.float32)
        self.aabb_min: np.ndarray = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        self.aabb_max: np.ndarray = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.is_error_mesh: bool = False
        self.sub_mesh_ranges: list[tuple[int, int]] = []
        self._vao: Optional[Any] = None
        self._vbo: Optional[Any] = None
        self._ibo: Optional[Any] = None
        self._outline_vao: Optional[Any] = None
        self._outline_vbo: Optional[Any] = None
        self._ctx: Optional[Any] = None
        self._vao_cache: dict[int, Any] = {}
        self._bounding_radius: Optional[float] = None

    def compute_aabb(self):
        if len(self.vertices) < 3:
            self._bounding_radius = 0.0
            return
        v = self.vertices.reshape(-1, 3)
        self.aabb_min = v.min(axis=0).astype(np.float32)
        self.aabb_max = v.max(axis=0).astype(np.float32)
        self._bounding_radius = float(np.linalg.norm(self.aabb_max - self.aabb_min) / 2.0)

    @property
    def bounding_radius(self) -> float:
        if self._bounding_radius is None:
            self.compute_aabb()
        return self._bounding_radius

    def build_gl(self, ctx: moderngl.Context, program: moderngl.Program):
        self._ctx = ctx
        if len(self.vertices) == 0:
            return
        n_verts = len(self.vertices) // 3
        data = np.zeros((n_verts, 8), dtype=np.float32)
        data[:, 0:3] = self.vertices.reshape(-1, 3)
        if len(self.normals) == len(self.vertices):
            data[:, 3:6] = self.normals.reshape(-1, 3)
        if len(self.uvs) * 3 == len(self.vertices) * 2:
            data[:, 6:8] = self.uvs.reshape(-1, 2)
        if self._vbo is None:
            self._vbo = ctx.buffer(data.tobytes())
        else:
            self._vbo.write(data.tobytes())
        if len(self.indices) > 0:
            if self._ibo is None:
                self._ibo = ctx.buffer(self.indices.astype(np.uint32).tobytes())
        self._build_vao_for_program(program)
        self._vao = self._vao_cache.get(id(program))

    def _build_vao_for_program(self, program: moderngl.Program):
        if self._ctx is None or self._vbo is None:
            return
        key = id(program)
        if key in self._vao_cache:
            return
        has_pos = "in_position" in program
        has_nrm = "in_normal" in program
        has_uv = "in_uv" in program
        fmt_parts = []
        attrib_names = []
        if has_pos:
            fmt_parts.append("3f")
            attrib_names.append("in_position")
        else:
            fmt_parts.append("3x4")
        if has_nrm:
            fmt_parts.append("3f")
            attrib_names.append("in_normal")
        else:
            fmt_parts.append("3x4")
        if has_uv:
            fmt_parts.append("2f")
            attrib_names.append("in_uv")
        else:
            fmt_parts.append("2x4")
        fmt = " ".join(fmt_parts)
        self._vao_cache[key] = self._ctx.vertex_array(
            program,
            [(self._vbo, fmt, *attrib_names)],
            self._ibo
        )

    def render(self, program: Optional[moderngl.Program] = None):
        if program is not None:
            self._build_vao_for_program(program)
            vao = self._vao_cache.get(id(program))
        else:
            vao = self._vao
        if vao:
            try:
                vao.render()
            except Exception as e:
                Logger.error("MeshData.render VAO render failed", e)

    def render_range(self, program: moderngl.Program, start: int, count: int):
        self._build_vao_for_program(program)
        vao = self._vao_cache.get(id(program))
        if vao:
            try:
                vao.render(vertices=count, first=start)
            except Exception as e:
                Logger.error("MeshData.render_range VAO render failed", e)

    def build_outline_vao(self, ctx: moderngl.Context, program: moderngl.Program):
        if len(self.vertices) == 0:
            return
        pos_data = self.vertices.copy()
        if self._outline_vbo is None:
            self._outline_vbo = ctx.buffer(pos_data.tobytes())
        else:
            self._outline_vbo.write(pos_data.tobytes())
        self._outline_vao = ctx.vertex_array(
            program,
            [(self._outline_vbo, "3f", "in_position")],
            self._ibo
        )

    def render_outline(self):
        if self._outline_vao:
            try:
                self._outline_vao.render()
            except Exception as e:
                Logger.error("MeshData.render_outline VAO render failed", e)

    def release(self):
        if self._vbo:
            self._vbo.release()
        if self._ibo:
            self._ibo.release()
        if self._outline_vao:
            self._outline_vao.release()
        if self._outline_vbo:
            self._outline_vbo.release()
        for v in self._vao_cache.values():
            if v:
                try:
                    v.release()
                except Exception:
                    pass
        self._vao_cache.clear()
        self._vao = None
        self._vbo = None
        self._ibo = None
        self._outline_vao = None
