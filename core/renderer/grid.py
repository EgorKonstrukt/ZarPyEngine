# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
import moderngl
from typing import Optional
from core.maths.math3d import Vec3


class GridRenderer:
    _TARGET_PIXEL_SPACING = 500.0
    _FADE_STEP_MIN_PX = 8.0
    _FADE_STEP_RANGE_PX = 100.0
    _MAJOR_MULT = 5.0
    _SUPER_MULT = 25.0

    def __init__(self, ctx: moderngl.Context, grid_prog: moderngl.Program):
        self._ctx = ctx
        self._prog = grid_prog
        self._grid_vbo: Optional[moderngl.Buffer] = None
        self._grid_ibo: Optional[moderngl.Buffer] = None
        self._grid_vao: Optional[moderngl.VertexArray] = None
        self._show_grid: bool = True
        self._grid_world_size: float = 2000.0
        self._grid_2d_mode: bool = False
        self._grid_zoom_distance: float = 5.0
        self._grid_size: float = 1.0
        self._grid_opacity: float = 0.33
        self._build()

    def _build(self):
        s = self._grid_world_size
        v = np.array([-s, 0.0, -s,  s, 0.0, -s,  s, 0.0, s,  -s, 0.0, s], dtype=np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        self._grid_vbo = self._ctx.buffer(v.tobytes(), dynamic=True)
        self._grid_ibo = self._ctx.buffer(idx.tobytes())
        self._grid_vao = self._ctx.vertex_array(
            self._prog,
            [(self._grid_vbo, "3f", "in_position")],
            self._grid_ibo
        )

    def _nice_step(self, raw: float) -> float:
        n = max(1e-9, raw / self._grid_size)
        exp = round(math.log(n) / math.log(4))
        return (4.0 ** exp) * self._grid_size

    def _alpha_from_px(self, step_px: float) -> float:
        return max(0.0, min(1.0, (step_px - self._FADE_STEP_MIN_PX) / self._FADE_STEP_RANGE_PX))

    def compute_grid_step(self, cam_pos: Vec3, viewport_h: int = 1080, fov: float = 60.0):
        if self._grid_2d_mode:
            height = max(self._grid_zoom_distance, 0.01)
        else:
            height = max(abs(cam_pos.y), 0.01)
        scaling = viewport_h / (2.0 * height * math.tan(math.radians(fov) * 0.5))
        target_world = self._TARGET_PIXEL_SPACING / max(scaling, 1e-10)
        minor_world = self._nice_step(max(target_world, 1e-10))
        minor_px = minor_world * scaling
        major_world = minor_world * self._MAJOR_MULT
        major_px = major_world * scaling
        super_world = minor_world * self._SUPER_MULT
        super_px = super_world * scaling
        return minor_world, minor_px, major_world, major_px, super_world, super_px

    def render(self, view_f32: np.ndarray, proj_f32: np.ndarray, cam_pos: Vec3,
               clear_color: Optional[list] = None, viewport_h: int = 1080, fov: float = 60.0):
        if not self._show_grid:
            return
        s = self._grid_world_size
        if self._grid_2d_mode:
            v = np.array([-s, -s, 0.0,  s, -s, 0.0,  s, s, 0.0,  -s, s, 0.0], dtype=np.float32)
        else:
            v = np.array([-s, 0.0, -s,  s, 0.0, -s,  s, 0.0, s,  -s, 0.0, s], dtype=np.float32)
        self._grid_vbo.write(v.tobytes())
        minor_w, minor_px, major_w, major_px, super_w, super_px = self.compute_grid_step(
            cam_pos, viewport_h, fov
        )

        if "u_view" in self._prog:
            self._prog["u_view"].write(view_f32.tobytes())
        if "u_proj" in self._prog:
            self._prog["u_proj"].write(proj_f32.tobytes())
        if "u_camera_pos" in self._prog:
            self._prog["u_camera_pos"].write(
                np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes()
            )
        if "u_grid_size" in self._prog:
            self._prog["u_grid_size"].value = float(minor_w)
        if "u_grid_alpha_minor" in self._prog:
            self._prog["u_grid_alpha_minor"].value = float(self._alpha_from_px(minor_px))
        if "u_grid_alpha_major" in self._prog:
            self._prog["u_grid_alpha_major"].value = float(self._alpha_from_px(major_px))
        if "u_grid_alpha_super" in self._prog:
            self._prog["u_grid_alpha_super"].value = float(self._alpha_from_px(super_px))
        if "u_grid_2d" in self._prog:
            self._prog["u_grid_2d"].value = 1.0 if self._grid_2d_mode else 0.0
        if "u_grid_step_major" in self._prog:
            self._prog["u_grid_step_major"].value = float(self._MAJOR_MULT)
        if "u_grid_step_super" in self._prog:
            self._prog["u_grid_step_super"].value = float(self._SUPER_MULT)
        if "u_grid_opacity" in self._prog:
            self._prog["u_grid_opacity"].value = float(self._grid_opacity)
        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.depth_mask = False
        self._grid_vao.render(moderngl.TRIANGLES)
        self._ctx.depth_mask = True
        self._ctx.enable(moderngl.CULL_FACE)

    @property
    def show(self) -> bool:
        return self._show_grid

    @show.setter
    def show(self, v: bool):
        self._show_grid = v

    @property
    def grid_2d_mode(self) -> bool:
        return self._grid_2d_mode

    @grid_2d_mode.setter
    def grid_2d_mode(self, v: bool):
        self._grid_2d_mode = v

    @property
    def grid_zoom_distance(self) -> float:
        return self._grid_zoom_distance

    @grid_zoom_distance.setter
    def grid_zoom_distance(self, v: float):
        self._grid_zoom_distance = v

    @property
    def grid_size(self) -> float:
        return self._grid_size

    @grid_size.setter
    def grid_size(self, v: float):
        self._grid_size = max(0.01, v)
