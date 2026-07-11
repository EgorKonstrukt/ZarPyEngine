# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from core.math.math3d import Mat4, Vec3
from core.renderer.mesh_data import MeshData


class SkyboxRenderer:

    def __init__(self, ctx: moderngl.Context, skybox_prog: moderngl.Program, skybox_cube: MeshData):
        self._ctx = ctx
        self._prog = skybox_prog
        self._cube = skybox_cube
        self._sun_size: float = 0.0008
        self._sun_convergence: float = 0.5
        self._sun_direction: np.ndarray = np.array([0.0, -0.3, -1.0], dtype=np.float32)
        self._sun_color: np.ndarray = np.array([1.0, 0.95, 0.85], dtype=np.float32)
        self._sun_intensity: float = 1.0
        self._enabled: bool = True

    def set_sun_from_light(self, light_dir: Vec3, light_color: list[float], light_intensity: float):
        self._sun_direction = np.array([-light_dir.x, -light_dir.y, -light_dir.z], dtype=np.float32)
        self._sun_color = np.array(light_color, dtype=np.float32)
        self._sun_intensity = light_intensity

    def render(self, view_mat: Mat4, proj_mat: Mat4):
        if not self._prog or not self._cube or not self._enabled:
            return
        prog = self._prog
        rot = view_mat._d[:3, :3].copy()
        sky_view = np.eye(4, dtype=np.float64)
        sky_view[:3, :3] = rot
        sky_mat4 = Mat4(sky_view)
        mvp = sky_mat4 * proj_mat
        if "u_mvp" in prog:
            prog["u_mvp"].write(mvp.to_f32().tobytes())

        sun_dir = self._sun_direction

        if "u_sun_direction" in prog:
            prog["u_sun_direction"].write(sun_dir.tobytes())
        if "u_sun_color" in prog:
            prog["u_sun_color"].write(self._sun_color.tobytes())
        if "u_sun_intensity" in prog:
            prog["u_sun_intensity"].value = self._sun_intensity
        if "u_sun_size" in prog:
            prog["u_sun_size"].value = self._sun_size
        if "u_sun_convergence" in prog:
            prog["u_sun_convergence"].value = self._sun_convergence

        self._ctx.disable(moderngl.CULL_FACE)
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._cube.render(prog)
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.enable(moderngl.CULL_FACE)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, v: bool):
        self._enabled = v


