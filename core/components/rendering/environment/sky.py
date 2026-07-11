# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
import moderngl
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.math.math3d import Mat4
from core.components.lighting.light import Light


@ComponentRegistry.register
class Sky(Component):
    _icon = "Sky.png"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("material_path", "Sky Shader", FieldType.RESOURCE_PATH, file_filter="Shader (*.shader)"),
        ]

    def __init__(self):
        super().__init__()
        self.material_path: str = "core/shaders/Sky.shader"

    def render_sky(self, ctx, shaders, view_mat, proj_mat, dir_light, cube_mesh):
        prog = shaders.get_or_compile(self.material_path) if shaders else None
        if not prog:
            return
        if dir_light:
            dl, dt = dir_light
            if dl.procedural_sky_lighting:
                sun_to = -dt.forward
                sky_color, sky_intensity = Light.compute_sun_light(sun_to)
            else:
                sky_color, sky_intensity = dl.color, dl.intensity
            if "_SunDirection" in prog:
                sun_dir = -dt.forward
                prog["_SunDirection"].write(np.array([sun_dir.x, sun_dir.y, sun_dir.z], dtype=np.float32).tobytes())
            if "_SunColor" in prog:
                prog["_SunColor"].write(np.array(sky_color, dtype=np.float32).tobytes())
            if "_SunIntensity" in prog:
                prog["_SunIntensity"].value = sky_intensity
            if "_SunSize" in prog:
                prog["_SunSize"].value = 0.0008
            if "_SunConvergence" in prog:
                prog["_SunConvergence"].value = 0.5
        sky_view = np.eye(4, dtype=np.float64)
        sky_view[:3, :3] = view_mat._d[:3, :3].copy()
        sky_mat4 = Mat4(sky_view)
        mvp = sky_mat4 * proj_mat
        if "u_mvp" in prog:
            prog["u_mvp"].write(mvp.to_f32().tobytes())
        ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        cube_mesh.render(prog)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)

    def serialize(self) -> dict:
        d = super().serialize()
        d["material_path"] = self.material_path
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Sky:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.material_path = data.get("material_path", "core/shaders/Sky.shader")
        return c
