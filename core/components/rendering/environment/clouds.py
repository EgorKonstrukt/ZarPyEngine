# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import numpy as np
import moderngl
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Mat4, Vec3
from core.components.lighting.light import Light


@ComponentRegistry.register
class Cloud(Component):
    _icon = "Cloud.png"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("material_path", "Cloud Material", FieldType.RESOURCE_PATH, file_filter="Shader (*.shader)"),
            InspectorField("coverage", "Coverage", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("density", "Density", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.01, decimals=2),
            InspectorField("speed", "Wind Speed", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.01, decimals=3),
            InspectorField("wind_direction", "Wind Direction", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
            InspectorField("height", "Height", FieldType.FLOAT, min_val=0.0, max_val=500.0, step=1.0, decimals=1),
            InspectorField("thickness", "Thickness", FieldType.FLOAT, min_val=0.0, max_val=200.0, step=1.0, decimals=1),
            InspectorField("scale", "Noise Scale", FieldType.FLOAT, min_val=0.001, max_val=1.0, step=0.001, decimals=3),
            InspectorField("opacity", "Opacity", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("shadow_strength", "Shadow Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("fog_density", "Fog Density", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("height_falloff", "Height Falloff", FieldType.FLOAT, min_val=0.0, max_val=4.0, step=0.01, decimals=3),
            InspectorField("scattering", "Scattering", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.01, decimals=2),
            InspectorField("absorption", "Absorption", FieldType.SLIDER, min_val=0.0, max_val=4.0, step=0.01, decimals=2),
            InspectorField("softness", "Softness", FieldType.SLIDER, min_val=0.02, max_val=0.8, step=0.01, decimals=2),
            InspectorField("detail_strength", "Detail Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("steps", "Volume Steps", FieldType.INT_SLIDER, min_val=4, max_val=32, step=1, decimals=0),
            InspectorField("evolution", "Evolution", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.01, decimals=2),
            InspectorField("contrast", "Contrast", FieldType.SLIDER, min_val=0.2, max_val=2.5, step=0.01, decimals=2),
            InspectorField("silver_lining", "Silver Lining", FieldType.SLIDER, min_val=0.0, max_val=2.0, step=0.01, decimals=2),
            InspectorField("seed", "Seed", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=0.1, decimals=2),
            InspectorField("tint", "Tint", FieldType.COLOR),
        ]

    def __init__(self):
        super().__init__()
        self._time_origin: float = time.time()
        self.material_path: str = "core/shaders/Clouds.shader"
        self.coverage: float = 0.78
        self.density: float = 1.12
        self.speed: float = 0.38
        self.wind_direction: float = 18.0
        self.height: float = 82.0
        self.thickness: float = 42.0
        self.scale: float = 0.014
        self.opacity: float = 0.68
        self.shadow_strength: float = 0.52
        self.fog_density: float = 0.0
        self.height_falloff: float = 0.025
        self.scattering: float = 1.05
        self.absorption: float = 0.92
        self.softness: float = 0.28
        self.detail_strength: float = 0.92
        self.steps: int = 16
        self.evolution: float = 1.1
        self.contrast: float = 1.7
        self.silver_lining: float = 1.35
        self.seed: float = 11.3
        self.tint: list[float] = [0.86, 0.92, 1.0]

    def render_clouds(self, ctx, shaders, view_mat, proj_mat, dir_light, cam_pos, quad_mesh, shadows=None, depth_tex=None, viewport_size=(1, 1)):
        cloud_prog = shaders.get_or_compile(self.material_path) if shaders else None
        if not cloud_prog:
            return
        if dir_light:
            dl, dt = dir_light
            sun_dir_obj = -dt.forward
            if "_SunDirection" in cloud_prog:
                cloud_prog["_SunDirection"].write(np.array([sun_dir_obj.x, sun_dir_obj.y, sun_dir_obj.z], dtype=np.float32).tobytes())
            if "_SunColor" in cloud_prog:
                if dl.procedural_sky_lighting:
                    sc, si = Light.compute_sun_light(-dt.forward)
                    cloud_prog["_SunColor"].write(np.array(sc, dtype=np.float32).tobytes())
                    if "_SunIntensity" in cloud_prog:
                        cloud_prog["_SunIntensity"].value = si
                else:
                    cloud_prog["_SunColor"].write(np.array(dl.color, dtype=np.float32).tobytes())
                    if "_SunIntensity" in cloud_prog:
                        cloud_prog["_SunIntensity"].value = dl.intensity
        if "u_time" in cloud_prog:
            cloud_prog["u_time"].value = time.time() - self._time_origin
        if "u_cam_pos" in cloud_prog:
            cloud_prog["u_cam_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
        vp = proj_mat * view_mat
        inv_vp = vp.inverted()
        if "u_inv_view_proj" in cloud_prog:
            cloud_prog["u_inv_view_proj"].write(inv_vp.to_f32().tobytes())
        if "u_view" in cloud_prog:
            cloud_prog["u_view"].write(view_mat.to_f32().tobytes())
        if "u_viewport_size" in cloud_prog:
            cloud_prog["u_viewport_size"].value = (float(viewport_size[0]), float(viewport_size[1]))
        if depth_tex is not None and "u_depth_tex" in cloud_prog:
            cloud_prog["u_depth_tex"] = 14
            depth_tex.use(14)
            if "u_has_depth" in cloud_prog:
                cloud_prog["u_has_depth"].value = 1
        elif "u_has_depth" in cloud_prog:
            cloud_prog["u_has_depth"].value = 0
        for prop_name in ("_Coverage", "_Density", "_Speed", "_Height", "_Thickness", "_Scale", "_Opacity"):
            if prop_name in cloud_prog:
                val = getattr(self, prop_name.lower().lstrip("_"))
                cloud_prog[prop_name].value = val
        if "_WindDirection" in cloud_prog:
            cloud_prog["_WindDirection"].value = self.wind_direction
        if "_ShadowStrength" in cloud_prog:
            cloud_prog["_ShadowStrength"].value = self.shadow_strength
        if "_FogDensity" in cloud_prog:
            cloud_prog["_FogDensity"].value = self.fog_density
        if "_HeightFalloff" in cloud_prog:
            cloud_prog["_HeightFalloff"].value = self.height_falloff
        if "_Scattering" in cloud_prog:
            cloud_prog["_Scattering"].value = self.scattering
        if "_Absorption" in cloud_prog:
            cloud_prog["_Absorption"].value = self.absorption
        if "_Softness" in cloud_prog:
            cloud_prog["_Softness"].value = self.softness
        if "_DetailStrength" in cloud_prog:
            cloud_prog["_DetailStrength"].value = self.detail_strength
        if "_Steps" in cloud_prog:
            cloud_prog["_Steps"].value = int(self.steps)
        if "_Evolution" in cloud_prog:
            cloud_prog["_Evolution"].value = self.evolution
        if "_Contrast" in cloud_prog:
            cloud_prog["_Contrast"].value = self.contrast
        if "_SilverLining" in cloud_prog:
            cloud_prog["_SilverLining"].value = self.silver_lining
        if "_Seed" in cloud_prog:
            cloud_prog["_Seed"].value = float(self.seed)
        if "_Tint" in cloud_prog:
            tint = self.tint[:3] if isinstance(self.tint, list) else [0.82, 0.90, 1.0]
            cloud_prog["_Tint"].write(np.array(tint, dtype=np.float32).tobytes())
        if shadows:
            shadows.set_uniforms(cloud_prog)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.CULL_FACE)
        ctx.disable(moderngl.DEPTH_TEST)
        quad_mesh.render(cloud_prog)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.disable(moderngl.BLEND)

    def render_cloud_layer(self, ctx, shaders, view_mat, proj_mat, dir_light, cam_pos, plane_mesh):
        if self.height < 8.0 or self.opacity <= 0.001:
            return
        cloud_prog = shaders.get_or_compile("core/shaders/CloudLayer.shader") if shaders else None
        if not cloud_prog:
            return
        if dir_light:
            dl, dt = dir_light
            sun_dir_obj = -dt.forward
            if "_SunDirection" in cloud_prog:
                cloud_prog["_SunDirection"].write(np.array([sun_dir_obj.x, sun_dir_obj.y, sun_dir_obj.z], dtype=np.float32).tobytes())
            if "_SunColor" in cloud_prog:
                if dl.procedural_sky_lighting:
                    sc, si = Light.compute_sun_light(-dt.forward)
                    cloud_prog["_SunColor"].write(np.array(sc, dtype=np.float32).tobytes())
                    if "_SunIntensity" in cloud_prog:
                        cloud_prog["_SunIntensity"].value = si
                else:
                    cloud_prog["_SunColor"].write(np.array(dl.color, dtype=np.float32).tobytes())
                    if "_SunIntensity" in cloud_prog:
                        cloud_prog["_SunIntensity"].value = dl.intensity
        if "u_time" in cloud_prog:
            cloud_prog["u_time"].value = time.time() - self._time_origin
        if "u_cam_pos" in cloud_prog:
            cloud_prog["u_cam_pos"].write(np.array([cam_pos.x, cam_pos.y, cam_pos.z], dtype=np.float32).tobytes())
        if "u_view" in cloud_prog:
            cloud_prog["u_view"].write(view_mat.to_f32().tobytes())
        if "u_proj" in cloud_prog:
            cloud_prog["u_proj"].write(proj_mat.to_f32().tobytes())
        size = 1400.0
        model = Mat4.scale(Vec3(size, 1.0, size)) * Mat4.translation(Vec3(cam_pos.x, self.height + self.thickness * 0.5, cam_pos.z))
        if "u_model" in cloud_prog:
            cloud_prog["u_model"].write(model.to_f32().tobytes())
        for prop_name in ("_Coverage", "_Density", "_Speed", "_Height", "_Thickness", "_Scale", "_Opacity"):
            if prop_name in cloud_prog:
                val = getattr(self, prop_name.lower().lstrip("_"))
                cloud_prog[prop_name].value = val
        if "_WindDirection" in cloud_prog:
            cloud_prog["_WindDirection"].value = self.wind_direction
        if "_ShadowStrength" in cloud_prog:
            cloud_prog["_ShadowStrength"].value = self.shadow_strength
        if "_Scattering" in cloud_prog:
            cloud_prog["_Scattering"].value = self.scattering
        if "_Absorption" in cloud_prog:
            cloud_prog["_Absorption"].value = self.absorption
        if "_Softness" in cloud_prog:
            cloud_prog["_Softness"].value = self.softness
        if "_DetailStrength" in cloud_prog:
            cloud_prog["_DetailStrength"].value = self.detail_strength
        if "_Evolution" in cloud_prog:
            cloud_prog["_Evolution"].value = self.evolution
        if "_Contrast" in cloud_prog:
            cloud_prog["_Contrast"].value = self.contrast
        if "_SilverLining" in cloud_prog:
            cloud_prog["_SilverLining"].value = self.silver_lining
        if "_Seed" in cloud_prog:
            cloud_prog["_Seed"].value = float(self.seed)
        if "_Tint" in cloud_prog:
            tint = self.tint[:3] if isinstance(self.tint, list) else [0.82, 0.90, 1.0]
            cloud_prog["_Tint"].write(np.array(tint, dtype=np.float32).tobytes())
        old_depth_mask = ctx.depth_mask
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.CULL_FACE)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_mask = False
        plane_mesh.render(cloud_prog)
        ctx.depth_mask = old_depth_mask
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.CULL_FACE)
        ctx.disable(moderngl.BLEND)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "material_path": self.material_path,
            "coverage": self.coverage,
            "density": self.density,
            "speed": self.speed,
            "wind_direction": self.wind_direction,
            "height": self.height,
            "thickness": self.thickness,
            "scale": self.scale,
            "opacity": self.opacity,
            "shadow_strength": self.shadow_strength,
            "fog_density": self.fog_density,
            "height_falloff": self.height_falloff,
            "scattering": self.scattering,
            "absorption": self.absorption,
            "softness": self.softness,
            "detail_strength": self.detail_strength,
            "steps": self.steps,
            "evolution": self.evolution,
            "contrast": self.contrast,
            "silver_lining": self.silver_lining,
            "seed": self.seed,
            "tint": self.tint,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Cloud:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.material_path = data.get("material_path", "core/shaders/Clouds.shader")
        c.coverage = data.get("coverage", 0.78)
        c.density = data.get("density", 1.12)
        c.speed = data.get("speed", 0.38)
        c.wind_direction = data.get("wind_direction", 18.0)
        c.height = data.get("height", 82.0)
        c.thickness = data.get("thickness", 42.0)
        c.scale = data.get("scale", 0.014)
        c.opacity = data.get("opacity", 0.68)
        c.shadow_strength = data.get("shadow_strength", 0.52)
        c.fog_density = data.get("fog_density", 0.0)
        c.height_falloff = data.get("height_falloff", 0.025)
        c.scattering = data.get("scattering", 1.05)
        c.absorption = data.get("absorption", 0.92)
        c.softness = data.get("softness", 0.28)
        c.detail_strength = data.get("detail_strength", 0.92)
        c.steps = data.get("steps", 16)
        c.evolution = data.get("evolution", 1.1)
        c.contrast = data.get("contrast", 1.7)
        c.silver_lining = data.get("silver_lining", 1.35)
        c.seed = data.get("seed", 11.3)
        c.tint = data.get("tint", [0.86, 0.92, 1.0])
        return c
