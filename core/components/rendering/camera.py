# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from enum import Enum
from core.ecs import Component, ComponentRegistry
from core.math3d import Mat4, Vec3
from core.components.inspector_meta import FieldType, InspectorField, ComponentInspectorMeta
class CameraProjection(Enum):
    PERSPECTIVE = "perspective"
    ORTHOGRAPHIC = "orthographic"

class CameraResolutionMode(Enum):
    NATIVE = "native"
    CUSTOM = "custom"
@ComponentRegistry.register
class Camera(Component):
    _icon = "Camera.png"
    _gizmo_pass = "camera"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("fov", "FOV", FieldType.FLOAT, min_val=1.0, max_val=179.0, step=1.0, decimals=1),
            InspectorField("near", "Near", FieldType.FLOAT, min_val=0.001, max_val=100.0, step=0.01),
            InspectorField("far", "Far", FieldType.FLOAT, min_val=0.1, max_val=100000.0, step=1.0, decimals=1),
            InspectorField("projection", "Projection", FieldType.ENUM, enum_class=CameraProjection),
            InspectorField("ortho_size", "Ortho Size", FieldType.FLOAT, min_val=0.001, max_val=1000.0),
            InspectorField("depth", "Depth", FieldType.INT, min_val=-100, max_val=100),
            InspectorField("render_scale", "Render Resolution", FieldType.FLOAT, min_val=0.1, max_val=1.0, step=0.05, decimals=2),
            InspectorField("resolution_mode", "Resolution Mode", FieldType.ENUM, enum_class=CameraResolutionMode),
            InspectorField("resolution_w", "Viewport Width", FieldType.INT, min_val=1, max_val=16384, step=1),
            InspectorField("resolution_h", "Viewport Height", FieldType.INT, min_val=1, max_val=16384, step=1),
        ]

    def __init__(self):
        super().__init__()
        self.fov: float = 60.0
        self.near: float = 0.01
        self.far: float = 1000.0
        self.projection: CameraProjection = CameraProjection.PERSPECTIVE
        self.ortho_size: float = 5.0
        self.clear_color: list[float] = [0.15, 0.15, 0.15, 1.0]
        self.depth: int = 0
        self.render_scale: float = 1.0
        self.resolution_mode: CameraResolutionMode = CameraResolutionMode.NATIVE
        self.resolution_w: int = 1920
        self.resolution_h: int = 1080
    def get_projection_matrix(self, aspect: float) -> Mat4:
        if self.projection == CameraProjection.PERSPECTIVE:
            return Mat4.perspective(self.fov, aspect, self.near, self.far)
        else:
            hw = self.ortho_size * aspect
            return Mat4.orthographic(-hw, hw, -self.ortho_size, self.ortho_size, self.near, self.far)
    def get_view_matrix(self) -> Mat4:
        t = self.transform
        if not t: return Mat4.identity()
        pos = t.position
        fwd = t.forward
        up = t.up
        return Mat4.look_at(pos, pos + fwd, up)
    def compute_render_size(self, display_w: int, display_h: int) -> tuple[int, int]:
        if self.resolution_mode == CameraResolutionMode.CUSTOM:
            return max(1, int(self.resolution_w)), max(1, int(self.resolution_h))
        scale = max(0.05, min(1.0, self.render_scale))
        return max(1, int(round(display_w * scale))), max(1, int(round(display_h * scale)))
    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "fov": self.fov, "near": self.near, "far": self.far,
            "projection": self.projection.value, "ortho_size": self.ortho_size,
            "clear_color": self.clear_color, "depth": self.depth,
            "render_scale": self.render_scale,
            "resolution_mode": self.resolution_mode.value,
            "resolution_w": self.resolution_w, "resolution_h": self.resolution_h,
        })
        return d
    @classmethod
    def deserialize(cls, data: dict) -> Camera:
        c = cls()
        c.enabled = data.get("enabled", True)
        c.fov = data.get("fov", 60.0)
        c.near = data.get("near", 0.01)
        c.far = data.get("far", 1000.0)
        c.projection = CameraProjection(data.get("projection", "perspective"))
        c.ortho_size = data.get("ortho_size", 5.0)
        c.clear_color = data.get("clear_color", [0.15, 0.15, 0.15, 1.0])
        c.depth = data.get("depth", 0)
        c.render_scale = float(data.get("render_scale", 1.0))
        c.resolution_mode = CameraResolutionMode(data.get("resolution_mode", "native"))
        c.resolution_w = int(data.get("resolution_w", 1920))
        c.resolution_h = int(data.get("resolution_h", 1080))
        return c

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        tr = self.transform
        if not tr:
            return []
        near = self.near
        far = self.far
        aspect = 16.0 / 9.0
        if self.projection == CameraProjection.ORTHOGRAPHIC:
            half_w = self.ortho_size * aspect * 0.5
            half_h = self.ortho_size * 0.5
            half_w_near = half_w_far = half_w
            half_h_near = half_h_far = half_h
        else:
            fov_rad = math.radians(self.fov)
            half_h_near = math.tan(fov_rad * 0.5) * near
            half_w_near = half_h_near * aspect
            half_h_far = math.tan(fov_rad * 0.5) * far
            half_w_far = half_h_far * aspect
        pos = tr.position
        fwd = tr.forward
        up = tr.up
        right = tr.right
        def local_to_world(lx: float, ly: float, lz: float) -> Vec3:
            return pos + right * lx + up * ly + fwd * lz
        corners = [
            local_to_world(-half_w_near,  half_h_near, near),
            local_to_world( half_w_near,  half_h_near, near),
            local_to_world( half_w_near, -half_h_near, near),
            local_to_world(-half_w_near, -half_h_near, near),
            local_to_world(-half_w_far,  half_h_far,  far),
            local_to_world( half_w_far,  half_h_far,  far),
            local_to_world( half_w_far, -half_h_far,  far),
            local_to_world(-half_w_far, -half_h_far,  far),
        ]
        color = [0.3, 0.7, 1.0, 0.6]
        lines: list[tuple[Vec3, Vec3, list[float]]] = []
        near_idx = [0, 1, 2, 3]
        far_idx = [4, 5, 6, 7]
        for i in range(4):
            j = (i + 1) % 4
            lines.append((corners[near_idx[i]], corners[near_idx[j]], color))
            lines.append((corners[far_idx[i]], corners[far_idx[j]], color))
            lines.append((corners[near_idx[i]], corners[far_idx[i]], color))
        return lines