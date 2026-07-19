# This Source Code Form is subject to terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import numpy as np
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry, GizmoPrimitive, InstancePrimitive
from core.math.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class TerrainCollider(Component):
    _icon = "TerrainCollider.png"
    _gizmo_icon_color = (80, 200, 120)
    _gizmo_icon_label = "T"
    _show_gizmo_icon: bool = False
    _gizmo_pass = "collider"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("layer", "Layer", FieldType.LAYER),
            InspectorField("mask", "Collision Mask", FieldType.LAYER_MASK),
            InspectorField("size", "World Size", FieldType.VEC3),
            InspectorField("height_scale", "Height Scale", FieldType.FLOAT),
            InspectorField("resolution", "Resolution", FieldType.INT),
            InspectorField("center", "Center", FieldType.VEC3),
            InspectorField("is_trigger", "Is Trigger", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.layer: int = 0
        self.mask: int = 0xFFFF
        self.center: Vec3 = Vec3.zero()
        self.size: Vec3 = Vec3(1000.0, 60.0, 1000.0)
        self.height_scale: float = 60.0
        self.resolution: int = 256
        self.is_trigger: bool = False
        self.material_friction: float = 0.9
        self.material_bounciness: float = 0.0
        self._height_data: Optional[np.ndarray] = None

    @property
    def height_data(self) -> Optional[np.ndarray]:
        return self._height_data

    @height_data.setter
    def height_data(self, value: Optional[np.ndarray]):
        self._height_data = value

    def set_height_data(self, data: np.ndarray):
        self._height_data = data
        if data is not None and data.ndim == 2:
            self.resolution = int(data.shape[0])

    def cell_size(self) -> tuple[float, float]:
        sx = self.size.x / max(1, self.resolution - 1) if self.resolution > 1 else self.size.x
        sz = self.size.z / max(1, self.resolution - 1) if self.resolution > 1 else self.size.z
        return float(sx), float(sz)

    @property
    def scaled_center(self) -> Vec3:
        tr = self.transform
        s = tr.local_scale if tr else Vec3.one()
        c = self.center if isinstance(self.center, Vec3) else Vec3(*self.center)
        return Vec3(c.x * s.x, c.y * s.y, c.z * s.z)

    def gizmo_instance_data(self):
        tr = self.transform
        if not tr:
            return None
        import numpy as np
        import math as m
        c = np.array([self.center.x, self.center.y, self.center.z], dtype=np.float32)
        h = np.array([self.size.x * 0.5, self.size.y * 0.5, self.size.z * 0.5], dtype=np.float32)
        T = np.array([tr.local_position.x, tr.local_position.y, tr.local_position.z], dtype=np.float32)
        q = tr.local_rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        n = m.sqrt(x * x + y * y + z * z + w * w)
        if n > 1e-10:
            inv = 1.0 / n
            x *= inv
            y *= inv
            z *= inv
            w *= inv
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                       [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                       [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], dtype=np.float32)
        S = np.array([tr.local_scale.x, tr.local_scale.y, tr.local_scale.z], dtype=np.float32)
        RS = R * S
        combined = np.eye(4, dtype=np.float32)
        combined[:3, :3] = RS * h
        combined[:3, 3] = RS @ c + T
        return InstancePrimitive('box', combined.ravel('F'), [0.0, 1.0, 0.0, 0.4])

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "center": self.center.to_list(),
            "size": self.size.to_list(),
            "height_scale": self.height_scale,
            "resolution": self.resolution,
            "is_trigger": self.is_trigger,
            "friction": self.material_friction,
            "bounciness": self.material_bounciness,
            "layer": self.layer,
            "mask": self.mask,
        })
        if self._height_data is not None and self._height_data.ndim == 2:
            d["height_data"] = self._height_data.tolist()
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TerrainCollider:
        tc = cls()
        tc.enabled = data.get("enabled", True)
        tc.center = Vec3(*data.get("center", [0, 0, 0]))
        tc.size = Vec3(*data.get("size", [1000.0, 60.0, 1000.0]))
        tc.height_scale = data.get("height_scale", 60.0)
        tc.resolution = data.get("resolution", 256)
        tc.is_trigger = data.get("is_trigger", False)
        tc.material_friction = data.get("friction", 0.9)
        tc.material_bounciness = data.get("bounciness", 0.0)
        tc.layer = data.get("layer", 0)
        tc.mask = data.get("mask", 0xFFFF)
        hd = data.get("height_data")
        if hd:
            try:
                arr = np.array(hd, dtype=np.float32)
                if arr.ndim == 2:
                    tc._height_data = arr
            except Exception:
                pass
        return tc
