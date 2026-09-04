# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry, InstancePrimitive
from core.maths.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class CapsuleCollider(Component):
    _icon = "CapsuleCollider.png"
    _allow_multiple = True
    _gizmo_icon_color = (200, 80, 80)
    _gizmo_icon_label = "C"
    _show_gizmo_icon: bool = False
    _gizmo_pass = "collider"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("layer", "Layer", FieldType.LAYER),
            InspectorField("mask", "Collision Mask", FieldType.LAYER_MASK),
            InspectorField("radius", "Radius", FieldType.FLOAT, min_val=0.001, max_val=10000.0, step=0.01),
            InspectorField("height", "Height", FieldType.FLOAT, min_val=0.001, max_val=10000.0, step=0.01),
            InspectorField("is_trigger", "Is Trigger", FieldType.BOOL),
            InspectorField("physic_material", "Physic Material", FieldType.ASSET, resource_type="physicmaterial"),
        ]

    def __init__(self):
        super().__init__()
        self.layer: int = 0
        self.mask: int = 0xFFFF
        self.center: Vec3 = Vec3.zero()
        self.radius: float = 0.5
        self.height: float = 2.0
        self.direction: int = 1
        self.is_trigger: bool = False
        self.physic_material: str = ""
        self.material_friction: float = 0.6
        self.material_bounciness: float = 0.0
    @property
    def scaled_radius(self) -> float:
        tr = self.transform
        s = tr.local_scale if tr else Vec3.one()
        return self.radius * max(s.x, s.y, s.z)
    @property
    def scaled_height(self) -> float:
        tr = self.transform
        s = tr.local_scale if tr else Vec3.one()
        scale_val = s.y
        if self.direction == 0:
            scale_val = s.x
        elif self.direction == 2:
            scale_val = s.z
        return self.height * scale_val
    @property
    def scaled_center(self) -> Vec3:
        tr = self.transform
        s = tr.local_scale if tr else Vec3.one()
        c = self.center if isinstance(self.center, Vec3) else Vec3(*self.center)
        return Vec3(c.x * s.x, c.y * s.y, c.z * s.z)
    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "center": self.center.to_list(), "radius": self.radius,
            "height": self.height, "direction": self.direction, "is_trigger": self.is_trigger,
            "physic_material": self.physic_material,
            "layer": self.layer, "mask": self.mask,
        })
        return d
    @classmethod
    def deserialize(cls, data: dict) -> CapsuleCollider:
        cc = cls()
        cc.enabled = data.get("enabled", True)
        cc.center = Vec3(*data.get("center", [0,0,0]))
        cc.radius = data.get("radius", 0.5)
        cc.height = data.get("height", 2.0)
        cc.direction = data.get("direction", 1)
        cc.is_trigger = data.get("is_trigger", False)
        cc.physic_material = data.get("physic_material", "") or ""
        cc.layer = data.get("layer", 0)
        cc.mask = data.get("mask", 0xFFFF)
        return cc

    def gizmo_instance_data(self):
        tr = self.transform
        if not tr:
            return None
        dir_idx = self.direction
        r = self.radius
        half_h = max(0, self.height * 0.5 - r)
        sc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        sc[dir_idx] = half_h + r
        sc[(dir_idx + 1) % 3] = r
        sc[(dir_idx + 2) % 3] = r
        c = np.array([self.center.x, self.center.y, self.center.z], dtype=np.float32)
        T = np.array([tr.local_position.x, tr.local_position.y, tr.local_position.z], dtype=np.float32)
        import math as m
        q = tr.local_rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        n = m.sqrt(x*x + y*y + z*z + w*w)
        if n > 1e-10:
            inv = 1.0/n; x *= inv; y *= inv; z *= inv; w *= inv
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                       [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                       [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]], dtype=np.float32)
        S = np.array([tr.local_scale.x, tr.local_scale.y, tr.local_scale.z], dtype=np.float32)
        RS = R * S
        combined = np.eye(4, dtype=np.float32)
        combined[:3, :3] = RS * sc
        combined[:3, 3] = RS @ c + T
        return InstancePrimitive('capsule', combined.ravel('F'), [0.0, 1.0, 0.0, 0.6])
