# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry, InstancePrimitive
from core.maths.math3d import Vec2, Vec3
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class BoxCollider2D(Component):
    _icon = "BoxCollider2D.png"
    _gizmo_icon_color = (200, 80, 80)
    _gizmo_icon_label = "C2"
    _show_gizmo_icon: bool = False
    _gizmo_pass = "collider"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("layer", "Layer", FieldType.LAYER),
            InspectorField("mask", "Collision Mask", FieldType.LAYER_MASK),
            InspectorField("offset", "Offset", FieldType.VEC2),
            InspectorField("size", "Size", FieldType.VEC2),
            InspectorField("is_trigger", "Is Trigger", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.layer: int = 0
        self.mask: int = 0xFFFF
        self.offset: Vec2 = Vec2.zero()
        self.size: Vec2 = Vec2.one()
        self.is_trigger: bool = False
        self.material_friction: float = 0.6
        self.material_bounciness: float = 0.0

    @property
    def scaled_size(self) -> Vec2:
        tr = self.transform
        s = tr.local_scale if tr else Vec2.one()
        sz = self.size if isinstance(self.size, Vec2) else Vec2(self.size.x, self.size.y)
        return Vec2(sz.x * s.x, sz.y * s.y)

    @property
    def scaled_offset(self) -> Vec2:
        tr = self.transform
        s = tr.local_scale if tr else Vec2.one()
        o = self.offset if isinstance(self.offset, Vec2) else Vec2(self.offset.x, self.offset.y)
        return Vec2(o.x * s.x, o.y * s.y)

    def gizmo_instance_data(self):
        tr = self.transform
        if not tr:
            return None
        hx = self.size.x * 0.5; hy = self.size.y * 0.5
        off = np.array([self.offset.x, self.offset.y, 0.0], dtype=np.float32)
        T = np.array([tr.local_position.x, tr.local_position.y, tr.local_position.z], dtype=np.float32)
        q = tr.local_rotation
        import math as m
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
        combined[:3, :3] = RS * np.array([hx, hy, 1.0], dtype=np.float32)
        combined[:3, 3] = RS @ off + T
        return InstancePrimitive('rect', combined.ravel('F'), [0.0, 1.0, 0.0, 0.6])

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "offset": self.offset.to_list(), "size": self.size.to_list(),
            "is_trigger": self.is_trigger, "friction": self.material_friction,
            "bounciness": self.material_bounciness,
            "layer": self.layer, "mask": self.mask,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> BoxCollider2D:
        bc = cls()
        bc.enabled = data.get("enabled", True)
        bc.offset = Vec2(*data.get("offset", [0, 0]))
        bc.size = Vec2(*data.get("size", [1, 1]))
        bc.is_trigger = data.get("is_trigger", False)
        bc.material_friction = data.get("friction", 0.6)
        bc.material_bounciness = data.get("bounciness", 0.0)
        bc.layer = data.get("layer", 0)
        bc.mask = data.get("mask", 0xFFFF)
        return bc
