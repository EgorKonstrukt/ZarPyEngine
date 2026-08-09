# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import enum
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3


class JointType(enum.Enum):
    HINGE = "hinge"
    FIXED = "fixed"
    SPRING = "spring"


@ComponentRegistry.register
class Joint(Component):
    _icon = "Joint.png"
    _gizmo_icon_color = (180, 100, 200)
    _gizmo_icon_label = "J"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("connected_entity_name", "Connected Entity", FieldType.STRING),
            InspectorField("anchor", "Anchor", FieldType.VEC3),
            InspectorField("axis", "Axis", FieldType.VEC3),
            InspectorField("joint_type", "Joint Type", FieldType.STRING),
            InspectorField("limit_low", "Limit Low", FieldType.FLOAT, min_val=-3.14159, max_val=0.0, step=0.01, decimals=3),
            InspectorField("limit_high", "Limit High", FieldType.FLOAT, min_val=0.0, max_val=3.14159, step=0.01, decimals=3),
            InspectorField("stiffness", "Stiffness", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=0.1, decimals=1),
            InspectorField("damping", "Damping", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.01, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.connected_entity_name: str = ""
        self._connected_entity_id: str = ""
        self.anchor: Vec3 = Vec3.zero()
        self.axis: Vec3 = Vec3(0, 0, 1)
        self.joint_type: str = JointType.HINGE.value
        self.limit_low: float = -3.14159
        self.limit_high: float = 3.14159
        self.stiffness: float = 10.0
        self.damping: float = 1.0

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "connected_entity_name": self.connected_entity_name,
            "anchor": self.anchor.to_list(),
            "axis": self.axis.to_list(),
            "joint_type": self.joint_type,
            "limit_low": self.limit_low,
            "limit_high": self.limit_high,
            "stiffness": self.stiffness,
            "damping": self.damping,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Joint:
        j = cls()
        j.enabled = data.get("enabled", True)
        j.connected_entity_name = data.get("connected_entity_name", "")
        j.anchor = Vec3(*data.get("anchor", [0, 0, 0]))
        j.axis = Vec3(*data.get("axis", [0, 0, 1]))
        j.joint_type = data.get("joint_type", JointType.HINGE.value)
        j.limit_low = data.get("limit_low", -3.14159)
        j.limit_high = data.get("limit_high", 3.14159)
        j.stiffness = data.get("stiffness", 10.0)
        j.damping = data.get("damping", 1.0)
        return j
