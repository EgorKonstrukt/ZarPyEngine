# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec2
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class Rigidbody2D(Component):
    _icon = "Rigidbody2D.png"
    _gizmo_icon_color = (80, 200, 220)
    _gizmo_icon_label = "R2"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mass", "Mass", FieldType.FLOAT, min_val=0.001, max_val=100000.0),
            InspectorField("drag", "Drag", FieldType.FLOAT, min_val=0.0, max_val=1000.0),
            InspectorField("angular_drag", "Angular Drag", FieldType.FLOAT, min_val=0.0, max_val=1000.0),
            InspectorField("gravity_scale", "Gravity Scale", FieldType.FLOAT, min_val=0.0, max_val=100.0),
            InspectorField("is_kinematic", "Is Kinematic", FieldType.BOOL),
            InspectorField("freeze_rotation", "Freeze Rotation", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.mass: float = 1.0
        self.drag: float = 0.0
        self.angular_drag: float = 0.05
        self.gravity_scale: float = 1.0
        self.is_kinematic: bool = False
        self.freeze_rotation: bool = False
        self._velocity: Vec2 = Vec2.zero()
        self._angular_velocity: float = 0.0
        self._force_accum: Vec2 = Vec2.zero()
        self._torque_accum: float = 0.0
        self._body_id: int = -1
        self._velocity_dirty: bool = False

    @property
    def velocity(self) -> Vec2:
        return self._velocity

    @velocity.setter
    def velocity(self, v: Vec2):
        self._velocity = v
        self._velocity_dirty = True

    @property
    def angular_velocity(self) -> float:
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, v: float):
        self._angular_velocity = v
        self._velocity_dirty = True

    def add_force(self, force: Vec2):
        self._force_accum = self._force_accum + force

    def add_torque(self, torque: float):
        self._torque_accum = self._torque_accum + torque

    def add_impulse(self, impulse: Vec2):
        if self.mass > 0 and not self.is_kinematic:
            self._velocity = self._velocity + impulse * (1.0 / self.mass)
            self._velocity_dirty = True

    def consume_velocity_dirty(self) -> bool:
        d = self._velocity_dirty
        self._velocity_dirty = False
        return d

    def _clear_forces(self):
        self._force_accum = Vec2.zero()
        self._torque_accum = 0.0

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "mass": self.mass, "drag": self.drag,
            "angular_drag": self.angular_drag,
            "gravity_scale": self.gravity_scale,
            "is_kinematic": self.is_kinematic,
            "freeze_rotation": self.freeze_rotation,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Rigidbody2D:
        rb = cls()
        rb.enabled = data.get("enabled", True)
        rb.mass = data.get("mass", 1.0)
        rb.drag = data.get("drag", 0.0)
        rb.angular_drag = data.get("angular_drag", 0.05)
        rb.gravity_scale = data.get("gravity_scale", 1.0)
        rb.is_kinematic = data.get("is_kinematic", False)
        rb.freeze_rotation = data.get("freeze_rotation", False)
        return rb
