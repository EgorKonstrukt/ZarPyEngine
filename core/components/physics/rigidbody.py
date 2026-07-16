# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class Rigidbody(Component):
    _icon = "Rigidbody.png"
    _gizmo_icon_color = (80, 200, 220)
    _gizmo_icon_label = "R"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mass", "Mass", FieldType.FLOAT, min_val=0.001, max_val=100000.0),
            InspectorField("drag", "Drag", FieldType.FLOAT, min_val=0.0, max_val=1000.0),
            InspectorField("angular_drag", "Angular Drag", FieldType.FLOAT, min_val=0.0, max_val=1000.0),
            InspectorField("use_gravity", "Use Gravity", FieldType.BOOL),
            InspectorField("is_kinematic", "Is Kinematic", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.mass: float = 1.0
        self.drag: float = 0.0
        self.angular_drag: float = 0.05
        self.use_gravity: bool = True
        self.is_kinematic: bool = False
        self.freeze_pos_x: bool = False
        self.freeze_pos_y: bool = False
        self.freeze_pos_z: bool = False
        self.freeze_rot_x: bool = False
        self.freeze_rot_y: bool = False
        self.freeze_rot_z: bool = False
        self._velocity: Vec3 = Vec3.zero()
        self._angular_velocity: Vec3 = Vec3.zero()
        self._force_accum: Vec3 = Vec3.zero()
        self._torque_accum: Vec3 = Vec3.zero()
        self._body_id: int = -1
        self._velocity_dirty: bool = False

    @property
    def velocity(self) -> Vec3: return self._velocity
    @velocity.setter
    def velocity(self, v: Vec3):
        self._velocity = v
        self._velocity_dirty = True
    @property
    def angular_velocity(self) -> Vec3: return self._angular_velocity
    @angular_velocity.setter
    def angular_velocity(self, v: Vec3):
        self._angular_velocity = v
        self._velocity_dirty = True
    def add_force(self, force: Vec3, world_space: bool = True):
        self._force_accum = self._force_accum + force
    def add_torque(self, torque: Vec3):
        self._torque_accum = self._torque_accum + torque
    def add_impulse(self, impulse: Vec3):
        if self.mass > 0 and not self.is_kinematic:
            self._velocity = self._velocity + impulse * (1.0 / self.mass)
            self._velocity_dirty = True
    def consume_velocity_dirty(self) -> bool:
        d = self._velocity_dirty
        self._velocity_dirty = False
        return d
    def _clear_forces(self):
        self._force_accum._x = 0.0; self._force_accum._y = 0.0; self._force_accum._z = 0.0
        self._torque_accum._x = 0.0; self._torque_accum._y = 0.0; self._torque_accum._z = 0.0
    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "mass": self.mass, "drag": self.drag, "angular_drag": self.angular_drag,
            "use_gravity": self.use_gravity, "is_kinematic": self.is_kinematic,
            "freeze_pos": [self.freeze_pos_x, self.freeze_pos_y, self.freeze_pos_z],
            "freeze_rot": [self.freeze_rot_x, self.freeze_rot_y, self.freeze_rot_z]
        })
        return d
    @classmethod
    def deserialize(cls, data: dict) -> Rigidbody:
        rb = cls()
        rb.enabled = data.get("enabled", True)
        rb.mass = data.get("mass", 1.0)
        rb.drag = data.get("drag", 0.0)
        rb.angular_drag = data.get("angular_drag", 0.05)
        rb.use_gravity = data.get("use_gravity", True)
        rb.is_kinematic = data.get("is_kinematic", False)
        fp = data.get("freeze_pos", [False,False,False])
        fr = data.get("freeze_rot", [False,False,False])
        rb.freeze_pos_x, rb.freeze_pos_y, rb.freeze_pos_z = fp
        rb.freeze_rot_x, rb.freeze_rot_y, rb.freeze_rot_z = fr
        return rb
