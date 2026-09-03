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


class MotorMode(enum.Enum):
    VELOCITY = "velocity"
    POSITION = "position"


class MotorType(enum.Enum):
    HINGE = "hinge"
    SLIDER = "slider"


def _get_physics_plugin():
    from core.engine.engine import Engine
    engine = Engine.instance()
    if engine is None:
        return None
    try:
        return engine.plugin_manager.get("PhysicsPlugin")
    except Exception:
        return None


def _resolve_solver():
    plugin = _get_physics_plugin()
    if plugin is None:
        return None
    solver = getattr(plugin, "_solver", None)
    if solver is not None and getattr(solver, "_world", None) is not None:
        return solver
    return None


@ComponentRegistry.register
class Motor(Component):
    _icon = "Motor.png"
    _gizmo_icon_color = (255, 160, 80)
    _gizmo_icon_label = "M"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("connected_entity_id", "Connected Entity", FieldType.GAMEOBJECT),
            InspectorField("motor_type", "Motor Type", FieldType.ENUM, enum_class=MotorType),
            InspectorField("mode", "Mode", FieldType.ENUM, enum_class=MotorMode),
            InspectorField("axis", "Axis", FieldType.VEC3),
            InspectorField("anchor", "Anchor", FieldType.VEC3),
            InspectorField("", "Drive", FieldType.HEADER),
            InspectorField("target", "Target", FieldType.FLOAT, min_val=-1000000.0, max_val=1000000.0, step=0.01, decimals=3),
            InspectorField("target_velocity", "Target Velocity", FieldType.FLOAT, min_val=-1000000.0, max_val=1000000.0, step=0.01, decimals=3),
            InspectorField("motor_force", "Motor Force", FieldType.FLOAT, min_val=0.0, max_val=1000000.0, step=1.0, decimals=1),
            InspectorField("motor_torque", "Motor Torque", FieldType.FLOAT, min_val=0.0, max_val=1000000.0, step=1.0, decimals=1),
            InspectorField("", "Limits", FieldType.HEADER),
            InspectorField("limit_low", "Limit Low", FieldType.FLOAT, min_val=-100000.0, max_val=0.0, step=0.01, decimals=3),
            InspectorField("limit_high", "Limit High", FieldType.FLOAT, min_val=0.0, max_val=100000.0, step=0.01, decimals=3),
            InspectorField("free_spin", "Free Spin", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.connected_entity_id: str = ""
        self._connected_entity_ref: str = ""
        self.motor_type: MotorType = MotorType.HINGE
        self.mode: MotorMode = MotorMode.POSITION
        self.axis: Vec3 = Vec3(0, 0, 1)
        self.anchor: Vec3 = Vec3.zero()
        self.target: float = 0.0
        self.target_velocity: float = 0.0
        self.motor_force: float = 500.0
        self.motor_torque: float = 500.0
        self.limit_low: float = -100000.0
        self.limit_high: float = 100000.0
        self.free_spin: bool = False

        self._solver = None
        self._constraint_id: int = -1
        self._created = False
        self._warned = False
        self._running_velocity = 0.0

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "connected_entity_id": self.connected_entity_id,
            "motor_type": self.motor_type.value,
            "mode": self.mode.value,
            "axis": self.axis.to_list(),
            "anchor": self.anchor.to_list(),
            "target": self.target,
            "target_velocity": self.target_velocity,
            "motor_force": self.motor_force,
            "motor_torque": self.motor_torque,
            "limit_low": self.limit_low,
            "limit_high": self.limit_high,
            "free_spin": self.free_spin,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> "Motor":
        m = cls()
        m.enabled = data.get("enabled", True)
        m.connected_entity_id = data.get("connected_entity_id", "")
        m.motor_type = MotorType(data.get("motor_type", MotorType.HINGE.value))
        m.mode = MotorMode(data.get("mode", MotorMode.POSITION.value))
        m.axis = Vec3(*data.get("axis", [0, 0, 1]))
        m.anchor = Vec3(*data.get("anchor", [0, 0, 0]))
        m.target = data.get("target", 0.0)
        m.target_velocity = data.get("target_velocity", 0.0)
        m.motor_force = data.get("motor_force", 500.0)
        m.motor_torque = data.get("motor_torque", 500.0)
        m.limit_low = data.get("limit_low", -100000.0)
        m.limit_high = data.get("limit_high", 100000.0)
        m.free_spin = data.get("free_spin", False)
        return m

    def set_target(self, value: float):
        self.target = float(value)
        self._push_target()

    def set_target_velocity(self, value: float):
        self.target_velocity = float(value)
        self._running_velocity = self.target if self._running_velocity == 0.0 else self._running_velocity

    def set_mode(self, mode: MotorMode):
        was_velocity = self.mode == MotorMode.VELOCITY
        self.mode = mode if isinstance(mode, MotorMode) else MotorMode(mode)
        if not was_velocity and self.mode == MotorMode.VELOCITY:
            self._running_velocity = self.target
        elif was_velocity and self.mode != MotorMode.VELOCITY:
            self.target = self._running_velocity
        self._push_target()

    def _motor_dict(self):
        return {
            "max_force": self.motor_force,
            "max_torque": self.motor_torque,
        }

    def _resolve_connected(self):
        eid = self.connected_entity_id
        if not eid:
            return None
        scene = getattr(self._entity, "_scene", None)
        if scene is None:
            return None
        return scene.get_entity(eid)

    def _create_constraint(self):
        solver = self._solver
        if solver is None:
            return
        entity_id = self._entity.id
        body_a = solver._entity_to_body.get(entity_id)
        if body_a is None:
            from core.foundation.logger import Logger
            Logger.warning(f"Motor: entity '{self._entity.name}' has no body")
            return
        connected = self._resolve_connected()
        if connected is None:
            from core.foundation.logger import Logger
            Logger.warning(f"Motor: connected entity '{self.connected_entity_id}' not found")
            return
        body_b = solver._entity_to_body.get(connected.id)
        if body_b is None:
            from core.foundation.logger import Logger
            Logger.warning(f"Motor: connected entity '{connected.name}' has no body")
            return
        anchor = (self.anchor.x, self.anchor.y, self.anchor.z)
        axis = (self.axis.x, self.axis.y, self.axis.z)
        if self.motor_type == MotorType.SLIDER:
            joint_type = "slider"
        else:
            joint_type = "hinge"
        self._constraint_id = solver.create_joint(
            joint_type=joint_type,
            body_a_id=body_a,
            body_b_id=body_b,
            anchor=anchor,
            axis=axis,
            limit_low=self.limit_low,
            limit_high=self.limit_high,
            stiffness=0.0,
            damping=0.0,
        )
        if self._constraint_id < 0:
            return
        solver.enable_constraint(self._constraint_id, self._motor_dict())
        self._created = True

    def _push_target(self):
        if not self._created or self._solver is None or self._constraint_id < 0:
            return
        if self.mode == MotorMode.VELOCITY:
            self._solver.set_motor_target(self._constraint_id, float(self._running_velocity))
        else:
            self._solver.set_motor_target(self._constraint_id, float(self.target))

    def on_start(self):
        self._solver = None
        self._constraint_id = -1
        self._created = False
        self._warned = False
        self._running_velocity = self.target
        if not self.enabled:
            return
        plugin = _get_physics_plugin()
        if plugin is None:
            return
        if plugin.ensure_single_mode():
            self._solver = _resolve_solver()
        if self._solver is not None:
            self._create_constraint()
            self._push_target()

    def on_enable(self):
        if not self._created:
            self._solver = _resolve_solver()
            if self._solver is not None:
                self._create_constraint()
                self._push_target()

    def on_disable(self):
        if self._solver is not None and self._constraint_id >= 0:
            self._solver.remove_joint(self._constraint_id)
        self._constraint_id = -1
        self._created = False

    def on_fixed_update(self, dt: float):
        if not self._entity or not self.enabled:
            return
        if not self._created:
            if self._solver is None:
                self._solver = _resolve_solver()
                if self._solver is not None:
                    self._create_constraint()
                    self._push_target()
            return
        if self.mode == MotorMode.VELOCITY:
            self._running_velocity += self.target_velocity * dt
            self._solver.set_motor_target(self._constraint_id, float(self._running_velocity))
            return
        self._push_target()
