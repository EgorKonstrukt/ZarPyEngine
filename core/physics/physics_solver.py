# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class IPhysicsSolver(ABC):
    """Abstract interface for physics engine solvers."""

    @abstractmethod
    def initialize(self, settings: Optional[dict] = None) -> bool:
        ...

    @abstractmethod
    def shutdown(self):
        ...

    @abstractmethod
    def step_simulation(self, dt: float):
        ...

    @abstractmethod
    def set_gravity(self, gravity: tuple[float, float, float]):
        ...

    @abstractmethod
    def create_rigid_body(
        self,
        entity_id: str,
        shape_type: str,
        shape_params: dict,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        mass: float,
        friction: float = 0.6,
        restitution: float = 0.0,
        is_trigger: bool = False,
        is_kinematic: bool = False,
        collision_layer: int = 0,
        collision_mask: int = 0xFFFF,
    ) -> int:
        ...

    def create_compound_rigid_body(
        self,
        entity_id: str,
        shapes: list,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        mass: float,
        friction: float = 0.6,
        restitution: float = 0.0,
        is_trigger: bool = False,
        is_kinematic: bool = False,
        collision_layer: int = 0,
        collision_mask: int = 0xFFFF,
    ) -> int:
        if not shapes:
            return -1
        first = shapes[0]
        return self.create_rigid_body(
            entity_id=entity_id,
            shape_type=first.get("type", "box"),
            shape_params=first.get("params", {}),
            position=position,
            rotation=rotation,
            mass=mass,
            friction=friction,
            restitution=restitution,
            is_trigger=is_trigger,
            is_kinematic=is_kinematic,
            collision_layer=collision_layer,
            collision_mask=collision_mask,
        )

    @abstractmethod
    def remove_rigid_body(self, body_id: int):
        ...

    @abstractmethod
    def remove_all_bodies(self):
        ...

    @abstractmethod
    def set_body_transform(
        self,
        body_id: int,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ):
        ...

    @abstractmethod
    def get_body_transform(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        ...

    def set_body_transform_quat(
        self,
        body_id: int,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ):
        import math
        qx, qy, qz, qw = quat
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)
        else:
            pitch = math.asin(max(-1.0, min(1.0, sinp)))
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        self.set_body_transform(body_id, position, (roll, pitch, yaw))

    def get_body_transform_quat(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        import math
        pos, euler = self.get_body_transform(body_id)
        rx, ry, rz = euler
        hx, hy, hz = rx * 0.5, ry * 0.5, rz * 0.5
        sx, cx = math.sin(hx), math.cos(hx)
        sy, cy = math.sin(hy), math.cos(hy)
        sz, cz = math.sin(hz), math.cos(hz)
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz
        qw = cx * cy * cz + sx * sy * sz
        return pos, (qx, qy, qz, qw)

    def set_velocities(
        self,
        body_id: int,
        linear: tuple[float, float, float] | None = None,
        angular: tuple[float, float, float] | None = None,
    ):
        if linear is not None:
            self.set_velocity(body_id, linear)
        if angular is not None:
            self.set_angular_velocity(body_id, angular)

    def get_velocities(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return self.get_velocity(body_id), self.get_angular_velocity(body_id)

    @abstractmethod
    def apply_force(
        self, body_id: int, force: tuple[float, float, float], local: bool = False
    ):
        ...

    @abstractmethod
    def apply_torque(self, body_id: int, torque: tuple[float, float, float]):
        ...

    @abstractmethod
    def apply_impulse(
        self, body_id: int, impulse: tuple[float, float, float], local: bool = False
    ):
        ...

    @abstractmethod
    def set_velocity(self, body_id: int, velocity: tuple[float, float, float]):
        ...

    @abstractmethod
    def get_velocity(
        self, body_id: int
    ) -> tuple[float, float, float]:
        ...

    @abstractmethod
    def set_angular_velocity(
        self, body_id: int, velocity: tuple[float, float, float]
    ):
        ...

    @abstractmethod
    def get_angular_velocity(
        self, body_id: int
    ) -> tuple[float, float, float]:
        ...

    @abstractmethod
    def ray_cast(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float = 100.0,
    ) -> Optional[dict]:
        ...

    @abstractmethod
    def get_collision_events(self) -> list[dict]:
        ...

    @abstractmethod
    def create_joint(
        self,
        joint_type: str,
        body_a_id: int,
        body_b_id: int,
        anchor: tuple[float, float, float],
        axis: tuple[float, float, float] = (0, 0, 1),
        limit_low: float = -3.14159,
        limit_high: float = 3.14159,
        stiffness: float = 10.0,
        damping: float = 1.0,
    ) -> int:
        ...

    @abstractmethod
    def remove_joint(self, joint_id: int):
        ...

    @abstractmethod
    def remove_all_joints(self):
        ...

    @abstractmethod
    def change_constraint(
        self,
        constraint_id: int,
        pivot: tuple[float, float, float],
        max_force: float = 500,
    ):
        ...

    def set_motor_target(self, constraint_id: int, target: float):
        ...

    def enable_constraint(self, constraint_id: int, motor: Optional[dict]) -> bool:
        ...

    @abstractmethod
    def add_plane(
        self,
        normal: tuple[float, float, float] = (0, 1, 0),
        distance: float = 0.0,
        friction: float = 0.6,
        restitution: float = 0.0,
    ) -> int:
        ...

    def create_soft_body(
        self,
        entity_id: str,
        vertices,
        indices,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        mass: float = 1.0,
        compliance: float = 0.001,
        bend_mode: int = 1,
        pressure: float = 0.0,
        damping: float = 0.1,
        iterations: int = 10,
        gravity_factor: float = 1.0,
        friction: float = 0.2,
        restitution: float = 0.0,
        vertex_radius: float = 0.05,
        max_velocity: float = 500.0,
        max_vertices: int = 0,
        pin_mode: str = "none",
        pin_fraction: float = 0.1,
        double_sided: bool = True,
        update_com: bool = True,
        collision_layer: int = 0,
        collision_mask: int = 0xFFFF,
    ) -> int:
        return -1

    def remove_soft_body(self, soft_id: int):
        ...

    def remove_all_soft_bodies(self):
        ...

    def get_soft_body_count(self, soft_id: int) -> int:
        return 0

    def get_soft_body_world_vertices(self, soft_id: int):
        return None

    def get_soft_body_geometry(self, soft_id: int):
        return None

    def get_soft_body_com(
        self, soft_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def apply_soft_body_force(self, soft_id: int, force: tuple[float, float, float]):
        ...

    def set_soft_body_velocity(self, soft_id: int, velocity: tuple[float, float, float]):
        ...

    def get_soft_body_velocity(self, soft_id: int) -> tuple[float, float, float]:
        return (0.0, 0.0, 0.0)

    def get_soft_body_sample(self, soft_id: int):
        return None

    @property
    @abstractmethod
    def body_count(self) -> int:
        ...

    @property
    @abstractmethod
    def debug_draw(self):
        ...

    @debug_draw.setter
    @abstractmethod
    def debug_draw(self, enabled: bool):
        ...

    def create_character(
        self,
        pos: tuple[float, float, float],
        height: float = 1.8,
        radius: float = 0.4,
        step_height: float = 0.4,
        max_slope: float = 45.0,
    ):
        return None

    def move_character(self, character, velocity: tuple[float, float, float], dt: float):
        pass

    def set_character_rotation(self, character, rot: tuple[float, float, float, float]):
        pass

    def get_character_position(self, character):
        return None

    def is_character_grounded(self, character) -> bool:
        return False

    def set_character_strength(self, character, strength: float):
        pass

    def destroy_character(self, character):
        pass
