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

    @abstractmethod
    def add_plane(
        self,
        normal: tuple[float, float, float] = (0, 1, 0),
        distance: float = 0.0,
        friction: float = 0.6,
        restitution: float = 0.0,
    ) -> int:
        ...

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
