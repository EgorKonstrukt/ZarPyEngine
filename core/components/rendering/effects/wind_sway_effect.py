# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from core.ecs.ecs import ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec3
from core.components.rendering.effects.object_effect import ObjectEffect


@ComponentRegistry.register
class WindSwayEffect(ObjectEffect):
    _gizmo_icon_label = "W"
    fx_uniform_defaults = {"u_wind_amount": 0.0}

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("direction", "Wind Direction", FieldType.VEC3, min_val=-1.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("speed", "Speed", FieldType.FLOAT, min_val=0.0, max_val=20.0, step=0.1, decimals=2),
            InspectorField("strength", "Sway", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.01, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 1.0
        self.direction: Vec3 = Vec3(1.0, 0.0, 0.0)
        self.speed: float = 2.0
        self.strength: float = 0.3
        self._dir_buf = np.zeros(3, dtype=np.float32)

    def _apply(self, prog):
        if not self.enabled or self.amount <= 0.0:
            self._set(prog, "u_wind_amount", 0.0)
            return
        d = self.direction
        length = (d.x * d.x + d.y * d.y + d.z * d.z) ** 0.5
        if length > 1e-6:
            inv = 1.0 / length
            self._dir_buf[0] = d.x * inv
            self._dir_buf[1] = d.y * inv
            self._dir_buf[2] = d.z * inv
        else:
            self._dir_buf[0] = 1.0
            self._dir_buf[1] = 0.0
            self._dir_buf[2] = 0.0
        self._set(prog, "u_wind_amount", float(self.amount))
        self._set_vec_bytes(prog, "u_wind_dir", self._dir_buf)
        self._set(prog, "u_wind_speed", float(self.speed))
        self._set(prog, "u_wind_strength", float(self.strength))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "direction": [self.direction.x, self.direction.y, self.direction.z],
            "speed": self.speed,
            "strength": self.strength,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> WindSwayEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 1.0)
        fd = data.get("direction", [1.0, 0.0, 0.0])
        fx.direction = Vec3(*fd[:3])
        fx.speed = data.get("speed", 2.0)
        fx.strength = data.get("strength", 0.3)
        return fx
