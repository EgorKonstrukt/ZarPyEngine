# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from core.ecs.ecs import ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.components.rendering.effects.object_effect import ObjectEffect


@ComponentRegistry.register
class EmissivePulseEffect(ObjectEffect):
    _gizmo_icon_label = "P"
    fx_uniform_defaults = {"u_pulse_amount": 0.0}

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("color", "Pulse Color", FieldType.COLOR),
            InspectorField("speed", "Speed", FieldType.FLOAT, min_val=0.0, max_val=20.0, step=0.1, decimals=2),
            InspectorField("strength", "Intensity", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.1, decimals=3),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 1.0
        self.color: list[float] = [1.0, 0.4, 0.1]
        self.speed: float = 3.0
        self.strength: float = 2.0
        self._color_buf = np.zeros(3, dtype=np.float32)

    def _apply(self, prog):
        if not self.enabled or self.amount <= 0.0:
            self._set(prog, "u_pulse_amount", 0.0)
            return
        self._color_buf[0] = self.color[0]
        self._color_buf[1] = self.color[1]
        self._color_buf[2] = self.color[2]
        self._set(prog, "u_pulse_amount", float(self.amount))
        self._set_vec_bytes(prog, "u_pulse_color", self._color_buf)
        self._set(prog, "u_pulse_speed", float(self.speed))
        self._set(prog, "u_pulse_strength", float(self.strength))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "color": list(self.color),
            "speed": self.speed,
            "strength": self.strength,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> EmissivePulseEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 1.0)
        fc = data.get("color", [1.0, 0.4, 0.1])
        fx.color = list(fc)
        fx.speed = data.get("speed", 3.0)
        fx.strength = data.get("strength", 2.0)
        return fx
