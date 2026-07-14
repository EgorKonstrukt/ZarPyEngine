# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import numpy as np
from core.ecs.ecs import ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.components.rendering.effects.object_effect import ObjectEffect


@ComponentRegistry.register
class FrostEffect(ObjectEffect):
    _gizmo_icon_label = "F"
    fx_uniform_defaults = {"u_frost_amount": 0.0}

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("color", "Ice Color", FieldType.COLOR),
            InspectorField("coverage", "Coverage", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("rim_power", "Rim Power", FieldType.FLOAT, min_val=0.5, max_val=8.0, step=0.1, decimals=2),
            InspectorField("animate", "Animate", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 1.0
        self.color: list[float] = [0.8, 0.92, 1.0]
        self.coverage: float = 0.7
        self.rim_power: float = 3.0
        self.animate: bool = False
        self._anim_active: bool = False
        self._color_buf = np.zeros(3, dtype=np.float32)

    def _apply(self, prog):
        if not self.enabled:
            self._set(prog, "u_frost_amount", 0.0)
            return
        if self.animate:
            if not self._anim_active:
                self._time_offset = time.time()
                self._anim_active = True
            t = (time.time() - self._time_offset) * 0.3
            self.amount = max(0.0, min(1.0, 0.5 + 0.5 * (t % 1.0)))
        else:
            self._anim_active = False
        if self.amount <= 0.0:
            self._set(prog, "u_frost_amount", 0.0)
            return
        self._color_buf[0] = self.color[0]
        self._color_buf[1] = self.color[1]
        self._color_buf[2] = self.color[2]
        self._set(prog, "u_frost_amount", float(self.amount))
        self._set_vec_bytes(prog, "u_frost_color", self._color_buf)
        self._set(prog, "u_frost_coverage", float(self.coverage))
        self._set(prog, "u_frost_rim", float(self.rim_power))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "color": list(self.color),
            "coverage": self.coverage,
            "rim_power": self.rim_power,
            "animate": self.animate,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> FrostEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 1.0)
        fc = data.get("color", [0.8, 0.92, 1.0])
        fx.color = list(fc)
        fx.coverage = data.get("coverage", 0.7)
        fx.rim_power = data.get("rim_power", 3.0)
        fx.animate = data.get("animate", False)
        return fx
