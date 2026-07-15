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
class HologramEffect(ObjectEffect):
    _gizmo_icon_label = "H"
    fx_uniform_defaults = {"u_holo_amount": 0.0}

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("scan_density", "Scan Density", FieldType.FLOAT, min_val=1.0, max_val=200.0, step=1.0, decimals=1),
            InspectorField("scan_speed", "Scan Speed", FieldType.FLOAT, min_val=0.0, max_val=20.0, step=0.1, decimals=2),
            InspectorField("flicker", "Flicker", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("rim_power", "Rim Power", FieldType.FLOAT, min_val=0.5, max_val=8.0, step=0.1, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 1.0
        self.color: list[float] = [0.2, 0.9, 1.0]
        self.scan_density: float = 40.0
        self.scan_speed: float = 3.0
        self.flicker: float = 0.15
        self.rim_power: float = 2.0
        self._color_buf = np.zeros(3, dtype=np.float32)

    def _apply(self, prog):
        if not self.enabled or self.amount <= 0.0:
            self._set(prog, "u_holo_amount", 0.0)
            return
        self._color_buf[0] = self.color[0]
        self._color_buf[1] = self.color[1]
        self._color_buf[2] = self.color[2]
        self._set(prog, "u_holo_amount", float(self.amount))
        self._set_vec_bytes(prog, "u_holo_color", self._color_buf)
        self._set(prog, "u_holo_scan", float(self.scan_density))
        self._set(prog, "u_holo_speed", float(self.scan_speed))
        self._set(prog, "u_holo_flicker", float(self.flicker))
        self._set(prog, "u_holo_rim", float(self.rim_power))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "color": list(self.color),
            "scan_density": self.scan_density,
            "scan_speed": self.scan_speed,
            "flicker": self.flicker,
            "rim_power": self.rim_power,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> HologramEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 1.0)
        fc = data.get("color", [0.2, 0.9, 1.0])
        fx.color = list(fc)
        fx.scan_density = data.get("scan_density", 40.0)
        fx.scan_speed = data.get("scan_speed", 3.0)
        fx.flicker = data.get("flicker", 0.15)
        fx.rim_power = data.get("rim_power", 2.0)
        return fx
