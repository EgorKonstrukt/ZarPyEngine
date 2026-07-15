# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import math
import numpy as np
from core.ecs.ecs import ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.math.math3d import Vec3
from core.components.rendering.effects.object_effect import ObjectEffect


@ComponentRegistry.register
class GlitchEffect(ObjectEffect):
    _gizmo_icon_label = "G"
    fx_uniform_defaults = {"u_glitch_amount": 0.0}

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("intensity", "Block Shift", FieldType.FLOAT, min_val=0.0, max_val=2.0, step=0.01, decimals=3),
            InspectorField("speed", "Speed", FieldType.FLOAT, min_val=0.0, max_val=20.0, step=0.1, decimals=2),
            InspectorField("block_size", "Block Size", FieldType.FLOAT, min_val=0.01, max_val=2.0, step=0.01, decimals=3),
            InspectorField("rgb_shift", "RGB Shift", FieldType.FLOAT, min_val=0.0, max_val=3.0, step=0.05, decimals=3),
            InspectorField("animate", "Animate", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 0.3
        self.intensity: float = 0.3
        self.speed: float = 6.0
        self.block_size: float = 0.2
        self.rgb_shift: float = 1.0
        self.animate: bool = True
        self._anim_active: bool = False

    def _apply(self, prog):
        if not self.enabled:
            self._set(prog, "u_glitch_amount", 0.0)
            return
        if self.animate:
            if not self._anim_active:
                self._time_offset = time.time()
                self._anim_active = True
            t = (time.time() - self._time_offset) * self.speed
            self.amount = 0.15 + 0.35 * (0.5 + 0.5 * math.sin(t * 1.3)) * (0.5 + 0.5 * math.sin(t * 0.37))
        else:
            self._anim_active = False
        if self.amount <= 0.0:
            self._set(prog, "u_glitch_amount", 0.0)
            return
        self._set(prog, "u_glitch_amount", float(self.amount))
        self._set(prog, "u_glitch_intensity", float(self.intensity))
        self._set(prog, "u_glitch_speed", float(self.speed))
        self._set(prog, "u_glitch_block", float(self.block_size))
        self._set(prog, "u_glitch_rgb", float(self.rgb_shift))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "intensity": self.intensity,
            "speed": self.speed,
            "block_size": self.block_size,
            "rgb_shift": self.rgb_shift,
            "animate": self.animate,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> GlitchEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 0.3)
        fx.intensity = data.get("intensity", 0.3)
        fx.speed = data.get("speed", 6.0)
        fx.block_size = data.get("block_size", 0.2)
        fx.rgb_shift = data.get("rgb_shift", 1.0)
        fx.animate = data.get("animate", True)
        return fx
