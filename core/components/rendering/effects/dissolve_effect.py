# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
from core.ecs.ecs import ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
from core.math.math3d import Vec3
from core.components.rendering.effects.object_effect import ObjectEffect


@ComponentRegistry.register
class DissolveEffect(ObjectEffect):
    _gizmo_icon_label = "D"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("noise_strength", "Noise Strength", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("noise_scale", "Noise Scale", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("direction", "Gradient Direction", FieldType.VEC3, min_val=-1.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("invert", "Invert", FieldType.BOOL),
            InspectorField("edge_width", "Edge Width", FieldType.SLIDER, min_val=0.0, max_val=0.5, step=0.01, decimals=3),
            InspectorField("edge_color", "Edge Color", FieldType.COLOR),
            InspectorField("edge_emission", "Edge Emission", FieldType.SLIDER, min_val=0.0, max_val=5.0, step=0.05, decimals=3),
            InspectorField("animate", "Animate", FieldType.BOOL),
            InspectorField("speed", "Speed", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=3),
            InspectorField("ping_pong", "Ping Pong", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 0.0
        self.noise_strength: float = 1.0
        self.noise_scale: float = 4.0
        self.direction: Vec3 = Vec3(0.0, 1.0, 0.0)
        self.invert: bool = False
        self.edge_width: float = 0.05
        self.edge_color: list[float] = [1.0, 0.4, 0.05]
        self.edge_emission: float = 2.5
        self.animate: bool = False
        self.speed: float = 0.5
        self.ping_pong: bool = False

    def on_awake(self):
        super().on_awake()
        if self.animate:
            self._time_offset = time.time()

    def _apply(self, prog):
        if self.animate:
            t = (time.time() - self._time_offset) * self.speed
            if self.ping_pong:
                tri = abs((t % 2.0) - 1.0)
            else:
                tri = t % 1.0
            self.amount = max(0.0, min(1.0, tri))
        self._set(prog, "u_dissolve_amount", float(self.amount))
        self._set(prog, "u_dissolve_edge", float(self.edge_width))
        self._set_vec(prog, "u_dissolve_color", self.edge_color[:3])
        self._set(prog, "u_dissolve_edge_emission", float(self.edge_emission))
        self._set(prog, "u_dissolve_noise_scale", float(self.noise_scale))
        self._set(prog, "u_dissolve_noise_strength", float(self.noise_strength))
        self._set_vec(prog, "u_dissolve_dir", [self.direction.x, self.direction.y, self.direction.z])
        self._set(prog, "u_dissolve_invert", 1.0 if self.invert else 0.0)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "noise_strength": self.noise_strength,
            "noise_scale": self.noise_scale,
            "direction": [self.direction.x, self.direction.y, self.direction.z],
            "invert": self.invert,
            "edge_width": self.edge_width,
            "edge_color": list(self.edge_color),
            "edge_emission": self.edge_emission,
            "animate": self.animate,
            "speed": self.speed,
            "ping_pong": self.ping_pong,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> DissolveEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 0.0)
        fx.noise_strength = data.get("noise_strength", 1.0)
        fx.noise_scale = data.get("noise_scale", 4.0)
        fd = data.get("direction", [0.0, 1.0, 0.0])
        fx.direction = Vec3(*fd[:3])
        fx.invert = data.get("invert", False)
        fx.edge_width = data.get("edge_width", 0.05)
        fc = data.get("edge_color", [1.0, 0.4, 0.05])
        fx.edge_color = list(fc)
        fx.edge_emission = data.get("edge_emission", 2.5)
        fx.animate = data.get("animate", False)
        fx.speed = data.get("speed", 0.5)
        fx.ping_pong = data.get("ping_pong", False)
        return fx
