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
class PolygonDisintegrationEffect(ObjectEffect):
    _gizmo_icon_label = "P"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("direction", "Eject Direction", FieldType.VEC3, min_val=-1.0, max_val=1.0, step=0.05, decimals=3),
            InspectorField("speed", "Eject Speed", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.1, decimals=3),
            InspectorField("drag", "Drag", FieldType.SLIDER, min_val=0.5, max_val=12.0, step=0.1, decimals=3),
            InspectorField("cell_size", "Cell Size", FieldType.FLOAT, min_val=0.01, max_val=5.0, step=0.01, decimals=3),
            InspectorField("noise_scale", "Shatter Noise", FieldType.FLOAT, min_val=0.1, max_val=20.0, step=0.1, decimals=2),
            InspectorField("edge_width", "Dissolve Edge", FieldType.SLIDER, min_val=0.0, max_val=0.5, step=0.01, decimals=3),
            InspectorField("edge_color", "Edge Color", FieldType.COLOR),
            InspectorField("edge_emission", "Edge Emission", FieldType.SLIDER, min_val=0.0, max_val=5.0, step=0.05, decimals=3),
            InspectorField("animate", "Animate", FieldType.BOOL),
            InspectorField("speed_anim", "Anim Speed", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05, decimals=3),
            InspectorField("ping_pong", "Ping Pong", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.amount: float = 0.0
        self.direction: Vec3 = Vec3(0.0, -1.0, 0.0)
        self.speed: float = 2.0
        self.drag: float = 4.0
        self.cell_size: float = 0.15
        self.noise_scale: float = 1.0
        self.edge_width: float = 0.06
        self.edge_color: list[float] = [1.0, 0.55, 0.15]
        self.edge_emission: float = 2.0
        self.animate: bool = False
        self.speed_anim: float = 0.5
        self.ping_pong: bool = False

    def on_awake(self):
        super().on_awake()
        if self.animate:
            self._time_offset = time.time()

    def _apply(self, prog):
        if self.animate:
            t = (time.time() - self._time_offset) * self.speed_anim
            if self.ping_pong:
                tri = abs((t % 2.0) - 1.0)
            else:
                tri = t % 1.0
            self.amount = max(0.0, min(1.0, tri))
        self._set(prog, "u_disint_amount", float(self.amount))
        self._set_vec(prog, "u_disint_dir", [self.direction.x, self.direction.y, self.direction.z])
        self._set(prog, "u_disint_speed", float(self.speed))
        self._set(prog, "u_disint_drag", float(self.drag))
        self._set(prog, "u_disint_cell", float(self.cell_size))
        self._set(prog, "u_disint_noise_scale", float(self.noise_scale))
        self._set(prog, "u_disint_edge", float(self.edge_width))
        self._set_vec(prog, "u_disint_edge_color", self.edge_color[:3])
        self._set(prog, "u_disint_edge_emission", float(self.edge_emission))

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "direction": [self.direction.x, self.direction.y, self.direction.z],
            "speed": self.speed,
            "drag": self.drag,
            "cell_size": self.cell_size,
            "noise_scale": self.noise_scale,
            "edge_width": self.edge_width,
            "edge_color": list(self.edge_color),
            "edge_emission": self.edge_emission,
            "animate": self.animate,
            "speed_anim": self.speed_anim,
            "ping_pong": self.ping_pong,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> PolygonDisintegrationEffect:
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 0.0)
        fd = data.get("direction", [0.0, -1.0, 0.0])
        fx.direction = Vec3(*fd[:3])
        fx.speed = data.get("speed", 2.0)
        fx.drag = data.get("drag", 4.0)
        fx.cell_size = data.get("cell_size", 0.15)
        fx.noise_scale = data.get("noise_scale", 1.0)
        fx.edge_width = data.get("edge_width", 0.06)
        fc = data.get("edge_color", [1.0, 0.55, 0.15])
        fx.edge_color = list(fc)
        fx.edge_emission = data.get("edge_emission", 2.0)
        fx.animate = data.get("animate", False)
        fx.speed_anim = data.get("speed_anim", 0.5)
        fx.ping_pong = data.get("ping_pong", False)
        return fx
