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
from core.math.math3d import Vec3
from core.components.rendering.effects.object_effect import ObjectEffect
from core.components.rendering.effects.voxel_cpu import compute_voxel_instances


@ComponentRegistry.register
class VoxelizeEffect(ObjectEffect):
    _gizmo_icon_label = "V"
    fx_uniform_defaults = {"u_vox_amount": 0.0, "u_vox_show_base": 1.0, "u_vox_world_grid": 0.0}

    @classmethod
    def fx_geometry_shader(cls) -> "str | None":
        return None

    @classmethod
    def fx_fragment_uniforms(cls) -> str:
        return ""

    @classmethod
    def fx_fragment_snippet(cls) -> str:
        return ""

    def __init__(self):
        super().__init__()
        self.amount: float = 0.5
        self.show_base_mesh: bool = True
        self.world_grid: bool = False
        self.voxel_size: float = 0.3
        self.height: float = 1.0
        self.jitter: float = 0.0
        self.emission: float = 2.5
        self.color: list[float] = [0.4, 1.0, 0.6]
        self.rim: float = 2.5
        self.cell: float = 0.3
        self.double_sided: bool = True
        self.animate: bool = False
        self.speed_anim: float = 0.5
        self.ping_pong: bool = False
        self._anim_active: bool = False
        self._col_buf = np.zeros(3, dtype=np.float32)
        self._vox_cache_key = None
        self._vox_cells = None

    def on_awake(self):
        super().on_awake()
        self._time_offset = time.time()

    def get_voxel_instances(self, verts, idx, model, size: float, world_grid: bool, jitter: float) -> np.ndarray:
        model_arr = np.asarray(model, dtype=np.float32).reshape(4, 4)
        if world_grid:
            mkey = tuple(np.round(model_arr, 2).reshape(16).tolist())
        else:
            mkey = None
        key = (id(verts), len(verts), size, world_grid, jitter, mkey)
        if self._vox_cache_key == key and self._vox_cells is not None:
            return self._vox_cells
        cells = compute_voxel_instances(verts, idx, model_arr, size, world_grid, jitter, seed=(id(verts) & 0xFFFF))
        self._vox_cache_key = key
        self._vox_cells = cells
        return cells

    def _apply(self, prog):
        if not self.enabled:
            self._set(prog, "u_vox_amount", 0.0)
            return
        if self.animate:
            if not self._anim_active:
                self._time_offset = time.time()
                self._anim_active = True
            t = (time.time() - self._time_offset) * self.speed_anim
            tri = abs((t % 2.0) - 1.0) if self.ping_pong else (t % 1.0)
            self.amount = max(0.0, min(1.0, tri))
        else:
            self._anim_active = False

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "amount": self.amount,
            "show_base_mesh": self.show_base_mesh,
            "world_grid": self.world_grid,
            "voxel_size": self.voxel_size,
            "height": self.height,
            "jitter": self.jitter,
            "emission": self.emission,
            "color": list(self.color),
            "rim": self.rim,
            "cell": self.cell,
            "double_sided": self.double_sided,
            "animate": self.animate,
            "speed_anim": self.speed_anim,
            "ping_pong": self.ping_pong,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> "VoxelizeEffect":
        fx = cls()
        fx.enabled = data.get("enabled", True)
        fx.amount = data.get("amount", 0.5)
        fx.show_base_mesh = data.get("show_base_mesh", True)
        fx.world_grid = data.get("world_grid", False)
        fx.voxel_size = data.get("voxel_size", 0.3)
        fx.height = data.get("height", 1.0)
        fx.jitter = data.get("jitter", 0.0)
        fx.emission = data.get("emission", 2.5)
        fc = data.get("color", [0.4, 1.0, 0.6])
        fx.color = list(fc)
        fx.rim = data.get("rim", 2.5)
        fx.cell = data.get("cell", 0.3)
        fx.double_sided = data.get("double_sided", True)
        fx.animate = data.get("animate", False)
        fx.speed_anim = data.get("speed_anim", 0.5)
        fx.ping_pong = data.get("ping_pong", False)
        return fx

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("amount", "Amount", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01, decimals=3),
            InspectorField("show_base_mesh", "Show Base Mesh", FieldType.BOOL),
            InspectorField("world_grid", "World Grid", FieldType.BOOL),
            InspectorField("voxel_size", "Voxel Size", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("height", "Voxel Height", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("jitter", "Jitter", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("emission", "Glow", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("color", "Voxel Color", FieldType.COLOR),
            InspectorField("rim", "Rim Power", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("cell", "Cell Size", FieldType.FLOAT, step=0.01, decimals=3),
            InspectorField("double_sided", "Double Sided", FieldType.BOOL),
            InspectorField("animate", "Animate", FieldType.BOOL),
            InspectorField("speed_anim", "Anim Speed", FieldType.FLOAT, step=0.05, decimals=3),
            InspectorField("ping_pong", "Ping Pong", FieldType.BOOL),
        ]
