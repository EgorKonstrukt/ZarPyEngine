# This Source Code Form is subject to terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField
from core.terrain.terrain_generator import TerrainSettings, build_terrain_mesh


@ComponentRegistry.register
class Terrain(Component):
    _icon = "Terrain.png"
    _gizmo_icon_color = (90, 170, 90)
    _gizmo_icon_label = "T"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("world_size", "World Size", FieldType.FLOAT),
            InspectorField("material_path", "Material", FieldType.RESOURCE_PATH, file_filter="Materials (*.mat *.zpem)"),
            InspectorField("graph_path", "Terrain Graph", FieldType.RESOURCE_PATH, file_filter="Terrain Graphs (*.zterr)"),
            InspectorField("auto_regenerate", "Auto Regenerate", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.world_size: float = 1000.0
        self.material_path: str = ""
        self.graph_path: str = ""
        self.auto_regenerate: bool = True
        self.settings: TerrainSettings = TerrainSettings()
        self._heightfield: Optional[np.ndarray] = None
        self._mesh_data: Optional[dict] = None
        self._gpu_dirty: bool = True
        self._generated: bool = False

    @property
    def heightfield(self) -> Optional[np.ndarray]:
        return self._heightfield

    def set_heightfield(self, hf: np.ndarray, mesh_data: Optional[dict] = None):
        self._heightfield = hf
        if mesh_data is not None:
            self._mesh_data = mesh_data
        self._gpu_dirty = True
        self._generated = True

    @property
    def mesh_data(self) -> Optional[dict]:
        return self._mesh_data

    @property
    def resolution(self) -> int:
        return int(self.settings.get("resolution"))

    def generate(self, size: Optional[float] = None):
        if size is not None:
            self.world_size = float(size)
        result = build_terrain_mesh(self.settings, self.world_size)
        if result is None:
            return False
        self._heightfield = result["heightfield"]
        self._mesh_data = result
        self._gpu_dirty = True
        self._generated = True
        return True

    def ensure_generated(self):
        if not self._generated or self._mesh_data is None:
            self.generate()

    def get_height_at(self, x: float, z: float) -> float:
        self.ensure_generated()
        if self._heightfield is None:
            return 0.0
        res = self._heightfield.shape[0]
        half = self.world_size * 0.5
        fx = (x + half) / self.world_size * (res - 1)
        fz = (z + half) / self.world_size * (res - 1)
        fx = max(0.0, min(res - 1.001, fx))
        fz = max(0.0, min(res - 1.001, fz))
        ix = int(fx)
        iz = int(fz)
        dx = fx - ix
        dz = fz - iz
        h = self._heightfield
        h00 = float(h[iz, ix])
        h10 = float(h[iz, ix + 1])
        h01 = float(h[iz + 1, ix])
        h11 = float(h[iz + 1, ix + 1])
        return (h00 * (1 - dx) + h10 * dx) * (1 - dz) + (h01 * (1 - dx) + h11 * dx) * dz

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "world_size": self.world_size,
            "material_path": self.material_path,
            "graph_path": self.graph_path,
            "auto_regenerate": self.auto_regenerate,
            "settings": self.settings.to_dict(),
        })
        if self._heightfield is not None and self._heightfield.ndim == 2:
            d["heightfield"] = self._heightfield.tolist()
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Terrain:
        t = cls()
        t.enabled = data.get("enabled", True)
        t.world_size = data.get("world_size", 1000.0)
        t.material_path = data.get("material_path", "") or ""
        t.graph_path = data.get("graph_path", "") or ""
        t.auto_regenerate = data.get("auto_regenerate", True)
        t.settings = TerrainSettings.from_dict(data.get("settings", {}))
        hf = data.get("heightfield")
        if hf:
            try:
                arr = np.array(hf, dtype=np.float32)
                if arr.ndim == 2:
                    t._heightfield = arr
                    t._generated = True
            except Exception:
                pass
        return t
