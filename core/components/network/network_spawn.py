# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import random
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3, Quat
from core.components.inspector_meta import FieldType, InspectorField, ListElementField


@ComponentRegistry.register
class NetworkSpawn(Component):
    _icon = "NetworkSpawn.png"
    _gizmo_icon_color = (100, 255, 120)
    _gizmo_icon_label = "S"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("spawnable_prefabs", "Spawnable Prefabs", FieldType.LIST, element_fields=[
                ListElementField("path", "Prefab", FieldType.RESOURCE_PATH, file_filter="Prefab (*.zpep)"),
            ]),
            InspectorField("spawn_on_start", "Spawn On Start", FieldType.BOOL),
            InspectorField("spawn_radius", "Spawn Radius", FieldType.FLOAT, min_val=0.0, max_val=100.0),
            InspectorField("randomize_rotation", "Randomize Rotation", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.spawnable_prefabs: list[dict] = []
        self.spawn_on_start: bool = False
        self.spawn_radius: float = 2.0
        self.randomize_rotation: bool = False

    def _get_manager(self):
        ent = self._entity
        if ent is not None:
            m = ent.get_component_by_name("NetworkManager")
            if m is not None:
                return m
        try:
            from core.components.network.network_manager import NetworkManager
            return NetworkManager.get()
        except Exception:
            return None

    def _pick_prefab(self, index: int = 0) -> str:
        if not self.spawnable_prefabs:
            return ""
        if index < 0 or index >= len(self.spawnable_prefabs):
            index = 0
        entry = self.spawnable_prefabs[index]
        if isinstance(entry, dict):
            return str(entry.get("path", ""))
        if isinstance(entry, str):
            return entry
        return ""

    def spawn(self, prefab_index: int = 0, pos: Vec3 | None = None, rot: Quat | None = None):
        mgr = self._get_manager()
        if mgr is None:
            return None
        prefab_path = self._pick_prefab(prefab_index)
        if not prefab_path:
            return None
        spawn_pos = pos
        if spawn_pos is None:
            tr = self.transform
            base = tr.local_position if tr is not None else Vec3.zero()
            if self.spawn_radius > 0:
                ang = random.uniform(0, 6.28318530718)
                rad = random.uniform(0, self.spawn_radius)
                import math
                ox = math.cos(ang) * rad
                oz = math.sin(ang) * rad
                spawn_pos = Vec3(base.x + ox, base.y, base.z + oz)
            else:
                spawn_pos = base
        spawn_rot = rot
        if spawn_rot is None and self.randomize_rotation:
            spawn_rot = Quat.from_euler(0.0, float(random.uniform(0, 360)), 0.0)
        return mgr.spawn_prefab(prefab_path, pos=spawn_pos, rot=spawn_rot)

    def spawn_by_path(self, prefab_path: str, pos: Vec3 | None = None, rot: Quat | None = None):
        mgr = self._get_manager()
        if mgr is None:
            return None
        spawn_pos = pos
        if spawn_pos is None:
            tr = self.transform
            spawn_pos = tr.local_position if tr is not None else Vec3.zero()
        return mgr.spawn_prefab(str(prefab_path), pos=spawn_pos, rot=rot)

    def spawn_player(self, prefab_path: str | None = None, owner_id: int | None = None):
        mgr = self._get_manager()
        if mgr is None:
            return None
        path = str(prefab_path) if prefab_path else self._pick_prefab(0)
        if not path:
            return None
        tr = self.transform
        base = tr.local_position if tr is not None else Vec3.zero()
        if self.spawn_radius > 0:
            ang = random.uniform(0, 6.28318530718)
            rad = random.uniform(0, self.spawn_radius)
            import math
            ox = math.cos(ang) * rad
            oz = math.sin(ang) * rad
            base = Vec3(base.x + ox, base.y, base.z + oz)
        rot = None
        if self.randomize_rotation:
            rot = Quat.from_euler(0.0, float(random.uniform(0, 360)), 0.0)
        return mgr.spawn_prefab(path, pos=base, rot=rot, owner_id=owner_id)

    def on_start(self):
        if self.spawn_on_start:
            self.spawn()

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "spawnable_prefabs": list(self.spawnable_prefabs),
            "spawn_on_start": self.spawn_on_start,
            "spawn_radius": self.spawn_radius,
            "randomize_rotation": self.randomize_rotation,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkSpawn:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        raw = data.get("spawnable_prefabs", [])
        if isinstance(raw, list):
            inst.spawnable_prefabs = list(raw)
        inst.spawn_on_start = bool(data.get("spawn_on_start", False))
        inst.spawn_radius = float(data.get("spawn_radius", 2.0))
        inst.randomize_rotation = bool(data.get("randomize_rotation", False))
        return inst
