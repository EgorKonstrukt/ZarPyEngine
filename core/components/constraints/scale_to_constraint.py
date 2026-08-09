# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3, Quat, Mat4
from core.components.inspector_meta import FieldType, InspectorField, ListElementField


@ComponentRegistry.register
class ScaleToConstraint(Component):
    _icon = "ScaleToConstraint.png"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("is_active", "Active", FieldType.BOOL),
            InspectorField(
                "sources",
                "Sources",
                FieldType.LIST,
                element_fields=[
                    ListElementField("entity_id", "Source", FieldType.GAMEOBJECT),
                    ListElementField("weight", "Weight", FieldType.FLOAT, 0.0, 1.0, 0.01, 2),
                ]
            ),
            InspectorField("scale_factor", "Scale Factor", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.01, decimals=2),
            InspectorField("min_scale", "Min Scale", FieldType.FLOAT, min_val=0.01, max_val=2.0, step=0.01, decimals=2),
            InspectorField("max_scale", "Max Scale", FieldType.FLOAT, min_val=1.0, max_val=10.0, step=0.1, decimals=1),
        ]

    def __init__(self):
        super().__init__()
        self.is_active = True
        self.sources: list[dict] = []
        self.scale_factor = 1.0
        self.min_scale = 0.5
        self.max_scale = 2.0

    def add_source(self, entity=None, weight: float = 1.0):
        self.sources.append({
            "entity_id": entity.id if entity else None,
            "weight": weight
        })

    def remove_source(self, index: int):
        if 0 <= index < len(self.sources):
            self.sources.pop(index)

    def _get_source_transform(self, src_data: dict):
        scene = self._entity._scene if self._entity else None
        if not scene or not src_data.get("entity_id"):
            return None
        entity = scene.get_entity(src_data["entity_id"])
        if not entity:
            return None
        return entity.transform

    def _compute_target_scale(self) -> Vec3 | None:
        valid = [s for s in self.sources if s.get("weight", 0.0) > 1e-6]
        if not valid:
            return None

        total_weight = sum(s["weight"] for s in valid)
        if total_weight < 1e-8:
            return None

        sx, sy, sz = 0.0, 0.0, 0.0
        for s in valid:
            st = self._get_source_transform(s)
            if not st:
                continue
            sc = st.local_scale
            w = s["weight"] / total_weight
            sx += sc.x * w
            sy += sc.y * w
            sz += sc.z * w

        return Vec3(sx, sy, sz)

    def on_update(self, dt: float):
        if not self.is_active or not self.transform:
            return

        target_scale = self._compute_target_scale()
        if target_scale is None:
            return

        t = self.transform

        new_x = max(self.min_scale, min(target_scale.x * self.scale_factor, self.max_scale))
        new_y = max(self.min_scale, min(target_scale.y * self.scale_factor, self.max_scale))
        new_z = max(self.min_scale, min(target_scale.z * self.scale_factor, self.max_scale))

        t.local_scale = Vec3(new_x, new_y, new_z)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "is_active": self.is_active,
            "sources": self.sources,
            "scale_factor": self.scale_factor,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ScaleToConstraint:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.is_active = data.get("is_active", True)
        inst.sources = data.get("sources", [])
        inst.scale_factor = data.get("scale_factor", 1.0)
        inst.min_scale = data.get("min_scale", 0.5)
        inst.max_scale = data.get("max_scale", 2.0)
        return inst
