# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3, Quat, Mat4
from core.components.inspector_meta import FieldType, InspectorField, ListElementField


@ComponentRegistry.register
class PositionConstraint(Component):
    _icon = "PositionConstraint.png"

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
            InspectorField("weight_x", "X Weight", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("weight_y", "Y Weight", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("weight_z", "Z Weight", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.is_active = True
        self.sources: list[dict] = []
        self.weight_x = 1.0
        self.weight_y = 1.0
        self.weight_z = 1.0
        self._offset = Vec3.zero()

    def add_source(self, entity=None, weight: float = 1.0):
        self.sources.append({
            "entity_id": entity.id if entity else None,
            "weight": weight
        })

    def remove_source(self, index: int):
        if 0 <= index < len(self.sources):
            self.sources.pop(index)

    def set_offset_from_current_position(self):
        t = self.transform
        if not t:
            return
        target_pos = self._compute_target_position()
        if target_pos is not None:
            self._offset = t.position - target_pos

    def _get_source_transform(self, src_data: dict):
        scene = self._entity._scene if self._entity else None
        if not scene or not src_data.get("entity_id"):
            return None
        entity = scene.get_entity(src_data["entity_id"])
        if not entity:
            return None
        return entity.transform

    def _compute_target_position(self) -> Vec3 | None:
        valid = [s for s in self.sources if s.get("weight", 0.0) > 1e-6]
        if not valid:
            return None

        total_weight = sum(s["weight"] for s in valid)
        if total_weight < 1e-8:
            return None

        px, py, pz = 0.0, 0.0, 0.0
        for s in valid:
            st = self._get_source_transform(s)
            if not st:
                continue
            wp = st.position
            w = s["weight"] / total_weight
            px += wp.x * w
            py += wp.y * w
            pz += wp.z * w

        return Vec3(px, py, pz)

    def on_update(self, dt: float):
        if not self.is_active or not self.transform:
            return

        target = self._compute_target_position()
        if target is None:
            return

        t = self.transform
        current = t.position
        result_x = current.x + (target.x + self._offset.x - current.x) * self.weight_x
        result_y = current.y + (target.y + self._offset.y - current.y) * self.weight_y
        result_z = current.z + (target.z + self._offset.z - current.z) * self.weight_z

        t.position = Vec3(result_x, result_y, result_z)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "is_active": self.is_active,
            "sources": self.sources,
            "weight_x": self.weight_x,
            "weight_y": self.weight_y,
            "weight_z": self.weight_z,
            "offset": self._offset.to_list(),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> PositionConstraint:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.is_active = data.get("is_active", True)
        inst.sources = data.get("sources", [])
        inst.weight_x = data.get("weight_x", 1.0)
        inst.weight_y = data.get("weight_y", 1.0)
        inst.weight_z = data.get("weight_z", 1.0)
        offset_data = data.get("offset", [0, 0, 0])
        inst._offset = Vec3(*offset_data)
        return inst
