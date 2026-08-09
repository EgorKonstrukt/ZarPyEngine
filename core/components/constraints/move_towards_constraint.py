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
class MoveTowardsConstraint(Component):
    _icon = "MoveTowardsConstraint.png"

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
            InspectorField("speed", "Speed", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.1, decimals=2),
            InspectorField("maintain_offset", "Maintain Offset", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.is_active = True
        self.sources: list[dict] = []
        self.speed = 5.0
        self.maintain_offset = False
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

        target_pos = self._compute_target_position()
        if target_pos is None:
            return

        t = self.transform
        current_pos = t.position
        effective_target = target_pos + (self._offset if self.maintain_offset else Vec3.zero())

        direction = effective_target - current_pos
        distance = direction.length()

        if distance < 1e-6:
            return

        move_distance = min(self.speed * dt, distance)
        new_pos = current_pos + (direction.normalized() * move_distance)
        t.position = new_pos

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "is_active": self.is_active,
            "sources": self.sources,
            "speed": self.speed,
            "maintain_offset": self.maintain_offset,
            "offset": self._offset.to_list(),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> MoveTowardsConstraint:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.is_active = data.get("is_active", True)
        inst.sources = data.get("sources", [])
        inst.speed = data.get("speed", 5.0)
        inst.maintain_offset = data.get("maintain_offset", False)
        offset_data = data.get("offset", [0, 0, 0])
        inst._offset = Vec3(*offset_data)
        return inst
