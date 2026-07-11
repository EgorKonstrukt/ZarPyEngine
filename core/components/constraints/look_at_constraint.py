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
class LookAtConstraint(Component):
    _icon = "LookAtConstraint.png"

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
            InspectorField("look_at_weight", "Look At Weight", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("world_up_weight", "World Up Weight", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.is_active = True
        self.sources: list[dict] = []
        self.look_at_weight = 1.0
        self.world_up_weight = 1.0

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
        direction = (target_pos - current_pos).normalized()
        if direction.length() < 1e-8:
            return

        world_up = Vec3.up()
        effective_up = world_up * self.world_up_weight + current_pos * (1.0 - self.world_up_weight)
        if effective_up.length() < 1e-6:
            effective_up = world_up

        aim_quat = Quat.look_rotation(direction, effective_up)
        current_euler = t.local_rotation.to_euler()
        target_euler = aim_quat.to_euler()

        lerped_x = current_euler.x + (target_euler.x - current_euler.x) * self.look_at_weight
        lerped_y = current_euler.y + (target_euler.y - current_euler.y) * self.look_at_weight
        lerped_z = current_euler.z + (target_euler.z - current_euler.z) * self.look_at_weight

        t.local_euler_angles = Vec3(lerped_x, lerped_y, lerped_z)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "is_active": self.is_active,
            "sources": self.sources,
            "look_at_weight": self.look_at_weight,
            "world_up_weight": self.world_up_weight,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> LookAtConstraint:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.is_active = data.get("is_active", True)
        inst.sources = data.get("sources", [])
        inst.look_at_weight = data.get("look_at_weight", 1.0)
        inst.world_up_weight = data.get("world_up_weight", 1.0)
        return inst
