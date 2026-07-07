# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs import Component, ComponentRegistry
from core.math3d import Vec3, Quat, Mat4
from core.components.inspector_meta import FieldType, InspectorField, ListElementField


@ComponentRegistry.register
class FollowTransformConstraint(Component):
    _icon = "FollowTransformConstraint.png"

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
            InspectorField("follow_position", "Follow Position", FieldType.BOOL),
            InspectorField("follow_rotation", "Follow Rotation", FieldType.BOOL),
            InspectorField("position_speed", "Position Speed", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.5, decimals=2),
            InspectorField("rotation_speed", "Rotation Speed", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
        ]

    def __init__(self):
        super().__init__()
        self.is_active = True
        self.sources: list[dict] = []
        self.follow_position = True
        self.follow_rotation = True
        self.position_speed = 10.0
        self.rotation_speed = 90.0

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

    def _compute_target_rotation(self) -> Quat | None:
        valid = [s for s in self.sources if s.get("weight", 0.0) > 1e-6]
        if not valid:
            return None

        if len(valid) == 1:
            st = self._get_source_transform(valid[0])
            if st:
                return st.local_rotation
            return None

        total_weight = sum(s["weight"] for s in valid)
        if total_weight < 1e-8:
            return None

        sx, sy, sz, sw = 0.0, 0.0, 0.0, 0.0
        for s in valid:
            st = self._get_source_transform(s)
            if not st:
                continue
            q = st.local_rotation
            w = s["weight"] / total_weight
            sx += q.x * w
            sy += q.y * w
            sz += q.z * w
            sw += q.w * w

        return Quat(sx, sy, sz, sw).normalized()

    def on_update(self, dt: float):
        if not self.is_active or not self.transform:
            return

        t = self.transform

        if self.follow_position:
            target_pos = self._compute_target_position()
            if target_pos is not None:
                current_pos = t.position
                direction = target_pos - current_pos
                distance = direction.length()

                if distance > 1e-6:
                    move_distance = min(self.position_speed * dt, distance)
                    new_pos = current_pos + (direction.normalized() * move_distance)
                    t.position = new_pos

        if self.follow_rotation:
            target_quat = self._compute_target_rotation()
            if target_quat is not None:
                current_quat = t.local_rotation
                max_angle_deg = self.rotation_speed * dt
                new_quat = current_quat.slerp(target_quat, min(max_angle_deg / 180.0, 1.0))
                t.local_rotation = new_quat.normalized()

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "is_active": self.is_active,
            "sources": self.sources,
            "follow_position": self.follow_position,
            "follow_rotation": self.follow_rotation,
            "position_speed": self.position_speed,
            "rotation_speed": self.rotation_speed,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> FollowTransformConstraint:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.is_active = data.get("is_active", True)
        inst.sources = data.get("sources", [])
        inst.follow_position = data.get("follow_position", True)
        inst.follow_rotation = data.get("follow_rotation", True)
        inst.position_speed = data.get("position_speed", 10.0)
        inst.rotation_speed = data.get("rotation_speed", 90.0)
        return inst
