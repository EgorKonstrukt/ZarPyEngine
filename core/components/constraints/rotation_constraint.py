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
class RotationConstraint(Component):
    _icon = "RotationConstraint.png"

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
        self._offset_rotation = Quat.identity()

    def add_source(self, entity=None, weight: float = 1.0):
        self.sources.append({
            "entity_id": entity.id if entity else None,
            "weight": weight
        })

    def remove_source(self, index: int):
        if 0 <= index < len(self.sources):
            self.sources.pop(index)

    def set_offset_from_current_rotation(self):
        t = self.transform
        if not t:
            return
        target_quat = self._compute_target_rotation()
        if target_quat is not None:
            diff = target_quat.conjugate() * t.local_rotation
            self._offset_rotation = diff.normalized()

    def _get_source_transform(self, src_data: dict):
        scene = self._entity._scene if self._entity else None
        if not scene or not src_data.get("entity_id"):
            return None
        entity = scene.get_entity(src_data["entity_id"])
        if not entity:
            return None
        return entity.transform

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

        target_quat = self._compute_target_rotation()
        if target_quat is None:
            return

        t = self.transform
        current_euler = t.local_rotation.to_euler()
        effective_target = target_quat * self._offset_rotation
        target_euler = effective_target.to_euler()

        new_x = current_euler.x + (target_euler.x - current_euler.x) * self.weight_x
        new_y = current_euler.y + (target_euler.y - current_euler.y) * self.weight_y
        new_z = current_euler.z + (target_euler.z - current_euler.z) * self.weight_z

        t.local_euler_angles = Vec3(new_x, new_y, new_z)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "is_active": self.is_active,
            "sources": self.sources,
            "weight_x": self.weight_x,
            "weight_y": self.weight_y,
            "weight_z": self.weight_z,
            "offset_rotation": self._offset_rotation.to_list(),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> RotationConstraint:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.is_active = data.get("is_active", True)
        inst.sources = data.get("sources", [])
        inst.weight_x = data.get("weight_x", 1.0)
        inst.weight_y = data.get("weight_y", 1.0)
        inst.weight_z = data.get("weight_z", 1.0)
        off_data = data.get("offset_rotation", [0, 0, 0, 1])
        inst._offset_rotation = Quat(off_data[0], off_data[1], off_data[2], off_data[3])
        return inst
