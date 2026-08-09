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
class ParentConstraint(Component):
    _icon = "ParentConstraint.png"

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
            InspectorField("constrain_position_x", "Constrain Position X", FieldType.BOOL),
            InspectorField("constrain_position_y", "Constrain Position Y", FieldType.BOOL),
            InspectorField("constrain_position_z", "Constrain Position Z", FieldType.BOOL),
            InspectorField("constrain_rotation_x", "Constrain Rotation X", FieldType.BOOL),
            InspectorField("constrain_rotation_y", "Constrain Rotation Y", FieldType.BOOL),
            InspectorField("constrain_rotation_z", "Constrain Rotation Z", FieldType.BOOL),
            InspectorField("constrain_scale_x", "Constrain Scale X", FieldType.BOOL),
            InspectorField("constrain_scale_y", "Constrain Scale Y", FieldType.BOOL),
            InspectorField("constrain_scale_z", "Constrain Scale Z", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.is_active = True
        self.sources: list[dict] = []

        self.constrain_position_x = True
        self.constrain_position_y = True
        self.constrain_position_z = True

        self.constrain_rotation_x = True
        self.constrain_rotation_y = True
        self.constrain_rotation_z = True

        self.constrain_scale_x = True
        self.constrain_scale_y = True
        self.constrain_scale_z = True

        self._position_offset = Vec3.zero()
        self._rotation_offset = Quat.identity()
        self._scale_offset = Vec3(1.0, 1.0, 1.0)

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
            self._position_offset = t.position - target_pos

    def set_offset_from_current_rotation(self):
        t = self.transform
        if not t:
            return
        target_quat = self._compute_target_rotation()
        if target_quat is not None:
            diff = target_quat.conjugate() * t.local_rotation
            self._rotation_offset = diff.normalized()

    def set_offset_from_current_scale(self):
        t = self.transform
        if not t:
            return
        target_scale = self._compute_target_scale()
        if target_scale is not None:
            self._scale_offset = Vec3(
                t.local_scale.x / max(target_scale.x, 1e-8),
                t.local_scale.y / max(target_scale.y, 1e-8),
                t.local_scale.z / max(target_scale.z, 1e-8)
            )

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

        t = self.transform

        pos_target = self._compute_target_position()
        rot_target = self._compute_target_rotation()
        scale_target = self._compute_target_scale()

        if pos_target is not None:
            current_pos = t.position
            new_x = current_pos.x
            new_y = current_pos.y
            new_z = current_pos.z

            if self.constrain_position_x:
                new_x = pos_target.x + self._position_offset.x
            if self.constrain_position_y:
                new_y = pos_target.y + self._position_offset.y
            if self.constrain_position_z:
                new_z = pos_target.z + self._position_offset.z

            t.position = Vec3(new_x, new_y, new_z)

        if rot_target is not None:
            current_euler = t.local_rotation.to_euler()
            effective_rot = rot_target * self._rotation_offset
            target_euler = effective_rot.to_euler()

            new_x = current_euler.x if not self.constrain_rotation_x else target_euler.x
            new_y = current_euler.y if not self.constrain_rotation_y else target_euler.y
            new_z = current_euler.z if not self.constrain_rotation_z else target_euler.z

            t.local_euler_angles = Vec3(new_x, new_y, new_z)

        if scale_target is not None:
            current_scale = t.local_scale
            effective_x = scale_target.x * self._scale_offset.x
            effective_y = scale_target.y * self._scale_offset.y
            effective_z = scale_target.z * self._scale_offset.z

            new_x = current_scale.x if not self.constrain_scale_x else effective_x
            new_y = current_scale.y if not self.constrain_scale_y else effective_y
            new_z = current_scale.z if not self.constrain_scale_z else effective_z

            t.local_scale = Vec3(new_x, new_y, new_z)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "is_active": self.is_active,
            "sources": self.sources,
            "constrain_position_x": self.constrain_position_x,
            "constrain_position_y": self.constrain_position_y,
            "constrain_position_z": self.constrain_position_z,
            "constrain_rotation_x": self.constrain_rotation_x,
            "constrain_rotation_y": self.constrain_rotation_y,
            "constrain_rotation_z": self.constrain_rotation_z,
            "constrain_scale_x": self.constrain_scale_x,
            "constrain_scale_y": self.constrain_scale_y,
            "constrain_scale_z": self.constrain_scale_z,
            "position_offset": self._position_offset.to_list(),
            "rotation_offset": self._rotation_offset.to_list(),
            "scale_offset": self._scale_offset.to_list(),
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ParentConstraint:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.is_active = data.get("is_active", True)
        inst.sources = data.get("sources", [])

        inst.constrain_position_x = data.get("constrain_position_x", True)
        inst.constrain_position_y = data.get("constrain_position_y", True)
        inst.constrain_position_z = data.get("constrain_position_z", True)

        inst.constrain_rotation_x = data.get("constrain_rotation_x", True)
        inst.constrain_rotation_y = data.get("constrain_rotation_y", True)
        inst.constrain_rotation_z = data.get("constrain_rotation_z", True)

        inst.constrain_scale_x = data.get("constrain_scale_x", True)
        inst.constrain_scale_y = data.get("constrain_scale_y", True)
        inst.constrain_scale_z = data.get("constrain_scale_z", True)

        pos_off = data.get("position_offset", [0, 0, 0])
        inst._position_offset = Vec3(*pos_off)

        rot_off = data.get("rotation_offset", [0, 0, 0, 1])
        inst._rotation_offset = Quat(rot_off[0], rot_off[1], rot_off[2], rot_off[3])

        scl_off = data.get("scale_offset", [1, 1, 1])
        inst._scale_offset = Vec3(*scl_off)

        return inst
