# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry, GizmoPrimitive
from core.math.math3d import Mat4, Vec3, Quat
from core.config.config import get_global_config
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class Bone(Component):
    _icon = "Bone.png"
    _show_gizmo_icon: bool = False
    _gizmo_icon_label = "B"
    _category = "Skinned Mesh"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("bone_name", "Bone Name", FieldType.STRING),
            InspectorField("bone_index", "Bone Index", FieldType.INT),
        ]

    def __init__(self):
        super().__init__()
        self.bone_name: str = ""
        self.bone_index: int = -1

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({"bone_name": self.bone_name, "bone_index": self.bone_index})
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Bone:
        b = cls()
        b.enabled = data.get("enabled", True)
        b.bone_name = data.get("bone_name", "")
        b.bone_index = int(data.get("bone_index", -1))
        return b


@ComponentRegistry.register
class Armature(Component):
    _icon = "Armature.png"
    _show_gizmo_icon: bool = False
    _gizmo_icon_label = "A"
    _category = "Skinned Mesh"
    _gizmo_pass = "armature"
    _gizmo_bone_color = (1.0, 0.78, 0.22, 1.0)
    _gizmo_root_color = (0.42, 0.9, 1.0, 1.0)

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("bone_count", "Bone Count", FieldType.INT),
            InspectorField("root_bone_name", "Root Bone", FieldType.STRING),
        ]

    def __init__(self):
        super().__init__()
        self.bone_names: list[str] = []
        self.bone_parents: list[int] = []
        self.bone_offset_matrices: list[np.ndarray] = []
        self.bone_bind_local: list[np.ndarray] = []
        self.bone_entity_ids: list[str] = []
        self.root_bone_name: str = ""

    def setup(self, skeleton) -> None:
        if isinstance(skeleton, dict):
            self.bone_names = list(skeleton.get("bone_names", []))
            self.bone_parents = list(skeleton.get("bone_parents", []))
            self.bone_offset_matrices = [np.array(m, dtype=np.float32) for m in skeleton.get("bone_offset_matrices", [])]
            self.bone_bind_local = [np.array(m, dtype=np.float32) for m in skeleton.get("bone_bind_local", [])]
        else:
            self.bone_names = list(getattr(skeleton, "bone_names", []))
            self.bone_parents = list(getattr(skeleton, "bone_parents", []))
            self.bone_offset_matrices = [np.array(m, dtype=np.float32) for m in getattr(skeleton, "bone_offset_matrices", [])]
            self.bone_bind_local = [np.array(m, dtype=np.float32) for m in getattr(skeleton, "bone_bind_local", [])]
        self.bone_entity_ids = [""] * len(self.bone_names)
        for name in self.bone_names:
            idx = self.bone_names.index(name)
            if self.bone_parents[idx] < 0:
                self.root_bone_name = name
                break

    def create_bone_entities(self, scene, root_entity) -> None:
        from core.components import Transform
        from core.components.rendering.skeleton.armature import Bone
        n = len(self.bone_names)
        if n == 0:
            return
        entities: list = [None] * n
        for i in range(n):
            be = scene.create_entity(self.bone_names[i])
            bt = Transform()
            local = Mat4(self.bone_bind_local[i])
            pos, rot, scale = local.decompose()
            bt.local_position = pos
            bt.local_rotation = rot
            bt.local_scale = scale
            be.add_component(bt)
            bn = Bone()
            bn.bone_name = self.bone_names[i]
            bn.bone_index = i
            be.add_component(bn)
            entities[i] = be
        for i in range(n):
            parent_idx = self.bone_parents[i]
            parent_ent = None
            if parent_idx >= 0 and parent_idx < n:
                parent_ent = entities[parent_idx]
            else:
                parent_ent = root_entity
            if parent_ent is not None and entities[i] is not None:
                entities[i].set_parent(parent_ent, preserve_world=False)
            self.bone_entity_ids[i] = entities[i].id if entities[i] is not None else ""

    def _bone_world_positions(self) -> list:
        n = len(self.bone_names)
        scene = self._entity._scene if self._entity else None
        positions = []
        for i in range(n):
            ent = None
            if scene is not None and i < len(self.bone_entity_ids) and self.bone_entity_ids[i]:
                ent = scene.get_entity(self.bone_entity_ids[i])
            if ent is not None and ent.transform is not None:
                wm = ent.transform.world_matrix._d
                positions.append((float(wm[3, 0]), float(wm[3, 1]), float(wm[3, 2])))
            else:
                positions.append(None)
        return positions

    def _append_joint_cross(self, s_list, e_list, c_list, p, size, color):
        x, y, z = p
        s_list.append([x - size, y, z]); e_list.append([x + size, y, z])
        s_list.append([x, y - size, z]); e_list.append([x, y + size, z])
        s_list.append([x, y, z - size]); e_list.append([x, y, z + size])
        for _ in range(3):
            c_list.append(color)

    def gizmo(self):
        if not get_global_config().get("gizmo.show_armature_bones", True):
            return []
        from core.engine.engine import Engine
        engine = Engine.instance()
        if engine:
            vp = getattr(engine, 'viewport', None)
            if vp is not None:
                selected = getattr(vp, '_selected_entities', None)
                if selected is not None:
                    selected_ids = {e.id for e in selected}
                    armature_id = self._entity.id if self._entity else None
                    bone_ids = set(self.bone_entity_ids)
                    if armature_id is not None and armature_id not in selected_ids and not bone_ids.intersection(selected_ids):
                        return []
                else:
                    return []
        n = len(self.bone_names)
        if n == 0:
            return []
        positions = self._bone_world_positions()
        parents = self.bone_parents
        bone_color = self._gizmo_bone_color
        root_color = self._gizmo_root_color
        s_list, e_list, c_list = [], [], []
        for i in range(n):
            p = positions[i]
            if p is None:
                continue
            pi = parents[i] if i < len(parents) else -1
            if pi >= 0 and pi < n and positions[pi] is not None:
                pp = positions[pi]
                s_list.append([pp[0], pp[1], pp[2]])
                e_list.append([p[0], p[1], p[2]])
                c_list.append(bone_color)
                self._append_joint_cross(s_list, e_list, c_list, p, 0.03, bone_color)
            else:
                self._append_joint_cross(s_list, e_list, c_list, p, 0.06, root_color)
        if not s_list:
            return []
        starts = np.array(s_list, dtype=np.float32)
        ends = np.array(e_list, dtype=np.float32)
        colors = np.array(c_list, dtype=np.float32)
        return [GizmoPrimitive(starts, ends, colors)]

    def bone_world_matrices(self, scene, renderer_world: Mat4) -> list[np.ndarray]:
        n = len(self.bone_offset_matrices)
        out: list[np.ndarray] = []
        inv = renderer_world.inverted()
        for i in range(n):
            ent = None
            if i < len(self.bone_entity_ids) and self.bone_entity_ids[i]:
                ent = scene.get_entity(self.bone_entity_ids[i])
            if ent is not None and ent.transform is not None:
                rel = (ent.transform.world_matrix * inv)._d
            else:
                rel = np.eye(4, dtype=np.float32)
            off = self.bone_offset_matrices[i]
            out.append((off @ rel).astype(np.float32))
        return out

    def compute_skinning_buffer(self, scene, renderer_world: Mat4) -> tuple[np.ndarray, int]:
        n = len(self.bone_offset_matrices)
        flat = np.zeros((n, 16), dtype=np.float32)
        inv = renderer_world.inverted()
        for i in range(n):
            ent = None
            if scene is not None and i < len(self.bone_entity_ids) and self.bone_entity_ids[i]:
                ent = scene.get_entity(self.bone_entity_ids[i])
            if ent is not None and ent.transform is not None:
                rel = (ent.transform.world_matrix * inv)._d
                off = self.bone_offset_matrices[i]
                skin = (off @ rel).astype(np.float32)
            else:
                skin = np.eye(4, dtype=np.float32)
            flat[i] = skin.flatten()
        return flat, n

    @property
    def bone_count(self) -> int:
        return len(self.bone_names)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "bone_names": self.bone_names,
            "bone_parents": self.bone_parents,
            "bone_offset_matrices": [m.tolist() for m in self.bone_offset_matrices],
            "bone_bind_local": [m.tolist() for m in self.bone_bind_local],
            "bone_entity_ids": self.bone_entity_ids,
            "root_bone_name": self.root_bone_name,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> Armature:
        a = cls()
        a.enabled = data.get("enabled", True)
        a.bone_names = list(data.get("bone_names", []))
        a.bone_parents = list(data.get("bone_parents", []))
        a.bone_offset_matrices = [np.array(m, dtype=np.float32) for m in data.get("bone_offset_matrices", [])]
        a.bone_bind_local = [np.array(m, dtype=np.float32) for m in data.get("bone_bind_local", [])]
        a.bone_entity_ids = list(data.get("bone_entity_ids", [""] * len(a.bone_names)))
        a.root_bone_name = data.get("root_bone_name", "")
        if len(a.bone_entity_ids) != len(a.bone_names):
            a.bone_entity_ids = [""] * len(a.bone_names)
        return a