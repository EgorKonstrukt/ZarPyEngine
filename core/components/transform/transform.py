# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3, Quat, Mat4, FLOAT_TYPE
from core.math_helpers import mat4_mul_fast, mat4_from_quaternion, mat4_translation, mat4_scale_mat, mat4_inv_fast
from core.components.inspector_meta import FieldType, InspectorField, ComponentInspectorMeta
@ComponentRegistry.register
class Transform(Component):
    _icon = "Transform.png"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("local_position", "Position", FieldType.VEC3),
            InspectorField("local_euler_angles", "Rotation", FieldType.VEC3),
            InspectorField("local_scale", "Scale", FieldType.VEC3),
        ]

    def __init__(self):
        super().__init__()
        self._local_pos: Vec3 = Vec3.zero()
        self._local_rot: Quat = Quat.identity()
        self._local_scale: Vec3 = Vec3.one()
        self._world_matrix: Mat4 = Mat4.identity()
        self._world_target: Mat4 | None = None
        self._dirty: bool = True
    def _mark_dirty(self):
        if self._dirty:
            return
        self._dirty = True
        if self._entity:
            scene = self._entity._scene
            if scene:
                scene._dirty_roots.add(self)
            for child in self._entity.children:
                ct = child.transform
                if ct:
                    ct._mark_dirty()
    def _update_world_matrix(self):
        if not self._dirty and self._world_matrix is not None:
            return
        if self._world_target is not None:
            self._resolve_world_target()
            return
        local = self._build_local_matrix()
        parent_entity = self._entity.parent if self._entity else None
        if parent_entity:
            pt = parent_entity.transform
            if pt:
                pt._update_world_matrix()
                self._world_matrix = local * pt._world_matrix
            else:
                self._world_matrix = local
        else:
            self._world_matrix = local
        self._dirty = False

    def _resolve_world_target(self):
        parent_entity = self._entity.parent if self._entity else None
        if parent_entity:
            pt = parent_entity.transform
            if pt:
                pt._update_world_matrix()
                inv = mat4_inv_fast(pt._world_matrix._d)
                local = Mat4(mat4_mul_fast(self._world_target._d, inv))
                pos, rot, scale = local.decompose()
                self._local_pos = pos
                self._local_rot = rot
                self._local_scale = scale
        else:
            pos, rot, scale = self._world_target.decompose()
            self._local_pos = pos
            self._local_rot = rot
            self._local_scale = scale
        self._world_matrix = self._world_target
        self._world_target = None
        self._dirty = False
    def _build_local_matrix(self) -> Mat4:
        r = mat4_from_quaternion(self._local_rot.x, self._local_rot.y, self._local_rot.z, self._local_rot.w)
        sx, sy, sz = self._local_scale.x, self._local_scale.y, self._local_scale.z
        m = np.array(r, dtype=FLOAT_TYPE)
        m[0, 0] *= sx; m[0, 1] *= sx; m[0, 2] *= sx
        m[1, 0] *= sy; m[1, 1] *= sy; m[1, 2] *= sy
        m[2, 0] *= sz; m[2, 1] *= sz; m[2, 2] *= sz
        m[3, 0] = self._local_pos.x
        m[3, 1] = self._local_pos.y
        m[3, 2] = self._local_pos.z
        m[3, 3] = 1.0
        return Mat4(m)
    @property
    def local_position(self) -> Vec3: return self._local_pos
    @local_position.setter
    def local_position(self, v: Vec3):
        if isinstance(v, (tuple, list)):
            self._local_pos = Vec3(*v)
        else:
            self._local_pos = v
        self._mark_dirty()
    @property
    def local_rotation(self) -> Quat: return self._local_rot
    @local_rotation.setter
    def local_rotation(self, v: Quat):
        self._local_rot = v.normalized()
        self._mark_dirty()
    @property
    def local_scale(self) -> Vec3: return self._local_scale
    @local_scale.setter
    def local_scale(self, v: Vec3):
        if isinstance(v, (tuple, list)):
            self._local_scale = Vec3(*v)
        else:
            self._local_scale = v
        self._mark_dirty()
    @property
    def local_euler_angles(self) -> Vec3: return self._local_rot.to_euler()
    @local_euler_angles.setter
    def local_euler_angles(self, v: Vec3):
        self._local_rot = Quat.from_euler(v.x, v.y, v.z)
        self._mark_dirty()
    @property
    def position(self) -> Vec3:
        self._update_world_matrix()
        return self._world_matrix.get_translation()
    @position.setter
    def position(self, world_pos: Vec3):
        if isinstance(world_pos, (tuple, list)):
            world_pos = Vec3(*world_pos)
        parent_entity = self._entity.parent if self._entity else None
        if parent_entity:
            pt = parent_entity.transform
            if pt:
                pt._update_world_matrix()
                inv = mat4_inv_fast(pt._world_matrix._d)
                world_arr = np.array([world_pos.x, world_pos.y, world_pos.z, 1.0], dtype=FLOAT_TYPE)
                local_arr = world_arr @ inv
                self._local_pos = Vec3(float(local_arr[0]), float(local_arr[1]), float(local_arr[2]))
                self._mark_dirty()
                return
        self._local_pos = world_pos
        self._mark_dirty()
    @property
    def world_matrix(self) -> Mat4:
        self._update_world_matrix()
        return self._world_matrix

    @world_matrix.setter
    def world_matrix(self, m: Mat4):
        self._world_target = Mat4(m._d.copy()) if isinstance(m, Mat4) else Mat4(m)
        self._dirty = True
    @property
    def forward(self) -> Vec3:
        self._update_world_matrix()
        m = self._world_matrix._d
        return Vec3(-float(m[2,0]), -float(m[2,1]), -float(m[2,2])).normalized()
    @property
    def right(self) -> Vec3:
        self._update_world_matrix()
        m = self._world_matrix._d
        return Vec3(float(m[0,0]), float(m[0,1]), float(m[0,2])).normalized()
    @property
    def up(self) -> Vec3:
        self._update_world_matrix()
        m = self._world_matrix._d
        return Vec3(float(m[1,0]), float(m[1,1]), float(m[1,2])).normalized()
    def translate(self, delta: Vec3, world_space: bool = False):
        if world_space:
            self.position = self.position + delta
        else:
            self.local_position = self._local_pos + delta
    def rotate(self, euler: Vec3):
        dq = Quat.from_euler(euler.x, euler.y, euler.z)
        self.local_rotation = (self._local_rot * dq).normalized()
    def look_at(self, target: Vec3, up: Vec3 = None):
        if up is None: up = Vec3.up()
        fwd = (target - self.position).normalized()
        self.local_rotation = Quat.look_rotation(fwd, up)
    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "local_position": self._local_pos.to_list(),
            "local_rotation": self._local_rot.to_list(),
            "local_scale": self._local_scale.to_list()
        })
        return d
    @classmethod
    def deserialize(cls, data: dict) -> Transform:
        t = cls()
        t.enabled = data.get("enabled", True)
        p = data.get("local_position", [0,0,0])
        r = data.get("local_rotation", [0,0,0,1])
        s = data.get("local_scale", [1,1,1])
        t._local_pos = Vec3(*p)
        t._local_rot = Quat(r[0],r[1],r[2],r[3])
        t._local_scale = Vec3(*s)
        return t

    @staticmethod
    def batch_update_world_matrices(transforms: list):
        from core._ecs_batch import batch_update_from_transforms
        batch_update_from_transforms(transforms)