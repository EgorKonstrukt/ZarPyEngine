# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
from typing import Optional
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


class SelectionMode(Enum):
    OBJECT = "Object"
    VERTEX = "Vertex"
    EDGE = "Edge"
    FACE = "Face"


class FaceData:
    def __init__(self, indices: list[int], normal: Optional[np.ndarray] = None):
        self.indices = indices
        self.normal = normal


@ComponentRegistry.register
class ProBuilderMesh(Component):
    _gizmo_icon_color = (255, 165, 0)
    _gizmo_icon_label = "PB"
    _show_gizmo_icon = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("selection_mode", "Selection", FieldType.ENUM, enum_class=SelectionMode),
            InspectorField("vertex_count", "Vertices", FieldType.INT, readonly=True),
            InspectorField("triangle_count", "Triangles", FieldType.INT, readonly=True),
            InspectorField("edge_count", "Edges", FieldType.INT, readonly=True),
        ]

    def __init__(self):
        super().__init__()
        self.positions: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.indices: np.ndarray = np.zeros((0, 3), dtype=np.uint32)
        self.normals: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.uvs: np.ndarray = np.zeros((0, 2), dtype=np.float32)
        self.colors: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.selection_mode: SelectionMode = SelectionMode.OBJECT
        self.selected_vertices: set[int] = set()
        self.selected_edges: set[tuple[int, int]] = set()
        self.selected_faces: set[int] = set()
        self.smooth_angle: float = 45.0
        self._gpu_dirty: bool = True

    @property
    def vertex_count(self) -> int:
        return self.positions.shape[0]

    @property
    def triangle_count(self) -> int:
        return self.indices.shape[0]

    @property
    def edge_count(self) -> int:
        if self.indices.size == 0:
            return 0
        edge_set = set()
        for i in range(self.indices.shape[0]):
            for a, b in [(self.indices[i, 0], self.indices[i, 1]),
                         (self.indices[i, 1], self.indices[i, 2]),
                         (self.indices[i, 2], self.indices[i, 0])]:
                edge_set.add(tuple(sorted((int(a), int(b)))))
        return len(edge_set)

    def get_faces(self) -> list[FaceData]:
        faces = []
        for i in range(self.indices.shape[0]):
            vi = [int(self.indices[i, j]) for j in range(3)]
            v0 = self.positions[vi[0]]
            v1 = self.positions[vi[1]]
            v2 = self.positions[vi[2]]
            n = np.cross(v1 - v0, v2 - v0)
            nlen = np.linalg.norm(n)
            normal = n / nlen if nlen > 1e-10 else np.array([0, 1, 0], dtype=np.float32)
            faces.append(FaceData(vi, normal))
        return faces

    def get_face_centers(self) -> np.ndarray:
        if self.indices.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        centers = np.zeros((self.indices.shape[0], 3), dtype=np.float32)
        for i in range(self.indices.shape[0]):
            centers[i] = self.positions[self.indices[i]].mean(axis=0)
        return centers

    def get_edges(self) -> list[tuple[int, int]]:
        edge_set = set()
        for i in range(self.indices.shape[0]):
            for a, b in [(self.indices[i, 0], self.indices[i, 1]),
                         (self.indices[i, 1], self.indices[i, 2]),
                         (self.indices[i, 2], self.indices[i, 0])]:
                edge_set.add(tuple(sorted((int(a), int(b)))))
        return sorted(edge_set, key=lambda x: (x[0], x[1]))

    def clear_selection(self):
        self.selected_vertices.clear()
        self.selected_edges.clear()
        self.selected_faces.clear()

    def rebuild_normals(self):
        if self.positions.size == 0 or self.indices.size == 0:
            return
        pos = self.positions
        idx = self.indices.astype(np.int32)
        n_verts = pos.shape[0]
        face_normals = np.zeros((idx.shape[0], 3), dtype=np.float32)
        for i in range(idx.shape[0]):
            v0 = pos[idx[i, 0]]
            v1 = pos[idx[i, 1]]
            v2 = pos[idx[i, 2]]
            n = np.cross(v1 - v0, v2 - v0)
            nlen = np.linalg.norm(n)
            if nlen > 1e-10:
                face_normals[i] = n / nlen
            else:
                face_normals[i] = np.array([0, 1, 0], dtype=np.float32)
        vertex_normals = np.zeros((n_verts, 3), dtype=np.float32)
        vertex_counts = np.zeros(n_verts, dtype=np.int32)
        angle_thresh = np.cos(np.radians(self.smooth_angle))
        for i in range(idx.shape[0]):
            fn = face_normals[i]
            for j in range(3):
                vi = idx[i, j]
                if vertex_counts[vi] == 0:
                    vertex_normals[vi] = fn
                    vertex_counts[vi] = 1
                else:
                    existing = vertex_normals[vi]
                    dot = np.dot(existing, fn)
                    if dot >= angle_thresh or dot < -angle_thresh:
                        vertex_normals[vi] = vertex_normals[vi] + fn
                    vertex_counts[vi] += 1
        for i in range(n_verts):
            if vertex_counts[i] > 0:
                nlen = np.linalg.norm(vertex_normals[i])
                if nlen > 1e-10:
                    vertex_normals[i] = vertex_normals[i] / nlen
                else:
                    vertex_normals[i] = np.array([0, 1, 0], dtype=np.float32)
        self.normals = vertex_normals

    def rebuild_uvs(self, world_scale: np.ndarray | None = None):
        if self.positions.size == 0:
            return
        pos = self.positions
        if world_scale is not None:
            pos = pos * world_scale
        n = pos.shape[0]
        uvs = np.zeros((n, 2), dtype=np.float32)
        bmin = pos.min(axis=0)
        bmax = pos.max(axis=0)
        size = bmax - bmin
        for axis in range(3):
            if size[axis] < 1e-8:
                size[axis] = 1.0
        for i in range(n):
            uvs[i, 0] = (pos[i, 0] - bmin[0]) / size[0]
            uvs[i, 1] = (pos[i, 2] - bmin[2]) / size[2] if size[2] > 0 else 0.0
        self.uvs = uvs

    def set_mesh_data(self, positions: np.ndarray, indices: np.ndarray):
        self.positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        self.indices = np.asarray(indices, dtype=np.uint32).reshape(-1, 3)
        self._flip_inward_faces()
        self.rebuild_normals()
        self.rebuild_uvs()
        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.clear_selection()
        self._gpu_dirty = True

    def _flip_inward_faces(self):
        if self.positions.size == 0 or self.indices.size == 0:
            return
        centroid = self.positions.mean(axis=0)
        for i in range(self.indices.shape[0]):
            v0 = self.positions[self.indices[i, 0]]
            v1 = self.positions[self.indices[i, 1]]
            v2 = self.positions[self.indices[i, 2]]
            normal = np.cross(v1 - v0, v2 - v0)
            center = (v0 + v1 + v2) / 3.0
            if np.dot(normal, center - centroid) < -1e-8:
                self.indices[i, 1], self.indices[i, 2] = self.indices[i, 2], self.indices[i, 1]
        sum_normal = np.zeros(3, dtype=np.float32)
        for i in range(self.indices.shape[0]):
            v0 = self.positions[self.indices[i, 0]]
            v1 = self.positions[self.indices[i, 1]]
            v2 = self.positions[self.indices[i, 2]]
            normal = np.cross(v1 - v0, v2 - v0)
            if np.linalg.norm(normal) > 1e-8:
                sum_normal += normal / np.linalg.norm(normal)
        sum_len = np.linalg.norm(sum_normal)
        if sum_len > 1e-8:
            ref_dir = sum_normal / sum_len
            counts = 0
            for i in range(self.indices.shape[0]):
                v0 = self.positions[self.indices[i, 0]]
                v1 = self.positions[self.indices[i, 1]]
                v2 = self.positions[self.indices[i, 2]]
                normal = np.cross(v1 - v0, v2 - v0)
                center = (v0 + v1 + v2) / 3.0
                if abs(np.dot(normal, center - centroid)) < 1e-8 and np.linalg.norm(normal) > 1e-8:
                    n = normal / np.linalg.norm(normal)
                    if np.dot(n, ref_dir) > 0:
                        counts += 1
                    else:
                        counts -= 1
            if counts < 0:
                for i in range(self.indices.shape[0]):
                    self.indices[i, 1], self.indices[i, 2] = self.indices[i, 2], self.indices[i, 1]

    def to_gpu_mesh(self):
        from core.renderer.mesh_data import MeshData
        mesh = MeshData()
        mesh.vertices = self.positions.flatten().copy()
        mesh.indices = self.indices.flatten().copy()
        if self.normals.shape == self.positions.shape:
            mesh.normals = self.normals.flatten().copy()
        else:
            self.rebuild_normals()
            mesh.normals = self.normals.flatten().copy()
        if self.uvs.shape[0] == self.vertex_count:
            mesh.uvs = self.uvs.flatten().copy()
        else:
            self.rebuild_uvs()
            mesh.uvs = self.uvs.flatten().copy()
        mesh.compute_aabb()
        return mesh

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "positions": self.positions.tolist() if self.positions.size > 0 else [],
            "indices": self.indices.tolist() if self.indices.size > 0 else [],
            "normals": self.normals.tolist() if self.normals.size > 0 else [],
            "uvs": self.uvs.tolist() if self.uvs.size > 0 else [],
            "smooth_angle": self.smooth_angle,
            "selection_mode": self.selection_mode.value,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ProBuilderMesh:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        if data.get("positions"):
            inst.positions = np.array(data["positions"], dtype=np.float32).reshape(-1, 3)
        if data.get("indices"):
            inst.indices = np.array(data["indices"], dtype=np.uint32).reshape(-1, 3)
        if data.get("normals"):
            inst.normals = np.array(data["normals"], dtype=np.float32).reshape(-1, 3)
        if data.get("uvs"):
            inst.uvs = np.array(data["uvs"], dtype=np.float32).reshape(-1, 2)
        inst.smooth_angle = data.get("smooth_angle", 45.0)
        try:
            inst.selection_mode = SelectionMode(data.get("selection_mode", "Object"))
        except ValueError:
            inst.selection_mode = SelectionMode.OBJECT
        return inst

    def get_aabb_corners(self) -> np.ndarray:
        if self.positions.size == 0:
            return np.zeros((0, 3), dtype=np.float32)
        bmin = self.positions.min(axis=0)
        bmax = self.positions.max(axis=0)
        corners = np.array([
            [bmin[0], bmin[1], bmin[2]],
            [bmax[0], bmin[1], bmin[2]],
            [bmax[0], bmax[1], bmin[2]],
            [bmin[0], bmax[1], bmin[2]],
            [bmin[0], bmin[1], bmax[2]],
            [bmax[0], bmin[1], bmax[2]],
            [bmax[0], bmax[1], bmax[2]],
            [bmin[0], bmax[1], bmax[2]],
        ], dtype=np.float32)
        return corners

    def scale_from_corner(self, corner_idx: int, old_world_corners: np.ndarray, new_world_corner_pos: np.ndarray,
                          world_to_local_offset: np.ndarray | None = None):
        corners = self.get_aabb_corners()
        opposite = corners[7 - corner_idx]
        if world_to_local_offset is not None:
            new_local = new_world_corner_pos - world_to_local_offset
        else:
            new_local = new_world_corner_pos
        for axis in range(3):
            denom = corners[corner_idx, axis] - opposite[axis]
            if abs(denom) < 1e-8:
                continue
            ratio = (new_local[axis] - opposite[axis]) / denom
            ratio = max(0.01, min(10.0, ratio))
            self.positions[:, axis] = opposite[axis] + (self.positions[:, axis] - opposite[axis]) * ratio
        self.rebuild_normals()
        self.rebuild_uvs()
        self._gpu_dirty = True

    def gizmo_meshes(self):
        n_verts = self.vertex_count
        if n_verts == 0:
            return []
        if self.normals.shape != self.positions.shape:
            self.rebuild_normals()
        if self.uvs.shape[0] != n_verts:
            self.rebuild_uvs()
        verts = [self.positions[i] for i in range(n_verts)]
        idx_list = self.indices.tolist()
        flat_idx = [item for tri in idx_list for item in tri]
        cols = []
        for i in range(n_verts):
            if self.colors.shape[0] == n_verts:
                cols.append(self.colors[i])
            else:
                cols.append([0.8, 0.8, 0.8])
        return [(verts, flat_idx, cols)]
