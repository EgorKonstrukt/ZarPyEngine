# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
import math
import numpy as np
from core.ecs.ecs import Component, ComponentRegistry, GizmoPrimitive
from core.components.inspector_meta import FieldType, InspectorField


class SoftBendMode(Enum):
    NONE = "none"
    DISTANCE = "distance"
    DIHEDRAL = "dihedral"


class SoftPinMode(Enum):
    NONE = "none"
    TOP = "top"
    BOTTOM = "bottom"


@ComponentRegistry.register
class SoftBody(Component):
    _icon = "SoftBody.png"
    _gizmo_icon_color = (80, 200, 220)
    _gizmo_icon_label = "S"
    _show_gizmo_icon: bool = False
    _gizmo_pass = "collider"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("mass", "Mass", FieldType.FLOAT, min_val=0.01, max_val=10000.0, step=0.1),
            InspectorField("stiffness", "Stiffness", FieldType.SLIDER, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("bend_mode", "Bend Mode", FieldType.ENUM, enum_class=SoftBendMode),
            InspectorField("pressure", "Pressure", FieldType.FLOAT, min_val=0.0, max_val=100000.0, step=1.0),
            InspectorField("damping", "Damping", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01),
            InspectorField("iterations", "Iterations", FieldType.INT, min_val=1, max_val=50, step=1, decimals=0),
            InspectorField("gravity_scale", "Gravity Scale", FieldType.FLOAT, min_val=0.0, max_val=5.0, step=0.05),
            InspectorField("vertex_radius", "Vertex Radius", FieldType.FLOAT, min_val=0.001, max_val=1.0, step=0.005),
            InspectorField("max_velocity", "Max Velocity", FieldType.FLOAT, min_val=1.0, max_val=5000.0, step=1.0),
            InspectorField("pin_mode", "Pin Mode", FieldType.ENUM, enum_class=SoftPinMode),
            InspectorField("pin_fraction", "Pin Fraction", FieldType.FLOAT, min_val=0.0, max_val=0.5, step=0.01),
            InspectorField("max_vertices", "Max Vertices", FieldType.INT, min_val=0, max_val=30000, step=100, decimals=0),
            InspectorField("double_sided", "Double Sided", FieldType.BOOL),
            InspectorField("update_com", "Update COM", FieldType.BOOL),
            InspectorField("recompute_normals", "Recompute Normals", FieldType.BOOL),
            InspectorField("layer", "Layer", FieldType.LAYER),
            InspectorField("mask", "Collision Mask", FieldType.LAYER_MASK),
        ]

    def __init__(self):
        super().__init__()
        self.layer: int = 0
        self.mask: int = 0xFFFF
        self.mass: float = 1.0
        self.stiffness: float = 0.85
        self.bend_mode: SoftBendMode = SoftBendMode.DISTANCE
        self.pressure: float = 0.0
        self.damping: float = 0.1
        self.iterations: int = 10
        self.gravity_scale: float = 1.0
        self.vertex_radius: float = 0.05
        self.max_velocity: float = 500.0
        self.pin_mode: SoftPinMode = SoftPinMode.NONE
        self.pin_fraction: float = 0.1
        self.max_vertices: int = 3000
        self.double_sided: bool = True
        self.update_com: bool = True
        self.recompute_normals: bool = True
        self.material_friction: float = 0.2
        self.material_bounciness: float = 0.0
        self._render_mesh = None
        self._soft_id: int = -1

    @property
    def compliance(self) -> float:
        try:
            s = max(0.0, min(1.0, float(self.stiffness)))
        except Exception:
            s = 0.85
        return (1.0 - s) * 0.01

    def _gizmo_sig(self):
        tr = self.transform
        if tr is None:
            return None
        mf = self.entity.get_component_by_name("MeshFilter") if self.entity else None
        mp = getattr(mf, "mesh_path", "") or ""
        mn = getattr(mf, "mesh_name", "") or ""
        ov = getattr(self, "_render_mesh", None)
        sim_ver = getattr(ov, "_soft_version", -1) if ov is not None else -1
        try:
            wx = round(float(tr.world_matrix._d[3, 0]), 4)
            wy = round(float(tr.world_matrix._d[3, 1]), 4)
            wz = round(float(tr.world_matrix._d[3, 2]), 4)
        except Exception:
            wx, wy, wz = 0.0, 0.0, 0.0
        return (
            mp, mn, int(sim_ver), wx, wy, wz,
            tr.local_position.x, tr.local_position.y, tr.local_position.z,
            tr.local_rotation.x, tr.local_rotation.y, tr.local_rotation.z, tr.local_rotation.w,
            tr.local_scale.x, tr.local_scale.y, tr.local_scale.z,
        )

    def _gizmo_from_sim(self, tr):
        ov = getattr(self, "_render_mesh", None)
        if ov is None:
            return None
        try:
            verts = np.asarray(getattr(ov, "vertices", None), dtype=np.float32).reshape(-1, 3)
            faces = np.asarray(getattr(ov, "_soft_faces", None), dtype=np.int64).reshape(-1, 3)
        except Exception:
            return None
        try:
            n = len(verts)
            if n < 3 or len(faces) == 0:
                return None
            faces = faces[((faces >= 0) & (faces < n)).all(axis=1)]
            if len(faces) == 0:
                return None
            if len(faces) > 8000:
                faces = faces[::int(np.ceil(len(faces) / 8000))]
            pairs = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
            edges = np.ascontiguousarray(verts[pairs].astype(np.float32))
            W = np.asarray(tr.world_matrix._d, dtype=np.float32)
            flat = edges.reshape(-1, 3)
            h = np.ones((flat.shape[0], 4), dtype=np.float32)
            h[:, :3] = flat
            world = (h @ W).astype(np.float32)[:, :3]
            n_edges = edges.shape[0]
            starts = world[0::2].reshape(n_edges, 3)
            ends = world[1::2].reshape(n_edges, 3)
            c_arr = np.empty((n_edges, 4), dtype=np.float32)
            c_arr[:] = [0.2, 0.8, 1.0, 0.6]
            return (starts, ends, c_arr)
        except Exception:
            return None

    def gizmo_primitives(self):
        tr = self.transform
        if tr is None or self.entity is None:
            return None
        sig = self._gizmo_sig()
        cached = getattr(self, "_soft_gizmo_cache", None)
        if cached is not None and cached[0] == sig:
            return cached[1]
        sim = self._gizmo_from_sim(tr)
        if sim is not None:
            self._soft_gizmo_cache = (sig, sim)
            return sim
        from core.components.rendering.renderers.mesh_filter import MeshFilter
        mf = self.entity.get_component(MeshFilter)
        path = (mf.mesh_path or "") if mf else ""
        if not path:
            return None
        try:
            from core.components.physics.mesh_collider import load_collision_geometry
            verts, indices, _ = load_collision_geometry(path)
        except Exception:
            return None
        if verts is None or indices is None or len(verts) == 0 or len(indices) < 3:
            return None
        try:
            tris = np.asarray(indices).reshape(-1, 3)
            n = len(verts)
            tris = tris[((tris >= 0) & (tris < n)).all(axis=1)]
            if len(tris) == 0:
                return None
            if len(tris) > 8000:
                tris = tris[::int(np.ceil(len(tris) / 8000))]
            pairs = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
            edges_np = np.ascontiguousarray(verts[pairs].astype(np.float32))
        except Exception:
            return None
        q = tr.local_rotation
        x, y, z, w = q.x, q.y, q.z, q.w
        nrm = math.sqrt(x * x + y * y + z * z + w * w)
        if nrm > 1e-10:
            inv = 1.0 / nrm
            x *= inv
            y *= inv
            z *= inv
            w *= inv
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]], dtype=np.float32)
        T = np.array([tr.local_position.x, tr.local_position.y, tr.local_position.z], dtype=np.float32)
        S = np.array([tr.local_scale.x, tr.local_scale.y, tr.local_scale.z], dtype=np.float32)
        n_edges = edges_np.shape[0]
        flat = edges_np.reshape(-1, 3)
        transformed = flat * S @ R.T + T
        starts = transformed[0::2].reshape(n_edges, 3)
        ends = transformed[1::2].reshape(n_edges, 3)
        c_arr = np.empty((n_edges, 4), dtype=np.float32)
        c_arr[:] = [0.2, 0.8, 1.0, 0.6]
        result = (starts, ends, c_arr)
        self._soft_gizmo_cache = (sig, result)
        return result

    def gizmo(self):
        prims = self.gizmo_primitives()
        if prims is None:
            return []
        s, e, c = prims
        if s.shape[0] == 0:
            return []
        return [GizmoPrimitive(s, e, c)]

    def serialize(self) -> dict:
        d = super().serialize()
        bend = self.bend_mode.value if isinstance(self.bend_mode, SoftBendMode) else self.bend_mode
        pin = self.pin_mode.value if isinstance(self.pin_mode, SoftPinMode) else self.pin_mode
        d.update({
            "mass": self.mass, "stiffness": self.stiffness,
            "bend_mode": bend, "pressure": self.pressure,
            "damping": self.damping, "iterations": self.iterations,
            "gravity_scale": self.gravity_scale,
            "vertex_radius": self.vertex_radius, "max_velocity": self.max_velocity,
            "pin_mode": pin, "pin_fraction": self.pin_fraction,
            "max_vertices": self.max_vertices,
            "double_sided": self.double_sided, "update_com": self.update_com,
            "recompute_normals": self.recompute_normals,
            "friction": self.material_friction, "bounciness": self.material_bounciness,
            "layer": self.layer, "mask": self.mask,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> SoftBody:
        sb = cls()
        sb.enabled = data.get("enabled", True)
        sb.mass = data.get("mass", 1.0)
        sb.stiffness = data.get("stiffness", 0.85)
        try:
            sb.bend_mode = SoftBendMode(data.get("bend_mode", "distance"))
        except Exception:
            sb.bend_mode = SoftBendMode.DISTANCE
        sb.pressure = data.get("pressure", 0.0)
        sb.damping = data.get("damping", 0.1)
        sb.iterations = data.get("iterations", 10)
        sb.gravity_scale = data.get("gravity_scale", 1.0)
        sb.vertex_radius = data.get("vertex_radius", 0.05)
        sb.max_velocity = data.get("max_velocity", 500.0)
        try:
            sb.pin_mode = SoftPinMode(data.get("pin_mode", "none"))
        except Exception:
            sb.pin_mode = SoftPinMode.NONE
        sb.pin_fraction = data.get("pin_fraction", 0.1)
        sb.max_vertices = data.get("max_vertices", 3000)
        sb.double_sided = data.get("double_sided", True)
        sb.update_com = data.get("update_com", True)
        sb.recompute_normals = data.get("recompute_normals", True)
        sb.material_friction = data.get("friction", 0.2)
        sb.material_bounciness = data.get("bounciness", 0.0)
        sb.layer = data.get("layer", 0)
        sb.mask = data.get("mask", 0xFFFF)
        return sb
