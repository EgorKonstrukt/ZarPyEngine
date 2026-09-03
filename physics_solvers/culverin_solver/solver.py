# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
from typing import Optional

import numpy as np

from core.foundation.logger import Logger
from core.physics.physics_solver import IPhysicsSolver

try:
    import culverin
    from culverin import (
        PhysicsWorld,
        SHAPE_BOX,
        SHAPE_SPHERE,
        SHAPE_CAPSULE,
        SHAPE_CYLINDER,
        SHAPE_PLANE,
        SHAPE_MESH,
        SHAPE_CONVEX_HULL,
        MOTION_STATIC,
        MOTION_KINEMATIC,
        MOTION_DYNAMIC,
        CONSTRAINT_FIXED,
        CONSTRAINT_POINT,
        CONSTRAINT_HINGE,
        CONSTRAINT_SLIDER,
        CONSTRAINT_DISTANCE,
        CONSTRAINT_CONE,
        euler_to_quat,
    )
    _HAS_CULVERIN = True
except ImportError:
    PhysicsWorld = None
    _HAS_CULVERIN = False

    SHAPE_BOX = 0
    SHAPE_SPHERE = 1
    SHAPE_CAPSULE = 2
    SHAPE_CYLINDER = 3
    SHAPE_PLANE = 4
    SHAPE_MESH = 5
    SHAPE_CONVEX_HULL = 6
    MOTION_STATIC = 0
    MOTION_KINEMATIC = 1
    MOTION_DYNAMIC = 2
    CONSTRAINT_FIXED = 0
    CONSTRAINT_POINT = 1
    CONSTRAINT_HINGE = 2
    CONSTRAINT_SLIDER = 3
    CONSTRAINT_DISTANCE = 4
    CONSTRAINT_CONE = 5

    def euler_to_quat(roll: float, pitch: float, yaw: float):
        cx, sx = math.cos(roll / 2), math.sin(roll / 2)
        cy, sy = math.cos(pitch / 2), math.sin(pitch / 2)
        cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
        return (
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
            cx * cy * cz + sx * sy * sz,
        )


_SHAPE_TYPE_MAP = {
    "box": SHAPE_BOX,
    "sphere": SHAPE_SPHERE,
    "capsule": SHAPE_CAPSULE,
    "cylinder": SHAPE_CYLINDER,
    "plane": SHAPE_PLANE,
    "mesh": SHAPE_MESH,
}

_JOINT_TYPE_MAP = {
    "fixed": CONSTRAINT_FIXED,
    "point": CONSTRAINT_POINT,
    "point2point": CONSTRAINT_POINT,
    "hinge": CONSTRAINT_HINGE,
    "revolute": CONSTRAINT_HINGE,
    "slider": CONSTRAINT_SLIDER,
    "prismatic": CONSTRAINT_SLIDER,
    "distance": CONSTRAINT_DISTANCE,
    "cone": CONSTRAINT_CONE,
    "spherical": CONSTRAINT_CONE,
    "spring": CONSTRAINT_DISTANCE,
}

def _engine_project_root() -> str:
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
        if eng is not None and getattr(eng, "project_root", None):
            return os.path.normpath(eng.project_root)
    except Exception:
        pass
    return os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_mesh_file(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.exists(path):
        return os.path.normpath(path)
    if os.path.isabs(path):
        root = _engine_project_root()
        if path[1:2] == ":":
            parts = path.replace("\\", "/").split("/")
            for i in range(len(parts)):
                sub = "/".join(parts[i:])
                if sub:
                    candidate = os.path.normpath(os.path.join(root, sub))
                    if os.path.exists(candidate):
                        return candidate
        return None
    candidate = os.path.normpath(os.path.join(_engine_project_root(), path))
    if os.path.exists(candidate):
        return candidate
    return None


def _layer_category(layer: int) -> int:
    try:
        return (1 << (int(layer) & 31)) & 0xFFFFFFFF
    except Exception:
        return 1


def _mask_value(mask: int) -> int:
    try:
        return int(mask) & 0xFFFFFFFF
    except Exception:
        return 0xFFFF


def _decimate_points(verts: np.ndarray, max_vertices: int) -> np.ndarray:
    n = len(verts)
    if n <= max_vertices or max_vertices < 4:
        return verts
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    extent = maxs - mins
    extent = np.where(extent < 1e-8, 1.0, extent)
    try:
        target_cell_vol = float(np.prod(extent)) / float(max_vertices)
    except Exception:
        return verts
    if not np.isfinite(target_cell_vol) or target_cell_vol <= 0.0:
        return verts
    cell_size = target_cell_vol ** (1.0 / 3.0)
    grid_res = np.maximum(1, np.ceil(extent / cell_size).astype(np.int32))
    indices = np.floor((verts - mins) / extent * grid_res).astype(np.int32)
    indices = np.clip(indices, 0, grid_res - 1)
    cell_ids = indices[:, 0] * grid_res[1] * grid_res[2] + indices[:, 1] * grid_res[2] + indices[:, 2]
    unique_ids, inverse = np.unique(cell_ids, return_inverse=True)
    centroids = np.zeros((len(unique_ids), 3), dtype=np.float32)
    np.add.at(centroids, inverse, verts.astype(np.float32))
    counts = np.bincount(inverse, minlength=len(unique_ids)).astype(np.float32)
    counts[counts == 0.0] = 1.0
    centroids /= counts[:, None]
    return centroids


def _unique_points(verts: np.ndarray) -> np.ndarray:
    if len(verts) > 1:
        try:
            return np.unique(np.ascontiguousarray(verts, dtype=np.float32), axis=0)
        except Exception:
            return verts
    return verts


def _clean_triangle_indices(verts: np.ndarray, indices: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if indices is None:
        return None
    try:
        tris = np.asarray(indices).reshape(-1, 3).astype(np.int64)
    except Exception:
        return None
    n = len(verts)
    if len(tris) == 0 or n == 0:
        return None
    tris = tris[((tris >= 0) & (tris < n)).all(axis=1)]
    if len(tris) == 0:
        return None
    tris = tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]
    if len(tris) == 0:
        return None
    t = verts[tris]
    area2 = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    try:
        ref = float(np.max(np.abs(verts)))
    except Exception:
        ref = 1.0
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    eps = max(1e-12, (ref * 1e-7) ** 2)
    tris = tris[area2 > eps]
    if len(tris) == 0:
        return None
    return np.ascontiguousarray(tris.astype(np.uint32)).reshape(-1)


def _load_mesh_geometry(path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        from core.components.physics.mesh_collider import load_collision_geometry
        verts, indices, _ = load_collision_geometry(path)
        if verts is not None and len(verts) > 0:
            return verts, indices
    except Exception:
        pass
    return None, None


def _load_mesh_tscale(path: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    try:
        from core.components.physics.mesh_collider import load_collision_geometry
        verts, indices, k = load_collision_geometry(path)
        if verts is not None and len(verts) > 0:
            return verts, indices, k
    except Exception:
        pass
    return None, None, 1.0


def _transform_scale(params_scale, import_scale: float) -> tuple[float, float, float]:
    try:
        k = float(import_scale or 1.0)
    except Exception:
        k = 1.0
    if abs(k) < 1e-9:
        k = 1.0
    try:
        return (float(params_scale[0]) / k, float(params_scale[1]) / k, float(params_scale[2]) / k)
    except Exception:
        return (1.0, 1.0, 1.0)


def _quat_to_euler(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    qx, qy, qz, qw = q
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return (roll, pitch, yaw)


def _rotate_vector_by_quat(
    v: tuple[float, float, float],
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    vx, vy, vz = v
    qw, qx, qy, qz = q
    uv_x = qy * vz - qz * vy
    uv_y = qz * vx - qx * vz
    uv_z = qx * vy - qy * vx
    uv2_x = qy * uv_z - qz * uv_y
    uv2_y = qz * uv_x - qx * uv_z
    uv2_z = qx * uv_y - qy * uv_x
    return (
        vx + 2.0 * (qw * uv_x + uv2_x),
        vy + 2.0 * (qw * uv_y + uv2_y),
        vz + 2.0 * (qw * uv_z + uv2_z),
    )


def _rot_vec_by_quat_xyzw(
    v: tuple[float, float, float],
    q: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    vx, vy, vz = v
    qx, qy, qz, qw = q
    uv_x = qy * vz - qz * vy
    uv_y = qz * vx - qx * vz
    uv_z = qx * vy - qy * vx
    uv2_x = qy * uv_z - qz * uv_y
    uv2_y = qz * uv_x - qx * uv_z
    uv2_z = qx * uv_y - qy * uv_x
    return (
        vx + 2.0 * (qw * uv_x + uv2_x),
        vy + 2.0 * (qw * uv_y + uv2_y),
        vz + 2.0 * (qw * uv_z + uv2_z),
    )


def _shifted_pos(
    position: tuple[float, float, float],
    rot_q: tuple[float, float, float, float],
    com: Optional[tuple[float, float, float]],
    sign: float,
) -> tuple[float, float, float]:
    if com is None:
        return position
    try:
        ox, oy, oz = _rot_vec_by_quat_xyzw((float(com[0]), float(com[1]), float(com[2])), rot_q)
    except Exception:
        return position
    return (position[0] + sign * ox, position[1] + sign * oy, position[2] + sign * oz)


def _load_mesh_verts(path: str) -> Optional[np.ndarray]:
    verts, _ = _load_mesh_geometry(path)
    return verts


class CulverinSolver(IPhysicsSolver):

    def __init__(self):
        self._world: Optional[PhysicsWorld] = None
        self._initialized = False
        self._body_count = 0
        self._next_body_id = 1
        self._all_body_ids: list[int] = []
        self._body_handles: list[int] = []
        self._handle_to_id: dict[int, int] = {}
        self._id_to_handle: dict[int, int] = {}
        self._entity_to_body: dict[str, int] = {}
        self._body_to_entity: dict[int, str] = {}
        self._body_specs: dict[int, dict] = {}
        self._gravity: tuple[float, float, float] = (0.0, -9.81, 0.0)
        self._debug_enabled = False
        self._next_joint_id = 1
        self._joint_id_to_handle: dict[int, int] = {}
        self._joint_handle_to_id: dict[int, int] = {}
        self._joint_specs: dict[int, dict] = {}
        self._num_threads = 0
        self._max_physics_jobs = 8
        self._max_physics_barriers = 8
        self._enable_ccd = False
        self._enable_sleeping = True

    def initialize(self, settings: Optional[dict] = None) -> bool:
        if self._initialized:
            return True
        if not _HAS_CULVERIN or PhysicsWorld is None:
            Logger.error("CulverinSolver: culverin not installed. Run: pip install culverin")
            return False
        try:
            opts = settings or {}
            world_opts: dict = {}

            gx = opts.get("gravity_x", 0.0)
            gy = opts.get("gravity_y", -9.81)
            gz = opts.get("gravity_z", 0.0)
            self._gravity = (gx, gy, gz)
            world_opts["gravity"] = (gx, gy, gz)

            self._num_threads = opts.get("culverin_num_threads", 0)
            self._max_physics_jobs = opts.get("culverin_max_physics_jobs", 0)
            self._max_physics_barriers = opts.get("culverin_max_physics_barriers", 0)
            self._enable_ccd = opts.get("culverin_enable_ccd", False)
            self._enable_sleeping = opts.get("culverin_enable_sleeping", True)

            if self._num_threads == 0:
                self._max_physics_jobs = 0
                self._max_physics_barriers = 0

            world_opts["num_threads"] = self._num_threads
            world_opts["max_physics_jobs"] = self._max_physics_jobs
            world_opts["max_physics_barriers"] = self._max_physics_barriers
            world_opts["penetration_slop"] = opts.get("culverin_penetration_slop", 0.02)
            world_opts["max_bodies"] = opts.get("culverin_max_bodies", 65536)
            world_opts["max_pairs"] = opts.get("culverin_max_pairs", 65536)
            world_opts["max_contact_constraints"] = opts.get("culverin_max_contact_constraints", 65536)
            world_opts["temp_allocator_size"] = opts.get("culverin_temp_allocator_size", 16777216)

            self._world = PhysicsWorld(settings=world_opts)
            self._world.set_gravity(gx, gy, gz)
            self._initialized = True
            Logger.info(f"CulverinSolver initialized (threads={self._num_threads}, jobs={self._max_physics_jobs}, ccd={self._enable_ccd}, sleep={self._enable_sleeping})")
            return True
        except Exception as e:
            Logger.error(f"CulverinSolver init failed: {e}")
            return False

    def shutdown(self):
        self._world = None
        self._initialized = False
        self._body_count = 0
        self._next_body_id = 1
        self._all_body_ids.clear()
        self._body_handles.clear()
        self._handle_to_id.clear()
        self._id_to_handle.clear()
        self._entity_to_body.clear()
        self._body_to_entity.clear()
        self._body_specs.clear()
        self._next_joint_id = 1
        self._joint_id_to_handle.clear()
        self._joint_handle_to_id.clear()
        self._joint_specs.clear()
        Logger.info("CulverinSolver shutdown.")

    @property
    def body_count(self) -> int:
        return self._body_count

    @property
    def debug_draw(self):
        return self._debug_enabled

    @debug_draw.setter
    def debug_draw(self, enabled: bool):
        self._debug_enabled = enabled

    def step_simulation(self, dt: float):
        if self._world is not None:
            self._world.step(dt)

    def set_gravity(self, gravity: tuple[float, float, float]):
        self._gravity = gravity
        if self._world is not None:
            self._world.set_gravity(gravity[0], gravity[1], gravity[2])

    def create_rigid_body(
        self,
        entity_id: str,
        shape_type: str,
        shape_params: dict,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        mass: float,
        friction: float = 0.6,
        restitution: float = 0.0,
        is_trigger: bool = False,
        is_kinematic: bool = False,
        collision_layer: int = 0,
        collision_mask: int = 0xFFFF,
    ) -> int:
        if self._world is None:
            return -1

        body_id = self._next_body_id
        self._next_body_id += 1

        if mass <= 0.0 or is_trigger:
            motion = MOTION_STATIC
            mass_val = 0.0
        elif is_kinematic:
            motion = MOTION_KINEMATIC
            mass_val = mass
        else:
            motion = MOTION_DYNAMIC
            mass_val = mass

        rot_q = euler_to_quat(rotation[0], rotation[1], rotation[2])

        handle = -1
        com = None

        if shape_type == "mesh":
            handle, com = self._create_mesh_collider(
                position, rot_q, mass_val, motion, shape_params,
                friction, restitution, is_trigger,
                collision_layer, collision_mask, body_id,
            )
        elif shape_type == "heightfield":
            handle = self._create_heightfield_collider(
                position, rot_q, mass_val, motion, shape_params,
                friction, restitution, is_trigger,
                collision_layer, collision_mask, body_id,
            )
            com = None
        else:
            handle, com = self._create_primitive_body(
                shape_type, shape_params, position, rot_q,
                mass_val, motion, friction, restitution, is_trigger,
                _layer_category(collision_layer), _mask_value(collision_mask),
                body_id,
            )

        if handle is None or handle < 0:
            return -1

        if motion == MOTION_DYNAMIC and self._enable_sleeping:
            self._world.set_linear_velocity(handle, 0.0, -0.001, 0.0)
            self._world.set_linear_velocity(handle, 0.0, 0.0, 0.0)

        self._body_count += 1
        self._all_body_ids.append(body_id)
        self._body_handles.append(handle)
        self._handle_to_id[handle] = body_id
        self._id_to_handle[body_id] = handle
        if entity_id:
            self._entity_to_body[entity_id] = body_id
        self._body_to_entity[body_id] = entity_id
        self._body_specs[body_id] = {
            "entity_id": entity_id,
            "shape_type": shape_type,
            "mass": mass_val,
            "friction": friction,
            "restitution": restitution,
            "is_trigger": is_trigger,
            "is_kinematic": is_kinematic,
            "com": com,
        }

        return body_id

    def create_compound_rigid_body(
        self,
        entity_id: str,
        shapes: list,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
        mass: float,
        friction: float = 0.6,
        restitution: float = 0.0,
        is_trigger: bool = False,
        is_kinematic: bool = False,
        collision_layer: int = 0,
        collision_mask: int = 0xFFFF,
    ) -> int:
        if self._world is None or not shapes:
            return -1

        body_id = self._next_body_id
        self._next_body_id += 1

        if mass <= 0.0 or is_trigger:
            motion = MOTION_STATIC
            mass_val = 0.0
        elif is_kinematic:
            motion = MOTION_KINEMATIC
            mass_val = mass
        else:
            motion = MOTION_DYNAMIC
            mass_val = mass

        try:
            if any(bool(s.get("is_trigger", False)) != bool(is_trigger) for s in shapes):
                Logger.warning("CulverinSolver: mixed trigger/solid colliders share one body, using the first")
        except Exception:
            pass

        rot_q = euler_to_quat(rotation[0], rotation[1], rotation[2])
        category = _layer_category(collision_layer)
        mask = _mask_value(collision_mask)

        try:
            from core.physics.shape_utils import part_volume, part_center, capsule_part_quat, capsule_section_height
        except Exception:
            return -1

        parts: list = []
        total_vol = 0.0
        com_acc = [0.0, 0.0, 0.0]
        for s in shapes:
            stype = s.get("type", "box")
            params = s.get("params", {})
            if stype not in ("box", "sphere", "capsule", "cylinder"):
                continue
            spec = self._compound_part_spec(stype, params, part_center, capsule_part_quat, capsule_section_height)
            if spec is None:
                continue
            pquat, cshape, size = spec
            cen = part_center(stype, params)
            try:
                vol = max(float(part_volume(stype, params)), 1e-9)
            except Exception:
                vol = 1.0
            parts.append(((float(cen[0]), float(cen[1]), float(cen[2])), pquat, cshape, size))
            total_vol += vol
            com_acc[0] += vol * cen[0]
            com_acc[1] += vol * cen[1]
            com_acc[2] += vol * cen[2]

        if not parts:
            return -1

        if total_vol > 0.0:
            com = (com_acc[0] / total_vol, com_acc[1] / total_vol, com_acc[2] / total_vol)
        else:
            com = None
        if motion == MOTION_STATIC:
            com = None

        try:
            handle = self._world.create_compound_body(
                parts=parts,
                pos=position,
                rot=rot_q,
                motion=motion, mass=mass_val,
                user_data=body_id, is_sensor=is_trigger,
                category=category, mask=mask,
                friction=friction, restitution=restitution,
            )
        except Exception as e:
            Logger.warning(f"CulverinSolver: create_compound_body failed: {e}")
            return -1
        if handle is None or handle < 0:
            return -1

        if motion == MOTION_DYNAMIC and self._enable_sleeping:
            self._world.set_linear_velocity(handle, 0.0, -0.001, 0.0)
            self._world.set_linear_velocity(handle, 0.0, 0.0, 0.0)

        self._body_count += 1
        self._all_body_ids.append(body_id)
        self._body_handles.append(handle)
        self._handle_to_id[handle] = body_id
        self._id_to_handle[body_id] = handle
        if entity_id:
            self._entity_to_body[entity_id] = body_id
        self._body_to_entity[body_id] = entity_id
        self._body_specs[body_id] = {
            "entity_id": entity_id,
            "shape_type": "compound",
            "mass": mass_val,
            "friction": friction,
            "restitution": restitution,
            "is_trigger": is_trigger,
            "is_kinematic": is_kinematic,
            "com": com,
        }
        return body_id

    def _compound_part_spec(self, shape_type, params, part_center, capsule_part_quat, capsule_section_height):
        try:
            cen = part_center(shape_type, params)
        except Exception:
            cen = [0.0, 0.0, 0.0]
        if shape_type == "box":
            s = params.get("size", [1, 1, 1])
            try:
                half = (max(float(s[0]) * 0.5, 1e-4), max(float(s[1]) * 0.5, 1e-4), max(float(s[2]) * 0.5, 1e-4))
            except Exception:
                return None
            return (0.0, 0.0, 0.0, 1.0), SHAPE_BOX, half
        if shape_type == "sphere":
            try:
                r = max(float(params.get("radius", 0.5)), 1e-4)
            except Exception:
                return None
            return (0.0, 0.0, 0.0, 1.0), SHAPE_SPHERE, r
        if shape_type == "capsule":
            try:
                r, hsec = capsule_section_height(float(params.get("radius", 0.5)), float(params.get("height", 2.0)))
            except Exception:
                return None
            try:
                direction = int(params.get("direction", 1))
            except Exception:
                direction = 1
            return capsule_part_quat(direction), SHAPE_CAPSULE, (r, hsec * 0.5)
        if shape_type == "cylinder":
            try:
                r = max(float(params.get("radius", 0.5)), 1e-4)
                h = max(float(params.get("height", 1.0)), 1e-4)
            except Exception:
                return None
            return (0.0, 0.0, 0.0, 1.0), SHAPE_CYLINDER, (r, h * 0.5)
        return None

    def _create_primitive_body(
        self, shape_type, shape_params, position, rot_q,
        mass_val, motion, friction, restitution, is_trigger,
        category, mask, body_id,
    ):
        culv_shape = _SHAPE_TYPE_MAP.get(shape_type, SHAPE_BOX)
        try:
            from core.physics.shape_utils import part_center, capsule_part_quat, capsule_section_height
        except Exception:
            return -1, None
        if shape_type in ("box", "sphere", "capsule", "cylinder"):
            try:
                cen = part_center(shape_type, shape_params)
            except Exception:
                cen = [0.0, 0.0, 0.0]
            spec = self._compound_part_spec(shape_type, shape_params, part_center, capsule_part_quat, capsule_section_height)
            if spec is None:
                return -1, None
            pquat, cshape, size = spec
            if abs(cen[0]) + abs(cen[1]) + abs(cen[2]) < 1e-9 and pquat == (0.0, 0.0, 0.0, 1.0):
                try:
                    handle = self._world.create_body(
                        pos=position,
                        rot=rot_q,
                        size=size,
                        shape=cshape,
                        motion=motion,
                        mass=mass_val,
                        friction=friction,
                        restitution=restitution,
                        category=category,
                        mask=mask,
                        is_sensor=is_trigger,
                        user_data=body_id,
                    )
                    return (handle if handle is not None else -1), None
                except Exception:
                    return -1, None
            com = (float(cen[0]), float(cen[1]), float(cen[2])) if motion != MOTION_STATIC else None
            try:
                handle = self._world.create_compound_body(
                    parts=[((float(cen[0]), float(cen[1]), float(cen[2])), pquat, cshape, size)],
                    pos=position,
                    rot=rot_q,
                    motion=motion, mass=mass_val,
                    user_data=body_id, is_sensor=is_trigger,
                    category=category, mask=mask,
                    friction=friction, restitution=restitution,
                )
                return (handle if handle is not None else -1), com
            except Exception as e:
                Logger.warning(f"CulverinSolver: offset primitive failed: {e}")
                return -1, None
        try:
            size = self._shape_size(shape_type, shape_params)
            handle = self._world.create_body(
                pos=position,
                rot=rot_q,
                size=size,
                shape=culv_shape,
                motion=motion,
                mass=mass_val,
                friction=friction,
                restitution=restitution,
                category=category,
                mask=mask,
                is_sensor=is_trigger,
                user_data=body_id,
            )
            return (handle if handle is not None else -1), None
        except Exception:
            return -1, None

    def _shape_size(self, shape_type: str, shape_params: dict):
        if shape_type == "box":
            s = shape_params.get("size", [1, 1, 1])
            return (s[0] / 2.0, s[1] / 2.0, s[2] / 2.0)
        if shape_type == "sphere":
            return shape_params.get("radius", 0.5)
        if shape_type == "capsule":
            radius = shape_params.get("radius", 0.5)
            height = shape_params.get("height", 2.0)
            try:
                from core.physics.shape_utils import capsule_section_height
                _, hsec = capsule_section_height(float(radius), float(height))
                return (max(float(radius), 1e-4), hsec * 0.5)
            except Exception:
                return (radius, height / 2.0)
        if shape_type == "cylinder":
            radius = shape_params.get("radius", 0.5)
            height = shape_params.get("height", 1.0)
            return (radius, height / 2.0)
        if shape_type == "plane":
            n = shape_params.get("normal", [0, 1, 0])
            return (n[0], n[1], n[2], 0.0)
        return (0.5, 0.5, 0.5)

    def _create_mesh_collider(
        self, position, rot_q, mass_val, motion, shape_params,
        friction, restitution, is_trigger, collision_layer, collision_mask, body_id,
    ):
        file_path = shape_params.get("file", "")
        resolved = _resolve_mesh_file(file_path)
        if resolved is None:
            Logger.warning(f"CulverinSolver: mesh file not found: {file_path}")
            return -1, None

        raw_mode = shape_params.get("collision_mode", "auto")
        try:
            mode = str(raw_mode or "auto").lower()
        except Exception:
            mode = "auto"
        if mode not in ("auto", "mesh", "convex_hull", "box", "sphere"):
            Logger.warning(f"CulverinSolver: unknown collision_mode '{raw_mode}', using 'auto'")
            mode = "auto"
        try:
            max_vertices = int(shape_params.get("max_vertices", 0) or 0)
        except Exception:
            max_vertices = 0

        scale = shape_params.get("scale", [1.0, 1.0, 1.0])
        try:
            sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
        except Exception:
            sx = sy = sz = 1.0
        if not all(np.isfinite((sx, sy, sz))) or min(abs(sx), abs(sy), abs(sz)) < 1e-9:
            Logger.warning(f"CulverinSolver: degenerate scale for mesh '{file_path}'")
            return -1, None
        center = shape_params.get("center", [0.0, 0.0, 0.0])
        try:
            cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        except Exception:
            cx = cy = cz = 0.0
        if not all(np.isfinite((cx, cy, cz))):
            cx = cy = cz = 0.0

        category = _layer_category(collision_layer)
        mask = _mask_value(collision_mask)

        if mode == "auto":
            mode = "mesh" if motion == MOTION_STATIC else "convex_hull"
        elif mode == "mesh" and motion != MOTION_STATIC:
            Logger.warning(f"CulverinSolver: concave mesh '{file_path}' requires a static body, using convex hull")
            mode = "convex_hull"

        verts, indices, import_scale = _load_mesh_tscale(resolved)
        if verts is None or len(verts) == 0:
            Logger.warning(f"CulverinSolver: could not read mesh '{file_path}'")
            return -1, None
        tx, ty, tz = _transform_scale(scale, import_scale)
        if not all(np.isfinite((tx, ty, tz))) or min(abs(tx), abs(ty), abs(tz)) < 1e-9:
            Logger.warning(f"CulverinSolver: degenerate scale for mesh '{file_path}'")
            return -1, None
        shaped = verts * np.array([tx, ty, tz], dtype=np.float32)
        if cx != 0.0 or cy != 0.0 or cz != 0.0:
            shaped = shaped + np.array([cx, cy, cz], dtype=np.float32)
        if not np.all(np.isfinite(shaped)):
            Logger.warning(f"CulverinSolver: mesh '{file_path}' contains invalid vertices")
            return -1, None

        if mode == "mesh":
            tris = _clean_triangle_indices(shaped, indices)
            if tris is None:
                Logger.warning(f"CulverinSolver: mesh '{file_path}' has no usable triangles, using convex hull")
            else:
                try:
                    buf_verts = np.ascontiguousarray(shaped, dtype=np.float32).tobytes()
                    buf_indices = np.ascontiguousarray(tris, dtype=np.uint32).tobytes()
                    handle = self._world.create_mesh_body(
                        pos=position, rot=rot_q,
                        vertices=buf_verts, indices=buf_indices,
                        user_data=body_id,
                        category=category, mask=mask,
                    )
                    return (handle if handle is not None else -1), None
                except Exception as e:
                    Logger.warning(f"CulverinSolver: create_mesh_body failed for '{file_path}': {e}")
                    return -1, None

        if mode == "box":
            return self._create_box_approx(
                shaped, position, rot_q, mass_val, motion,
                friction, restitution, is_trigger,
                category, mask, body_id, file_path,
            )

        if mode == "sphere":
            return self._create_sphere_approx(
                shaped, position, rot_q, mass_val, motion,
                friction, restitution, is_trigger,
                category, mask, body_id, file_path,
            )

        points = _unique_points(shaped)
        if max_vertices >= 4 and len(points) > max_vertices:
            points = _decimate_points(points, max_vertices)
        if len(points) < 4:
            Logger.warning(f"CulverinSolver: mesh '{file_path}' is degenerate, using box approximation")
            return self._create_box_approx(
                shaped, position, rot_q, mass_val, motion,
                friction, restitution, is_trigger,
                category, mask, body_id, file_path,
            )
        try:
            com = None
            if motion != MOTION_STATIC and len(points) > 0:
                try:
                    mean = points.astype(np.float64).mean(axis=0)
                    if np.all(np.isfinite(mean)):
                        com = (float(mean[0]), float(mean[1]), float(mean[2]))
                except Exception:
                    com = None
            buf = np.ascontiguousarray(points, dtype=np.float32).tobytes()
            handle = self._world.create_convex_hull(
                pos=position, rot=rot_q, points=buf,
                motion=motion, mass=mass_val,
                user_data=body_id, is_sensor=is_trigger,
                category=category, mask=mask,
                friction=friction, restitution=restitution,
            )
            return (handle if handle is not None else -1), com
        except Exception as e:
            Logger.warning(f"CulverinSolver: create_convex_hull failed for '{file_path}': {e}, using box approximation")
            return self._create_box_approx(
                shaped, position, rot_q, mass_val, motion,
                friction, restitution, is_trigger,
                category, mask, body_id, file_path,
            )

    def _create_box_approx(
        self, shaped, position, rot_q, mass_val, motion,
        friction, restitution, is_trigger, category, mask, body_id, file_path,
    ):
        try:
            mins = shaped.min(axis=0).astype(np.float64)
            maxs = shaped.max(axis=0).astype(np.float64)
        except Exception:
            return -1, None
        half = np.maximum((maxs - mins) * 0.5, 1e-4)
        off = (mins + maxs) * 0.5
        com = (float(off[0]), float(off[1]), float(off[2])) if motion != MOTION_STATIC else None
        try:
            handle = self._world.create_compound_body(
                parts=[((float(off[0]), float(off[1]), float(off[2])),
                        (0.0, 0.0, 0.0, 1.0), SHAPE_BOX,
                        (float(half[0]), float(half[1]), float(half[2])))],
                pos=position, rot=rot_q,
                motion=motion, mass=mass_val,
                user_data=body_id, is_sensor=is_trigger,
                category=category, mask=mask,
                friction=friction, restitution=restitution,
            )
            return (handle if handle is not None else -1), com
        except Exception as e:
            Logger.warning(f"CulverinSolver: box approximation failed for '{file_path}': {e}")
            return -1, None

    def _create_sphere_approx(
        self, shaped, position, rot_q, mass_val, motion,
        friction, restitution, is_trigger, category, mask, body_id, file_path,
    ):
        try:
            sc = shaped.mean(axis=0).astype(np.float64)
            radius = float(np.max(np.linalg.norm(shaped.astype(np.float64) - sc, axis=1)))
        except Exception:
            return -1, None
        radius = max(radius, 1e-4)
        com = (float(sc[0]), float(sc[1]), float(sc[2])) if motion != MOTION_STATIC else None
        try:
            handle = self._world.create_compound_body(
                parts=[((float(sc[0]), float(sc[1]), float(sc[2])),
                        (0.0, 0.0, 0.0, 1.0), SHAPE_SPHERE, radius)],
                pos=position, rot=rot_q,
                motion=motion, mass=mass_val,
                user_data=body_id, is_sensor=is_trigger,
                category=category, mask=mask,
                friction=friction, restitution=restitution,
            )
            return (handle if handle is not None else -1), com
        except Exception as e:
            Logger.warning(f"CulverinSolver: sphere approximation failed for '{file_path}': {e}")
            return -1, None

    def _create_heightfield_collider(
        self, position, rot_q, mass_val, motion, shape_params,
        friction, restitution, is_trigger, collision_layer, collision_mask, body_id,
    ) -> int:
        hd = shape_params.get("height_data")
        size = shape_params.get("size", [1000.0, 60.0, 1000.0])
        resolution = int(shape_params.get("resolution", 0))
        if hd is None or resolution < 2:
            Logger.warning("CulverinSolver: terrain height field not available")
            return -1
        try:
            arr = np.asarray(hd, dtype=np.float32)
            if arr.ndim == 1:
                side = int(np.sqrt(len(arr)))
                arr = arr.reshape(side, side)
            res = arr.shape[0]
            max_verts = 8000
            step = 1
            if res * res > max_verts:
                step = max(1, int(np.ceil(res / int(np.sqrt(max_verts)))))
            sx = size[0] / max(1, res - 1)
            sz = size[2] / max(1, res - 1)
            verts = []
            indices = []
            for z in range(0, res, step):
                for x in range(0, res, step):
                    px = (x - (res - 1) * 0.5) * sx
                    pz = (z - (res - 1) * 0.5) * sz
                    verts.append(px)
                    verts.append(float(arr[z, x]))
                    verts.append(pz)
            w = (res + step - 1) // step
            for r in range(w - 1):
                for c in range(w - 1):
                    a = r * w + c
                    b = r * w + c + 1
                    d = (r + 1) * w + c
                    e = (r + 1) * w + c + 1
                    indices.append(a)
                    indices.append(d)
                    indices.append(b)
                    indices.append(b)
                    indices.append(d)
                    indices.append(e)
            buf_verts = np.array(verts, dtype=np.float32).tobytes()
            buf_indices = np.array(indices, dtype=np.uint32).tobytes()
            return self._world.create_mesh_body(
                pos=position, rot=rot_q,
                vertices=buf_verts, indices=buf_indices,
                user_data=body_id,
                category=_layer_category(collision_layer), mask=_mask_value(collision_mask),
            )
        except Exception as e:
            Logger.warning(f"CulverinSolver: heightfield body failed: {e}")
            return -1

    def remove_rigid_body(self, body_id: int):
        handle = self._id_to_handle.pop(body_id, None)
        if handle is not None and self._world is not None:
            self._world.destroy_body(handle)
            self._handle_to_id.pop(handle, None)
        self._body_specs.pop(body_id, None)
        entity_id = self._body_to_entity.pop(body_id, None)
        if entity_id:
            self._entity_to_body.pop(entity_id, None)
        if body_id in self._all_body_ids:
            self._all_body_ids.remove(body_id)
        self._body_count = max(0, self._body_count - 1)

    def remove_all_bodies(self):
        if self._world is not None:
            for handle in list(self._body_handles):
                self._world.destroy_body(handle)
        self._body_handles.clear()
        self._handle_to_id.clear()
        self._id_to_handle.clear()
        self._all_body_ids.clear()
        self._entity_to_body.clear()
        self._body_to_entity.clear()
        self._body_specs.clear()
        self._body_count = 0

    def _body_com(self, body_id: int) -> Optional[tuple[float, float, float]]:
        try:
            spec = self._body_specs.get(body_id)
            if spec is None:
                return None
            return spec.get("com")
        except Exception:
            return None

    def set_body_transform(
        self,
        body_id: int,
        position: tuple[float, float, float],
        rotation: tuple[float, float, float],
    ):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        rot_q = euler_to_quat(rotation[0], rotation[1], rotation[2])
        self._world.set_transform(handle, position, rot_q)

    def set_body_transform_quat(
        self,
        body_id: int,
        position: tuple[float, float, float],
        quat: tuple[float, float, float, float],
    ):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        self._world.set_transform(handle, position, quat)

    def get_body_transform(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        pos, quat = self.get_body_transform_quat(body_id)
        euler = _quat_to_euler((quat[0], quat[1], quat[2], quat[3]))
        return (pos, euler)

    def get_body_transform_quat(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        pos = self._world.get_position(handle)
        rot = self._world.get_rotation(handle)
        if pos is None or rot is None:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        quat = (rot[0], rot[1], rot[2], rot[3])
        pos = _shifted_pos((pos[0], pos[1], pos[2]), quat, self._body_com(body_id), -1.0)
        return (pos, quat)

    def apply_force(
        self, body_id: int, force: tuple[float, float, float], local: bool = False
    ):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        if local:
            rot = self._world.get_rotation(handle)
            if rot is not None:
                force = _rotate_vector_by_quat(force, rot)
        self._world.apply_force(handle, force[0], force[1], force[2])

    def apply_torque(self, body_id: int, torque: tuple[float, float, float]):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        self._world.apply_torque(handle, torque[0], torque[1], torque[2])

    def activate(self, body_id: int):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        try:
            self._world.activate(handle)
        except Exception:
            pass

    def apply_impulse(
        self, body_id: int, impulse: tuple[float, float, float], local: bool = False
    ):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        if local:
            rot = self._world.get_rotation(handle)
            if rot is not None:
                impulse = _rotate_vector_by_quat(impulse, rot)
        self._world.apply_impulse(handle, impulse[0], impulse[1], impulse[2])

    def set_velocity(self, body_id: int, velocity: tuple[float, float, float]):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        self._world.set_linear_velocity(handle, velocity[0], velocity[1], velocity[2])

    def set_velocities(
        self, body_id: int,
        linear: Optional[tuple[float, float, float]] = None,
        angular: Optional[tuple[float, float, float]] = None,
    ):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        if linear is not None:
            self._world.set_linear_velocity(handle, linear[0], linear[1], linear[2])
        if angular is not None:
            self._world.set_angular_velocity(handle, angular[0], angular[1], angular[2])

    def get_velocity(self, body_id: int) -> tuple[float, float, float]:
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return (0.0, 0.0, 0.0)
        v = self._world.get_velocity(handle)
        return v if v is not None else (0.0, 0.0, 0.0)

    def get_velocities(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        v = self._world.get_velocity(handle)
        a = self._world.get_angular_velocity(handle)
        return (
            v if v is not None else (0.0, 0.0, 0.0),
            a if a is not None else (0.0, 0.0, 0.0),
        )

    def set_angular_velocity(
        self, body_id: int, velocity: tuple[float, float, float]
    ):
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return
        self._world.set_angular_velocity(handle, velocity[0], velocity[1], velocity[2])

    def get_angular_velocity(
        self, body_id: int
    ) -> tuple[float, float, float]:
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return (0.0, 0.0, 0.0)
        av = self._world.get_angular_velocity(handle)
        return av if av is not None else (0.0, 0.0, 0.0)

    def ray_cast(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float = 100.0,
    ) -> Optional[dict]:
        if self._world is None:
            return None
        dx, dy, dz = direction
        mag = (dx * dx + dy * dy + dz * dz) ** 0.5
        if mag < 1e-10:
            return None
        result = self._world.raycast(origin, direction, max_distance)
        if result is None:
            return None
        handle, fraction, position = result[0], result[1], result[2]
        body_id = self._handle_to_id.get(handle, -1)
        return {
            "body_id": body_id,
            "position": position,
            "normal": (0.0, 0.0, 0.0),
            "fraction": fraction,
        }

    def get_collision_events(self) -> list[dict]:
        if self._world is None:
            return []
        raw = self._world.get_contact_events()
        out = []
        for ev in raw:
            ba = self._handle_to_id.get(ev[0], -1)
            bb = self._handle_to_id.get(ev[1], -1)
            if ba < 0 or bb < 0:
                continue
            out.append({"body_a": ba, "body_b": bb})
        return out

    def add_plane(
        self,
        normal: tuple[float, float, float] = (0, 1, 0),
        distance: float = 0.0,
        friction: float = 0.6,
        restitution: float = 0.0,
    ) -> int:
        if self._world is None:
            return -1
        body_id = self._next_body_id
        self._next_body_id += 1

        handle = self._world.create_body(
            pos=(0.0, -distance, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            size=(normal[0], normal[1], normal[2], 0.0),
            shape=SHAPE_PLANE,
            motion=MOTION_STATIC,
            mass=0.0,
            friction=friction,
            restitution=restitution,
            user_data=body_id,
        )
        if handle is None or handle < 0:
            return -1

        self._body_count += 1
        self._all_body_ids.append(body_id)
        self._body_handles.append(handle)
        self._handle_to_id[handle] = body_id
        self._id_to_handle[body_id] = handle
        self._body_specs[body_id] = {
            "entity_id": "",
            "shape_type": "plane",
            "mass": 0.0,
            "friction": friction,
            "restitution": restitution,
            "is_trigger": False,
            "is_kinematic": False,
        }
        return body_id

    def create_joint(
        self,
        joint_type: str,
        body_a_id: int,
        body_b_id: int,
        anchor: tuple[float, float, float],
        axis: tuple[float, float, float] = (0, 0, 1),
        limit_low: float = -3.14159,
        limit_high: float = 3.14159,
        stiffness: float = 10.0,
        damping: float = 1.0,
    ) -> int:
        if self._world is None:
            return -1

        handle_a = self._id_to_handle.get(body_a_id)
        if handle_a is None:
            return -1

        handle_b = self._id_to_handle.get(body_b_id) if body_b_id >= 0 else -1
        if handle_b is None:
            return -1

        ctype = _JOINT_TYPE_MAP.get(joint_type, CONSTRAINT_FIXED)

        if handle_b < 0:
            Logger.warning("CulverinSolver: world constraints not supported")
            return -1

        params = None
        if ctype == CONSTRAINT_FIXED:
            params = None
        elif ctype == CONSTRAINT_POINT:
            params = anchor
        elif ctype in (CONSTRAINT_HINGE, CONSTRAINT_SLIDER):
            params = (anchor, axis, limit_low, limit_high)
        elif ctype == CONSTRAINT_DISTANCE:
            params = (limit_low, limit_high)
        elif ctype == CONSTRAINT_CONE:
            params = (anchor, axis, limit_high)

        motor = None
        if joint_type == "spring" and stiffness > 0:
            motor = {"type": 0, "target": stiffness}

        c_handle = self._world.create_constraint(
            ctype, handle_a, handle_b, params=params, motor=motor,
        )
        if c_handle is None or c_handle < 0:
            return -1

        joint_id = self._next_joint_id
        self._next_joint_id += 1
        self._joint_id_to_handle[joint_id] = c_handle
        self._joint_handle_to_id[c_handle] = joint_id
        self._joint_specs[joint_id] = {
            "type": ctype,
            "body_a": handle_a,
            "body_b": handle_b,
            "params": params,
            "motor": motor,
        }

        return joint_id

    def remove_joint(self, joint_id: int):
        c_handle = self._joint_id_to_handle.pop(joint_id, None)
        if c_handle is not None and self._world is not None:
            self._world.destroy_constraint(c_handle)
            self._joint_handle_to_id.pop(c_handle, None)
        self._joint_specs.pop(joint_id, None)

    def remove_all_joints(self):
        if self._world is not None:
            for c_handle in list(self._joint_handle_to_id.keys()):
                self._world.destroy_constraint(c_handle)
        self._joint_id_to_handle.clear()
        self._joint_handle_to_id.clear()
        self._joint_specs.clear()

    def change_constraint(
        self,
        constraint_id: int,
        pivot: tuple[float, float, float],
        max_force: float = 500,
    ):
        spec = self._joint_specs.get(constraint_id)
        if spec is None or self._world is None:
            return
        old_handle = self._joint_id_to_handle.get(constraint_id)
        if old_handle is None:
            return
        self._world.destroy_constraint(old_handle)
        self._joint_handle_to_id.pop(old_handle, None)

        new_params = spec["params"]
        if spec["type"] == CONSTRAINT_POINT:
            new_params = pivot
        elif spec["type"] in (CONSTRAINT_HINGE, CONSTRAINT_SLIDER):
            new_params = (pivot, spec["params"][1], spec["params"][2], spec["params"][3])

        new_handle = self._world.create_constraint(
            spec["type"],
            spec["body_a"],
            spec["body_b"],
            params=new_params,
            motor=spec["motor"],
        )
        if new_handle is not None and new_handle >= 0:
            self._joint_id_to_handle[constraint_id] = new_handle
            self._joint_handle_to_id[new_handle] = constraint_id
            spec["params"] = new_params

    def set_motor_target(self, constraint_id: int, target: float):
        if self._world is None:
            return
        c_handle = self._joint_id_to_handle.get(constraint_id)
        if c_handle is None:
            return
        try:
            self._world.set_constraint_target(c_handle, float(target))
        except Exception as e:
            Logger.debug(f"CulverinSolver.set_motor_target({constraint_id}) failed: {e}")

    def enable_constraint(self, constraint_id: int, motor: Optional[dict]) -> bool:
        if self._world is None:
            return False
        spec = self._joint_specs.get(constraint_id)
        if spec is None:
            return False
        old = self._joint_id_to_handle.get(constraint_id)
        if old is None:
            return False
        self._world.destroy_constraint(old)
        self._joint_handle_to_id.pop(old, None)
        spec["motor"] = motor
        new_handle = self._world.create_constraint(
            spec["type"], spec["body_a"], spec["body_b"],
            params=spec["params"], motor=motor,
        )
        if new_handle is None or new_handle < 0:
            return False
        self._joint_id_to_handle[constraint_id] = new_handle
        self._joint_handle_to_id[new_handle] = constraint_id
        return True

    def create_character(
        self,
        pos: tuple[float, float, float],
        height: float = 1.8,
        radius: float = 0.4,
        step_height: float = 0.4,
        max_slope: float = 45.0,
    ):
        if self._world is None:
            return None
        try:
            return self._world.create_character(
                pos,
                height=height,
                radius=radius,
                step_height=step_height,
                max_slope=max_slope,
            )
        except Exception as e:
            Logger.error(f"CulverinSolver.create_character failed: {e}")
            return None

    def move_character(self, character, velocity: tuple[float, float, float], dt: float):
        if character is None:
            return
        try:
            character.move(velocity, dt)
        except Exception as e:
            Logger.error(f"CulverinSolver.move_character failed: {e}")

    def set_character_rotation(self, character, rot: tuple[float, float, float, float]):
        if character is None:
            return
        try:
            character.set_rotation(rot)
        except Exception:
            pass

    def get_character_position(self, character):
        if character is None:
            return None
        try:
            return character.get_position()
        except Exception:
            return None

    def is_character_grounded(self, character) -> bool:
        if character is None:
            return False
        try:
            return bool(character.is_grounded())
        except Exception:
            return False

    def set_character_strength(self, character, strength: float):
        if character is None:
            return
        try:
            character.set_strength(strength)
        except Exception:
            pass

    def destroy_character(self, character):
        if character is None or self._world is None:
            return
        try:
            self._world.destroy_body(character.handle)
        except Exception:
            pass
