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

_LOADED_MESH_VERTS: dict[str, np.ndarray] = {}


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


def _load_mesh_verts(path: str) -> Optional[np.ndarray]:
    key = path.lower().replace("\\", "/")
    if key in _LOADED_MESH_VERTS:
        return _LOADED_MESH_VERTS[key]
    try:
        from core.assets.asset_importer import load_mesh
        data = load_mesh(path)
    except Exception:
        return None
    if data is None or len(data.vertices) == 0:
        return None
    verts = data.vertices.reshape(-1, 3).astype(np.float32)
    _LOADED_MESH_VERTS[key] = verts
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
            self._enable_sleeping = opts.get("culverin_enable_sleeping", False)

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
        culv_shape = _SHAPE_TYPE_MAP.get(shape_type, SHAPE_BOX)

        if shape_type == "mesh":
            handle = self._create_mesh_collider(
                position, rot_q, mass_val, motion, shape_params,
                friction, restitution, is_trigger,
                collision_layer, collision_mask, body_id,
            )
        else:
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
                category=collision_mask,
                mask=collision_mask,
                is_sensor=is_trigger,
                user_data=body_id,
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
        }

        return body_id

    def _shape_size(self, shape_type: str, shape_params: dict):
        if shape_type == "box":
            s = shape_params.get("size", [1, 1, 1])
            return (s[0] / 2.0, s[1] / 2.0, s[2] / 2.0)
        if shape_type == "sphere":
            return shape_params.get("radius", 0.5)
        if shape_type == "capsule":
            radius = shape_params.get("radius", 0.5)
            height = shape_params.get("height", 2.0)
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
    ) -> int:
        file_path = shape_params.get("file", "")
        if not file_path:
            return -1
        resolved = file_path
        if not os.path.isabs(resolved):
            proj_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
            candidate = os.path.normpath(os.path.join(proj_root, resolved))
            if os.path.exists(candidate):
                resolved = candidate
        if not os.path.exists(resolved):
            Logger.warning(f"CulverinSolver: mesh file not found: {file_path}")
            return -1

        collision_mode = shape_params.get("collision_mode", "convex_hull")
        verts = _load_mesh_verts(resolved)
        scale = tuple(shape_params.get("scale", [1, 1, 1]))
        if verts is not None and scale != (1.0, 1.0, 1.0):
            verts = verts * np.array(scale, dtype=np.float32)

        if motion == MOTION_STATIC and collision_mode == "mesh":
            if verts is None:
                return -1
            try:
                from core.assets.asset_importer import load_mesh
                data = load_mesh(resolved)
                if data is not None and hasattr(data, 'indices') and len(data.indices) > 0:
                    indices = data.indices.reshape(-1).astype(np.uint32)
                else:
                    tri_count = len(verts) // 3
                    indices = np.arange(tri_count * 3, dtype=np.uint32)
                buf_verts = verts.reshape(-1).astype(np.float32).tobytes()
                buf_indices = indices.tobytes()
                return self._world.create_mesh_body(
                    pos=position, rot=rot_q,
                    vertices=buf_verts, indices=buf_indices,
                    user_data=body_id,
                    category=collision_mask, mask=collision_mask,
                )
            except Exception as e:
                Logger.warning(f"CulverinSolver: create_mesh_body failed: {e}")
                return -1

        if verts is None:
            return -1

        try:
            buf = verts.reshape(-1).astype(np.float32).tobytes()
            handle = self._world.create_convex_hull(
                pos=position, rot=rot_q, points=buf,
                motion=motion, mass=mass_val,
                user_data=body_id, is_sensor=is_trigger,
                category=collision_mask, mask=collision_mask,
                friction=friction, restitution=restitution,
            )
            return handle
        except Exception as e:
            Logger.warning(f"CulverinSolver: create_convex_hull failed: {e}")
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

    def get_body_transform(
        self, body_id: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        handle = self._id_to_handle.get(body_id)
        if handle is None or self._world is None:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        pos = self._world.get_position(handle)
        rot = self._world.get_rotation(handle)
        if pos is None or rot is None:
            return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        euler = _quat_to_euler((rot[0], rot[1], rot[2], rot[3]))
        return (pos, euler)

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
