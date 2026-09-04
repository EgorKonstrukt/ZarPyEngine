# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
from typing import Optional, TYPE_CHECKING
from core.foundation.logger import Logger
from core.physics.shape_utils import find_shape_info, find_shapes_info, make_shape_key, make_shapes_key, partition_compound_shapes

if TYPE_CHECKING:
    from core.ecs.ecs import Entity, Scene
    from core.physics.physics_solver import IPhysicsSolver


def create_soft_body_in_solver(solver, *, vertices, indices, scale_xyz, pos_xyz,
                               euler_rad_xyz, params, entity_id=""):
    try:
        sx, sy, sz = float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])
        lv = (np.asarray(vertices).reshape(-1, 3).astype(np.float64)
              * np.array([sx, sy, sz], dtype=np.float64)).astype(np.float32)
        if not np.all(np.isfinite(lv)):
            return None, 0, None, None
        p = params or {}
        bend = p.get("bend_mode", "distance")
        bend = bend.value if hasattr(bend, "value") else str(bend)
        pin = p.get("pin_mode", "none")
        pin = pin.value if hasattr(pin, "value") else str(pin)
        sid = solver.create_soft_body(
            entity_id=entity_id,
            vertices=lv,
            indices=np.asarray(indices).reshape(-1),
            position=(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])),
            rotation=(float(euler_rad_xyz[0]), float(euler_rad_xyz[1]), float(euler_rad_xyz[2])),
            mass=p.get("mass", 1.0),
            compliance=p.get("compliance", 0.001),
            bend_mode=bend,
            pressure=p.get("pressure", 0.0),
            damping=p.get("damping", 0.1),
            iterations=p.get("iterations", 10),
            gravity_factor=p.get("gravity_factor", 1.0),
            friction=p.get("friction", 0.6),
            restitution=p.get("restitution", 0.0),
            vertex_radius=p.get("vertex_radius", 0.05),
            max_velocity=p.get("max_velocity", 500.0),
            max_vertices=p.get("max_vertices", 0),
            pin_mode=pin,
            pin_fraction=p.get("pin_fraction", 0.1),
            double_sided=p.get("double_sided", True),
            update_com=p.get("update_com", True),
            collision_layer=p.get("layer", 0),
            collision_mask=p.get("mask", 0xFFFF),
        )
    except Exception as e:
        Logger.warning(f"[SoftBody] '{entity_id}' creation exception: {e}")
        return None, 0, None, None
    if sid is None or sid >= 0:
        Logger.warning(f"[SoftBody] '{entity_id}' solver returned invalid id: {sid}")
        return None, 0, None, None
    try:
        n = solver.get_soft_body_count(sid)
    except Exception:
        n = 0
    if not n or n <= 0:
        Logger.warning(f"[SoftBody] '{entity_id}' solver returned 0 vertices, removing")
        try:
            solver.remove_soft_body(sid)
        except Exception:
            pass
        return None, 0, None, None
    try:
        geom = solver.get_soft_body_geometry(sid)
    except Exception:
        geom = None
    sverts, sfaces = None, None
    if geom is not None:
        try:
            sverts, sfaces = geom
        except Exception:
            sverts, sfaces = None, None
    return sid, int(n), sverts, sfaces


class PhysicsScene:
    _ZERO_VEC3 = None
    _ZERO_VEC2 = None

    def __init__(self, solver: IPhysicsSolver):
        self._solver = solver
        self._entity_to_body: dict[str, int] = {}
        self._body_to_entity: dict[int, str] = {}
        self._entity_to_extra_bodies: dict[str, list[int]] = {}
        self._entity_to_soft: dict[str, int] = {}
        self._cached_soft_sig: dict[str, tuple] = {}
        self._entity_body_cache: dict[str, tuple] = {}
        self._entity_to_joint: dict[str, int] = {}
        self._joint_to_entity: dict[int, str] = {}
        self._cached_shape: dict[str, tuple] = {}
        self._cached_shape_info: dict[str, list] = {}
        self._prev_frame_contacts: set[frozenset[int]] = set()
        self._scene: Optional[Scene] = None
        self._2d_bodies: set[int] = set()
        self._shape_check_counter: int = 0
        self._body_items: list[tuple[str, int]] = []
        self._body_items_dirty: bool = False
        self._has_collision_scripts: bool = False
        self._collision_scripts_checked: bool = False
        self._collision_listener_sig: tuple = None


    @property
    def solver(self) -> IPhysicsSolver:
        return self._solver

    def initialize(self, scene: Scene):
        self._scene = scene
        Logger.info("PhysicsScene initialized.")

    def shutdown(self):
        self._solver.remove_all_joints()
        self._solver.remove_all_bodies()
        try:
            self._solver.remove_all_soft_bodies()
        except Exception:
            pass
        self._entity_to_body.clear()
        self._body_to_entity.clear()
        self._entity_to_extra_bodies.clear()
        self._entity_to_soft.clear()
        self._cached_soft_sig.clear()
        self._entity_body_cache.clear()
        self._entity_to_joint.clear()
        self._joint_to_entity.clear()
        self._cached_shape.clear()
        self._cached_shape_info.clear()
        self._prev_frame_contacts.clear()
        self._2d_bodies.clear()
        self._has_collision_scripts = False
        self._collision_scripts_checked = False
        self._scene = None

    def load_scene(self, scene: Scene):
        self._scene = scene
        self._solver.remove_all_joints()
        self._solver.remove_all_bodies()
        try:
            self._solver.remove_all_soft_bodies()
        except Exception:
            pass
        self._entity_to_body.clear()
        self._body_to_entity.clear()
        self._entity_to_extra_bodies.clear()
        self._entity_to_soft.clear()
        self._cached_soft_sig.clear()
        self._entity_body_cache.clear()
        self._entity_to_joint.clear()
        self._joint_to_entity.clear()
        self._cached_shape.clear()
        self._cached_shape_info.clear()
        self._2d_bodies.clear()
        self._has_collision_scripts = False
        self._collision_scripts_checked = False
        entities = scene.get_all_entities()
        for entity in entities:
            self._create_entity_bodies(entity)
        for entity in entities:
            self._create_entity_soft(entity)
        for entity in entities:
            self._create_entity_joints(entity)

    def _soft_component(self, entity):
        try:
            from core.components.physics.soft_body import SoftBody
            comp = entity.get_component(SoftBody)
            if comp is not None and getattr(comp, "enabled", True):
                return comp
        except Exception:
            pass
        return None

    def _diag_soft_once(self, msg: str):
        if not getattr(self, "_soft_diag", False):
            self._soft_diag = True
            Logger.warning(f"[SoftBody] {msg}")

    def _create_entity_bodies(self, entity: Entity):
        from core.components import Rigidbody, Rigidbody2D

        if self._soft_component(entity) is not None:
            return
        rb = entity.get_component(Rigidbody)
        rb2d = entity.get_component(Rigidbody2D)
        tr = entity.transform
        if (not rb and not rb2d) or not tr:
            return

        is_2d = rb2d is not None
        effective_rb = rb2d if is_2d else rb

        shapes = find_shapes_info(entity, tr)
        if not shapes:
            return

        if entity.id in self._entity_to_body:
            return

        if is_2d:
            pos = (tr.local_position.x, tr.local_position.y, 0.0)
            euler = tr.local_euler_angles
            rot = (0.0, 0.0, math.radians(euler.z))
        else:
            pos = (tr.local_position.x, tr.local_position.y, tr.local_position.z)
            euler = tr.local_euler_angles
            rot = (math.radians(euler.x), math.radians(euler.y), math.radians(euler.z))

        if is_2d:
            mass = 0.0 if rb2d.is_kinematic else rb2d.mass
            is_kinematic = rb2d.is_kinematic
        else:
            mass = 0.0 if rb.is_kinematic else rb.mass
            is_kinematic = rb.is_kinematic

        if is_2d or len(shapes) == 1:
            shape_info = shapes[0]
            body_id = self._solver.create_rigid_body(
                entity_id=entity.id,
                shape_type=shape_info["type"],
                shape_params=shape_info["params"],
                position=pos,
                rotation=rot,
                mass=mass,
                friction=shape_info.get("friction", 0.6),
                restitution=shape_info.get("restitution", 0.0),
                is_trigger=shape_info.get("is_trigger", False),
                is_kinematic=is_kinematic,
                collision_layer=shape_info.get("layer", 0),
                collision_mask=shape_info.get("mask", 0xFFFF),
            )
            extra_shapes: list[dict] = []
        else:
            dynamic = (mass > 0.0) and (not is_kinematic) and (not shapes[0].get("is_trigger", False))
            prims, extra_shapes = partition_compound_shapes(shapes, dynamic)
            if len(prims) == 1 and not extra_shapes:
                shape_info = prims[0]
                body_id = self._solver.create_rigid_body(
                    entity_id=entity.id,
                    shape_type=shape_info["type"],
                    shape_params=shape_info["params"],
                    position=pos,
                    rotation=rot,
                    mass=mass,
                    friction=shape_info.get("friction", 0.6),
                    restitution=shape_info.get("restitution", 0.0),
                    is_trigger=shape_info.get("is_trigger", False),
                    is_kinematic=is_kinematic,
                    collision_layer=shape_info.get("layer", 0),
                    collision_mask=shape_info.get("mask", 0xFFFF),
                )
            else:
                first = shapes[0]
                if not prims:
                    shape_info = first
                    body_id = self._solver.create_rigid_body(
                        entity_id=entity.id,
                        shape_type=shape_info["type"],
                        shape_params=shape_info["params"],
                        position=pos,
                        rotation=rot,
                        mass=mass,
                        friction=shape_info.get("friction", 0.6),
                        restitution=shape_info.get("restitution", 0.0),
                        is_trigger=shape_info.get("is_trigger", False),
                        is_kinematic=is_kinematic,
                        collision_layer=shape_info.get("layer", 0),
                        collision_mask=shape_info.get("mask", 0xFFFF),
                    )
                    extra_shapes = [s for s in shapes if s is not first]
                else:
                    anchor = prims[0]
                    body_id = self._solver.create_compound_rigid_body(
                        entity_id=entity.id,
                        shapes=prims,
                        position=pos,
                        rotation=rot,
                        mass=mass,
                        friction=anchor.get("friction", 0.6),
                        restitution=anchor.get("restitution", 0.0),
                        is_trigger=anchor.get("is_trigger", False),
                        is_kinematic=is_kinematic,
                        collision_layer=anchor.get("layer", 0),
                        collision_mask=anchor.get("mask", 0xFFFF),
                    )
        if body_id >= 0:
            self._entity_to_body[entity.id] = body_id
            self._body_to_entity[body_id] = entity.id
            self._entity_body_cache[entity.id] = (entity, effective_rb, tr, is_2d)
            effective_rb._body_id = body_id
            self._mark_body_items_dirty()
            if is_2d:
                self._2d_bodies.add(body_id)
            key = make_shapes_key(shapes)
            self._cached_shape[entity.id] = key
            self._cached_shape_info[entity.id] = shapes
            if extra_shapes:
                body_static = (mass <= 0.0) or is_kinematic or shapes[0].get("is_trigger", False)
                for extra in extra_shapes:
                    self._create_extra_body(entity, extra, pos, rot, mass, is_kinematic, body_static)

    def _create_extra_body(self, entity, shape_info: dict, pos, rot, mass: float, is_kinematic: bool, body_static: bool):
        extra_static = body_static or shape_info["type"] == "heightfield"
        try:
            extra_id = self._solver.create_rigid_body(
                entity_id=entity.id,
                shape_type=shape_info["type"],
                shape_params=shape_info["params"],
                position=pos,
                rotation=rot,
                mass=0.0 if extra_static else (0.0 if is_kinematic else mass),
                friction=shape_info.get("friction", 0.6),
                restitution=shape_info.get("restitution", 0.0),
                is_trigger=shape_info.get("is_trigger", False),
                is_kinematic=True if extra_static else is_kinematic,
                collision_layer=shape_info.get("layer", 0),
                collision_mask=shape_info.get("mask", 0xFFFF),
            )
        except Exception:
            return
        if extra_id is not None and extra_id >= 0:
            self._body_to_entity[extra_id] = entity.id
            self._entity_to_extra_bodies.setdefault(entity.id, []).append(extra_id)

    def _sync_extra_bodies(self):
        if not self._entity_to_extra_bodies:
            return
        try:
            from core.math_helpers import quat_to_euler_rad
        except ImportError:
            return
        for entity_id, extras in self._entity_to_extra_bodies.items():
            cached = self._entity_body_cache.get(entity_id)
            if not cached:
                continue
            entity, rb, tr, is_2d = cached
            if not entity._active:
                continue
            if not getattr(tr, "_physics_dirty", False):
                continue
            p = tr._local_pos
            if is_2d:
                bpos = (p._x, p._y, 0.0)
                brot = (0.0, 0.0, quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y, tr._local_rot._z, tr._local_rot._w)[2])
            else:
                e = quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y, tr._local_rot._z, tr._local_rot._w)
                bpos = (p._x, p._y, p._z)
                brot = e
            for extra_id in extras:
                try:
                    self._solver.set_body_transform(extra_id, bpos, brot)
                except Exception:
                    pass

    @staticmethod
    def _soft_sig(entity, soft, tr) -> Optional[tuple]:
        try:
            from core.components.rendering.renderers.mesh_filter import MeshFilter
            mf = entity.get_component(MeshFilter)
            mesh_path = (mf.mesh_path or "") if mf else ""
            bend = soft.bend_mode.value if hasattr(soft.bend_mode, "value") else soft.bend_mode
            pin = soft.pin_mode.value if hasattr(soft.pin_mode, "value") else soft.pin_mode
            return (
                mesh_path, bool(soft.enabled),
                float(soft.mass), float(soft.compliance), str(bend), float(soft.pressure),
                float(soft.damping), int(soft.iterations), float(soft.gravity_scale),
                float(soft.vertex_radius), float(soft.max_velocity), str(pin), float(soft.pin_fraction),
                int(soft.max_vertices), bool(soft.double_sided), bool(soft.update_com),
                int(soft.layer), int(soft.mask),
                float(tr.local_scale.x), float(tr.local_scale.y), float(tr.local_scale.z),
            )
        except Exception:
            return None

    @staticmethod
    def _soft_applied_pose(tr):
        try:
            p = tr._local_pos
            q = tr._local_rot
            s = tr._local_scale
            return (float(p._x), float(p._y), float(p._z),
                    float(q._x), float(q._y), float(q._z), float(q._w),
                    float(s._x), float(s._y), float(s._z))
        except Exception:
            return None

    def _soft_rebuild_for_teleport(self, entity, soft, tr) -> bool:
        try:
            tr._physics_dirty = False
        except Exception:
            pass
        self._remove_entity_soft(entity.id)
        ok = self._build_entity_soft(entity, soft, tr)
        if not ok:
            return False
        try:
            self._cached_soft_sig[entity.id] = self._soft_sig(entity, soft, tr)
        except Exception:
            pass
        try:
            pose = self._soft_applied_pose(tr)
            if pose is not None:
                soft._soft_applied_pose = pose
        except Exception:
            pass
        return True

    def _soft_check_user_teleport(self, entity, soft, tr) -> bool:
        try:
            if not getattr(tr, "_physics_dirty", False):
                return False
        except Exception:
            return False
        try:
            prev = getattr(soft, "_soft_applied_pose", None)
        except Exception:
            prev = None
        if prev is None:
            return self._soft_rebuild_for_teleport(entity, soft, tr)
        cur = self._soft_applied_pose(tr)
        if cur is None:
            try:
                tr._physics_dirty = False
            except Exception:
                pass
            return False
        try:
            dp = abs(cur[0] - prev[0]) + abs(cur[1] - prev[1]) + abs(cur[2] - prev[2])
            dq = 1.0 - abs(cur[3] * prev[3] + cur[4] * prev[4] + cur[5] * prev[5] + cur[6] * prev[6])
            ds = abs(cur[7] - prev[7]) + abs(cur[8] - prev[8]) + abs(cur[9] - prev[9])
        except Exception:
            return False
        if dp > 1e-6 or dq > 1e-8 or ds > 1e-9:
            return self._soft_rebuild_for_teleport(entity, soft, tr)
        try:
            tr._physics_dirty = False
        except Exception:
            pass
        return False

    @staticmethod
    def _soft_pin_active(soft) -> bool:
        try:
            pm = getattr(soft, "pin_mode", None)
            v = pm.value if hasattr(pm, "value") else pm
            return str(v or "none").lower() in ("top", "bottom")
        except Exception:
            return False

    def _soft_apply_com(self, entity, soft, tr, sid=None, com=None) -> bool:
        try:
            if not bool(getattr(soft, "update_com", True)):
                return False
        except Exception:
            return False
        if self._soft_pin_active(soft):
            return False
        if com is None:
            if sid is None:
                return False
            try:
                com = self._solver.get_soft_body_com(sid)
            except Exception:
                return False
        try:
            pos, rot = com
            cx, cy, cz = float(pos[0]), float(pos[1]), float(pos[2])
            qx, qy, qz, qw = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
        except Exception:
            return False
        if not all(map(lambda v: v == v and abs(v) != float("inf"), (cx, cy, cz, qx, qy, qz, qw))):
            return False
        try:
            from core.maths.math3d import Vec3, Quat
            n = qx * qx + qy * qy + qz * qz + qw * qw
            if n < 1e-20:
                return False
            inv = 1.0 / (n ** 0.5)
            wq = Quat(qx * inv, qy * inv, qz * inv, qw * inv)
            parent = entity._parent if hasattr(entity, "_parent") else None
            if parent is None and hasattr(entity, "parent"):
                try:
                    parent = entity.parent
                except Exception:
                    parent = None
            if parent is not None:
                try:
                    pt = parent.transform
                except Exception:
                    pt = None
                if pt is not None:
                    try:
                        pt._update_world_matrix()
                        pwq = pt._world_matrix.decompose()[1]
                        pwc = pwq.conjugate().normalized()
                        lq = (pwc * wq).normalized()
                    except Exception:
                        return False
                    try:
                        tr.position = Vec3(cx, cy, cz)
                    except Exception:
                        return False
                    try:
                        tr._local_rot = lq
                        tr._mark_dirty()
                    except Exception:
                        pass
                else:
                    try:
                        tr.position = Vec3(cx, cy, cz)
                    except Exception:
                        return False
                    try:
                        tr._local_rot = wq
                        tr._mark_dirty()
                    except Exception:
                        pass
            else:
                try:
                    tr._local_pos = Vec3(cx, cy, cz)
                    tr._local_rot = wq
                    tr._mark_dirty()
                except Exception:
                    return False
            try:
                tr._physics_dirty = False
            except Exception:
                pass
            try:
                pose = self._soft_applied_pose(tr)
                if pose is not None:
                    soft._soft_applied_pose = pose
            except Exception:
                pass
            return True
        except Exception:
            return False

    def _separate_soft_bodies(self, dt: float):
        try:
            items = list(self._entity_to_soft.items())
        except Exception:
            return
        if len(items) < 2:
            return
        infos = []
        for entity_id, sid in items:
            try:
                entity = self._get_entity(entity_id)
            except Exception:
                continue
            if entity is None:
                continue
            try:
                active = bool(entity._active)
            except Exception:
                active = True
            if not active:
                continue
            try:
                soft = self._soft_component(entity)
            except Exception:
                continue
            if soft is None:
                continue
            try:
                sample = self._solver.get_soft_body_sample(sid)
            except Exception:
                continue
            if not sample:
                continue
            try:
                lay = int(getattr(soft, "layer", 0))
                msk = int(getattr(soft, "mask", 0xFFFF)) & 0xFFFFFFFF
            except Exception:
                lay = 0
                msk = 0xFFFF
            try:
                maxv = float(getattr(soft, "max_velocity", 50.0))
            except Exception:
                maxv = 50.0
            infos.append({
                "key": entity_id,
                "center": sample.get("center", (0.0, 0.0, 0.0)),
                "aabb_min": sample.get("aabb_min", (0.0, 0.0, 0.0)),
                "aabb_max": sample.get("aabb_max", (0.0, 0.0, 0.0)),
                "velocity": sample.get("velocity", (0.0, 0.0, 0.0)),
                "pinned": bool(sample.get("pinned", False)),
                "mass": sample.get("mass", 1.0),
                "max_velocity": maxv,
                "layer": lay,
                "mask": msk,
            })
        try:
            out = PhysicsScene._soft_separation_corrections(infos, dt)
        except Exception:
            return
        for key, vel in out.items():
            try:
                sid = self._entity_to_soft.get(key)
                if sid is None:
                    continue
                self._solver.set_soft_body_velocity(sid, vel)
            except Exception:
                pass

    @staticmethod
    def _soft_separation_corrections(items, dt: float):
        out = {}
        try:
            step_dt = max(float(dt), 1e-6)
        except Exception:
            step_dt = 1.0 / 60.0
        try:
            n = len(items)
        except Exception:
            return out
        if n < 2:
            return out
        vel = {}
        for it in items:
            try:
                v = it.get("velocity", (0.0, 0.0, 0.0))
                vel[it["key"]] = [float(v[0]), float(v[1]), float(v[2])]
            except Exception:
                continue
        changed = set()
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    a = items[i]
                    b = items[j]
                    ka = a["key"]
                    kb = b["key"]
                    if ka not in vel or kb not in vel:
                        continue
                    cat_a = (1 << (int(a.get("layer", 0)) & 31)) & 0xFFFFFFFF
                    cat_b = (1 << (int(b.get("layer", 0)) & 31)) & 0xFFFFFFFF
                    if (int(a.get("mask", 0xFFFF)) & cat_b) == 0 or (int(b.get("mask", 0xFFFF)) & cat_a) == 0:
                        continue
                    if bool(a.get("pinned", False)) and bool(b.get("pinned", False)):
                        continue
                    amn = a["aabb_min"]
                    amx = a["aabb_max"]
                    bmn = b["aabb_min"]
                    bmx = b["aabb_max"]
                    ox = min(float(amx[0]), float(bmx[0])) - max(float(amn[0]), float(bmn[0]))
                    oy = min(float(amx[1]), float(bmx[1])) - max(float(amn[1]), float(bmn[1]))
                    oz = min(float(amx[2]), float(bmx[2])) - max(float(amn[2]), float(bmn[2]))
                except Exception:
                    continue
                if not (ox == ox and oy == oy and oz == oz):
                    continue
                if ox <= 0.0 or oy <= 0.0 or oz <= 0.0:
                    continue
                try:
                    ca = a["center"]
                    cb = b["center"]
                    if ox <= oy and ox <= oz:
                        nx, ny, nz = (1.0, 0.0, 0.0) if float(cb[0]) >= float(ca[0]) else (-1.0, 0.0, 0.0)
                        overlap = ox
                    elif oy <= oz:
                        nx, ny, nz = (0.0, 1.0, 0.0) if float(cb[1]) >= float(ca[1]) else (0.0, -1.0, 0.0)
                        overlap = oy
                    else:
                        nx, ny, nz = (0.0, 0.0, 1.0) if float(cb[2]) >= float(ca[2]) else (0.0, 0.0, -1.0)
                        overlap = oz
                    va = vel[ka]
                    vb = vel[kb]
                    rvx = float(vb[0]) - float(va[0])
                    rvy = float(vb[1]) - float(va[1])
                    rvz = float(vb[2]) - float(va[2])
                    vn = rvx * nx + rvy * ny + rvz * nz
                    bias = overlap / step_dt * 0.15
                    if vn >= bias:
                        continue
                    corr = bias - vn
                    try:
                        ma = max(float(a.get("mass", 1.0)), 1e-6)
                        mb = max(float(b.get("mass", 1.0)), 1e-6)
                    except Exception:
                        ma, mb = 1.0, 1.0
                    ima = 0.0 if bool(a.get("pinned", False)) else 1.0 / ma
                    imb = 0.0 if bool(b.get("pinned", False)) else 1.0 / mb
                    denom = ima + imb
                    if denom <= 0.0:
                        continue
                    wa = ima / denom
                    wb = imb / denom
                    try:
                        maxv = max(float(a.get("max_velocity", 50.0)), 1.0)
                        corr = min(corr, maxv)
                    except Exception:
                        pass
                    va[0] -= nx * corr * wa
                    va[1] -= ny * corr * wa
                    va[2] -= nz * corr * wa
                    vb[0] += nx * corr * wb
                    vb[1] += ny * corr * wb
                    vb[2] += nz * corr * wb
                    changed.add(ka)
                    changed.add(kb)
                except Exception:
                    continue
        for k in changed:
            try:
                v = vel.get(k)
                if v is None:
                    continue
                out[k] = (float(v[0]), float(v[1]), float(v[2]))
            except Exception:
                pass
        return out

    def _create_entity_soft(self, entity):
        if entity.id in self._entity_to_soft:
            return
        soft = self._soft_component(entity)
        if soft is None:
            return
        tr = entity.transform
        if tr is None:
            Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' has no transform, skipping")
            return
        sig = self._soft_sig(entity, soft, tr)
        if sig is None:
            Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' signature is None, skipping")
            return
        if not self._build_entity_soft(entity, soft, tr):
            return
        self._cached_soft_sig[entity.id] = sig
        try:
            pose = self._soft_applied_pose(tr)
            if pose is not None:
                soft._soft_applied_pose = pose
        except Exception:
            pass
        try:
            from core.components import Rigidbody
            if entity.get_component(Rigidbody) is not None:
                Logger.warning(f"SoftBody owns '{getattr(entity, 'name', entity.id)}', rigid body skipped")
        except Exception:
            pass

    def _build_entity_soft(self, entity, soft, tr) -> bool:
        try:
            from core.components.rendering.renderers.mesh_filter import MeshFilter
            from core.components.physics.mesh_collider import load_collision_geometry
            mf = entity.get_component(MeshFilter)
            mesh_path = (mf.mesh_path or "") if mf else ""
            if not mesh_path:
                Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' needs a MeshFilter mesh")
                return False
            Logger.info(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' loading mesh '{mesh_path}'")
            verts, indices, _ = load_collision_geometry(mesh_path)
            if verts is None or indices is None or len(verts) < 3:
                Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' could not read mesh '{mesh_path}' (verts={verts is not None}, indices={indices is not None})")
                return False
            Logger.info(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' loaded {len(verts)} verts, {len(indices)} indices")
            s = tr.local_scale
            euler = tr.local_euler_angles
            bend = soft.bend_mode.value if hasattr(soft.bend_mode, "value") else str(soft.bend_mode)
            pin = soft.pin_mode.value if hasattr(soft.pin_mode, "value") else str(soft.pin_mode)
            params = {
                "mass": soft.mass, "compliance": soft.compliance, "bend_mode": bend,
                "pressure": soft.pressure, "damping": soft.damping, "iterations": soft.iterations,
                "gravity_factor": soft.gravity_scale, "friction": soft.material_friction,
                "restitution": soft.material_bounciness, "vertex_radius": soft.vertex_radius,
                "max_velocity": soft.max_velocity, "max_vertices": soft.max_vertices,
                "pin_mode": pin, "pin_fraction": soft.pin_fraction,
                "double_sided": soft.double_sided, "update_com": soft.update_com,
                "layer": soft.layer, "mask": soft.mask,
            }
            sid, n, sverts, sfaces = create_soft_body_in_solver(
                self._solver,
                vertices=verts,
                indices=indices,
                scale_xyz=(s.x, s.y, s.z),
                pos_xyz=(tr.local_position.x, tr.local_position.y, tr.local_position.z),
                euler_rad_xyz=(math.radians(euler.x), math.radians(euler.y), math.radians(euler.z)),
                params=params,
                entity_id=entity.id,
            )
        except Exception as e:
            Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' creation exception: {e}")
            return False
        if sid is None or n <= 0:
            return False
        self._entity_to_soft[entity.id] = sid
        self._body_to_entity[sid] = entity.id
        try:
            if sverts is not None and len(sverts) >= 3:
                soft._soft_verts = np.ascontiguousarray(sverts, dtype=np.float32)
            if sfaces is not None and len(sfaces) >= 3:
                soft._soft_faces = np.ascontiguousarray(sfaces, dtype=np.int64)
        except Exception:
            pass
        try:
            Logger.info(f"SoftBody on '{getattr(entity, 'name', entity.id)}' simulating {n} vertices")
        except Exception:
            pass
        try:
            soft._soft_id = sid
        except Exception:
            pass
        return True

    def _remove_entity_soft(self, entity_id: str):
        sid = self._entity_to_soft.pop(entity_id, None)
        if sid is not None:
            try:
                self._solver.remove_soft_body(sid)
            except Exception:
                pass
            self._body_to_entity.pop(sid, None)
        self._cached_soft_sig.pop(entity_id, None)
        try:
            entity = self._get_entity(entity_id)
            soft = self._soft_component(entity) if entity is not None else None
            if soft is not None:
                for attr in ("_soft_skin_idx", "_soft_skin_w", "_soft_skin_nsim",
                             "_soft_skin_nfull", "_soft_skin_scale", "_soft_skin_mesh",
                             "_soft_use_skin", "_soft_needs_rebuild"):
                    try:
                        if hasattr(soft, attr):
                            delattr(soft, attr)
                    except Exception:
                        pass
        except Exception:
            pass

    def _check_soft_changes(self):
        for entity_id in list(self._entity_to_soft.keys()):
            entity = self._get_entity(entity_id)
            if entity is None:
                self._remove_entity_soft(entity_id)
                continue
            soft = self._soft_component(entity)
            if soft is None:
                self._remove_entity_soft(entity_id)
                continue
            tr = entity.transform
            if tr is None:
                continue
            sig = self._soft_sig(entity, soft, tr)
            if sig is not None and sig != self._cached_soft_sig.get(entity_id):
                self._remove_entity_soft(entity_id)
                self._create_entity_soft(entity)

    @staticmethod
    def _entity_frame_inv(tr):
        import numpy as np
        try:
            wm = tr.world_matrix
            m = np.asarray(wm._d, dtype=np.float64)
            if m.shape != (4, 4):
                raise ValueError("bad world matrix shape")
            return np.linalg.inv(m)
        except Exception:
            pass
        import numpy as np
        try:
            p = tr.local_position
            q = tr.local_rotation
            x, y, z, w = q.x, q.y, q.z, q.w
            n = x * x + y * y + z * z + w * w
            if n > 1e-20:
                inv = 1.0 / (n ** 0.5)
                x *= inv
                y *= inv
                z *= inv
                w *= inv
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)],
                [2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
                [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)],
            ], dtype=np.float64)
            F = np.eye(4, dtype=np.float64)
            F[:3, :3] = R.T
            F[3, :3] = [p.x, p.y, p.z]
            try:
                return np.linalg.inv(F)
            except Exception:
                return None
        except Exception:
            return None

    def _sync_soft_bodies(self):
        if not self._entity_to_soft or not self._scene:
            return
        import numpy as np
        if not getattr(self, '_soft_sync_logged', False):
            Logger.info(f"[PhysicsScene] _sync_soft_bodies called, {len(self._entity_to_soft)} entities")
            self._soft_sync_logged = True
        for entity_id, sid in list(self._entity_to_soft.items()):
            entity = self._get_entity(entity_id)
            if entity is None or not entity._active:
                self._diag_soft_once("sync: entity missing/inactive")
                continue
            soft = self._soft_component(entity)
            if soft is None:
                self._remove_entity_soft(entity_id)
                continue
            tr = entity.transform
            if tr is None:
                self._diag_soft_once(f"sync '{entity_id}': no transform")
                continue
            try:
                if self._soft_check_user_teleport(entity, soft, tr):
                    sid = self._entity_to_soft.get(entity_id, sid)
            except Exception:
                pass
            try:
                world = self._solver.get_soft_body_world_vertices(sid)
            except Exception:
                self._diag_soft_once(f"sync '{entity_id}': get_soft_body_world_vertices raised")
                continue
            if world is None or len(world) == 0:
                self._diag_soft_once(f"sync '{entity_id}': no world verts (world={world is not None})")
                continue
            try:
                self._soft_apply_com(entity, soft, tr, sid)
            except Exception:
                pass
            invW = self._entity_frame_inv(tr)
            if invW is None:
                self._diag_soft_once(f"sync '{entity_id}': _entity_frame_inv returned None")
                continue
            try:
                w4 = np.ones((len(world), 4), dtype=np.float64)
                w4[:, :3] = np.asarray(world, dtype=np.float64)
                local = (w4 @ invW)[:, :3].astype(np.float32)
            except Exception:
                continue
            if not np.all(np.isfinite(local)):
                self._diag_soft_once(f"sync '{entity_id}': local has NaN/inf")
                continue
            self._sync_soft_mesh(entity_id, soft, local)

    def _sync_soft_mesh(self, entity_id: str, soft, local):
        import numpy as np
        try:
            local = np.ascontiguousarray(local, dtype=np.float32)
        except Exception:
            return
        try:
            soft._soft_latest_local = np.ascontiguousarray(local, dtype=np.float32)
        except Exception:
            pass
        ov = getattr(soft, "_render_mesh", None)
        if ov is None:
            self._diag_soft_once(f"sync '{entity_id}': soft._render_mesh is None (renderer override not built)")
            return
        try:
            sk_idx = getattr(soft, "_soft_skin_idx", None)
            sk_w = getattr(soft, "_soft_skin_w", None)
            sk_nsim = int(getattr(soft, "_soft_skin_nsim", -1))
            sk_nfull = int(getattr(soft, "_soft_skin_nfull", -1))
            sk_rows = int(np.asarray(sk_idx).shape[0]) if sk_idx is not None else -1
            sk_cols = int(np.asarray(sk_idx).shape[1]) if sk_idx is not None else -1
            ov_rows = int(len(np.asarray(ov.vertices).reshape(-1, 3)))
            skin_valid = (
                sk_idx is not None and sk_w is not None
                and np.asarray(sk_w).shape == np.asarray(sk_idx).shape
                and sk_rows == sk_nfull and sk_nsim == len(local)
                and ov_rows == sk_rows and sk_cols >= 1
            )
        except Exception:
            skin_valid = False
        if skin_valid:
            try:
                L = np.ascontiguousarray(local, dtype=np.float32)
                ii = np.asarray(sk_idx, dtype=np.int64)
                ww = np.ascontiguousarray(sk_w, dtype=np.float32)
                full = np.zeros((len(ii), 3), dtype=np.float32)
                for c in range(ii.shape[1]):
                    full += ww[:, c:c + 1] * L[ii[:, c]]
            except Exception:
                return
            if not np.all(np.isfinite(full)):
                return
            try:
                ov.vertices = np.ascontiguousarray(full.reshape(-1))
                if getattr(soft, "recompute_normals", True):
                    try:
                        from core.assets.asset_importer import _compute_smooth_normals
                        faces = getattr(ov, "_soft_faces", None)
                        if faces is not None:
                            ov.normals = np.ascontiguousarray(
                                _compute_smooth_normals(full.astype(np.float32), np.asarray(faces)).reshape(-1))
                    except Exception:
                        pass
                ov.compute_aabb()
                try:
                    ov._soft_dirty = True
                    ov._soft_version = int(getattr(ov, "_soft_version", 0) or 0) + 1
                except Exception:
                    pass
                self._diag_frame(entity_id, np.asarray(full, dtype=np.float32))
            except Exception:
                pass
            return
        try:
            use_skin = getattr(soft, "_soft_use_skin", None)
        except Exception:
            use_skin = None
        if use_skin is False:
            try:
                ov_n = len(np.asarray(ov.vertices).reshape(-1, 3))
            except Exception:
                ov_n = -1
            if ov_n != len(local):
                self._diag_soft_once(f"sync '{entity_id}': count mismatch render={ov_n} physics={len(local)}, rebuilding render mesh to physics count")
            try:
                if len(local) != len(np.asarray(ov.vertices).reshape(-1, 3)):
                    sv_compact = getattr(soft, "_soft_verts", None)
                    sf_compact = getattr(soft, "_soft_faces", None)
                    if sv_compact is not None and len(sv_compact) == len(local):
                        ov.vertices = np.ascontiguousarray(sv_compact, dtype=np.float32).reshape(-1).copy()
                        ov.normals = np.zeros_like(ov.vertices)
                        n_verts = len(local)
                        try:
                            ov.uvs = np.zeros((n_verts * 2,), dtype=np.float32)
                        except Exception:
                            pass
                        try:
                            ov.sub_mesh_ranges = []
                            ov.sub_mesh_names = []
                        except Exception:
                            pass
                        if sf_compact is not None and len(sf_compact) >= 3:
                            ov.indices = np.ascontiguousarray(sf_compact, dtype=np.uint32).reshape(-1).copy()
                            ov._soft_faces = np.ascontiguousarray(sf_compact, dtype=np.int64).reshape(-1).copy()
                        ov._soft_n = len(local)
                    else:
                        sv_len = len(sv_compact) if sv_compact is not None else -1
                        self._diag_soft_once(f"sync '{entity_id}': cannot rebuild (sv_compact={sv_len} local={len(local)})")
                        return
            except Exception:
                self._diag_soft_once(f"sync '{entity_id}': rebuild exception")
                return
            try:
                ov.vertices = np.ascontiguousarray(local.reshape(-1))
                if getattr(soft, "recompute_normals", True):
                    try:
                        from core.assets.asset_importer import _compute_smooth_normals
                        faces = getattr(ov, "_soft_faces", None)
                        if faces is not None:
                            ov.normals = np.ascontiguousarray(
                                _compute_smooth_normals(local.astype(np.float32), np.asarray(faces)).reshape(-1))
                    except Exception:
                        pass
                ov.compute_aabb()
                try:
                    ov._soft_dirty = True
                    ov._soft_version = int(getattr(ov, "_soft_version", 0) or 0) + 1
                except Exception:
                    pass
                self._diag_frame(entity_id, np.asarray(local, dtype=np.float32))
            except Exception:
                pass
            return
        try:
            soft._soft_needs_rebuild = True
        except Exception:
            pass

    def _diag_frame(self, entity_id: str, local):
        seen = getattr(self, "_diag_seen", {})
        if entity_id not in seen:
            seen[entity_id] = 0
        seen[entity_id] += 1
        self._diag_seen = seen
        if seen[entity_id] in (1, 2, 3, 4, 5) or seen[entity_id] % 15 == 0:
            try:
                ymin = float(local[:, 1].min()); ymax = float(local[:, 1].max())
            except Exception:
                ymin = ymax = float("nan")
            Logger.info(f"[SoftBody] frame#{seen[entity_id]} '{entity_id}' local y-range: {round(ymin,3)}..{round(ymax,3)}")

    def _find_shape(self, entity: Entity, transform=None) -> Optional[dict]:
        return find_shape_info(entity, transform)

    def _find_shapes(self, entity: Entity, transform=None) -> list[dict]:
        return find_shapes_info(entity, transform)

    def _make_shape_key(self, entity: Entity, shape_info: dict) -> tuple:
        return make_shape_key(entity, shape_info)

    def remove_entity_bodies(self, entity_id: str):
        self._remove_entity_soft(entity_id)
        body_id = self._entity_to_body.pop(entity_id, None)
        if body_id is not None:
            self._solver.remove_rigid_body(body_id)
            self._body_to_entity.pop(body_id, None)
            self._2d_bodies.discard(body_id)
        for extra_id in self._entity_to_extra_bodies.pop(entity_id, []):
            try:
                self._solver.remove_rigid_body(extra_id)
            except Exception:
                pass
            self._body_to_entity.pop(extra_id, None)
            self._2d_bodies.discard(extra_id)
        self._entity_body_cache.pop(entity_id, None)
        self._mark_body_items_dirty()
        joint_id = self._entity_to_joint.pop(entity_id, None)
        if joint_id is not None:
            self._solver.remove_joint(joint_id)
            self._joint_to_entity.pop(joint_id, None)

    def step(self, dt: float):
        if not self._scene:
            return
        eng = self._scene._engine
        if eng is None:
            return
        if self._entity_to_soft and not getattr(self, '_soft_step_logged', False):
            Logger.info(f"[PhysicsScene] step with {len(self._entity_to_soft)} soft bodies, dt={dt:.4f}")
            self._soft_step_logged = True
        prof = eng._profiler if hasattr(eng, '_profiler') else None

        if prof: prof.start("phys_register")
        self._register_new_entities()
        if prof: prof.stop("phys_register")

        self._shape_check_counter += 1
        if self._shape_check_counter >= 60:
            self._shape_check_counter = 0
            if prof: prof.start("phys_shape_check")
            self._check_shape_changes()
            if prof: prof.stop("phys_shape_check")

        if prof: prof.start("phys_sync_to_solver")
        self._sync_ecs_to_physics()
        if prof: prof.stop("phys_sync_to_solver")

        if self._entity_to_soft and len(self._entity_to_soft) >= 2:
            if prof: prof.start("phys_soft_separate")
            try:
                self._separate_soft_bodies(dt)
            except Exception:
                pass
            if prof: prof.stop("phys_soft_separate")

        if prof: prof.start("phys_step_sim")
        self._solver.step_simulation(dt)
        if prof: prof.stop("phys_step_sim")

        if self._2d_bodies:
            if prof: prof.start("phys_constrain_2d")
            self._constrain_2d_bodies()
            if prof: prof.stop("phys_constrain_2d")

        if prof: prof.start("phys_sync_to_ecs")
        self._sync_physics_to_ecs()
        if prof: prof.stop("phys_sync_to_ecs")

        if self._entity_to_soft:
            if prof: prof.start("phys_sync_soft")
            self._sync_soft_bodies()
            if prof: prof.stop("phys_sync_soft")

        if prof: prof.start("phys_collision_events")
        self._process_collision_events()
        if prof: prof.stop("phys_collision_events")

    def _register_new_entities(self):
        if not self._scene:
            return
        # NOTE: no count-based early-out. A remove+add in the same frame keeps
        # the count identical but still leaves a new (soft-only) entity without
        # a body -> it would never simulate. The loop below is just dict
        # lookups, cheap even every frame.
        need_reset = False
        for entity in self._scene.get_all_entities():
            if entity.id not in self._entity_to_body:
                before = len(self._entity_to_body)
                self._create_entity_bodies(entity)
                if len(self._entity_to_body) != before:
                    need_reset = True
            if entity.id not in self._entity_to_soft:
                before = len(self._entity_to_soft)
                self._create_entity_soft(entity)
                if len(self._entity_to_soft) != before:
                    need_reset = True
        if need_reset:
            self._has_collision_scripts = False
            self._collision_scripts_checked = False

    def _constrain_2d_bodies(self):
        for body_id in list(self._2d_bodies):
            pos, rot = self._solver.get_body_transform(body_id)
            clamped = False
            if pos[2] != 0.0:
                clamped = True
            if rot[0] != 0.0 or rot[1] != 0.0:
                clamped = True
            if clamped:
                self._solver.set_body_transform(body_id, (pos[0], pos[1], 0.0), (0.0, 0.0, rot[2]))
            vel = self._solver.get_velocity(body_id)
            if vel[2] != 0.0:
                self._solver.set_velocity(body_id, (vel[0], vel[1], 0.0))
            ang_vel = self._solver.get_angular_velocity(body_id)
            if ang_vel[0] != 0.0 or ang_vel[1] != 0.0:
                self._solver.set_angular_velocity(body_id, (0.0, 0.0, ang_vel[2]))

    def _has_collision_listeners(self) -> bool:
        if not self._scene:
            return False
        sig = self._listener_signature()
        if sig != self._collision_listener_sig:
            self._collision_listener_sig = sig
            self._collision_scripts_checked = False
            self._has_collision_scripts = False
        if self._collision_scripts_checked:
            return self._has_collision_scripts
        self._collision_scripts_checked = True
        for entity in self._scene.get_all_entities():
            for comp in entity._components.values():
                if type(comp).__name__ == "ScriptComponent":
                    inst = comp._py_instance if hasattr(comp, '_py_instance') else None
                    if inst is None:
                        continue
                    if (hasattr(inst, 'on_collision_enter') or
                        hasattr(inst, 'on_collision_stay') or
                        hasattr(inst, 'on_collision_exit')):
                        self._has_collision_scripts = True
                        return True
                elif hasattr(comp, 'on_collision_enter'):
                    self._has_collision_scripts = True
                    return True
        return False

    def _listener_signature(self) -> tuple:
        entities = self._scene._entities
        count = 0
        for ent in entities.values():
            count += len(ent._components)
        return (len(entities), count)

    def _process_collision_events(self):
        from core.components import ScriptComponent
        if not self._has_collision_listeners():
            self._prev_frame_contacts.clear()
            return
        raw = self._solver.get_collision_events()
        current: set[frozenset[int]] = set()
        forces: dict[frozenset[int], float] = {}
        for ev in raw:
            ba, bb = ev["body_a"], ev["body_b"]
            if ba < 0 or bb < 0:
                continue
            pair = frozenset([ba, bb])
            current.add(pair)
            force = float(ev.get("force", 0.0) or 0.0)
            forces[pair] = max(forces.get(pair, 0.0), force)

        entered = current - self._prev_frame_contacts
        exited = self._prev_frame_contacts - current
        stayed = current & self._prev_frame_contacts

        def _dispatch(pairs, callback_name):
            for pair in pairs:
                bodies = list(pair)
                e0 = self._body_to_entity.get(bodies[0], "")
                e1 = self._body_to_entity.get(bodies[1], "")
                if not e0 or not e1:
                    continue
                for sc in self._get_entity(e0).get_components(ScriptComponent):
                    inst = sc._py_instance
                    if inst and hasattr(inst, callback_name):
                        try: getattr(inst, callback_name)(e1)
                        except Exception as ex: Logger.error(f"Script {callback_name} error: {ex}")
                for sc in self._get_entity(e1).get_components(ScriptComponent):
                    inst = sc._py_instance
                    if inst and hasattr(inst, callback_name):
                        try: getattr(inst, callback_name)(e0)
                        except Exception as ex: Logger.error(f"Script {callback_name} error: {ex}")

        def _dispatch_components(pairs, callback_name):
            for pair in pairs:
                bodies = list(pair)
                e0 = self._body_to_entity.get(bodies[0], "")
                e1 = self._body_to_entity.get(bodies[1], "")
                if not e0 or not e1:
                    continue
                force = forces.get(pair, 0.0)
                self._invoke_component_collision(self._get_entity(e0), e1, callback_name, force)
                self._invoke_component_collision(self._get_entity(e1), e0, callback_name, force)

        _dispatch(entered, 'on_collision_enter')
        _dispatch(exited, 'on_collision_exit')
        _dispatch(stayed, 'on_collision_stay')

        _dispatch_components(entered, 'on_collision_enter')
        _dispatch_components(exited, 'on_collision_exit')
        _dispatch_components(stayed, 'on_collision_stay')

        self._prev_frame_contacts = current

    def _invoke_component_collision(self, entity, other_eid: str, callback_name: str, force: float):
        if not entity:
            return
        from core.components import ScriptComponent
        for comp in entity.get_all_components():
            if isinstance(comp, ScriptComponent):
                continue
            if hasattr(comp, callback_name):
                try:
                    getattr(comp, callback_name)(other_eid, force)
                except Exception as ex:
                    Logger.error(f"Component {callback_name} error: {ex}")

    def _check_shape_changes(self):
        for entity_id in list(self._entity_to_body.keys()):
            entity = self._get_entity(entity_id)
            if not entity:
                continue
            shapes = self._find_shapes(entity, entity.transform)
            if not shapes:
                continue
            current_key = make_shapes_key(shapes)
            cached = self._cached_shape.get(entity_id)
            if cached is not None and current_key != cached:
                self.rebuild_entity(entity)
        self._check_soft_changes()

    def _mark_body_items_dirty(self):
        self._body_items_dirty = True

    def _get_body_items(self):
        if self._body_items_dirty or not self._body_items:
            self._body_items = list(self._entity_to_body.items())
            self._body_items_dirty = False
        return self._body_items

    def _sync_ecs_to_physics(self):
        try:
            from core._physics_sync import batch_sync_ecs_to_physics
            self._sync_extra_bodies()
            cache = self._entity_body_cache
            items = []
            for entity_id, body_id in self._entity_to_body.items():
                cached = cache.get(entity_id)
                if cached is not None:
                    items.append((entity_id, body_id, cached[0], cached[1], cached[2], cached[3]))
            if items:
                batch_sync_ecs_to_physics(items, self._solver)
            from core.math_helpers import quat_to_euler_rad
            for entity_id, body_id in list(self._entity_to_body.items()):
                cached = cache.get(entity_id)
                if not cached:
                    continue
                entity, rb, tr, is_2d = cached
                if not entity._active:
                    continue
                if not getattr(tr, "_physics_dirty", False):
                    continue
                p = tr._local_pos
                if is_2d:
                    self._solver.set_body_transform(body_id, (p._x, p._y, 0.0), (0.0, 0.0, quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y, tr._local_rot._z, tr._local_rot._w)[2]))
                else:
                    e = quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y, tr._local_rot._z, tr._local_rot._w)
                    self._solver.set_body_transform(body_id, (p._x, p._y, p._z), e)
                try:
                    self._solver.activate(body_id)
                except Exception:
                    pass
                tr._physics_dirty = False
        except ImportError:
            from core.math_helpers import quat_to_euler_rad
            self._sync_extra_bodies()
            cache = self._entity_body_cache
            for entity_id, body_id in self._entity_to_body.items():
                cached = cache.get(entity_id)
                if not cached:
                    continue
                entity, rb, tr, is_2d = cached
                if not entity._active:
                    continue
                if getattr(tr, "_physics_dirty", False):
                    p = tr._local_pos
                    if is_2d:
                        self._solver.set_body_transform(body_id, (p._x, p._y, 0.0), (0.0, 0.0, quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y, tr._local_rot._z, tr._local_rot._w)[2]))
                    else:
                        e = quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y, tr._local_rot._z, tr._local_rot._w)
                        self._solver.set_body_transform(body_id, (p._x, p._y, p._z), e)
                    try:
                        self._solver.activate(body_id)
                    except Exception:
                        pass
                    tr._physics_dirty = False
                fa = rb._force_accum
                if is_2d:
                    if fa._x != 0.0 or fa._y != 0.0:
                        self._solver.apply_force(body_id, (fa._x, fa._y, 0.0))
                    if abs(rb._torque_accum) > 1e-10:
                        self._solver.apply_torque(body_id, (0.0, 0.0, rb._torque_accum))
                else:
                    if fa._x * fa._x + fa._y * fa._y + fa._z * fa._z > 1e-10:
                        self._solver.apply_force(body_id, (fa._x, fa._y, fa._z))
                    ta = rb._torque_accum
                    if ta._x * ta._x + ta._y * ta._y + ta._z * ta._z > 1e-10:
                        self._solver.apply_torque(body_id, (ta._x, ta._y, ta._z))

    def _sync_physics_to_ecs(self):
        try:
            from core._physics_sync import batch_sync_physics_to_ecs
            cache = self._entity_body_cache
            items = []
            for entity_id, body_id in self._entity_to_body.items():
                cached = cache.get(entity_id)
                if cached is not None:
                    items.append((entity_id, body_id, cached[0], cached[1], cached[2], cached[3]))
            if items:
                batch_sync_physics_to_ecs(items, self._solver)
        except ImportError:
            from core.math_helpers import quat_from_euler_rad
            cache = self._entity_body_cache
            for entity_id, body_id in self._entity_to_body.items():
                cached = cache.get(entity_id)
                if not cached:
                    continue
                entity, rb, tr, is_2d = cached
                if not entity._active or rb.is_kinematic or getattr(tr, "_physics_dirty", False):
                    continue
                pos, rot = self._solver.get_body_transform(body_id)
                vel = self._solver.get_velocity(body_id)
                ang_vel = self._solver.get_angular_velocity(body_id)
                tr._local_pos._x = pos[0]
                tr._local_pos._y = pos[1]
                tr._local_pos._z = 0.0 if is_2d else pos[2]
                if is_2d:
                    q = quat_from_euler_rad(0.0, 0.0, rot[2])
                else:
                    q = quat_from_euler_rad(rot[0], rot[1], rot[2])
                tr._local_rot._x = q[0]
                tr._local_rot._y = q[1]
                tr._local_rot._z = q[2]
                tr._local_rot._w = q[3]
                tr._dirty = True
                rb._velocity._x = vel[0]
                rb._velocity._y = vel[1]
                if not is_2d:
                    rb._velocity._z = vel[2]
                    rb._angular_velocity._x = ang_vel[0]
                    rb._angular_velocity._y = ang_vel[1]
                    rb._angular_velocity._z = ang_vel[2]
                else:
                    rb._angular_velocity = ang_vel[2]
                rb._force_accum._x = 0.0
                rb._force_accum._y = 0.0
                if not is_2d:
                    rb._force_accum._z = 0.0
                    rb._torque_accum._x = 0.0
                    rb._torque_accum._y = 0.0
                    rb._torque_accum._z = 0.0
                else:
                    rb._torque_accum = 0.0

    def _create_entity_joints(self, entity: Entity):
        from core.components import Joint

        joint = entity.get_component(Joint)
        if not joint or not joint.enabled:
            return

        body_a_id = self._entity_to_body.get(entity.id)
        if body_a_id is None:
            return

        connected = self._find_entity_by_name(joint.connected_entity_name)
        if connected is None:
            Logger.warning(f"Joint: connected entity '{joint.connected_entity_name}' not found")
            return

        body_b_id = self._entity_to_body.get(connected.id)
        if body_b_id is None:
            Logger.warning(f"Joint: connected entity '{joint.connected_entity_name}' has no body")
            return

        joint_id = self._solver.create_joint(
            joint_type=joint.joint_type,
            body_a_id=body_a_id,
            body_b_id=body_b_id,
            anchor=(joint.anchor.x, joint.anchor.y, joint.anchor.z),
            axis=(joint.axis.x, joint.axis.y, joint.axis.z),
            limit_low=joint.limit_low,
            limit_high=joint.limit_high,
            stiffness=joint.stiffness,
            damping=joint.damping,
        )
        if joint_id >= 0:
            self._entity_to_joint[entity.id] = joint_id
            self._joint_to_entity[joint_id] = entity.id

    def _find_entity_by_name(self, name: str):
        if not self._scene:
            return None
        for e in self._scene.get_all_entities():
            if e.name == name:
                return e
        return None

    def _get_entity(self, entity_id: str):
        if not self._scene:
            return None
        return self._scene.get_entity(entity_id)

    def ray_cast(self, origin: Vec3, direction: Vec3, max_distance: float = 100.0) -> Optional[dict]:
        result = self._solver.ray_cast(
            (origin.x, origin.y, origin.z),
            (direction.x, direction.y, direction.z),
            max_distance,
        )
        if result:
            body_id = result.get("body_id")
            if body_id is not None and body_id in self._body_to_entity:
                result["entity_id"] = self._body_to_entity[body_id]
        return result

    def create_drag_constraint(
        self,
        body_id: int,
        hit_world: Vec3,
        max_force: float = 500,
    ) -> Optional[int]:
        try:
            if int(body_id) < 0:
                return None
        except Exception:
            return None
        constraint_id = self._solver.create_joint(
            joint_type="point2point",
            body_a_id=body_id,
            body_b_id=-1,
            anchor=(hit_world.x, hit_world.y, hit_world.z),
        )
        if constraint_id >= 0:
            self._solver.change_constraint(constraint_id, (hit_world.x, hit_world.y, hit_world.z), max_force)
        return constraint_id if constraint_id >= 0 else None

    def update_drag_constraint(self, constraint_id: int, world_pos: Vec3):
        self._solver.change_constraint(constraint_id, (world_pos.x, world_pos.y, world_pos.z))

    def remove_drag_constraint(self, constraint_id: int):
        self._solver.remove_joint(constraint_id)

    def get_collision_events(self) -> list[dict]:
        return self._solver.get_collision_events()

    def rebuild_entity(self, entity: Entity):
        self._cached_shape.pop(entity.id, None)
        self._cached_shape_info.pop(entity.id, None)
        self.remove_entity_bodies(entity.id)
        self._create_entity_bodies(entity)
