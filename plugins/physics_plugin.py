# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
import numpy as np
from typing import TYPE_CHECKING
from core.foundation.plugin_manager import PluginBase
from core.foundation.logger import Logger
from core.physics import PhysicsProcess, PhysicsScene
from core.physics.shared_buffer import MAX_ENTITIES
from core.physics.physics_solver import IPhysicsSolver
from core.maths.math3d import Vec2, Vec3
from core.physics.shape_utils import find_shapes_info
from core.config.config import get_project_config

if TYPE_CHECKING:
    from core.ecs.ecs import Entity

_RAD = math.radians
_DEG = math.degrees

try:
    from core._physics_sync import (
        sync_read_to_ecs as _COMPILED_SYNC_READ,
        sync_write_from_ecs as _COMPILED_SYNC_WRITE,
    )
except (ImportError, AttributeError):
    _COMPILED_SYNC_READ = None
    _COMPILED_SYNC_WRITE = None


class PhysicsPlugin(PluginBase):
    NAME = "PhysicsPlugin"
    VERSION = "0.4.0"
    DESCRIPTION = "Physics system with shared-memory parallel simulation."
    SYSTEM = True

    def __init__(self):
        super().__init__()
        self._enabled: bool = True
        self._scanned_entity_ids: set[str] = set()
        self._last_entity_count: int = -1
        self._physics_process: Optional[PhysicsProcess] = None
        self._physics_scene: Optional[PhysicsScene] = None
        self._solver: Optional[IPhysicsSolver] = None
        self._simulation_mode: str = "multi_threaded"
        self._layer_processes: dict[int, PhysicsProcess] = {}
        self._prev_frame_contacts: set = set()
        self._project_root: str = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        self._step_caches: dict[int, tuple] = {}
        self._cache_version: int = 0
        self._collision_listener_sig: tuple = None
        self._has_collision_cache: bool = False
        self._collision_cache_valid: bool = False
        self._soft_remote: dict[str, dict] = {}
        self._soft_sync_helper = None
        self._soft_over_cap_warned: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, v: bool):
        self._enabled = v

    @property
    def physics_scene(self) -> Optional[PhysicsScene]:
        return self._physics_scene

    def _get_physics_settings(self) -> dict:
        project_path = getattr(self._engine, "_project_path", None) or "." if self._engine else "."
        cfg = get_project_config(project_path)
        out = {}
        prefix = "physics."
        for k in (
            "solver", "physx_device", "simulation_mode",
            "gravity_x", "gravity_y", "gravity_z",
            "fixed_time_step", "num_sub_steps", "solver_iterations",
            "erp", "contact_erp", "friction_erp",
            "contact_breaking_threshold", "restitution",
            "linear_damping", "angular_damping", "max_contacts_per_body",
            "culverin_max_bodies", "culverin_max_pairs",
            "culverin_max_contact_constraints", "culverin_temp_allocator_size",
            "culverin_max_physics_jobs", "culverin_max_physics_barriers",
            "culverin_num_threads", "culverin_penetration_slop",
            "culverin_enable_ccd", "culverin_enable_sleeping",
        ):
            v = cfg.get(prefix + k)
            if v is not None:
                out[k] = v
        return out

    def initialize(self, engine):
        super().initialize(engine)
        settings = self._get_physics_settings()
        self._simulation_mode = settings.get("simulation_mode", "multi_threaded")

        solver_name = settings.get("solver", "culverin")
        solver_module = ""
        solver_class = ""
        if solver_name == "physx":
            solver_module = "physics_solvers.physx_solver"
            solver_class = "PhysXSolver"
        elif solver_name == "culverin":
            solver_module = "physics_solvers.culverin_solver"
            solver_class = "CulverinSolver"
        else:
            solver_module = "physics_solvers.pybullet_solver"
            solver_class = "PyBulletSolver"

        Logger.info(f"[PhysicsPlugin] using solver={solver_name} mode={self._simulation_mode}")

        if self._simulation_mode == "single":
            self._init_single(solver_module, solver_class, settings, solver_name)
        elif self._simulation_mode == "per_layer_process":
            Logger.info(f"PhysicsPlugin: per-layer process mode (processes spawned on demand).")
        else:
            self._physics_process = PhysicsProcess(project_root=self._project_root)
            ok = self._physics_process.start(solver_module, solver_class, settings)
            if not ok:
                Logger.warning("PhysicsPlugin: multi-threaded init failed, falling back to single-threaded mode")
                self._physics_process = None
                self._simulation_mode = "single"
                self._init_single(solver_module, solver_class, settings, solver_name)
            else:
                Logger.info(f"PhysicsPlugin: solver {solver_name} started (shared-memory).")

    def ensure_single_mode(self) -> bool:
        if self._simulation_mode == "single" and self._solver is not None:
            return True
        self._clear_soft_remote()
        if self._simulation_mode == "single" and self._solver is None:
            self._init_single(*self._solver_module_class_with_settings())
            return self._solver is not None
        try:
            if self._physics_process is not None:
                self._physics_process.shutdown(5000)
                self._physics_process = None
        except Exception:
            self._physics_process = None
        self._simulation_mode = "single"
        self._init_single(*self._solver_module_class_with_settings())
        if self._solver is not None:
            Logger.info("[PhysicsPlugin] Switched to single-threaded mode for in-process physics (characters).")
        return self._solver is not None

    def _solver_module_class_with_settings(self) -> tuple:
        settings = self._get_physics_settings()
        solver_name = settings.get("solver", "culverin")
        sm, sc = self._solver_module_class()
        return (sm, sc, settings, solver_name)

    def _init_single(self, solver_module: str, solver_class: str, settings: dict, solver_name: str):
        import importlib
        try:
            mod = importlib.import_module(solver_module)
            cls = getattr(mod, solver_class)
            self._solver = cls()
            if not self._solver.initialize(settings):
                Logger.error(f"PhysicsPlugin: single-threaded {solver_name} solver initialize failed")
                self._solver = None
                self._physics_scene = None
                return
            self._physics_scene = PhysicsScene(self._solver)
            Logger.info(f"PhysicsPlugin: {solver_name} in-process (single-threaded).")
        except Exception as e:
            Logger.error(f"PhysicsPlugin: single-threaded init failed: {e}")
            self._solver = None
            self._physics_scene = None

    def _solver_module_class(self) -> tuple[str, str]:
        settings = self._get_physics_settings()
        solver_name = settings.get("solver", "culverin")
        if solver_name == "physx":
            return "physics_solvers.physx_solver", "PhysXSolver"
        if solver_name == "culverin":
            return "physics_solvers.culverin_solver", "CulverinSolver"
        return "physics_solvers.pybullet_solver", "PyBulletSolver"

    def _get_layer_process(self, layer: int) -> PhysicsProcess:
        if layer not in self._layer_processes:
            sm, sc = self._solver_module_class()
            settings = self._get_physics_settings()
            proc = PhysicsProcess(project_root=self._project_root)
            if proc.start(sm, sc, settings):
                self._layer_processes[layer] = proc
                Logger.info(f"PhysicsPlugin: spawned process for layer {layer}")
            else:
                raise RuntimeError(f"Failed to start physics process for layer {layer}")
        return self._layer_processes[layer]

    def _body_with_slot(self, entity, tr) -> Optional[dict]:
        return self._body_with_slot_in_process(entity, tr, self._physics_process)

    def _body_with_slot_in_process(self, entity, tr, proc: PhysicsProcess) -> Optional[dict]:
        rb = entity._components.get("Rigidbody")
        rb2d = entity._components.get("Rigidbody2D")
        try:
            from core.components.physics.soft_body import SoftBody
            soft = entity.get_component(SoftBody)
            if soft is not None and getattr(soft, "enabled", True):
                return None
        except Exception:
            pass
        shapes = find_shapes_info(entity, tr)
        if not shapes:
            return None
        shape_info = shapes[0]
        is_2d = rb2d is not None
        lp = tr._local_pos
        q = tr._local_rot
        if is_2d:
            pos = (lp.x, lp.y, 0.0)
            sz = 2.0 * math.asin(max(-1.0, min(1.0, q.z)))
            rot = (0.0, 0.0, sz)
            mass = 0.0 if rb2d.is_kinematic else rb2d.mass
            is_kinematic = rb2d.is_kinematic
        else:
            pos = (lp.x, lp.y, lp.z)
            euler = tr.local_euler_angles
            rot = (_RAD(euler.x), _RAD(euler.y), _RAD(euler.z))
            mass = 0.0 if rb.is_kinematic else rb.mass
            is_kinematic = rb.is_kinematic

        slot = proc.alloc_slot()
        proc.entity_slot_map[entity.id] = slot
        self._cache_version += 1
        return {
            "slot": slot,
            "entity_id": entity.id,
            "is_2d": is_2d,
            "shape_type": shape_info["type"],
            "shape_params": shape_info["params"],
            "shapes": shapes,
            "position": pos,
            "rotation": rot,
            "mass": mass,
            "friction": shape_info.get("friction", 0.6),
            "restitution": shape_info.get("restitution", 0.0),
            "is_trigger": shape_info.get("is_trigger", False),
            "is_kinematic": is_kinematic,
            "collision_layer": shape_info.get("layer", 0),
            "collision_mask": shape_info.get("mask", 0xFFFF),
        }

    def _get_entity_layer(self, entity) -> int:
        for comp in entity.get_all_components():
            if hasattr(comp, 'layer'):
                return int(getattr(comp, 'layer', 0))
        return 0

    def _get_entity_layer(self, entity) -> int:
        for comp in entity.get_all_components():
            if hasattr(comp, 'layer'):
                return int(getattr(comp, 'layer', 0))
        return 0

    def _get_soft_sync_helper(self):
        if self._soft_sync_helper is None:
            try:
                self._soft_sync_helper = PhysicsScene(None)
            except Exception:
                return None
        return self._soft_sync_helper

    def _clear_soft_remote(self):
        for eid in list(self._soft_remote.keys()):
            self._drop_soft_remote(eid)
        self._soft_remote.clear()
        self._soft_over_cap_warned.clear()

    def _drop_soft_remote(self, entity_id: str):
        info = self._soft_remote.pop(entity_id, None)
        if info is None:
            return
        try:
            proc = info.get("proc")
            slot = info.get("slot", -1)
            if proc is not None and slot is not None and slot >= 0:
                try:
                    proc.entity_soft_slot_map.pop(entity_id, None)
                except Exception:
                    pass
                try:
                    proc.free_soft_slot(slot)
                except Exception:
                    pass
                try:
                    if proc.is_alive():
                        proc.send({"type": "remove_bodies", "entity_ids": [entity_id]})
                except Exception:
                    pass
        except Exception:
            pass

    def _build_soft_dict(self, entity, tr, soft, proc, slot=None, reuse_mesh=None) -> Optional[dict]:
        from core.components.rendering.renderers.mesh_filter import MeshFilter
        from core.components.physics.mesh_collider import load_collision_geometry
        mf = entity.get_component(MeshFilter)
        mesh_path = (mf.mesh_path or "") if mf else ""
        if not mesh_path:
            Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' needs a MeshFilter mesh")
            return None
        verts_b, indices_b = (None, None)
        if reuse_mesh is not None:
            try:
                verts_b, indices_b = reuse_mesh
            except Exception:
                verts_b, indices_b = (None, None)
        if verts_b is None or indices_b is None:
            try:
                verts, indices, _k = load_collision_geometry(mesh_path)
            except Exception:
                verts, indices = None, None
            if verts is None or indices is None or len(verts) < 3:
                Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' could not read mesh '{mesh_path}'")
                return None
            try:
                verts_b = np.ascontiguousarray(verts, dtype=np.float32).tobytes()
                indices_b = np.ascontiguousarray(np.asarray(indices).reshape(-1), dtype=np.int64).tobytes()
            except Exception:
                return None
        if slot is None:
            try:
                slot = proc.alloc_soft_slot()
            except Exception as e:
                Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' no free soft slot: {e}")
                return None
            try:
                proc.entity_soft_slot_map[entity.id] = slot
            except Exception:
                pass
        lp = tr._local_pos
        s = tr.local_scale
        euler = tr.local_euler_angles
        bend = soft.bend_mode.value if hasattr(soft.bend_mode, "value") else str(soft.bend_mode)
        pin = soft.pin_mode.value if hasattr(soft.pin_mode, "value") else str(soft.pin_mode)
        return {
            "entity_id": entity.id, "slot": int(slot),
            "verts_b": verts_b, "indices_b": indices_b,
            "scale": (float(s.x), float(s.y), float(s.z)),
            "position": (float(lp._x), float(lp._y), float(lp._z)),
            "rotation": (_RAD(euler.x), _RAD(euler.y), _RAD(euler.z)),
            "mass": float(soft.mass), "compliance": float(soft.compliance),
            "bend_mode": str(bend), "pressure": float(soft.pressure),
            "damping": float(soft.damping), "iterations": int(soft.iterations),
            "gravity_factor": float(soft.gravity_scale),
            "friction": float(soft.material_friction),
            "restitution": float(soft.material_bounciness),
            "vertex_radius": float(soft.vertex_radius),
            "max_velocity": float(soft.max_velocity),
            "max_vertices": int(soft.max_vertices),
            "pin_mode": str(pin), "pin_fraction": float(soft.pin_fraction),
            "double_sided": bool(soft.double_sided), "update_com": bool(soft.update_com),
            "layer": int(soft.layer), "mask": int(soft.mask) & 0xFFFFFFFF,
            "mesh_path": mesh_path,
        }

    def _track_soft_dict(self, entity, soft, tr, proc, slot: int, d: dict):
        try:
            sig = PhysicsScene._soft_sig(entity, soft, tr)
        except Exception:
            sig = None
        try:
            pose = PhysicsScene._soft_applied_pose(tr)
        except Exception:
            pose = None
        try:
            pinned = str(d.get("pin_mode", "none")).lower() in ("top", "bottom")
        except Exception:
            pinned = False
        self._soft_remote[entity.id] = {
            "slot": int(slot), "proc": proc, "sig": sig,
            "count": -1, "pending": True,
            "layer": int(d.get("layer", 0)), "mask": int(d.get("mask", 0xFFFF)) & 0xFFFFFFFF,
            "mass": float(d.get("mass", 1.0)), "max_velocity": float(d.get("max_velocity", 50.0)),
            "pinned": bool(pinned), "radius": float(d.get("vertex_radius", 0.05)),
            "mesh_path": d.get("mesh_path", ""),
            "scale": (float(d["scale"][0]), float(d["scale"][1]), float(d["scale"][2])),
            "max_vertices": int(d.get("max_vertices", 0)),
            "mesh_b": (d.get("verts_b"), d.get("indices_b")),
        }
        if pose is not None:
            try:
                soft._soft_applied_pose = pose
            except Exception:
                pass

    def _collect_soft_dicts(self, scene, proc) -> list:
        from core.components.physics.soft_body import SoftBody
        out = []
        try:
            entities = scene.get_all_entities()
        except Exception:
            return out
        for entity in entities:
            try:
                if entity.id in self._soft_remote:
                    continue
                soft = entity.get_component(SoftBody)
                if soft is None or not getattr(soft, "enabled", True):
                    continue
                tr = entity._components.get("Transform")
                if tr is None:
                    continue
                d = self._build_soft_dict(entity, tr, soft, proc)
                if d is None:
                    continue
                self._track_soft_dict(entity, soft, tr, proc, d["slot"], d)
                out.append(d)
            except Exception:
                continue
        return out

    def _harvest_soft_created(self, proc, scene, results):
        if not results:
            return
        try:
            entities = scene._entities
        except Exception:
            return
        from core.components.physics.soft_body import SoftBody
        for r in results:
            try:
                items = r.get("soft_created", None) if isinstance(r, dict) else None
            except Exception:
                continue
            if not items:
                continue
            for item in items:
                try:
                    eid = item.get("entity_id")
                    info = self._soft_remote.get(eid)
                    if info is None or info.get("proc") is not proc:
                        continue
                    if not item.get("ok"):
                        Logger.warning(f"[SoftBody] '{eid}' worker creation failed: {item.get('error', '?')}")
                        self._drop_soft_remote(eid)
                        continue
                    entity = entities.get(eid)
                    if entity is None:
                        self._drop_soft_remote(eid)
                        continue
                    soft = entity.get_component(SoftBody)
                    if soft is None:
                        self._drop_soft_remote(eid)
                        continue
                    verts = np.frombuffer(item["verts_b"], dtype=np.float32).reshape(-1, 3).copy()
                    faces = np.frombuffer(item["faces_b"], dtype=np.int32).astype(np.int64, copy=True)
                    if len(verts) < 3 or int(item.get("count", 0)) <= 0:
                        self._drop_soft_remote(eid)
                        continue
                    soft._soft_verts = np.ascontiguousarray(verts, dtype=np.float32)
                    soft._soft_faces = np.ascontiguousarray(faces, dtype=np.int64)
                    info["count"] = int(item["count"])
                    info["pending"] = False
                    try:
                        info["radius"] = max(float(item.get("radius", info["radius"])), 1e-4)
                    except Exception:
                        pass
                except Exception:
                    continue

    def _soft_remote_check_update(self, entity, soft, tr, info, proc) -> bool:
        try:
            sig = PhysicsScene._soft_sig(entity, soft, tr)
        except Exception:
            sig = None
        try:
            dirty = bool(getattr(tr, "_physics_dirty", False))
        except Exception:
            dirty = False
        pose_moved = False
        if dirty:
            try:
                cur = PhysicsScene._soft_applied_pose(tr)
            except Exception:
                cur = None
            if cur is None:
                try:
                    tr._physics_dirty = False
                except Exception:
                    pass
            else:
                try:
                    prev = getattr(soft, "_soft_applied_pose", None)
                except Exception:
                    prev = None
                if prev is None:
                    pose_moved = True
                else:
                    try:
                        dp = abs(cur[0] - prev[0]) + abs(cur[1] - prev[1]) + abs(cur[2] - prev[2])
                        dq = 1.0 - abs(cur[3] * prev[3] + cur[4] * prev[4] + cur[5] * prev[5] + cur[6] * prev[6])
                        ds = abs(cur[7] - prev[7]) + abs(cur[8] - prev[8]) + abs(cur[9] - prev[9])
                    except Exception:
                        dp, dq, ds = 0.0, 0.0, 0.0
                    pose_moved = (dp > 1e-6 or dq > 1e-8 or ds > 1e-9)
        if (sig is not None and sig != info.get("sig")) or pose_moved:
            try:
                from core.components.rendering.renderers.mesh_filter import MeshFilter
                mf = entity.get_component(MeshFilter)
                mesh_path = (mf.mesh_path or "") if mf else ""
            except Exception:
                mesh_path = info.get("mesh_path", "")
            reuse = None
            try:
                if mesh_path and mesh_path == info.get("mesh_path"):
                    reuse = info.get("mesh_b")
            except Exception:
                reuse = None
            d = self._build_soft_dict(entity, tr, soft, proc, slot=info["slot"], reuse_mesh=reuse)
            if d is None:
                info["sig"] = sig
                try:
                    tr._physics_dirty = False
                except Exception:
                    pass
                return False
            try:
                proc.send({"type": "update_soft_body", "soft": d})
            except Exception:
                return False
            info["sig"] = sig
            info["pending"] = True
            for k in ("layer", "mask", "mass", "max_velocity", "mesh_path", "max_vertices"):
                try:
                    if k in ("mask",):
                        info[k] = int(d.get(k, info.get(k, 0xFFFF))) & 0xFFFFFFFF
                    elif k in ("layer", "max_vertices"):
                        info[k] = int(d.get(k, info.get(k, 0)))
                    elif k in ("mass", "max_velocity"):
                        info[k] = float(d.get(k, info.get(k, 1.0)))
                    else:
                        info[k] = d.get(k, info.get(k))
                except Exception:
                    pass
            try:
                info["pinned"] = str(d.get("pin_mode", "none")).lower() in ("top", "bottom")
                sc = d.get("scale", info.get("scale", (1.0, 1.0, 1.0)))
                info["scale"] = (float(sc[0]), float(sc[1]), float(sc[2]))
                info["mesh_b"] = (d.get("verts_b"), d.get("indices_b"))
            except Exception:
                pass
            try:
                tr._physics_dirty = False
            except Exception:
                pass
            return True
        if dirty:
            try:
                tr._physics_dirty = False
            except Exception:
                pass
        return False

    def _sync_soft_remote_entity(self, proc, entity, soft, tr, info, helper):
        slot = info.get("slot", -1)
        try:
            sbuf = proc.soft_shared
        except Exception:
            return None
        try:
            count = sbuf.read_count(slot)
        except Exception:
            return None
        if count <= 0:
            if count < 0 and entity.id not in self._soft_over_cap_warned:
                self._soft_over_cap_warned.add(entity.id)
                Logger.warning(f"[SoftBody] '{getattr(entity, 'name', entity.id)}' vertex count exceeds shared buffer cap, visual frozen")
            return None
        if info.get("count", -1) != count:
            return None
        try:
            data = sbuf.read_soft(slot)
        except Exception:
            return None
        if data is None:
            return None
        world, pos, quat, vel = data
        if world is None or len(world) != count:
            return None
        try:
            helper._soft_apply_com(entity, soft, tr, None, com=(pos, quat))
        except Exception:
            pass
        try:
            invW = PhysicsScene._entity_frame_inv(tr)
        except Exception:
            invW = None
        if invW is None:
            return None
        try:
            w4 = np.ones((len(world), 4), dtype=np.float64)
            w4[:, :3] = np.asarray(world, dtype=np.float64)
            local = (w4 @ invW)[:, :3].astype(np.float32)
        except Exception:
            return None
        if not np.all(np.isfinite(local)):
            return None
        try:
            helper._sync_soft_mesh(entity.id, soft, local)
        except Exception:
            return None
        return data

    def _sync_soft_remote(self, proc, scene, dt: float, pending_results) -> dict:
        vels: dict = {}
        try:
            self._harvest_soft_created(proc, scene, pending_results)
        except Exception:
            pass
        try:
            tracked = [(eid, info) for eid, info in self._soft_remote.items() if info.get("proc") is proc]
        except Exception:
            return vels
        if not tracked:
            return vels
        try:
            helper = self._get_soft_sync_helper()
        except Exception:
            helper = None
        if helper is None:
            return vels
        try:
            entities = scene._entities
        except Exception:
            return vels
        from core.components.physics.soft_body import SoftBody
        try:
            step_dt = max(float(dt), 1e-6)
        except Exception:
            step_dt = 1.0 / 60.0
        datas = {}
        for eid, info in tracked:
            try:
                entity = entities.get(eid)
                if entity is None or not getattr(entity, "_active", True):
                    continue
                soft = entity.get_component(SoftBody)
                if soft is None or not getattr(soft, "enabled", True):
                    continue
                tr = entity._components.get("Transform")
                if tr is None:
                    continue
                try:
                    if self._soft_remote_check_update(entity, soft, tr, info, proc):
                        continue
                except Exception:
                    pass
                data = self._sync_soft_remote_entity(proc, entity, soft, tr, info, helper)
                if data is not None:
                    datas[eid] = (data, info)
            except Exception:
                continue
        if len(datas) >= 2:
            try:
                items = []
                for eid, (data, info) in datas.items():
                    world, _pos, _quat, vel = data
                    v = np.asarray(world, dtype=np.float64)
                    c = v.mean(axis=0)
                    bmin = v.min(axis=0)
                    bmax = v.max(axis=0)
                    try:
                        r = max(float(info.get("radius", 0.05)), 1e-4)
                    except Exception:
                        r = 0.05
                    items.append({
                        "key": eid,
                        "center": (float(c[0]), float(c[1]), float(c[2])),
                        "aabb_min": (float(bmin[0]) - r, float(bmin[1]) - r, float(bmin[2]) - r),
                        "aabb_max": (float(bmax[0]) + r, float(bmax[1]) + r, float(bmax[2]) + r),
                        "velocity": (float(vel[0]), float(vel[1]), float(vel[2])),
                        "pinned": bool(info.get("pinned", False)),
                        "mass": float(info.get("mass", 1.0)),
                        "max_velocity": float(info.get("max_velocity", 50.0)),
                        "layer": int(info.get("layer", 0)),
                        "mask": int(info.get("mask", 0xFFFF)),
                    })
                out = PhysicsScene._soft_separation_corrections(items, step_dt)
                for k, vv in out.items():
                    vels[k] = (float(vv[0]), float(vv[1]), float(vv[2]))
            except Exception:
                pass
        return vels

    def on_scene_loaded(self, scene):
        self._scanned_entity_ids.clear()
        self._clear_soft_remote()
        self._last_entity_count = -1
        self._prev_frame_contacts.clear()
        self._step_caches.clear()
        self._collision_listener_sig = None
        self._collision_cache_valid = False

    def on_project_opened(self):
        settings = self._get_physics_settings()
        new_mode = settings.get("simulation_mode", "multi_threaded")
        if new_mode == self._simulation_mode:
            return
        Logger.info(f"[PhysicsPlugin] project opened: simulation_mode {self._simulation_mode} -> {new_mode}")
        self._clear_soft_remote()
        if self._simulation_mode == "per_layer_process":
            for proc in self._layer_processes.values():
                proc.shutdown(1000)
            self._layer_processes.clear()
        elif self._simulation_mode == "single":
            if self._physics_scene is not None:
                self._physics_scene.shutdown()
                self._physics_scene = None
            self._solver = None
        else:
            if self._physics_process is not None:
                self._physics_process.shutdown(1000)
                self._physics_process = None
        self._simulation_mode = new_mode
        if new_mode == "single":
            self._init_single(*self._solver_module_class_with_settings())
        elif new_mode == "per_layer_process":
            Logger.info("PhysicsPlugin: per-layer process mode (processes spawned on demand).")
        else:
            solver_module, solver_class = self._solver_module_class()
            self._physics_process = PhysicsProcess(project_root=self._project_root)
            if not self._physics_process.start(solver_module, solver_class, settings):
                Logger.warning("PhysicsPlugin: multi-threaded init failed, falling back to single-threaded mode")
                self._physics_process = None
                self._simulation_mode = "single"
                self._init_single(*self._solver_module_class_with_settings())

    def on_scene_unloaded(self, scene):
        self._scanned_entity_ids.clear()
        self._last_entity_count = -1
        self._prev_frame_contacts.clear()
        self._step_caches.clear()
        self._collision_listener_sig = None
        self._collision_cache_valid = False
        self._clear_soft_remote()
        self._reset_entity_velocities(scene)
        if self._simulation_mode == "per_layer_process":
            for proc in self._layer_processes.values():
                proc.clear_slots()
                proc.clear_soft_slots()
                proc.send({"type": "unload_all"})
        elif self._simulation_mode == "single":
            if self._physics_scene:
                self._physics_scene.shutdown()
        else:
            if self._physics_process:
                self._physics_process.clear_slots()
                self._physics_process.clear_soft_slots()
                self._physics_process.send({"type": "unload_all"})

    def _start_fresh_process(self) -> bool:
        if self._physics_process is not None:
            self._physics_process.shutdown(500)
            self._physics_process = None
        self._step_caches.clear()
        self._clear_soft_remote()
        sm, sc = self._solver_module_class()
        settings = self._get_physics_settings()
        new_proc = PhysicsProcess(project_root=self._project_root)
        if new_proc.start(sm, sc, settings):
            self._physics_process = new_proc
            return True
        Logger.error("PhysicsPlugin: failed to start physics process")
        return False

    def on_play_start(self):
        from core.foundation.logger import Logger
        Logger.info(f"[PhysicsPlugin] on_play_start called, mode={self._simulation_mode}")
        self._scanned_entity_ids.clear()
        self._last_entity_count = -1
        self._prev_frame_contacts.clear()
        self._collision_listener_sig = None
        self._collision_cache_valid = False
        self._clear_soft_remote()
        if self._engine is None:
            Logger.info("[PhysicsPlugin] on_play_start: engine is None, returning")
            return
        scene = self._engine.scene
        if scene is None:
            Logger.info("[PhysicsPlugin] on_play_start: scene is None, returning")
            return

        if self._simulation_mode == "single":
            if self._physics_scene is None:
                Logger.info("[PhysicsPlugin] on_play_start: physics_scene is None, returning")
                return
            self._physics_scene.load_scene(scene)
            Logger.info(f"[PhysicsPlugin] Scene loaded (single-threaded).")
        elif self._simulation_mode == "per_layer_process":
            for proc in self._layer_processes.values():
                proc.drain()
            self._layer_processes.clear()
            layer_bodies: dict[int, list[dict]] = {}
            for entity in scene.get_all_entities():
                rb = entity._components.get("Rigidbody")
                rb2d = entity._components.get("Rigidbody2D")
                tr = entity._components.get("Transform")
                if (not rb and not rb2d) or not tr:
                    continue
                layer = self._get_entity_layer(entity)
                proc = self._get_layer_process(layer)
                bd = self._body_with_slot_in_process(entity, tr, proc)
                if bd:
                    layer_bodies.setdefault(layer, []).append(bd)
            for layer, bodies in layer_bodies.items():
                proc = self._layer_processes[layer]
                proc.send({"type": "load_bodies", "bodies": bodies})
            for layer, proc in list(self._layer_processes.items()):
                if proc.wait_for_result("load_bodies", timeout=5.0) is None:
                    Logger.error(f"PhysicsPlugin: load_bodies timed out for layer {layer}")
            total = sum(len(v) for v in layer_bodies.values())
            Logger.info(f"[PhysicsPlugin] Scene loaded with {total} bodies across {len(self._layer_processes)} layer processes.")
            layer_softs: dict[int, list[dict]] = {}
            try:
                for entity in scene.get_all_entities():
                    try:
                        from core.components.physics.soft_body import SoftBody
                        soft = entity.get_component(SoftBody)
                    except Exception:
                        continue
                    if soft is None or not getattr(soft, "enabled", True):
                        continue
                    tr = entity._components.get("Transform")
                    if tr is None:
                        continue
                    if entity.id in self._soft_remote:
                        continue
                    layer = self._get_entity_layer(entity)
                    proc = self._get_layer_process(layer)
                    d = self._build_soft_dict(entity, tr, soft, proc)
                    if d is None:
                        continue
                    self._track_soft_dict(entity, soft, tr, proc, d["slot"], d)
                    layer_softs.setdefault(layer, []).append(d)
            except Exception as e:
                Logger.error(f"PhysicsPlugin: per-layer soft collect failed: {e}")
                layer_softs = {}
            for layer, softs in layer_softs.items():
                proc = self._layer_processes[layer]
                proc.send({"type": "load_soft_bodies", "softs": softs})
            for layer, proc in list(self._layer_processes.items()):
                if layer not in layer_softs:
                    continue
                ack = proc.wait_for_result("soft_loaded", timeout=60.0)
                if ack is None:
                    Logger.error(f"PhysicsPlugin: load_soft_bodies timed out for layer {layer}")
                else:
                    self._harvest_soft_created(proc, scene, [ack])
        else:
            if not self._start_fresh_process():
                return
            bodies = []
            for entity in scene.get_all_entities():
                rb = entity._components.get("Rigidbody")
                rb2d = entity._components.get("Rigidbody2D")
                tr = entity._components.get("Transform")
                if (not rb and not rb2d) or not tr:
                    continue
                bd = self._body_with_slot(entity, tr)
                if bd:
                    bodies.append(bd)
            softs = self._collect_soft_dicts(scene, self._physics_process)
            if not bodies and not softs:
                return
            if bodies:
                self._physics_process.send({"type": "load_bodies", "bodies": bodies})
                if self._physics_process.wait_for_result("load_bodies", timeout=5.0) is None:
                    Logger.error("PhysicsPlugin: load_bodies timed out, shutting down process")
                    self._physics_process.shutdown(500)
                    self._physics_process = None
                    self._clear_soft_remote()
                    return
                Logger.info(f"[PhysicsPlugin] Scene loaded with {len(bodies)} bodies (shared-memory).")
            if softs:
                self._physics_process.send({"type": "load_soft_bodies", "softs": softs})
                ack = self._physics_process.wait_for_result("soft_loaded", timeout=60.0)
                if ack is None:
                    Logger.error("PhysicsPlugin: load_soft_bodies timed out")
                else:
                    self._harvest_soft_created(self._physics_process, scene, [ack])
                Logger.info(f"[PhysicsPlugin] Scene loaded with {len(softs)} soft bodies (shared-memory).")

    def _reset_entity_velocities(self, scene=None):
        if scene is None:
            if not self._engine or not self._engine.scene:
                return
            scene = self._engine.scene
        for entity in scene.get_all_entities():
            rb = entity._components.get("Rigidbody")
            if rb:
                rb._velocity = Vec3.zero()
                rb._angular_velocity = Vec3.zero()
                rb._force_accum = Vec3.zero()
                rb._torque_accum = Vec3.zero()
            rb2d = entity._components.get("Rigidbody2D")
            if rb2d:
                rb2d._velocity = Vec2.zero()
                rb2d._angular_velocity = 0.0
                rb2d._force_accum = Vec2.zero()
                rb2d._torque_accum = 0.0

    def _unload_and_wait(self, proc: PhysicsProcess):
        proc.send({"type": "unload_all"})
        if proc.wait_for_result("unload_all", timeout=3.0) is None:
            Logger.warning("PhysicsPlugin: unload_all timed out, terminating process")
            proc.shutdown(500)

    def on_play_stop(self):
        self._scanned_entity_ids.clear()
        self._last_entity_count = -1
        self._prev_frame_contacts.clear()
        self._clear_soft_remote()
        self._reset_entity_velocities()
        if self._simulation_mode == "per_layer_process":
            for proc in self._layer_processes.values():
                proc.clear_slots()
                proc.clear_soft_slots()
                self._unload_and_wait(proc)
        elif self._simulation_mode == "single":
            if self._physics_scene:
                self._physics_scene.shutdown()
        else:
            if self._physics_process:
                self._physics_process.clear_slots()
                self._physics_process.clear_soft_slots()
                self._unload_and_wait(self._physics_process)
            self._step_caches.clear()

    def pre_step(self, dt: float):
        if not self._enabled:
            return
        if not self._engine or not self._engine.scene:
            return
        scene = self._engine.scene

        if self._simulation_mode == "single":
            if self._physics_scene is None:
                return
            entity_count = len(scene._entities)
            if entity_count != self._last_entity_count or not self._scanned_entity_ids:
                self._last_entity_count = entity_count
                scanned = self._scanned_entity_ids
                for eid, entity in scene._entities.items():
                    if eid in scanned:
                        continue
                    scanned.add(eid)
                    # Soft-only entities have no Rigidbody: they must still get
                    # a soft body (otherwise they never simulate / never fall).
                    try:
                        self._physics_scene._create_entity_soft(entity)
                    except Exception:
                        pass
                    rb = entity._components.get("Rigidbody")
                    rb2d = entity._components.get("Rigidbody2D")
                    tr = entity._components.get("Transform")
                    if (not rb and not rb2d) or not tr:
                        continue
                    if eid in self._physics_scene._entity_to_body:
                        continue
                    self._physics_scene._create_entity_bodies(entity)
            return

        if self._simulation_mode == "per_layer_process":
            entities_dict = scene._entities
            for eid, entity in entities_dict.items():
                if eid in self._scanned_entity_ids:
                    continue
                self._scanned_entity_ids.add(eid)
                tr = entity._components.get("Transform")
                if tr is not None and eid not in self._soft_remote:
                    try:
                        from core.components.physics.soft_body import SoftBody
                        soft = entity.get_component(SoftBody)
                    except Exception:
                        soft = None
                    if soft is not None and getattr(soft, "enabled", True):
                        try:
                            layer = self._get_entity_layer(entity)
                            proc = self._get_layer_process(layer)
                        except Exception:
                            proc = None
                        if proc is not None:
                            try:
                                d = self._build_soft_dict(entity, tr, soft, proc)
                            except Exception:
                                d = None
                            if d is not None:
                                self._track_soft_dict(entity, soft, tr, proc, d["slot"], d)
                                try:
                                    proc.send({"type": "add_soft_body", "soft": d})
                                except Exception:
                                    pass
                rb = entity._components.get("Rigidbody")
                rb2d = entity._components.get("Rigidbody2D")
                if (not rb and not rb2d) or not tr:
                    continue
                layer = self._get_entity_layer(entity)
                if layer in self._layer_processes:
                    proc = self._layer_processes[layer]
                else:
                    proc = self._get_layer_process(layer)
                if eid in proc.entity_slot_map:
                    continue
                bd = self._body_with_slot_in_process(entity, tr, proc)
                if bd:
                    proc.send({"type": "add_body", "body": bd})
            return

        if self._physics_process is None:
            return
        entities_dict = scene._entities
        entity_count = len(entities_dict)
        if entity_count != self._last_entity_count or not self._scanned_entity_ids:
            self._last_entity_count = entity_count
            scanned = self._scanned_entity_ids
            for eid, entity in entities_dict.items():
                if eid in scanned:
                    continue
                scanned.add(eid)
                tr = entity._components.get("Transform")
                if tr is not None and eid not in self._soft_remote:
                    try:
                        from core.components.physics.soft_body import SoftBody
                        soft = entity.get_component(SoftBody)
                    except Exception:
                        soft = None
                    if soft is not None and getattr(soft, "enabled", True):
                        try:
                            d = self._build_soft_dict(entity, tr, soft, self._physics_process)
                        except Exception:
                            d = None
                        if d is not None:
                            self._track_soft_dict(entity, soft, tr, self._physics_process, d["slot"], d)
                            try:
                                self._physics_process.send({"type": "add_soft_body", "soft": d})
                            except Exception:
                                pass
                rb = entity._components.get("Rigidbody")
                rb2d = entity._components.get("Rigidbody2D")
                if (not rb and not rb2d) or not tr:
                    continue
                if eid in self._physics_process.entity_slot_map:
                    continue
                bd = self._body_with_slot(entity, tr)
                if bd:
                    self._physics_process.send({"type": "add_body", "body": bd})

    def _rebuild_step_cache(self, proc, entities):
        _cache = []
        for eid, slot in proc.entity_slot_map.items():
            entity = entities.get(eid)
            if not entity or not entity._active:
                continue
            rb = entity._components.get("Rigidbody")
            rb2d = entity._components.get("Rigidbody2D")
            tr = entity._components.get("Transform")
            if not tr:
                continue
            _cache.append((entity, rb, rb2d, tr, slot))
        return _cache

    def _read_results_python(self, shared, cache) -> None:
        _flags = shared._flags_nd
        _rdata = shared._rdata_nd
        for entity, rb, rb2d, tr, slot in cache:
            if getattr(tr, "_physics_dirty", False):
                continue
            if rb2d is not None and rb2d.is_kinematic:
                continue
            if rb is not None and rb.is_kinematic:
                continue
            fl = int(_flags[slot])
            if not (fl & 1) or (fl & 4):
                continue
            if rb2d:
                nx = float(_rdata[slot, 0])
                ny = float(_rdata[slot, 1])
                nqz = float(_rdata[slot, 5])
                nqw = float(_rdata[slot, 6])
                if abs(tr._local_pos._x - nx) < 1e-5 and abs(tr._local_pos._y - ny) < 1e-5 and abs(tr._local_rot._z - nqz) < 1e-4 and abs(tr._local_rot._w - nqw) < 1e-4:
                    if not getattr(rb2d, "_velocity_dirty", False):
                        rb2d._velocity._x = float(_rdata[slot, 7])
                        rb2d._velocity._y = float(_rdata[slot, 8])
                        rb2d._angular_velocity = float(_rdata[slot, 12])
                    rb2d._force_accum._x = 0.0
                    rb2d._force_accum._y = 0.0
                    rb2d._torque_accum = 0.0
                    continue
                tr._local_pos._x = nx
                tr._local_pos._y = ny
                tr._local_pos._z = 0.0
                tr._local_rot._x = 0.0
                tr._local_rot._y = 0.0
                tr._local_rot._z = nqz
                tr._local_rot._w = nqw
                tr._dirty = True
                scn = entity._scene
                if scn is not None:
                    scn._dirty_roots.add(tr)
                    scn._spatial_dirty_entities.add(entity.id)
                    scn._spatial_dirty = True
                if not getattr(rb2d, "_velocity_dirty", False):
                    rb2d._velocity._x = float(_rdata[slot, 7])
                    rb2d._velocity._y = float(_rdata[slot, 8])
                    rb2d._angular_velocity = float(_rdata[slot, 12])
                rb2d._force_accum._x = 0.0
                rb2d._force_accum._y = 0.0
                rb2d._torque_accum = 0.0
            elif rb:
                nx = float(_rdata[slot, 0])
                ny = float(_rdata[slot, 1])
                nz = float(_rdata[slot, 2])
                nqx = float(_rdata[slot, 3])
                nqy = float(_rdata[slot, 4])
                nqz = float(_rdata[slot, 5])
                nqw = float(_rdata[slot, 6])
                if abs(tr._local_pos._x - nx) < 1e-5 and abs(tr._local_pos._y - ny) < 1e-5 and abs(tr._local_pos._z - nz) < 1e-5 and abs(tr._local_rot._x - nqx) < 1e-4 and abs(tr._local_rot._y - nqy) < 1e-4 and abs(tr._local_rot._z - nqz) < 1e-4 and abs(tr._local_rot._w - nqw) < 1e-4:
                    if not getattr(rb, "_velocity_dirty", False):
                        rb._velocity._x = float(_rdata[slot, 7])
                        rb._velocity._y = float(_rdata[slot, 8])
                        rb._velocity._z = float(_rdata[slot, 9])
                        rb._angular_velocity._x = float(_rdata[slot, 10])
                        rb._angular_velocity._y = float(_rdata[slot, 11])
                        rb._angular_velocity._z = float(_rdata[slot, 12])
                    continue
                tr._local_pos._x = nx
                tr._local_pos._y = ny
                tr._local_pos._z = nz
                tr._local_rot._x = nqx
                tr._local_rot._y = nqy
                tr._local_rot._z = nqz
                tr._local_rot._w = nqw
                tr._dirty = True
                scn = entity._scene
                if scn is not None:
                    scn._dirty_roots.add(tr)
                    scn._spatial_dirty_entities.add(entity.id)
                    scn._spatial_dirty = True
                if not getattr(rb, "_velocity_dirty", False):
                    rb._velocity._x = float(_rdata[slot, 7])
                    rb._velocity._y = float(_rdata[slot, 8])
                    rb._velocity._z = float(_rdata[slot, 9])
                    rb._angular_velocity._x = float(_rdata[slot, 10])
                    rb._angular_velocity._y = float(_rdata[slot, 11])
                    rb._angular_velocity._z = float(_rdata[slot, 12])

    def _write_inputs_python(self, shared, cache) -> int:
        _flags = shared._flags_nd
        _edata = shared._edata_nd
        _fdata = shared._fdata_nd
        slots, pos_x, pos_y, pos_z = [], [], [], []
        qx_l, qy_l, qz_l, qw_l = [], [], [], []
        vel_x, vel_y, vel_z = [], [], []
        av_x, av_y, av_z = [], [], []
        f_x, f_y, f_z = [], [], []
        t_x, t_y, t_z = [], [], []
        flv = []
        max_slot = -1
        for entity, rb, rb2d, tr, slot in cache:
            if not entity._active:
                continue
            lp = tr._local_pos
            slots.append(slot)
            if rb2d:
                if getattr(tr, "_physics_dirty", False):
                    q = tr._local_rot
                    pos_x.append(lp._x)
                    pos_y.append(lp._y)
                    pos_z.append(0.0)
                    qx_l.append(0.0)
                    qy_l.append(0.0)
                    qz_l.append(q._z)
                    qw_l.append(q._w)
                    vel_x.append(rb2d._velocity._x)
                    vel_y.append(rb2d._velocity._y)
                    vel_z.append(0.0)
                    av_x.append(0.0)
                    av_y.append(0.0)
                    av_z.append(rb2d._angular_velocity)
                    fa = rb2d._force_accum
                    f_x.append(fa._x)
                    f_y.append(fa._y)
                    f_z.append(0.0)
                    t_x.append(0.0)
                    t_y.append(0.0)
                    t_z.append(rb2d._torque_accum)
                    flv.append(15)
                    tr._physics_dirty = False
                elif rb2d.is_kinematic:
                    q = tr._local_rot
                    pos_x.append(lp._x)
                    pos_y.append(lp._y)
                    pos_z.append(0.0)
                    qx_l.append(0.0)
                    qy_l.append(0.0)
                    qz_l.append(q._z)
                    qw_l.append(q._w)
                    vel_x.append(rb2d._velocity._x)
                    vel_y.append(rb2d._velocity._y)
                    vel_z.append(0.0)
                    av_x.append(0.0)
                    av_y.append(0.0)
                    av_z.append(rb2d._angular_velocity)
                    fa = rb2d._force_accum
                    f_x.append(fa._x)
                    f_y.append(fa._y)
                    f_z.append(0.0)
                    t_x.append(0.0)
                    t_y.append(0.0)
                    t_z.append(rb2d._torque_accum)
                    flv.append(15)
                else:
                    pos_x.append(0.0)
                    pos_y.append(0.0)
                    pos_z.append(0.0)
                    qx_l.append(0.0)
                    qy_l.append(0.0)
                    qz_l.append(0.0)
                    qw_l.append(0.0)
                    vel_x.append(rb2d._velocity._x)
                    vel_y.append(rb2d._velocity._y)
                    vel_z.append(0.0)
                    av_x.append(0.0)
                    av_y.append(0.0)
                    av_z.append(rb2d._angular_velocity)
                    fa = rb2d._force_accum
                    f_x.append(fa._x)
                    f_y.append(fa._y)
                    f_z.append(0.0)
                    t_x.append(0.0)
                    t_y.append(0.0)
                    t_z.append(rb2d._torque_accum)
                    flv.append(11 if rb2d.consume_velocity_dirty() else 9)
            elif rb:
                if getattr(tr, "_physics_dirty", False):
                    q = tr._local_rot
                    pos_x.append(lp._x)
                    pos_y.append(lp._y)
                    pos_z.append(lp._z)
                    qx_l.append(q._x)
                    qy_l.append(q._y)
                    qz_l.append(q._z)
                    qw_l.append(q._w)
                    vel_x.append(rb._velocity._x)
                    vel_y.append(rb._velocity._y)
                    vel_z.append(rb._velocity._z)
                    av_x.append(rb._angular_velocity._x)
                    av_y.append(rb._angular_velocity._y)
                    av_z.append(rb._angular_velocity._z)
                    fa = rb._force_accum
                    ta = rb._torque_accum
                    f_x.append(fa._x)
                    f_y.append(fa._y)
                    f_z.append(fa._z)
                    t_x.append(ta._x)
                    t_y.append(ta._y)
                    t_z.append(ta._z)
                    fa._x = 0.0
                    fa._y = 0.0
                    fa._z = 0.0
                    ta._x = 0.0
                    ta._y = 0.0
                    ta._z = 0.0
                    flv.append(7)
                    tr._physics_dirty = False
                elif rb.is_kinematic:
                    q = tr._local_rot
                    pos_x.append(lp._x)
                    pos_y.append(lp._y)
                    pos_z.append(lp._z)
                    qx_l.append(q._x)
                    qy_l.append(q._y)
                    qz_l.append(q._z)
                    qw_l.append(q._w)
                    vel_x.append(rb._velocity._x)
                    vel_y.append(rb._velocity._y)
                    vel_z.append(rb._velocity._z)
                    av_x.append(rb._angular_velocity._x)
                    av_y.append(rb._angular_velocity._y)
                    av_z.append(rb._angular_velocity._z)
                    fa = rb._force_accum
                    ta = rb._torque_accum
                    f_x.append(fa._x)
                    f_y.append(fa._y)
                    f_z.append(fa._z)
                    t_x.append(ta._x)
                    t_y.append(ta._y)
                    t_z.append(ta._z)
                    fa._x = 0.0
                    fa._y = 0.0
                    fa._z = 0.0
                    ta._x = 0.0
                    ta._y = 0.0
                    ta._z = 0.0
                    flv.append(7)
                else:
                    pos_x.append(0.0)
                    pos_y.append(0.0)
                    pos_z.append(0.0)
                    qx_l.append(0.0)
                    qy_l.append(0.0)
                    qz_l.append(0.0)
                    qw_l.append(0.0)
                    vel_x.append(rb._velocity._x)
                    vel_y.append(rb._velocity._y)
                    vel_z.append(rb._velocity._z)
                    av_x.append(rb._angular_velocity._x)
                    av_y.append(rb._angular_velocity._y)
                    av_z.append(rb._angular_velocity._z)
                    fa = rb._force_accum
                    ta = rb._torque_accum
                    f_x.append(fa._x)
                    f_y.append(fa._y)
                    f_z.append(fa._z)
                    t_x.append(ta._x)
                    t_y.append(ta._y)
                    t_z.append(ta._z)
                    fa._x = 0.0
                    fa._y = 0.0
                    fa._z = 0.0
                    ta._x = 0.0
                    ta._y = 0.0
                    ta._z = 0.0
                    flv.append(3 if rb.consume_velocity_dirty() else 1)
            if slot > max_slot:
                max_slot = slot

        if slots:
            _edata[slots, 0] = pos_x
            _edata[slots, 1] = pos_y
            _edata[slots, 2] = pos_z
            _edata[slots, 3] = qx_l
            _edata[slots, 4] = qy_l
            _edata[slots, 5] = qz_l
            _edata[slots, 6] = qw_l
            _edata[slots, 7] = vel_x
            _edata[slots, 8] = vel_y
            _edata[slots, 9] = vel_z
            _edata[slots, 10] = av_x
            _edata[slots, 11] = av_y
            _edata[slots, 12] = av_z
            _fdata[slots, 0] = f_x
            _fdata[slots, 1] = f_y
            _fdata[slots, 2] = f_z
            _fdata[slots, 3] = t_x
            _fdata[slots, 4] = t_y
            _fdata[slots, 5] = t_z
            _flags[slots] = flv
        return max_slot

    def _step_process(self, proc: PhysicsProcess, scene, dt: float, prof) -> list:
        shared = proc.shared
        ets = proc.entity_slot_map
        entities = scene._entities

        proc_id = id(proc)
        gen_key = (len(ets), self._cache_version)
        entry = self._step_caches.get(proc_id)
        if entry is None or entry[0] != gen_key:
            self._step_caches[proc_id] = (gen_key, self._rebuild_step_cache(proc, entities))
        _cache = self._step_caches[proc_id][1]

        # 1) Drain result queue FIRST.
        #    multiprocessing.Queue.get() provides acquire semantics (internal mutex),
        #    guaranteeing all preceding shared memory writes from the physics process
        #    are visible — portable across x86 and ARM.
        pending_results = []
        result = proc.poll()
        while result is not None:
            if result.get("type") == "step_result":
                pending_results.append(result)
            result = proc.poll()

        # 2) Read transforms only from the latest result
        #    (shared memory contains the most recent write).
        if pending_results:
            _read = _COMPILED_SYNC_READ
            if _read is not None:
                _read(shared, _cache)
            else:
                self._read_results_python(shared, _cache)

        if not self._has_collision_listeners(scene):
            for r in pending_results:
                if r.get("collision_events"):
                    self._prev_frame_contacts.clear()
                    break
            events_accum = []
        else:
            events_accum = []
            for r in pending_results:
                ev = r.get("collision_events", [])
                if ev:
                    if len(events_accum) + len(ev) > 2048:
                        events_accum.extend(ev[: max(0, 2048 - len(events_accum))])
                    else:
                        events_accum.extend(ev)

        _write = _COMPILED_SYNC_WRITE
        if _write is not None:
            max_slot = _write(shared, _cache)
        else:
            max_slot = self._write_inputs_python(shared, _cache)
        shared.set_num_entities(max_slot + 1 if max_slot >= 0 else 0)

        need_coll = self._has_collision_listeners(scene)
        soft_velocities = None
        try:
            soft_velocities = self._sync_soft_remote(proc, scene, dt, pending_results)
        except Exception:
            soft_velocities = None
        msg = {"type": "step", "dt": dt, "need_collisions": need_coll}
        if soft_velocities:
            msg["soft_velocities"] = soft_velocities
        proc.send(msg)
        return events_accum

    def step(self, dt: float):
        if not self._enabled:
            return
        if not self._engine or not self._engine.scene:
            return
        scene = self._engine.scene
        prof = self._engine.profiler

        if self._simulation_mode == "single":
            if self._physics_scene is None:
                return
            prof.start("physics_step")
            self._physics_scene.step(dt)
            prof.stop("physics_step")
            return

        prof.start("physics_collect_results")
        if self._simulation_mode == "per_layer_process":
            for proc in list(self._layer_processes.values()):
                events = self._step_process(proc, scene, dt, prof)
                if events:
                    self._process_collisions(scene, events)
        elif self._physics_process:
            events = self._step_process(self._physics_process, scene, dt, prof)
            if events:
                self._process_collisions(scene, events)
        prof.stop("physics_collect_results")

    def _has_collision_listeners(self, scene) -> bool:
        if scene is None:
            return False
        sig = (len(scene._entities), sum(len(e._components) for e in scene._entities.values()))
        if sig != self._collision_listener_sig:
            self._collision_listener_sig = sig
            self._collision_cache_valid = False
        if self._collision_cache_valid:
            return self._has_collision_cache
        self._collision_cache_valid = True
        self._has_collision_cache = False
        for ent in scene._entities.values():
            for comp in ent._components.values():
                if type(comp).__name__ == "ScriptComponent":
                    inst = getattr(comp, "_py_instance", None)
                    if inst is None:
                        continue
                    if hasattr(inst, "on_collision_enter") or hasattr(inst, "on_collision_stay") or hasattr(inst, "on_collision_exit"):
                        self._has_collision_cache = True
                        return True
                elif hasattr(comp, "on_collision_enter"):
                    self._has_collision_cache = True
                    return True
        return False

    def _process_collisions(self, scene, events: list):
        if not events:
            return
        if len(events) > 2048:
            events = events[:2048]
        entities = scene._entities
        current: set = set()
        forces: dict = {}
        for ev in events:
            ea, eb = ev.get("entity_a", ""), ev.get("entity_b", "")
            if ea and eb:
                if ea > eb:
                    ea, eb = eb, ea
                pair = (ea, eb)
                if pair not in current:
                    current.add(pair)
                    forces[pair] = float(ev.get("force", 0.0) or 0.0)
                else:
                    f = float(ev.get("force", 0.0) or 0.0)
                    if f > forces[pair]:
                        forces[pair] = f
        prev = self._prev_frame_contacts
        entered = current - prev
        exited = prev - current
        stayed = current & prev
        for pair in entered:
            e0, e1 = pair
            self._dispatch_collision(entities, e0, e1, "on_collision_enter", forces.get(pair, 0.0))
            self._dispatch_collision(entities, e1, e0, "on_collision_enter", forces.get(pair, 0.0))
        for pair in exited:
            e0, e1 = pair
            self._dispatch_collision(entities, e0, e1, "on_collision_exit", forces.get(pair, 0.0))
            self._dispatch_collision(entities, e1, e0, "on_collision_exit", forces.get(pair, 0.0))
        for pair in stayed:
            e0, e1 = pair
            self._dispatch_collision(entities, e0, e1, "on_collision_stay", forces.get(pair, 0.0))
            self._dispatch_collision(entities, e1, e0, "on_collision_stay", forces.get(pair, 0.0))
        self._prev_frame_contacts = current

    def _dispatch_collision(self, entities, eid: str, other_eid: str, callback: str, force: float = 0.0):
        entity = entities.get(eid)
        if not entity:
            return
        from core.components import ScriptComponent
        for sc in entity.get_components(ScriptComponent):
            inst = sc._py_instance
            if inst and hasattr(inst, callback):
                try:
                    getattr(inst, callback)(other_eid)
                except Exception as e:
                    Logger.error(f"Script {callback} error: {e}")
        for comp in entity.get_all_components():
            if isinstance(comp, ScriptComponent):
                continue
            if hasattr(comp, callback):
                try:
                    getattr(comp, callback)(other_eid, force)
                except Exception as e:
                    Logger.error(f"Component {callback} error: {e}")

    def shutdown(self):
        if self._simulation_mode == "per_layer_process":
            for proc in self._layer_processes.values():
                proc.shutdown(5000)
            self._layer_processes.clear()
        elif self._simulation_mode == "single":
            if self._physics_scene:
                self._physics_scene.shutdown()
                self._physics_scene = None
            if self._solver:
                self._solver.shutdown()
                self._solver = None
        else:
            if self._physics_process:
                self._physics_process.shutdown(5000)
                self._physics_process = None
