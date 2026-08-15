# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
from typing import TYPE_CHECKING
from core.foundation.plugin_manager import PluginBase
from core.foundation.logger import Logger
from core.physics import PhysicsProcess, PhysicsScene
from core.physics.shared_buffer import MAX_ENTITIES
from core.physics.physics_solver import IPhysicsSolver
from core.maths.math3d import Vec2, Vec3
from core.physics.shape_utils import find_shape_info
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
        shape_info = find_shape_info(entity, tr)
        if not shape_info:
            return None
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

    def on_scene_loaded(self, scene):
        self._scanned_entity_ids.clear()
        self._last_entity_count = -1
        self._prev_frame_contacts.clear()
        self._step_caches.clear()

    def on_project_opened(self):
        settings = self._get_physics_settings()
        new_mode = settings.get("simulation_mode", "multi_threaded")
        if new_mode == self._simulation_mode:
            return
        Logger.info(f"[PhysicsPlugin] project opened: simulation_mode {self._simulation_mode} -> {new_mode}")
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
        self._reset_entity_velocities(scene)
        if self._simulation_mode == "per_layer_process":
            for proc in self._layer_processes.values():
                proc.clear_slots()
                proc.send({"type": "unload_all"})
        elif self._simulation_mode == "single":
            if self._physics_scene:
                self._physics_scene.shutdown()
        else:
            if self._physics_process:
                self._physics_process.clear_slots()
                self._physics_process.send({"type": "unload_all"})

    def _start_fresh_process(self) -> bool:
        if self._physics_process is not None:
            self._physics_process.shutdown(500)
            self._physics_process = None
        self._step_caches.clear()
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
            if not bodies:
                return
            self._physics_process.send({"type": "load_bodies", "bodies": bodies})
            if self._physics_process.wait_for_result("load_bodies", timeout=5.0) is None:
                Logger.error("PhysicsPlugin: load_bodies timed out, shutting down process")
                self._physics_process.shutdown(500)
                self._physics_process = None
                return
            Logger.info(f"[PhysicsPlugin] Scene loaded with {len(bodies)} bodies (shared-memory).")

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
        self._reset_entity_velocities()
        if self._simulation_mode == "per_layer_process":
            for proc in self._layer_processes.values():
                proc.clear_slots()
                self._unload_and_wait(proc)
        elif self._simulation_mode == "single":
            if self._physics_scene:
                self._physics_scene.shutdown()
        else:
            if self._physics_process:
                self._physics_process.clear_slots()
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
                rb = entity._components.get("Rigidbody")
                rb2d = entity._components.get("Rigidbody2D")
                tr = entity._components.get("Transform")
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
                rb = entity._components.get("Rigidbody")
                rb2d = entity._components.get("Rigidbody2D")
                tr = entity._components.get("Transform")
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
            fl = int(_flags[slot])
            if not (fl & 1) or (fl & 4):
                continue
            row = _rdata[slot].tolist()
            if rb2d:
                tr._local_pos._x = row[0]
                tr._local_pos._y = row[1]
                tr._local_pos._z = 0.0
                hz = row[5] * 0.5
                tr._local_rot._x = 0.0
                tr._local_rot._y = 0.0
                tr._local_rot._z = math.sin(hz)
                tr._local_rot._w = math.cos(hz)
                tr._dirty = True
                if not getattr(rb2d, "_velocity_dirty", False):
                    rb2d._velocity._x = row[6]
                    rb2d._velocity._y = row[7]
                    rb2d._angular_velocity = row[11]
                rb2d._force_accum._x = 0.0
                rb2d._force_accum._y = 0.0
                rb2d._torque_accum = 0.0
            elif rb:
                tr._local_pos._x = row[0]
                tr._local_pos._y = row[1]
                tr._local_pos._z = row[2]
                r0 = row[3]
                r1 = row[4]
                r2 = row[5]
                sr, cr = math.sin(r0 * 0.5), math.cos(r0 * 0.5)
                sp, cp = math.sin(r1 * 0.5), math.cos(r1 * 0.5)
                sy, cy = math.sin(r2 * 0.5), math.cos(r2 * 0.5)
                tr._local_rot._x = sr * cp * cy - cr * sp * sy
                tr._local_rot._y = cr * sp * cy + sr * cp * sy
                tr._local_rot._z = cr * cp * sy - sr * sp * cy
                tr._local_rot._w = cr * cp * cy + sr * sp * sy
                tr._dirty = True
                if not getattr(rb, "_velocity_dirty", False):
                    rb._velocity._x = row[6]
                    rb._velocity._y = row[7]
                    rb._velocity._z = row[8]
                    rb._angular_velocity._x = row[9]
                    rb._angular_velocity._y = row[10]
                    rb._angular_velocity._z = row[11]

    def _write_inputs_python(self, shared, cache) -> int:
        _flags = shared._flags_nd
        _edata = shared._edata_nd
        _fdata = shared._fdata_nd
        slots, pos_x, pos_y, pos_z = [], [], [], []
        rot_x, rot_y, rot_z = [], [], []
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
            pos_x.append(lp._x)
            pos_y.append(lp._y)
            if rb2d:
                q = tr._local_rot
                pos_z.append(0.0)
                rot_x.append(0.0)
                rot_y.append(0.0)
                rot_z.append(2.0 * math.asin(max(-1.0, min(1.0, q._z))))
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
                if rb2d.is_kinematic:
                    flv.append(15)
                else:
                    flv.append(11 if rb2d.consume_velocity_dirty() else 9)
            elif rb:
                q = tr._local_rot
                pos_z.append(lp._z)
                qx, qy, qz, qw = q._x, q._y, q._z, q._w
                rot_x.append(math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)))
                rot_y.append(math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))))
                rot_z.append(math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
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
                if rb.is_kinematic:
                    flv.append(7)
                else:
                    flv.append(3 if rb.consume_velocity_dirty() else 1)
            if slot > max_slot:
                max_slot = slot

        if slots:
            _edata[slots, 0] = pos_x
            _edata[slots, 1] = pos_y
            _edata[slots, 2] = pos_z
            _edata[slots, 3] = rot_x
            _edata[slots, 4] = rot_y
            _edata[slots, 5] = rot_z
            _edata[slots, 6] = vel_x
            _edata[slots, 7] = vel_y
            _edata[slots, 8] = vel_z
            _edata[slots, 9] = av_x
            _edata[slots, 10] = av_y
            _edata[slots, 11] = av_z
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

        # 3) Accumulate collision events from all pending results
        events_accum = []
        for r in pending_results:
            events_accum.extend(r.get("collision_events", []))

        # 4) Write input data to shared memory
        _write = _COMPILED_SYNC_WRITE
        if _write is not None:
            max_slot = _write(shared, _cache)
        else:
            max_slot = self._write_inputs_python(shared, _cache)
        shared.set_num_entities(max_slot + 1 if max_slot >= 0 else 0)

        proc.send({"type": "step", "dt": dt})
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

    def _process_collisions(self, scene, events: list):
        if not events:
            return
        entities = scene._entities
        current: set = set()
        forces: dict = {}
        for ev in events:
            ea, eb = ev.get("entity_a", ""), ev.get("entity_b", "")
            if ea and eb:
                pair = frozenset((ea, eb))
                current.add(pair)
                force = float(ev.get("force", 0.0) or 0.0)
                forces[pair] = max(forces.get(pair, 0.0), force)
        entered = current - self._prev_frame_contacts
        exited = self._prev_frame_contacts - current
        stayed = current & self._prev_frame_contacts
        for pair in entered:
            e0, e1 = tuple(pair)
            self._dispatch_collision(entities, e0, e1, "on_collision_enter", forces.get(pair, 0.0))
            self._dispatch_collision(entities, e1, e0, "on_collision_enter", forces.get(pair, 0.0))
        for pair in exited:
            e0, e1 = tuple(pair)
            self._dispatch_collision(entities, e0, e1, "on_collision_exit", forces.get(pair, 0.0))
            self._dispatch_collision(entities, e1, e0, "on_collision_exit", forces.get(pair, 0.0))
        for pair in stayed:
            e0, e1 = tuple(pair)
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
