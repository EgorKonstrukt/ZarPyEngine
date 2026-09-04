# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import multiprocessing
import queue
import sys
import os
import time
from typing import Optional
from core.physics.shared_buffer import SharedPhysicsBuffer, MAX_ENTITIES
from core.physics.soft_shared_buffer import SoftSharedBuffer


class PhysicsProcess:
    def __init__(self, project_root: str = ""):
        self._project_root = project_root or os.getcwd()
        self._cmd_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._result_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process: Optional[multiprocessing.Process] = None
        self._solver_module: str = ""
        self._solver_class: str = ""
        self._shared = SharedPhysicsBuffer()
        self._shared.create()
        self._soft_shared = SoftSharedBuffer()
        self._soft_shared.create()
        self._entity_to_slot: dict[str, int] = {}
        self._slot_free: list[int] = []
        self._entity_to_soft_slot: dict[str, int] = {}
        self._soft_slot_free: list[int] = []

    @property
    def shared(self) -> SharedPhysicsBuffer:
        return self._shared

    @property
    def soft_shared(self) -> SoftSharedBuffer:
        return self._soft_shared

    @property
    def entity_slot_map(self) -> dict[str, int]:
        return self._entity_to_slot

    @property
    def entity_soft_slot_map(self) -> dict[str, int]:
        return self._entity_to_soft_slot

    def clear_slots(self):
        self._entity_to_slot.clear()
        self._slot_free.clear()

    def clear_soft_slots(self):
        self._entity_to_soft_slot.clear()
        self._soft_slot_free.clear()

    def alloc_slot(self) -> int:
        if self._slot_free:
            return self._slot_free.pop()
        slot = len(self._entity_to_slot)
        if slot >= MAX_ENTITIES:
            raise RuntimeError(f"Max {MAX_ENTITIES} physics entities exceeded")
        return slot

    def free_slot(self, slot: int):
        self._slot_free.append(slot)
        self._shared.set_active(slot, False)

    def alloc_soft_slot(self) -> int:
        from core.physics.soft_shared_buffer import MAX_SOFT_BODIES
        if self._soft_slot_free:
            return self._soft_slot_free.pop()
        slot = len(self._entity_to_soft_slot)
        if slot >= MAX_SOFT_BODIES:
            raise RuntimeError(f"Max {MAX_SOFT_BODIES} soft bodies exceeded")
        return slot

    def free_soft_slot(self, slot: int):
        self._soft_slot_free.append(slot)
        try:
            self._soft_shared.clear_slot(slot)
        except Exception:
            pass

    def start(self, solver_module: str, solver_class: str, settings: dict) -> bool:
        self._solver_module = solver_module
        self._solver_class = solver_class
        self._shared.set_num_entities(0)
        from core.foundation.logger import Logger
        self._process = multiprocessing.Process(
            target=_physics_loop,
            args=(self._cmd_queue, self._result_queue,
                  self._shared.name, self._soft_shared.name,
                  self._project_root, solver_module, solver_class, settings),
            daemon=True,
        )
        self._process.start()
        result = self.wait_for_result("init", timeout=10.0)
        ok = result is not None and result.get("success", False)
        if not ok:
            Logger.warning(f"  PhysicsProcess.start FAILED, killing process")
            if self._process and self._process.is_alive():
                self._process.terminate()
                self._process.join(2)
            self._process = None
            self._shared.close()
            self._shared.unlink()
            self._soft_shared.close()
            self._soft_shared.unlink()
        return ok

    def send(self, cmd: dict):
        self._cmd_queue.put(cmd)

    def poll(self) -> Optional[dict]:
        try:
            return self._result_queue.get_nowait()
        except queue.Empty:
            return None

    def drain(self) -> list[dict]:
        results = []
        while True:
            r = self.poll()
            if r is None:
                break
            results.append(r)
        return results

    def wait_for_result(self, expected_type: str, timeout: float = 5.0) -> Optional[dict]:
        deadline = time.monotonic() + timeout
        while True:
            r = self.poll()
            if r and r.get("type") == expected_type:
                return r
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.001)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def shutdown(self, timeout: int = 5000):
        if self._process is None or not self._process.is_alive():
            self._process = None
            self._shared.close()
            self._shared.unlink()
            self._soft_shared.close()
            self._soft_shared.unlink()
            return
        try:
            self._cmd_queue.put({"type": "shutdown"})
            self._process.join(timeout / 1000)
        except Exception:
            pass
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join(2)
        self._process = None
        self._shared.close()
        self._shared.unlink()
        self._soft_shared.close()
        self._soft_shared.unlink()


def _physics_loop(
    cmd_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shared_name: str,
    soft_shared_name: str,
    project_root: str,
    solver_module: str,
    solver_class_name: str,
    settings: dict,
):
    import sys
    import os
    project_root = os.path.normpath(project_root)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    os.chdir(project_root)

    import importlib
    from core.foundation.logger import Logger

    shared = SharedPhysicsBuffer()
    shared.attach(shared_name)

    from core.physics.soft_shared_buffer import SoftSharedBuffer
    soft_shared = SoftSharedBuffer()
    soft_shared.attach(soft_shared_name)

    try:
        mod = importlib.import_module(solver_module)
        SolverCls = getattr(mod, solver_class_name)
    except Exception as e:
        Logger.error(f"PhysicsProcess: cannot import solver: {e}")
        result_queue.put({"type": "init", "success": False})
        shared.close()
        soft_shared.close()
        return

    solver = SolverCls()
    if not solver.initialize(settings):
        Logger.error("PhysicsProcess: solver init failed")
        result_queue.put({"type": "init", "success": False})
        shared.close()
        soft_shared.close()
        return

    from core.physics.physics_scene import PhysicsScene
    physics_scene = PhysicsScene(solver)
    _slot_to_body: dict[int, int] = {}
    _soft_states: dict[str, dict] = {}
    _pending_soft_created: list[dict] = []
    result_queue.put({"type": "init", "success": True})

    running = True
    while running:
        try:
            cmd = cmd_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break

        t = cmd.get("type")

        try:
            if t == "step":
                dt = float(cmd.get("dt", 0.02))
                while True:
                    try:
                        nxt = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if nxt.get("type") == "step":
                        dt += float(nxt.get("dt", 0.02))
                    else:
                        cmd_queue.put(nxt)
                        break
                cmd["dt"] = dt
                _process_step_shared(cmd, solver, physics_scene, result_queue, shared, _slot_to_body,
                                     soft_shared, _soft_states, _pending_soft_created)

            elif t == "load_bodies":
                _slot_to_body.clear()
                solver.remove_all_joints()
                solver.remove_all_bodies()
                physics_scene._entity_to_body.clear()
                physics_scene._body_to_entity.clear()
                if hasattr(physics_scene, "_entity_to_extra_bodies"):
                    physics_scene._entity_to_extra_bodies.clear()
                physics_scene._entity_to_joint.clear()
                physics_scene._joint_to_entity.clear()
                physics_scene._cached_shape.clear()
                physics_scene._cached_shape_info.clear()
                physics_scene._prev_frame_contacts.clear()
                for body in cmd.get("bodies", []):
                    _create_body(body, solver, physics_scene, shared, _slot_to_body)
                result_queue.put({"type": "load_bodies"})

            elif t == "add_body":
                _create_body(cmd["body"], solver, physics_scene, shared, _slot_to_body)

            elif t == "load_soft_bodies":
                _soft_states.clear()
                _pending_soft_created.clear()
                try:
                    solver.remove_all_soft_bodies()
                except Exception:
                    pass
                created = []
                for soft in cmd.get("softs", []):
                    created.append(_create_soft_worker(solver, soft, _soft_states))
                _pending_soft_created.extend(created)
                result_queue.put({"type": "soft_loaded", "soft_created": created})

            elif t == "add_soft_body":
                _pending_soft_created.append(_create_soft_worker(solver, cmd["soft"], _soft_states))

            elif t == "update_soft_body":
                soft = cmd["soft"]
                _drop_entity_soft(_soft_states, solver, soft.get("entity_id", ""))
                _pending_soft_created.append(_create_soft_worker(solver, soft, _soft_states))

            elif t == "remove_bodies":
                for eid in cmd.get("entity_ids", []):
                    _drop_entity_bodies(physics_scene, solver, eid)
                    _drop_entity_soft(_soft_states, solver, eid)
                for slot in cmd.get("slots", []):
                    _slot_to_body.pop(slot, None)
                    shared.set_active(slot, False)

            elif t == "unload_all":
                _slot_to_body.clear()
                _soft_states.clear()
                _pending_soft_created.clear()
                solver.remove_all_joints()
                solver.remove_all_bodies()
                try:
                    solver.remove_all_soft_bodies()
                except Exception:
                    pass
                physics_scene._entity_to_body.clear()
                physics_scene._body_to_entity.clear()
                if hasattr(physics_scene, "_entity_to_extra_bodies"):
                    physics_scene._entity_to_extra_bodies.clear()
                physics_scene._entity_to_joint.clear()
                physics_scene._joint_to_entity.clear()
                physics_scene._cached_shape.clear()
                physics_scene._cached_shape_info.clear()
                physics_scene._prev_frame_contacts.clear()
                shared.set_num_entities(0)
                result_queue.put({"type": "unload_all"})

            elif t == "shutdown":
                running = False
        except Exception as e:
            Logger.error(f"PhysicsProcess: error processing cmd '{t}': {e}")
            import traceback
            Logger.error(traceback.format_exc())
            result_queue.put({"type": t, "error": str(e)})

    solver.shutdown()
    shared.close()
    soft_shared.close()


def _create_body(body: dict, solver, physics_scene, shared, _slot_to_body: dict):
    shapes = body.get("shapes")
    if shapes is not None and len(shapes) > 1 and not body.get("is_2d", False):
        from core.physics.shape_utils import partition_compound_shapes
        dynamic = (body.get("mass", 0.0) > 0.0) and (not body.get("is_kinematic", False)) and (not shapes[0].get("is_trigger", False))
        if not dynamic:
            dynamic = bool(body.get("is_kinematic", False))
        prims, separate = partition_compound_shapes(shapes, dynamic)
        extras = list(separate)
        if not prims:
            prims = [shapes[0]]
            extras = [s for s in extras if s is not shapes[0]]
        if extras and not dynamic:
            p0 = prims[0]
            if len(prims) == 1:
                bid = solver.create_rigid_body(
                    entity_id=body["entity_id"],
                    shape_type=p0["type"],
                    shape_params=p0["params"],
                    position=body["position"],
                    rotation=body["rotation"],
                    mass=body["mass"],
                    friction=p0.get("friction", 0.6),
                    restitution=p0.get("restitution", 0.0),
                    is_trigger=p0.get("is_trigger", False),
                    is_kinematic=body.get("is_kinematic", False),
                    collision_layer=p0.get("layer", 0),
                    collision_mask=p0.get("mask", 0xFFFF),
                )
            else:
                bid = solver.create_compound_rigid_body(
                    entity_id=body["entity_id"],
                    shapes=prims,
                    position=body["position"],
                    rotation=body["rotation"],
                    mass=body["mass"],
                    friction=p0.get("friction", 0.6),
                    restitution=p0.get("restitution", 0.0),
                    is_trigger=p0.get("is_trigger", False),
                    is_kinematic=body.get("is_kinematic", False),
                    collision_layer=p0.get("layer", 0),
                    collision_mask=p0.get("mask", 0xFFFF),
                )
            if bid >= 0:
                _register_slot_body(body, bid, solver, physics_scene, shared, _slot_to_body)
                for extra in extras:
                    try:
                       ibid = solver.create_rigid_body(
                            entity_id=body["entity_id"],
                            shape_type=extra["type"],
                            shape_params=extra["params"],
                            position=body["position"],
                            rotation=body["rotation"],
                            mass=0.0,
                            friction=extra.get("friction", 0.6),
                            restitution=extra.get("restitution", 0.0),
                            is_trigger=extra.get("is_trigger", False),
                            is_kinematic=True,
                            collision_layer=extra.get("layer", 0),
                            collision_mask=extra.get("mask", 0xFFFF),
                        )
                    except Exception:
                        ibid = -1
                    if ibid is not None and ibid >= 0:
                        physics_scene._body_to_entity[ibid] = body["entity_id"]
                        physics_scene._entity_to_extra_bodies.setdefault(body["entity_id"], []).append(ibid)
            return
        p0 = prims[0]
        if p0.get("type") not in ("box", "sphere", "capsule", "cylinder"):
            bid = solver.create_rigid_body(
                entity_id=body["entity_id"],
                shape_type=p0["type"],
                shape_params=p0["params"],
                position=body["position"],
                rotation=body["rotation"],
                mass=body["mass"],
                friction=p0.get("friction", 0.6),
                restitution=p0.get("restitution", 0.0),
                is_trigger=p0.get("is_trigger", False),
                is_kinematic=body.get("is_kinematic", False),
                collision_layer=p0.get("layer", 0),
                collision_mask=p0.get("mask", 0xFFFF),
            )
            if bid >= 0:
                _register_slot_body(body, bid, solver, physics_scene, shared, _slot_to_body)
            return
        bid = solver.create_compound_rigid_body(
            entity_id=body["entity_id"],
            shapes=prims,
            position=body["position"],
            rotation=body["rotation"],
            mass=body["mass"],
            friction=p0.get("friction", 0.6),
            restitution=p0.get("restitution", 0.0),
            is_trigger=p0.get("is_trigger", False),
            is_kinematic=body.get("is_kinematic", False),
            collision_layer=p0.get("layer", 0),
            collision_mask=p0.get("mask", 0xFFFF),
        )
        if bid >= 0:
            _register_slot_body(body, bid, solver, physics_scene, shared, _slot_to_body)
        return
    bid = solver.create_rigid_body(
        entity_id=body["entity_id"],
        shape_type=body["shape_type"],
        shape_params=body["shape_params"],
        position=body["position"],
        rotation=body["rotation"],
        mass=body["mass"],
        friction=body.get("friction", 0.6),
        restitution=body.get("restitution", 0.0),
        is_trigger=body.get("is_trigger", False),
        is_kinematic=body.get("is_kinematic", False),
        collision_layer=body.get("collision_layer", 0),
        collision_mask=body.get("collision_mask", 0xFFFF),
    )
    if bid >= 0:
        _register_slot_body(body, bid, solver, physics_scene, shared, _slot_to_body)


def _register_slot_body(body: dict, bid: int, solver, physics_scene, shared, _slot_to_body: dict):
    slot = body.get("slot", -1)
    if slot >= 0:
        _slot_to_body[slot] = bid
        shared.set_body_id(slot, bid)
        shared.set_active(slot, True)
        shared.set_kinematic(slot, body.get("is_kinematic", False))
        shared.set_2d(slot, body.get("is_2d", False))
        shared.set_dirty(slot, True)
        if slot >= shared.get_num_entities():
            shared.set_num_entities(slot + 1)
    physics_scene._entity_to_body[body["entity_id"]] = bid
    physics_scene._body_to_entity[bid] = body["entity_id"]
    physics_scene._cached_shape[body["entity_id"]] = ()


def _drop_entity_bodies(physics_scene, solver, entity_id: str):
    bid = physics_scene._entity_to_body.pop(entity_id, None)
    if bid is not None:
        try:
            solver.remove_rigid_body(bid)
        except Exception:
            pass
        physics_scene._body_to_entity.pop(bid, None)
    extras = physics_scene._entity_to_extra_bodies.pop(entity_id, []) if hasattr(physics_scene, "_entity_to_extra_bodies") else []
    for ibid in extras:
        try:
            solver.remove_rigid_body(ibid)
        except Exception:
            pass
        physics_scene._body_to_entity.pop(ibid, None)


def _create_soft_worker(solver, soft: dict, soft_states: dict) -> dict:
    from core.physics.physics_scene import create_soft_body_in_solver
    import numpy as np
    entity_id = soft.get("entity_id", "")
    slot = int(soft.get("slot", -1))
    try:
        verts = np.frombuffer(soft["verts_b"], dtype=np.float32).reshape(-1, 3).copy()
        indices = np.frombuffer(soft["indices_b"], dtype=np.int64).reshape(-1).copy()
    except Exception as e:
        return {"entity_id": entity_id, "slot": slot, "ok": False, "error": f"bad mesh data: {e}"}
    params = {
        "mass": soft.get("mass", 1.0), "compliance": soft.get("compliance", 0.001),
        "bend_mode": soft.get("bend_mode", "distance"), "pressure": soft.get("pressure", 0.0),
        "damping": soft.get("damping", 0.1), "iterations": soft.get("iterations", 10),
        "gravity_factor": soft.get("gravity_factor", 1.0), "friction": soft.get("friction", 0.2),
        "restitution": soft.get("restitution", 0.0), "vertex_radius": soft.get("vertex_radius", 0.05),
        "max_velocity": soft.get("max_velocity", 500.0), "max_vertices": soft.get("max_vertices", 0),
        "pin_mode": soft.get("pin_mode", "none"), "pin_fraction": soft.get("pin_fraction", 0.1),
        "double_sided": soft.get("double_sided", True), "update_com": soft.get("update_com", True),
        "layer": soft.get("layer", 0), "mask": soft.get("mask", 0xFFFF),
    }
    try:
        sid, count, rest_verts, rest_faces = create_soft_body_in_solver(
            solver,
            vertices=verts,
            indices=indices,
            scale_xyz=soft.get("scale", (1.0, 1.0, 1.0)),
            pos_xyz=soft.get("position", (0.0, 0.0, 0.0)),
            euler_rad_xyz=soft.get("rotation", (0.0, 0.0, 0.0)),
            params=params,
            entity_id=entity_id,
        )
    except Exception as e:
        return {"entity_id": entity_id, "slot": slot, "ok": False, "error": str(e)}
    if sid is None or count <= 0 or rest_verts is None:
        try:
            if sid is not None:
                solver.remove_soft_body(sid)
        except Exception:
            pass
        return {"entity_id": entity_id, "slot": slot, "ok": False, "error": "solver rejected soft body"}
    soft_states[entity_id] = {"sid": sid, "slot": slot}
    try:
        stored_mass = max(float(params.get("mass", 1.0)), 1e-6)
    except Exception:
        stored_mass = 1.0
    soft_states[entity_id]["mass"] = stored_mass
    try:
        rad = float(params.get("vertex_radius", 0.05))
    except Exception:
        rad = 0.05
    try:
        spec_rad = solver.get_soft_body_radius(sid)
        if spec_rad and spec_rad > 0.0:
            rad = float(spec_rad)
    except Exception:
        pass
    try:
        rv = np.ascontiguousarray(rest_verts, dtype=np.float32)
        rf = np.ascontiguousarray(rest_faces, dtype=np.int32)
        item = {"entity_id": entity_id, "slot": slot, "ok": True, "count": int(count),
                "verts_b": rv.tobytes(), "faces_b": rf.tobytes(), "radius": float(rad)}
    except Exception as e:
        return {"entity_id": entity_id, "slot": slot, "ok": False, "error": f"readback failed: {e}"}
    return item


def _drop_entity_soft(soft_states: dict, solver, entity_id: str):
    st = soft_states.pop(entity_id, None)
    if st is not None:
        try:
            solver.remove_soft_body(st["sid"])
        except Exception:
            pass


def _write_soft_results(soft_states: dict, solver, soft_shared):
    for st in soft_states.values():
        try:
            verts = solver.get_soft_body_world_vertices(st["sid"])
        except Exception:
            verts = None
        try:
            com = solver.get_soft_body_com(st["sid"])
            pos, quat = com
        except Exception:
            pos, quat = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)
        try:
            vel = solver.get_soft_body_velocity(st["sid"])
        except Exception:
            vel = (0.0, 0.0, 0.0)
        try:
            soft_shared.write_soft(st["slot"], verts, pos, quat, vel)
        except Exception:
            pass


def _process_step_shared(cmd, solver, physics_scene, result_queue, shared, _slot_to_body,
                           soft_shared=None, soft_states=None, pending_soft_created=None):
    dt = cmd["dt"]
    num = shared.get_num_entities()
    result_ver = shared.get_result_version()
    flags_arr = shared._flags_nd
    edata = shared._edata_nd
    fdata = shared._fdata_nd
    rdata = shared._rdata_nd
    for slot in range(num):
        fl = int(flags_arr[slot])
        if not (fl & 1):
            continue
        bid = _slot_to_body.get(slot, -1)
        if bid < 0:
            continue
        if fl & 4:
            pos = (float(edata[slot, 0]), float(edata[slot, 1]), float(edata[slot, 2]))
            quat = (float(edata[slot, 3]), float(edata[slot, 4]), float(edata[slot, 5]), float(edata[slot, 6]))
            try:
                solver.set_body_transform_quat(bid, pos, quat)
            except Exception:
                solver.set_body_transform(bid, pos, (0,0,0))
        else:
            if fl & 2:
                try:
                    solver.activate(bid)
                except Exception:
                    pass
                vel = (float(edata[slot, 7]), float(edata[slot, 8]), float(edata[slot, 9]))
                ang_vel = (float(edata[slot, 10]), float(edata[slot, 11]), float(edata[slot, 12]))
                solver.set_velocities(bid, linear=vel, angular=ang_vel)
                flags_arr[slot] = fl & 0xFD
            fx = float(fdata[slot, 0]); fy = float(fdata[slot, 1]); fz = float(fdata[slot, 2])
            if fx or fy or fz:
                solver.apply_force(bid, (fx, fy, fz))
            tx = float(fdata[slot, 3]); ty = float(fdata[slot, 4]); tz = float(fdata[slot, 5])
            if tx or ty or tz:
                solver.apply_torque(bid, (tx, ty, tz))

    if soft_states and cmd.get("soft_velocities"):
        try:
            fdt = max(float(cmd.get("dt", 1.0 / 60.0)), 1e-6)
        except Exception:
            fdt = 1.0 / 60.0
        for eid, vel in cmd["soft_velocities"].items():
            try:
                st = soft_states.get(eid)
                if st is None:
                    continue
                cur = solver.get_soft_body_velocity(st["sid"])
                m = max(float(st.get("mass", 1.0)), 1e-6)
                solver.apply_soft_body_force(st["sid"], (
                    (float(vel[0]) - float(cur[0])) * m / fdt,
                    (float(vel[1]) - float(cur[1])) * m / fdt,
                    (float(vel[2]) - float(cur[2])) * m / fdt,
                ))
            except Exception:
                pass

    solver.step_simulation(dt)

    for slot in range(num):
        fl = int(flags_arr[slot])
        if not (fl & 1):
            continue
        bid = _slot_to_body.get(slot, -1)
        if bid < 0:
            continue
        if fl & 4:
            continue
        if fl & 8:
            vel, ang_vel = solver.get_velocities(bid)
            if vel[0] or vel[1] or vel[2]:
                solver.set_velocities(bid, linear=(vel[0], vel[1], 0.0))
            if ang_vel[0] or ang_vel[1] or ang_vel[2]:
                solver.set_velocities(bid, angular=(0.0, 0.0, ang_vel[2]))
        try:
            pos, quat = solver.get_body_transform_quat(bid)
        except Exception:
            pos, rot = solver.get_body_transform(bid)
            import math
            rx, ry, rz = rot
            hx, hy, hz = rx*0.5, ry*0.5, rz*0.5
            sx, cx = math.sin(hx), math.cos(hx)
            sy, cy = math.sin(hy), math.cos(hy)
            sz, cz = math.sin(hz), math.cos(hz)
            quat = (sx*cy*cz - cx*sy*sz, cx*sy*cz + sx*cy*sz, cx*cy*sz - sx*sy*cz, cx*cy*cz + sx*sy*sz)
        vel, ang_vel = solver.get_velocities(bid)
        rdata[slot, 0] = pos[0]; rdata[slot, 1] = pos[1]; rdata[slot, 2] = pos[2]
        rdata[slot, 3] = quat[0]; rdata[slot, 4] = quat[1]; rdata[slot, 5] = quat[2]; rdata[slot, 6] = quat[3]
        rdata[slot, 7] = vel[0]; rdata[slot, 8] = vel[1]; rdata[slot, 9] = vel[2]
        rdata[slot, 10] = ang_vel[0]; rdata[slot, 11] = ang_vel[1]; rdata[slot, 12] = ang_vel[2]

    shared.set_result_version(result_ver + 1)

    if soft_states is not None and soft_shared is not None and len(soft_states) > 0:
        try:
            _write_soft_results(soft_states, solver, soft_shared)
        except Exception:
            pass

    need_coll = bool(cmd.get("need_collisions", True))
    if need_coll:
        raw_events = solver.get_collision_events() if hasattr(solver, 'get_collision_events') else []
        events = []
        for ev in raw_events:
            ba, bb = ev.get("body_a", -1), ev.get("body_b", -1)
            events.append({
                "body_a": ba,
                "body_b": bb,
                "entity_a": physics_scene._body_to_entity.get(ba, ""),
                "entity_b": physics_scene._body_to_entity.get(bb, ""),
                "position": ev.get("position", (0, 0, 0)),
                "normal": ev.get("normal", (0, 0, 0)),
                "distance": ev.get("distance", 0.0),
                "force": ev.get("force", 0.0),
            })
    else:
        events = []

    result_queue.put({"type": "step_result", "collision_events": events, "version": result_ver + 1,
                      "soft_created": list(pending_soft_created) if pending_soft_created else []})
    if pending_soft_created:
        try:
            pending_soft_created.clear()
        except Exception:
            pass
