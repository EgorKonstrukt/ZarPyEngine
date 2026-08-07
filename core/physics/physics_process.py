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
        self._entity_to_slot: dict[str, int] = {}
        self._slot_free: list[int] = []

    @property
    def shared(self) -> SharedPhysicsBuffer:
        return self._shared

    @property
    def entity_slot_map(self) -> dict[str, int]:
        return self._entity_to_slot

    def clear_slots(self):
        self._entity_to_slot.clear()
        self._slot_free.clear()

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

    def start(self, solver_module: str, solver_class: str, settings: dict) -> bool:
        self._solver_module = solver_module
        self._solver_class = solver_class
        self._shared.set_num_entities(0)
        from core.foundation.logger import Logger
        self._process = multiprocessing.Process(
            target=_physics_loop,
            args=(self._cmd_queue, self._result_queue,
                  self._shared.name,
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
            time.sleep(0.0005)

    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def shutdown(self, timeout: int = 5000):
        if self._process is None or not self._process.is_alive():
            self._process = None
            self._shared.close()
            self._shared.unlink()
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


def _physics_loop(
    cmd_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    shared_name: str,
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

    try:
        mod = importlib.import_module(solver_module)
        SolverCls = getattr(mod, solver_class_name)
    except Exception as e:
        Logger.error(f"PhysicsProcess: cannot import solver: {e}")
        result_queue.put({"type": "init", "success": False})
        shared.close()
        return

    solver = SolverCls()
    if not solver.initialize(settings):
        Logger.error("PhysicsProcess: solver init failed")
        result_queue.put({"type": "init", "success": False})
        shared.close()
        return

    from core.physics.physics_scene import PhysicsScene
    physics_scene = PhysicsScene(solver)
    _slot_to_body: dict[int, int] = {}
    result_queue.put({"type": "init", "success": True})

    running = True
    while running:
        try:
            cmd = cmd_queue.get(timeout=0.004)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break

        t = cmd.get("type")

        try:
            if t == "step":
                _process_step_shared(cmd, solver, physics_scene, result_queue, shared, _slot_to_body)

            elif t == "load_bodies":
                _slot_to_body.clear()
                solver.remove_all_joints()
                solver.remove_all_bodies()
                physics_scene._entity_to_body.clear()
                physics_scene._body_to_entity.clear()
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

            elif t == "remove_bodies":
                for eid in cmd.get("entity_ids", []):
                    bid = physics_scene._entity_to_body.pop(eid, None)
                    if bid is not None:
                        solver.remove_rigid_body(bid)
                        physics_scene._body_to_entity.pop(bid, None)
                for slot in cmd.get("slots", []):
                    _slot_to_body.pop(slot, None)
                    shared.set_active(slot, False)

            elif t == "unload_all":
                _slot_to_body.clear()
                solver.remove_all_joints()
                solver.remove_all_bodies()
                physics_scene._entity_to_body.clear()
                physics_scene._body_to_entity.clear()
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


def _create_body(body: dict, solver, physics_scene, shared, _slot_to_body: dict):
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


def _process_step_shared(cmd, solver, physics_scene, result_queue, shared, _slot_to_body):
    dt = cmd["dt"]
    num = shared.get_num_entities()
    result_ver = shared.get_result_version()

    for slot in range(num):
        flags = shared.get_flags(slot)
        if not (flags & 1):
            continue
        bid = _slot_to_body.get(slot, -1)
        if bid < 0:
            continue

        is_kinematic = bool(flags & 4)
        if is_kinematic:
            pos, rot, _, _ = shared.read_entity_data(slot)
            solver.set_body_transform(bid, pos, rot)
        else:
            if flags & 2:
                try:
                    solver.activate(bid)
                except Exception:
                    pass
                _, _, vel, ang_vel = shared.read_entity_data(slot)
                solver.set_velocities(bid, linear=vel, angular=ang_vel)
                shared.set_dirty(slot, False)
            force, torque = shared.read_force_data(slot)
            if force[0] or force[1] or force[2]:
                solver.apply_force(bid, force)
            if torque[0] or torque[1] or torque[2]:
                solver.apply_torque(bid, torque)

    solver.step_simulation(dt)

    for slot in range(num):
        flags = shared.get_flags(slot)
        if not (flags & 1):
            continue
        bid = _slot_to_body.get(slot, -1)
        if bid < 0:
            continue
        if flags & 4:
            continue

        is_2d = bool(flags & 8)
        if is_2d:
            vel, ang_vel = solver.get_velocities(bid)
            if vel[0] or vel[1] or vel[2]:
                solver.set_velocities(bid, linear=(vel[0], vel[1], 0.0))
            if ang_vel[0] or ang_vel[1] or ang_vel[2]:
                solver.set_velocities(bid, angular=(0.0, 0.0, ang_vel[2]))

        pos, rot = solver.get_body_transform(bid)
        vel, ang_vel = solver.get_velocities(bid)
        shared.write_result(slot, pos, rot, vel, ang_vel)

    shared.set_result_version(result_ver + 1)

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

    result_queue.put({"type": "step_result", "collision_events": events, "version": result_ver + 1})
