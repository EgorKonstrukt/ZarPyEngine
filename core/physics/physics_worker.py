# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import threading
from queue import Queue, Empty
from typing import Optional, Callable

from core.logger import Logger


class PhysicsWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._cmd_queue: Queue = Queue()
        self._result_queue: Queue = Queue()
        self._running = False
        self._initialized = False
        self._solver = None
        self._physics_scene = None
        self._on_step_finished: Optional[Callable] = None

    def run(self):
        self._running = True
        while self._running:
            try:
                cmd = self._cmd_queue.get(timeout=0.004)
            except Empty:
                continue
            try:
                self._process(cmd)
            except Exception as e:
                Logger.error(f"PhysicsWorker cmd error: {e}")

    def _process(self, cmd: dict):
        t = cmd["type"]

        if t == "init":
            if self._initialized:
                self._result_queue.put({"type": "init"})
                return
            solver_class = cmd["solver_class"]
            settings = cmd.get("settings", {})
            solver = solver_class()
            if solver.initialize(settings):
                from core.physics.physics_scene import PhysicsScene
                self._solver = solver
                self._physics_scene = PhysicsScene(solver)
                self._initialized = True
            self._result_queue.put({"type": "init", "success": self._initialized})

        elif t == "load_bodies":
            self._solver.remove_all_joints()
            self._solver.remove_all_bodies()
            self._physics_scene._entity_to_body.clear()
            self._physics_scene._body_to_entity.clear()
            self._physics_scene._entity_to_joint.clear()
            self._physics_scene._joint_to_entity.clear()
            self._physics_scene._cached_shape.clear()
            self._physics_scene._cached_shape_info.clear()
            self._physics_scene._prev_frame_contacts.clear()
            for body in cmd["bodies"]:
                bid = self._solver.create_rigid_body(
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
                    vel = body.get("velocity")
                    if vel:
                        self._solver.set_velocity(bid, vel)
                    ang_vel = body.get("angular_velocity")
                    if ang_vel:
                        self._solver.set_angular_velocity(bid, ang_vel)
                    self._physics_scene._entity_to_body[body["entity_id"]] = bid
                    self._physics_scene._body_to_entity[bid] = body["entity_id"]
                    self._physics_scene._cached_shape[body["entity_id"]] = ()
            self._result_queue.put({"type": "load_bodies"})

        elif t == "step":
            ecs_data = cmd.get("ecs_data", {})
            dt = cmd["dt"]
            ps = self._physics_scene
            solver = self._solver
            if ps is None or solver is None:
                return

            for eid, bid in ps._entity_to_body.items():
                data = ecs_data.get(eid)
                if data is None:
                    continue
                if data.get("is_kinematic", False):
                    pos = data.get("pos", (0, 0, 0))
                    rot = data.get("rot", (0, 0, 0))
                    solver.set_body_transform(bid, pos, rot)
                else:
                    vel = data.get("vel")
                    if vel is not None:
                        solver.set_velocity(bid, vel)
                    ang_vel = data.get("ang_vel")
                    if ang_vel is not None:
                        solver.set_angular_velocity(bid, ang_vel)
                force = data.get("force", (0, 0, 0))
                fx, fy, fz = force
                if fx or fy or fz:
                    solver.apply_force(bid, force)
                torque = data.get("torque", (0, 0, 0))
                tx, ty, tz = torque
                if tx or ty or tz:
                    solver.apply_torque(bid, torque)

            solver.step_simulation(dt)

            transforms = {}
            entity_to_body = ps._entity_to_body
            for eid, bid in entity_to_body.items():
                data = ecs_data.get(eid)
                if data and data.get("is_kinematic", False):
                    continue
                if data and data.get("is_2d", False):
                    vel = solver.get_velocity(bid)
                    vx, vy, vz = vel
                    if vx or vy or vz:
                        solver.set_velocity(bid, (vx, vy, 0.0))
                    ang_vel = solver.get_angular_velocity(bid)
                    ax, ay, az = ang_vel
                    if ax or ay or az:
                        solver.set_angular_velocity(bid, (0.0, 0.0, az))
                pos, rot = solver.get_body_transform(bid)
                vel = solver.get_velocity(bid)
                ang_vel = solver.get_angular_velocity(bid)
                transforms[eid] = {
                    "pos": pos,
                    "rot": rot,
                    "vel": vel,
                    "ang_vel": ang_vel,
                }

            if cmd.get("need_collisions", True):
                raw_events = solver.get_collision_events() if hasattr(solver, 'get_collision_events') else []
            else:
                raw_events = []

            body_to_entity = ps._body_to_entity
            events = [None] * len(raw_events)
            for i, ev in enumerate(raw_events):
                ba = ev.get("body_a", -1)
                bb = ev.get("body_b", -1)
                events[i] = {
                    "body_a": ba,
                    "body_b": bb,
                    "entity_a": body_to_entity.get(ba, ""),
                    "entity_b": body_to_entity.get(bb, ""),
                    "position": ev.get("position", (0, 0, 0)),
                    "normal": ev.get("normal", (0, 0, 0)),
                    "distance": ev.get("distance", 0.0),
                    "force": ev.get("force", 0.0),
                }

            result = {
                "type": "step_result",
                "transforms": transforms,
                "collision_events": events,
            }

            self._result_queue.put(result)

        elif t == "remove_bodies":
            for eid in cmd["entity_ids"]:
                bid = self._physics_scene._entity_to_body.pop(eid, None)
                if bid is not None:
                    self._solver.remove_rigid_body(bid)
                    self._physics_scene._body_to_entity.pop(bid, None)

        elif t == "add_body":
            body = cmd["body"]
            bid = self._solver.create_rigid_body(
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
                vel = body.get("velocity")
                if vel:
                    self._solver.set_velocity(bid, vel)
                ang_vel = body.get("angular_velocity")
                if ang_vel:
                    self._solver.set_angular_velocity(bid, ang_vel)
                self._physics_scene._entity_to_body[body["entity_id"]] = bid
                self._physics_scene._body_to_entity[bid] = body["entity_id"]
                self._physics_scene._cached_shape[body["entity_id"]] = ()

        elif t == "unload_all":
            if self._solver:
                self._solver.remove_all_joints()
                self._solver.remove_all_bodies()
            if self._physics_scene:
                self._physics_scene._entity_to_body.clear()
                self._physics_scene._body_to_entity.clear()
                self._physics_scene._entity_to_joint.clear()
                self._physics_scene._joint_to_entity.clear()
                self._physics_scene._cached_shape.clear()
                self._physics_scene._cached_shape_info.clear()
                self._physics_scene._prev_frame_contacts.clear()

        elif t == "shutdown":
            if self._physics_scene:
                self._physics_scene.shutdown()
                self._physics_scene = None
            if self._solver:
                self._solver.shutdown()
                self._solver = None
            self._running = False

    def send(self, cmd: dict):
        self._cmd_queue.put(cmd)

    def poll(self) -> Optional[dict]:
        try:
            return self._result_queue.get_nowait()
        except Empty:
            return None

    def drain_results(self) -> list[dict]:
        results = []
        while True:
            r = self.poll()
            if r is None:
                break
            results.append(r)
        return results

    def wait_for_result(self, expected_type: str, timeout: float = 5.0) -> Optional[dict]:
        import time
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            r = self.poll()
            if r and r.get("type") == expected_type:
                return r
        return None

    def shutdown(self, timeout: float = 3.0):
        self.send({"type": "shutdown"})
        self.join(timeout=timeout)
