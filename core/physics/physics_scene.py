# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Optional, TYPE_CHECKING
from core.logger import Logger
from core.physics.shape_utils import find_shape_info, make_shape_key

if TYPE_CHECKING:
    from core.ecs import Entity, Scene
    from core.physics.physics_solver import IPhysicsSolver


class PhysicsScene:
    _ZERO_VEC3 = None
    _ZERO_VEC2 = None

    def __init__(self, solver: IPhysicsSolver):
        self._solver = solver
        self._entity_to_body: dict[str, int] = {}
        self._body_to_entity: dict[int, str] = {}
        self._entity_body_cache: dict[str, tuple] = {}
        self._entity_to_joint: dict[str, int] = {}
        self._joint_to_entity: dict[int, str] = {}
        self._cached_shape: dict[str, tuple] = {}
        self._cached_shape_info: dict[str, dict] = {}
        self._prev_frame_contacts: set[frozenset[int]] = set()
        self._scene: Optional[Scene] = None
        self._2d_bodies: set[int] = set()
        self._shape_check_counter: int = 0
        self._body_items: list[tuple[str, int]] = []
        self._body_items_dirty: bool = False
        self._has_collision_scripts: bool = False
        self._collision_scripts_checked: bool = False


    @property
    def solver(self) -> IPhysicsSolver:
        return self._solver

    def initialize(self, scene: Scene):
        self._scene = scene
        Logger.info("PhysicsScene initialized.")

    def shutdown(self):
        self._solver.remove_all_joints()
        self._solver.remove_all_bodies()
        self._entity_to_body.clear()
        self._body_to_entity.clear()
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
        self._entity_to_body.clear()
        self._body_to_entity.clear()
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
            self._create_entity_joints(entity)

    def _create_entity_bodies(self, entity: Entity):
        from core.components import Rigidbody, Rigidbody2D

        rb = entity.get_component(Rigidbody)
        rb2d = entity.get_component(Rigidbody2D)
        tr = entity.transform
        if (not rb and not rb2d) or not tr:
            return

        is_2d = rb2d is not None
        effective_rb = rb2d if is_2d else rb

        shape_info = find_shape_info(entity, tr)
        if not shape_info:
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
        if body_id >= 0:
            self._entity_to_body[entity.id] = body_id
            self._body_to_entity[body_id] = entity.id
            self._entity_body_cache[entity.id] = (entity, effective_rb, tr, is_2d)
            effective_rb._body_id = body_id
            self._mark_body_items_dirty()
            if is_2d:
                self._2d_bodies.add(body_id)
            key = self._make_shape_key(entity, shape_info)
            self._cached_shape[entity.id] = key
            self._cached_shape_info[entity.id] = shape_info

    def _find_shape(self, entity: Entity, transform=None) -> Optional[dict]:
        return find_shape_info(entity, transform)

    def _make_shape_key(self, entity: Entity, shape_info: dict) -> tuple:
        return make_shape_key(entity, shape_info)

    def remove_entity_bodies(self, entity_id: str):
        body_id = self._entity_to_body.pop(entity_id, None)
        if body_id is not None:
            self._solver.remove_rigid_body(body_id)
            self._body_to_entity.pop(body_id, None)
            self._2d_bodies.discard(body_id)
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

        if prof: prof.start("phys_collision_events")
        self._process_collision_events()
        if prof: prof.stop("phys_collision_events")

    def _register_new_entities(self):

        if not self._scene or len(self._scene._entities) == len(self._entity_to_body):
            return
        self._has_collision_scripts = False
        self._collision_scripts_checked = False
        for entity in self._scene.get_all_entities():
            if entity.id not in self._entity_to_body:
                self._create_entity_bodies(entity)

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
        return False

    def _process_collision_events(self):
        from core.components import ScriptComponent
        if not self._has_collision_listeners():
            self._prev_frame_contacts.clear()
            return
        raw = self._solver.get_collision_events()
        current: set[frozenset[int]] = set()
        for ev in raw:
            ba, bb = ev["body_a"], ev["body_b"]
            if ba < 0 or bb < 0:
                continue
            current.add(frozenset([ba, bb]))

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

        _dispatch(entered, 'on_collision_enter')
        _dispatch(exited, 'on_collision_exit')
        _dispatch(stayed, 'on_collision_stay')

        self._prev_frame_contacts = current

    def _check_shape_changes(self):
        for entity_id in list(self._entity_to_body.keys()):
            entity = self._get_entity(entity_id)
            if not entity:
                continue
            shape_info = self._find_shape(entity, entity.transform)
            if shape_info is None:
                continue
            current_key = self._make_shape_key(entity, shape_info)
            cached = self._cached_shape.get(entity_id)
            if cached is not None and current_key != cached:
                self.rebuild_entity(entity)

    def _mark_body_items_dirty(self):
        self._body_items_dirty = True

    def _get_body_items(self):
        if self._body_items_dirty or not self._body_items:
            self._body_items = list(self._entity_to_body.items())
            self._body_items_dirty = False
        return self._body_items

    def _sync_ecs_to_physics(self):
        from core.math_helpers import quat_to_euler_rad
        cache = self._entity_body_cache
        for entity_id, body_id in self._entity_to_body.items():
            cached = cache.get(entity_id)
            if not cached:
                continue
            entity, rb, tr, is_2d = cached
            if not entity._active:
                continue
            if rb.is_kinematic:
                p = tr._local_pos
                if is_2d:
                    self._solver.set_body_transform(body_id,
                        (p._x, p._y, 0.0),
                        (0.0, 0.0, quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y,
                                                      tr._local_rot._z, tr._local_rot._w)[2]))
                else:
                    e = quat_to_euler_rad(tr._local_rot._x, tr._local_rot._y,
                                          tr._local_rot._z, tr._local_rot._w)
                    self._solver.set_body_transform(body_id,
                        (p._x, p._y, p._z), e)
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
        from core.math_helpers import quat_from_euler_rad
        cache = self._entity_body_cache
        for entity_id, body_id in self._entity_to_body.items():
            cached = cache.get(entity_id)
            if not cached:
                continue
            entity, rb, tr, is_2d = cached
            if not entity._active or rb.is_kinematic:
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
