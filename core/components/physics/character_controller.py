# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3, Quat
from core.components.inspector_meta import FieldType, InspectorField
from core.input.input_system import Input, KeyCode

try:
    import culverin
    from culverin import Character as CulverinCharacter
    _HAS_CULVERIN = True
except Exception:
    culverin = None
    CulverinCharacter = None
    _HAS_CULVERIN = False


def _quat_yaw(yaw_deg: float) -> Quat:
    a = math.radians(yaw_deg) * 0.5
    return Quat(0.0, math.sin(a), 0.0, math.cos(a))


def _quat_pitch_yaw(pitch_deg: float, yaw_deg: float) -> Quat:
    px = math.radians(pitch_deg) * 0.5
    py = math.radians(yaw_deg) * 0.5
    sp, cp = math.sin(px), math.cos(px)
    sy, cy = math.sin(py), math.cos(py)
    return Quat(sp * sy, cp * sy, -sp * cy, cp * cy)


def _get_physics_plugin():
    from core.engine.engine import Engine
    engine = Engine.instance()
    if engine is None:
        return None
    try:
        return engine.plugin_manager.get("PhysicsPlugin")
    except Exception:
        return None


def _resolve_solver():
    plugin = _get_physics_plugin()
    if plugin is None:
        return None
    solver = getattr(plugin, "_solver", None)
    if solver is not None and getattr(solver, "_world", None) is not None:
        return solver
    return None


@ComponentRegistry.register
class CharacterController(Component):
    _icon = "CharacterController.png"
    _gizmo_icon_color = (100, 180, 255)
    _gizmo_icon_label = "CC"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("", "Movement", FieldType.HEADER),
            InspectorField("walk_speed", "Walk Speed", FieldType.FLOAT, min_val=0.0, max_val=2000.0),
            InspectorField("run_speed", "Run Speed", FieldType.FLOAT, min_val=0.0, max_val=2000.0),
            InspectorField("crouch_speed", "Crouch Speed", FieldType.FLOAT, min_val=0.0, max_val=2000.0),
            InspectorField("acceleration", "Acceleration", FieldType.FLOAT, min_val=0.0, max_val=200.0),
            InspectorField("air_acceleration", "Air Acceleration", FieldType.FLOAT, min_val=0.0, max_val=200.0),
            InspectorField("friction", "Friction", FieldType.FLOAT, min_val=0.0, max_val=20.0),
            InspectorField("stop_speed", "Stop Speed", FieldType.FLOAT, min_val=0.0, max_val=500.0),
            InspectorField("", "Jump", FieldType.HEADER),
            InspectorField("jump_power", "Jump Power", FieldType.FLOAT, min_val=0.0, max_val=2000.0),
            InspectorField("jump_buffer_time", "Jump Buffer", FieldType.FLOAT, min_val=0.0, max_val=1.0),
            InspectorField("coyote_time", "Coyote Time", FieldType.FLOAT, min_val=0.0, max_val=1.0),
            InspectorField("", "Crouch", FieldType.HEADER),
            InspectorField("crouch_toggle", "Crouch Toggle", FieldType.BOOL),
            InspectorField("crouch_speed_mult", "Crouch Speed Mult", FieldType.FLOAT, min_val=0.0, max_val=1.0),
            InspectorField("crouch_eye_offset", "Crouch Eye Offset", FieldType.FLOAT, min_val=0.0, max_val=2.0),
            InspectorField("", "Mouse Look", FieldType.HEADER),
            InspectorField("camera_entity_id", "Camera", FieldType.GAMEOBJECT),
            InspectorField("sensitivity", "Sensitivity", FieldType.FLOAT, min_val=0.0, max_val=50.0),
            InspectorField("sensitivity_x", "Sensitivity X", FieldType.FLOAT, min_val=0.0, max_val=50.0),
            InspectorField("sensitivity_y", "Sensitivity Y", FieldType.FLOAT, min_val=0.0, max_val=50.0),
            InspectorField("invert_x", "Invert X", FieldType.BOOL),
            InspectorField("invert_y", "Invert Y", FieldType.BOOL),
            InspectorField("mouse_smoothing", "Mouse Smoothing", FieldType.FLOAT, min_val=0.0, max_val=1.0),
            InspectorField("smoothing_strength", "Smoothing Strength", FieldType.FLOAT, min_val=0.0, max_val=60.0),
            InspectorField("cursor_lock_on_start", "Lock Cursor On Start", FieldType.BOOL),
            InspectorField("unlock_key", "Unlock Key", FieldType.KEYBINDING),
            InspectorField("", "Camera", FieldType.HEADER),
            InspectorField("camera_fov", "Field of View", FieldType.FLOAT, min_val=1.0, max_val=179.0),
            InspectorField("camera_near", "Near Plane", FieldType.FLOAT, min_val=0.001, max_val=100.0),
            InspectorField("camera_far", "Far Plane", FieldType.FLOAT, min_val=0.1, max_val=100000.0),
            InspectorField("scroll_zoom", "Scroll Zoom", FieldType.BOOL),
            InspectorField("min_fov", "Min FOV (Zoom)", FieldType.FLOAT, min_val=1.0, max_val=179.0),
            InspectorField("max_fov", "Max FOV", FieldType.FLOAT, min_val=1.0, max_val=179.0),
            InspectorField("aim_fov", "Aim FOV", FieldType.FLOAT, min_val=1.0, max_val=179.0),
            InspectorField("aim_zoom_speed", "Aim Zoom Speed", FieldType.FLOAT, min_val=0.0, max_val=30.0),
            InspectorField("", "Physics", FieldType.HEADER),
            InspectorField("gravity", "Gravity", FieldType.FLOAT, min_val=0.0, max_val=5000.0),
            InspectorField("capsule_radius", "Radius", FieldType.FLOAT, min_val=0.01, max_val=10.0),
            InspectorField("capsule_height", "Height", FieldType.FLOAT, min_val=0.01, max_val=20.0),
            InspectorField("crouch_height", "Crouch Height", FieldType.FLOAT, min_val=0.01, max_val=20.0),
            InspectorField("step_height", "Step Height", FieldType.FLOAT, min_val=0.0, max_val=2.0),
            InspectorField("max_slope", "Max Slope", FieldType.FLOAT, min_val=0.0, max_val=90.0),
            InspectorField("push_strength", "Push Strength", FieldType.FLOAT, min_val=0.0, max_val=200.0),
        ]

    def __init__(self):
        super().__init__()
        self.walk_speed: float = 260.0
        self.run_speed: float = 440.0
        self.crouch_speed: float = 130.0
        self.acceleration: float = 10.0
        self.air_acceleration: float = 10.0
        self.friction: float = 4.0
        self.stop_speed: float = 80.0

        self.jump_power: float = 300.0
        self.jump_buffer_time: float = 0.1
        self.coyote_time: float = 0.1

        self.crouch_toggle: bool = True
        self.crouch_speed_mult: float = 0.5
        self.crouch_eye_offset: float = 0.6

        self.sensitivity: float = 5.0
        self.sensitivity_x: float = 5.0
        self.sensitivity_y: float = 5.0
        self.invert_x: bool = False
        self.invert_y: bool = False
        self.mouse_smoothing: float = 0.0
        self.smoothing_strength: float = 18.0
        self.cursor_lock_on_start: bool = True
        self.unlock_key: int = KeyCode.ESCAPE

        self.camera_fov: float = 60.0
        self.camera_near: float = 0.05
        self.camera_far: float = 1000.0
        self.scroll_zoom: bool = True
        self.min_fov: float = 20.0
        self.max_fov: float = 90.0
        self.aim_fov: float = 45.0
        self.aim_zoom_speed: float = 12.0

        self.gravity: float = 800.0
        self.capsule_radius: float = 0.5
        self.capsule_height: float = 2.0
        self.crouch_height: float = 1.2
        self.step_height: float = 0.3
        self.max_slope: float = 45.0
        self.push_strength: float = 30.0

        self._solver = None
        self._character: Optional["CulverinCharacter"] = None
        self._velocity: Vec3 = Vec3.zero()
        self._pitch: float = 0.0
        self._yaw: float = 0.0
        self._target_pitch: float = 0.0
        self._target_yaw: float = 0.0
        self._is_crouching: bool = False
        self._wants_to_crouch: bool = False
        self._grounded: bool = False
        self._coyote_timer: float = 0.0
        self._jump_buffer_timer: float = 0.0
        self._eye_height: float = 1.7
        self._target_eye_height: float = 1.7
        self._camera_entity_id: Optional[str] = None
        self._current_fov: float = 60.0
        self._warned_no_world: bool = False
        self._cursor_locked: bool = False

    @property
    def velocity(self) -> Vec3:
        return self._velocity

    @property
    def is_grounded(self) -> bool:
        return self._grounded

    @property
    def is_crouching(self) -> bool:
        return self._is_crouching

    def _capsule_total_height(self) -> float:
        return max(self.capsule_radius * 2.0 + 0.1, self.capsule_height)

    def _create_character(self, solver, pos: Vec3):
        total = self._capsule_total_height()
        char = solver.create_character(
            (pos.x, pos.y, pos.z),
            height=total,
            radius=self.capsule_radius,
            step_height=self.step_height,
            max_slope=self.max_slope,
        )
        if char is not None:
            solver.set_character_rotation(char, _quat_yaw(self._yaw).to_list())
            solver.set_character_strength(char, self.push_strength)
        return char

    def get_move_speed(self) -> float:
        if self._is_crouching:
            return self.crouch_speed * self.crouch_speed_mult
        if Input.GetKey(KeyCode.LEFT_SHIFT) or Input.GetKey(KeyCode.RIGHT_SHIFT):
            return self.run_speed
        return self.walk_speed

    def get_wish_dir(self) -> Vec3:
        fwd = self.transform.forward if self.transform else Vec3.forward()
        right = self.transform.right if self.transform else Vec3.right()
        fwd.y = 0.0
        right.y = 0.0
        fwd = fwd.normalized()
        right = right.normalized()

        move_x = 0.0
        move_z = 0.0
        if Input.GetKey(KeyCode.W): move_z += 1.0
        if Input.GetKey(KeyCode.S): move_z -= 1.0
        if Input.GetKey(KeyCode.A): move_x -= 1.0
        if Input.GetKey(KeyCode.D): move_x += 1.0

        wish = (fwd * move_z + right * move_x)
        if wish.length_sq() > 0.001:
            wish = wish.normalized()
        return wish

    def _accelerate(self, wish_dir: Vec3, wish_speed: float, accel: float, dt: float):
        vel = self._velocity
        cur_speed = vel.dot(wish_dir)
        add = wish_speed - cur_speed
        if add <= 0:
            return
        add = min(add, accel * dt * wish_speed)
        self._velocity = vel + wish_dir * add

    def _apply_friction(self, dt: float):
        vel = self._velocity
        speed = vel.length()
        if speed < 0.001:
            self._velocity = Vec3.zero()
            return
        control = self.stop_speed if speed < self.stop_speed else speed
        drop = control * self.friction * dt
        new_speed = max(0.0, speed - drop) / speed
        self._velocity = vel * new_speed

    def _update_crouch(self):
        if self._is_crouching:
            self._target_eye_height = self.crouch_eye_offset
        else:
            self._target_eye_height = self.capsule_height - 0.3
        diff = self._target_eye_height - self._eye_height
        self._eye_height += diff * min(1.0, 10.0 * 0.016)

    def _find_camera(self):
        from core.engine.engine import Engine
        engine = Engine.instance()
        if not engine or not engine._scene:
            return
        from core.components.rendering.cameras.camera import Camera
        for ent in engine._scene.get_entities_with_component(Camera):
            if ent.active:
                self._camera_entity_id = ent.id
                return

    def _set_cursor_locked(self, locked: bool):
        if locked == self._cursor_locked:
            return
        self._cursor_locked = locked
        Input.set_cursor_locked(locked)
        Input.set_cursor_visible(not locked)

    def on_start(self):
        self._solver = None
        self._character = None
        self._velocity = Vec3.zero()
        self._pitch = 0.0
        self._yaw = 0.0
        self._target_pitch = 0.0
        self._target_yaw = 0.0
        self._is_crouching = False
        self._wants_to_crouch = False
        self._grounded = False
        self._coyote_timer = 0.0
        self._jump_buffer_timer = 0.0
        self._camera_entity_id = None
        self._warned_no_world = False
        self._cursor_locked = False
        self._current_fov = self.camera_fov
        self._eye_height = self.capsule_height - 0.3
        self._target_eye_height = self._eye_height

        if self.camera_entity_id:
            self._camera_entity_id = self.camera_entity_id

        if not _HAS_CULVERIN:
            return

        self._solver = _resolve_solver()
        if self._solver is None:
            plugin = _get_physics_plugin()
            if plugin is not None and plugin.ensure_single_mode():
                self._solver = _resolve_solver()

        if self._solver is None:
            return

        tr = self.transform
        start = tr.local_position if tr else Vec3.zero()
        self._character = self._create_character(self._solver, start)
        if tr and self._character is not None:
            tr.local_rotation = _quat_yaw(self._yaw)

        if self.cursor_lock_on_start:
            self._set_cursor_locked(True)

    def on_disable(self):
        self._set_cursor_locked(False)
        if self._solver is not None and self._character is not None:
            self._solver.destroy_character(self._character)
            self._character = None

    def on_update(self, dt: float):
        if not self._entity or not self.enabled:
            return
        if Input.GetKeyDown(self.unlock_key):
            self._set_cursor_locked(not self._cursor_locked)
        self._handle_mouse_look(dt)
        self._update_crouch()
        if self._camera_entity_id is None:
            self._find_camera()
        self._update_camera(dt)

    def on_fixed_update(self, dt: float):
        if not self._entity or not self.enabled:
            return
        char = self._character
        if char is None or self._solver is None:
            if self._solver is None:
                self._solver = _resolve_solver()
                if self._solver is None:
                    plugin = _get_physics_plugin()
                    if plugin is not None and plugin.ensure_single_mode():
                        self._solver = _resolve_solver()
            if self._solver is not None and self.transform is not None:
                self._character = self._create_character(self._solver, self.transform.local_position)
                if self.transform:
                    self.transform.local_rotation = _quat_yaw(self._yaw)
            if self._character is None:
                if not self._warned_no_world:
                    from core.foundation.logger import Logger
                    Logger.warning(
                        "CharacterController: physics world is not available in-process. "
                        "Use single-threaded physics mode (simulation_mode='single') for characters."
                    )
                    self._warned_no_world = True
                return
            char = self._character

        tr = self.transform
        if not tr:
            return

        self._grounded = self._solver.is_character_grounded(char)

        if self._grounded:
            self._coyote_timer = self.coyote_time
        else:
            self._coyote_timer -= dt

        if Input.GetKeyDown(KeyCode.SPACE):
            self._jump_buffer_timer = self.jump_buffer_time
        else:
            self._jump_buffer_timer -= dt

        if Input.GetKeyDown(KeyCode.C):
            self._wants_to_crouch = not self._wants_to_crouch
            if self.crouch_toggle:
                self._is_crouching = self._wants_to_crouch
            else:
                self._is_crouching = not self._is_crouching
        if not self.crouch_toggle:
            self._is_crouching = Input.GetKey(KeyCode.C)

        can_jump = self._jump_buffer_timer > 0.0 and self._coyote_timer > 0.0
        if can_jump:
            vel = self._velocity
            self._velocity = Vec3(vel.x, self.jump_power, vel.z)
            self._jump_buffer_timer = 0.0
            self._coyote_timer = 0.0

        vel = self._velocity
        if self._grounded:
            self._accelerate(self.get_wish_dir(), self.get_move_speed(), self.acceleration, dt)
            self._apply_friction(dt)
            vel = self._velocity
            if vel.y < 0.0:
                vel = Vec3(vel.x, 0.0, vel.z)
            self._velocity = vel
        else:
            self._velocity = Vec3(vel.x, vel.y - self.gravity * dt, vel.z)
            self._accelerate(self.get_wish_dir(), self.get_move_speed(), self.air_acceleration, dt)
            vel = self._velocity

        self._solver.move_character(char, (vel.x, vel.y, vel.z), dt)

        pos = self._solver.get_character_position(char)
        if pos is not None:
            tr.local_position = Vec3(pos[0], pos[1], pos[2])
        self._solver.set_character_rotation(char, _quat_yaw(self._yaw).to_list())

    def _update_camera(self, dt: float):
        if self._camera_entity_id is None:
            return
        from core.engine.engine import Engine
        engine = Engine.instance()
        if not engine or not engine._scene:
            return
        cam_ent = engine._scene.get_entity(self._camera_entity_id)
        if not cam_ent or not cam_ent.active:
            self._camera_entity_id = None
            return
        cam_tr = cam_ent.transform
        player_tr = self.transform
        if not cam_tr or not player_tr:
            return

        cam_comp = cam_ent.get_component(_CAMERA_TYPE) if (_CAMERA_TYPE is not None) else None

        if cam_comp is not None:
            self._apply_camera_settings(cam_comp, dt)

        cam_is_child = cam_tr._entity.parent is self._entity if cam_tr._entity else False

        if cam_is_child:
            cam_tr.local_position = Vec3(0, self._eye_height, 0)
            px = math.radians(self._pitch) * 0.5
            cam_tr.local_rotation = Quat(math.sin(px), 0.0, 0.0, math.cos(px))
        else:
            cam_tr.local_position = player_tr.local_position + Vec3(0, self._eye_height, 0)
            cam_tr.local_rotation = _quat_pitch_yaw(self._pitch, self._yaw)

    def _apply_camera_settings(self, cam_comp, dt: float):
        scroll = Input.mouseScrollDelta
        if self.scroll_zoom and scroll is not None:
            self._current_fov -= scroll[1] * 2.0
            self._current_fov = max(self.min_fov, min(self.max_fov, self._current_fov))

        aiming = Input.GetKey(KeyCode.MOUSE_RIGHT)
        target_fov = self.aim_fov if aiming else self.camera_fov
        rate = min(1.0, self.aim_zoom_speed * dt)
        self._current_fov += (target_fov - self._current_fov) * rate
        self._current_fov = max(1.0, min(179.0, self._current_fov))

        cam_comp.fov = self._current_fov
        cam_comp.near = self.camera_near
        cam_comp.far = self.camera_far

    def _handle_mouse_look(self, dt: float):
        if not self._entity:
            return
        dx, dy = Input.mouseDelta
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            return
        sx = self.sensitivity_x if self.sensitivity_x > 0.0 else self.sensitivity
        sy = self.sensitivity_y if self.sensitivity_y > 0.0 else self.sensitivity
        yaw_speed = sx * 0.022
        pitch_speed = sy * 0.022
        self._target_yaw -= dx * yaw_speed
        if self.invert_x:
            self._target_yaw = -self._target_yaw
        pitch_delta = dy * pitch_speed
        if self.invert_y:
            pitch_delta = -pitch_delta
        self._target_pitch -= pitch_delta
        self._target_pitch = max(-89.0, min(89.0, self._target_pitch))

        smoothing = self.mouse_smoothing
        if smoothing <= 0.0:
            self._yaw = self._target_yaw
            self._pitch = self._target_pitch
        else:
            k = min(1.0, self.smoothing_strength * dt)
            self._yaw += (self._target_yaw - self._yaw) * k
            self._pitch += (self._target_pitch - self._pitch) * k

        tr = self.transform
        if tr:
            tr.local_rotation = _quat_yaw(self._yaw)

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "walk_speed": self.walk_speed, "run_speed": self.run_speed,
            "crouch_speed": self.crouch_speed, "acceleration": self.acceleration,
            "air_acceleration": self.air_acceleration, "friction": self.friction,
            "stop_speed": self.stop_speed, "jump_power": self.jump_power,
            "jump_buffer_time": self.jump_buffer_time, "coyote_time": self.coyote_time,
            "crouch_toggle": self.crouch_toggle, "crouch_speed_mult": self.crouch_speed_mult,
            "crouch_eye_offset": self.crouch_eye_offset, "sensitivity": self.sensitivity,
            "sensitivity_x": self.sensitivity_x, "sensitivity_y": self.sensitivity_y,
            "invert_x": self.invert_x, "invert_y": self.invert_y,
            "mouse_smoothing": self.mouse_smoothing, "smoothing_strength": self.smoothing_strength,
            "cursor_lock_on_start": self.cursor_lock_on_start, "unlock_key": self.unlock_key,
            "camera_fov": self.camera_fov, "camera_near": self.camera_near, "camera_far": self.camera_far,
            "scroll_zoom": self.scroll_zoom, "min_fov": self.min_fov, "max_fov": self.max_fov,
            "aim_fov": self.aim_fov, "aim_zoom_speed": self.aim_zoom_speed,
            "gravity": self.gravity,
            "capsule_radius": self.capsule_radius, "capsule_height": self.capsule_height,
            "crouch_height": self.crouch_height, "step_height": self.step_height,
            "max_slope": self.max_slope, "push_strength": self.push_strength,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> CharacterController:
        cc = cls()
        cc.enabled = data.get("enabled", True)
        for key in ("walk_speed", "run_speed", "crouch_speed", "acceleration",
                     "air_acceleration", "friction", "stop_speed", "jump_power",
                     "jump_buffer_time", "coyote_time", "crouch_speed_mult",
                     "crouch_eye_offset", "sensitivity", "sensitivity_x", "sensitivity_y",
                     "camera_near", "camera_far", "min_fov", "max_fov",
                     "aim_fov", "aim_zoom_speed", "gravity",
                     "capsule_radius", "capsule_height", "crouch_height",
                     "step_height", "max_slope", "push_strength", "smoothing_strength"):
            setattr(cc, key, data.get(key, getattr(cc, key)))
        for bkey in ("crouch_toggle", "invert_x", "invert_y", "scroll_zoom",
                     "cursor_lock_on_start"):
            setattr(cc, bkey, data.get(bkey, getattr(cc, bkey)))
        cc.camera_fov = data.get("camera_fov", cc.camera_fov)
        cc.mouse_smoothing = data.get("mouse_smoothing", cc.mouse_smoothing)
        cc.unlock_key = data.get("unlock_key", cc.unlock_key)
        cc.camera_entity_id = data.get("camera_entity_id", "")
        return cc


_CAMERA_TYPE = None
def _load_camera_type():
    global _CAMERA_TYPE
    if _CAMERA_TYPE is not None:
        return _CAMERA_TYPE
    try:
        from core.components.rendering.cameras.camera import Camera
        _CAMERA_TYPE = Camera
    except Exception:
        _CAMERA_TYPE = None
    return _CAMERA_TYPE
_load_camera_type()
