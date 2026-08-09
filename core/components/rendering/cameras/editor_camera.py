# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.ecs.ecs import Component, ComponentRegistry
from core.maths.math3d import Vec3, Quat
from core.input.input_manager import InputManager
from core.input.constants import KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E, KEY_SHIFT, MOUSE_R, MOUSE_M, MOUSE_L
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class EditorCamera(Component):
    _icon = "Camera.png"
    _show_gizmo_icon = False
    _allow_multiple = True

    MOVE_SPEED = 5.0
    FAST_MULT = 3.0
    ROTATE_SPEED = 0.3
    ZOOM_SPEED = 4.0
    PAN_SPEED = 0.01
    ACCEL = 12.0
    DAMPING = 8.0
    SPEED_BOOST_RAMP = 2.0

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("move_speed", "Move Speed", FieldType.FLOAT, min_val=0.1, max_val=100.0, step=0.5),
            InspectorField("rotate_speed", "Rotate Speed", FieldType.FLOAT, min_val=0.01, max_val=5.0, step=0.05, decimals=2),
            InspectorField("zoom_speed", "Zoom Speed", FieldType.FLOAT, min_val=0.1, max_val=50.0, step=0.5),
            InspectorField("pan_speed", "Pan Speed", FieldType.FLOAT, min_val=0.001, max_val=0.1, step=0.005, decimals=3),
            InspectorField("damping", "Damping", FieldType.FLOAT, min_val=0.1, max_val=50.0, step=0.5),
            InspectorField("acceleration", "Acceleration", FieldType.FLOAT, min_val=1.0, max_val=100.0, step=1.0),
        ]

    def __init__(self):
        super().__init__()
        self._yaw: float = 0.0
        self._pitch: float = -15.0
        self.move_speed: float = self.MOVE_SPEED
        self.rotate_speed: float = self.ROTATE_SPEED
        self.zoom_speed: float = self.ZOOM_SPEED
        self.pan_speed: float = self.PAN_SPEED
        self.damping: float = self.DAMPING
        self.acceleration: float = self.ACCEL
        self._vel: Vec3 = Vec3.zero()
        self._right_mouse: bool = False
        self._middle_mouse: bool = False
        self._alt_left: bool = False
        self._last_mx: int = 0
        self._last_my: int = 0
        self._scroll_accumulator: float = 0.0
        self._speed_boost_time: float = 0.0

    def _forward(self) -> Vec3:
        pr = math.radians(self._pitch)
        yr = math.radians(self._yaw)
        return Vec3(
            -math.cos(pr) * math.sin(yr),
            -math.sin(pr),
            -math.cos(pr) * math.cos(yr)
        ).normalized()

    def _right(self) -> Vec3:
        return self._forward().cross(Vec3.up()).normalized()

    def _up(self) -> Vec3:
        return self._right().cross(self._forward()).normalized()

    def on_mouse_press(self, btn, x, y, alt_pressed=False):
        if btn == MOUSE_R:
            self._right_mouse = True
        elif btn == MOUSE_M:
            self._middle_mouse = True
        elif btn == MOUSE_L and alt_pressed:
            self._alt_left = True
        self._last_mx = x
        self._last_my = y

    def on_mouse_release(self, btn):
        if btn == MOUSE_R:
            self._right_mouse = False
        elif btn == MOUSE_M:
            self._middle_mouse = False
        elif btn == MOUSE_L:
            self._alt_left = False

    def on_mouse_move(self, x, y):
        t = self.transform
        if not t:
            return
        dx = float(x - self._last_mx)
        dy = float(y - self._last_my)
        self._last_mx = x
        self._last_my = y
        if self._right_mouse or self._alt_left:
            self._yaw -= dx * self.rotate_speed
            self._pitch = max(-89.0, min(89.0, self._pitch + dy * self.rotate_speed))
            t.local_rotation = Quat.from_euler(-self._pitch, self._yaw, 0.0)
        elif self._middle_mouse:
            r = self._right()
            u = self._up()
            amt = self.move_speed * self.pan_speed
            t.position = t.position - r * (dx * amt) + u * (dy * amt)

    def on_mouse_delta(self, dx: float, dy: float):
        t = self.transform
        if not t:
            return
        if self._right_mouse or self._alt_left:
            self._yaw -= dx * self.rotate_speed
            self._pitch = max(-89.0, min(89.0, self._pitch + dy * self.rotate_speed))
            t.local_rotation = Quat.from_euler(-self._pitch, self._yaw, 0.0)
        elif self._middle_mouse:
            r = self._right()
            u = self._up()
            amt = self.move_speed * self.pan_speed
            t.position = t.position - r * (dx * amt) + u * (dy * amt)

    def on_scroll(self, delta):
        t = self.transform
        if not t:
            return
        self._scroll_accumulator += delta * 0.1
        accel = max(1.0, pow(1.05, abs(self._scroll_accumulator)) - 1.0)
        scaled_delta = delta * (1.0 + accel)
        t.position = t.position + self._forward() * scaled_delta * self.zoom_speed

    def on_update(self, dt):
        t = self.transform
        if not t:
            return
        if dt > 0.05:
            dt = 0.05
        im = InputManager.instance()
        if self._right_mouse:
            fwd = self._forward()
            right = self._right()
            up = Vec3.up()
            any_move = im.is_key_pressed(KEY_W) or im.is_key_pressed(KEY_A) or im.is_key_pressed(KEY_S) or im.is_key_pressed(KEY_D) or im.is_key_pressed(KEY_Q) or im.is_key_pressed(KEY_E)
            if any_move:
                self._speed_boost_time += dt
            else:
                self._speed_boost_time = 0.0
            boost_t = min(self._speed_boost_time / max(self.SPEED_BOOST_RAMP, 0.001), 1.0)
            boost_factor = 1.0 + (self.FAST_MULT - 1.0) * boost_t
            speed = self.move_speed * boost_factor
            accel = Vec3.zero()
            if im.is_key_pressed(KEY_W):
                accel = accel + fwd * speed
            if im.is_key_pressed(KEY_S):
                accel = accel - fwd * speed
            if im.is_key_pressed(KEY_A):
                accel = accel - right * speed
            if im.is_key_pressed(KEY_D):
                accel = accel + right * speed
            if im.is_key_pressed(KEY_E):
                accel = accel + up * speed
            if im.is_key_pressed(KEY_Q):
                accel = accel - up * speed
            facc = dt * self.acceleration
            self._vel = self._vel + (accel - self._vel) * min(facc, 1.0)
            t.position = t.position + self._vel * dt
        else:
            self._speed_boost_time = 0.0
            self._vel = self._vel * 0.85
            if self._vel.length() > 0.001:
                t.position = t.position + self._vel * dt

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "yaw": self._yaw, "pitch": self._pitch,
            "move_speed": self.move_speed, "rotate_speed": self.rotate_speed,
            "zoom_speed": self.zoom_speed, "pan_speed": self.pan_speed,
            "damping": self.damping, "acceleration": self.acceleration,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> EditorCamera:
        c = cls()
        c.enabled = data.get("enabled", True)
        c._yaw = data.get("yaw", 0.0)
        c._pitch = data.get("pitch", -15.0)
        c.move_speed = float(data.get("move_speed", cls.MOVE_SPEED))
        c.rotate_speed = float(data.get("rotate_speed", cls.ROTATE_SPEED))
        c.zoom_speed = float(data.get("zoom_speed", cls.ZOOM_SPEED))
        c.pan_speed = float(data.get("pan_speed", cls.PAN_SPEED))
        c.damping = float(data.get("damping", cls.DAMPING))
        c.acceleration = float(data.get("acceleration", cls.ACCEL))
        return c
