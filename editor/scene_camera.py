# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Optional
from core.math3d import Vec3, Mat4
from core.input.input_manager import InputManager
from core.input.constants import KEY_W, KEY_A, KEY_S, KEY_D, KEY_Q, KEY_E, KEY_SHIFT, MOUSE_R, MOUSE_M, MOUSE_L
class SceneCamera:
    PERSPECTIVE_TO_ORTHO_FOV = 179.0
    TRANSITION_SPEED = 2.5
    DEFAULT_FOV = 60.0
    ORTHO_SIZE = 5.0
    DEFAULT_NEAR = 0.01
    DEFAULT_FAR = 1000.0
    MOVE_SPEED = 5.0
    FAST_MULT = 3.0
    ROTATE_SPEED = 0.3
    ZOOM_SPEED = 4.0
    PAN_SPEED = 0.01
    DAMPING = 8.0
    ACCEL = 12.0
    ZOOM_SMOOTH_SPEED = 15.0
    SPEED_BOOST_MULT = 3.0
    SPEED_BOOST_RAMP_TIME = 2.0
    def __init__(self):
        self._position: Vec3 = Vec3(0.0, 3.0, 10.0)
        self._yaw: float = 0.0
        self._pitch: float = -15.0
        self._fov: float = self.DEFAULT_FOV
        self._near: float = self.DEFAULT_NEAR
        self._far: float = self.DEFAULT_FAR
        self._move_speed: float = self.MOVE_SPEED
        self._fast_mult: float = self.FAST_MULT
        self._rotate_speed: float = self.ROTATE_SPEED
        self._zoom_speed: float = self.ZOOM_SPEED
        self._pan_speed: float = self.PAN_SPEED
        self._damping: float = self.DAMPING
        self._acceleration: float = self.ACCEL
        self._transition_speed: float = self.TRANSITION_SPEED
        self._zoom_smooth_speed: float = self.ZOOM_SMOOTH_SPEED
        self._zoom_strength: float = 0.3
        self._use_ortho_in_2d: bool = True
        self._right_mouse: bool = False
        self._middle_mouse: bool = False
        self._alt_left: bool = False
        self._render_scale: float = 1.0
        self._resolution_mode: str = "native"
        self._resolution_w: int = 1920
        self._resolution_h: int = 1080
        self._last_mx: int = 0
        self._last_my: int = 0
        self._vel: Vec3 = Vec3.zero()
        self._orbit_target: Vec3 = Vec3.zero()
        self._orbit_dist: float = 10.0
        self._orbiting: bool = False
        self._is_orthographic: bool = False
        self._transition_blend: float = 0.0
        self._transition_active: bool = False
        self._transition_to_ortho: bool = False
        self._ortho_zoom_distance: float = 0.0
        self._stored_ortho_size: float = 5.0
        self._is_2d_mode: bool = False
        self._mode_2d_transition: bool = False
        self._mode_2d_blend: float = 0.0
        self._mode_2d_to_2d: bool = False
        self._stored_yaw: float = 0.0
        self._stored_pitch: float = -15.0
        self._stored_fov: float = self.DEFAULT_FOV
        self._stored_ortho: bool = False
        self._viewport_w: int = 800
        self._viewport_h: int = 600
        self._zoom_target_distance: float = 0.0
        self._zoom_cursor_ndc: Optional[tuple[float, float]] = None
        self._on_2d_mode_changed = None
        self._release_damping: float = 1.0
        self._focus_transition_blend: float = 1.0
        self._focus_start_pos: Vec3 = self._position
        self._focus_target_pos: Vec3 = self._position
        self._focus_start_orbit: Vec3 = self._orbit_target
        self._focus_target_orbit: Vec3 = self._orbit_target
        self._focus_start_yaw: float = self._yaw
        self._focus_start_pitch: float = self._pitch
        self._focus_target_yaw: float = self._yaw
        self._focus_target_pitch: float = self._pitch
        self._focus_active: bool = False
        self._scroll_accumulator: float = 0.0
        self._speed_boost_time: float = 0.0
        self._speed_boost_enabled: bool = True
        self._speed_boost_mult: float = self.SPEED_BOOST_MULT
        self._speed_boost_ramp_time: float = self.SPEED_BOOST_RAMP_TIME
    def load_config(self, config) -> None:
        self._fov = config.get("camera.fov", self.DEFAULT_FOV)
        self._near = config.get("camera.near", self.DEFAULT_NEAR)
        self._far = config.get("camera.far", self.DEFAULT_FAR)
        self._move_speed = config.get("camera.move_speed", self.MOVE_SPEED)
        self._fast_mult = config.get("camera.fast_mult", self.FAST_MULT)
        self._rotate_speed = config.get("camera.rotate_speed", self.ROTATE_SPEED)
        self._zoom_speed = config.get("camera.zoom_speed", self.ZOOM_SPEED)
        self._pan_speed = config.get("camera.pan_speed", self.PAN_SPEED)
        self._damping = config.get("camera.damping", self.DAMPING)
        self._acceleration = config.get("camera.acceleration", self.ACCEL)
        self._transition_speed = config.get("camera.transition_speed", self.TRANSITION_SPEED)
        self._zoom_smooth_speed = config.get("camera.zoom_smooth_speed", self.ZOOM_SMOOTH_SPEED)
        self._zoom_strength = config.get("camera.zoom_strength", 0.3)
        self._use_ortho_in_2d = config.get("camera.use_ortho_in_2d", True)
        self._speed_boost_enabled = config.get("camera.speed_boost_enabled", True)
        self._speed_boost_mult = config.get("camera.speed_boost_mult", self.SPEED_BOOST_MULT)
        self._speed_boost_ramp_time = config.get("camera.speed_boost_ramp_time", self.SPEED_BOOST_RAMP_TIME)
        self._render_scale = float(config.get("camera.render_scale", 1.0))
        self._resolution_mode = config.get("camera.resolution_mode", "native")
        self._resolution_w = int(config.get("camera.resolution_w", 1920))
        self._resolution_h = int(config.get("camera.resolution_h", 1080))
    @property
    def forward(self) -> Vec3: return self._forward()
    @property
    def position(self) -> Vec3: return self._position
    @property
    def yaw(self) -> float: return self._yaw
    @property
    def pitch(self) -> float: return self._pitch
    @property
    def fov(self) -> float: return self._fov
    @property
    def near(self) -> float: return self._near
    @property
    def far(self) -> float: return self._far
    @property
    def is_orthographic(self) -> bool: return self._is_orthographic
    @property
    def is_2d_mode(self) -> bool: return self._is_2d_mode
    def _forward(self) -> Vec3:
        if self._is_2d_mode:
            return Vec3(0.0, 0.0, -1.0)
        pr = math.radians(self._pitch)
        yr = math.radians(self._yaw)
        return Vec3(
            -math.cos(pr) * math.sin(yr),
            -math.sin(pr),
            -math.cos(pr) * math.cos(yr)
        ).normalized()
    def _right(self) -> Vec3:
        if self._is_2d_mode:
            return Vec3(1.0, 0.0, 0.0)
        return self._forward().cross(Vec3.up()).normalized()
    def _up(self) -> Vec3:
        if self._is_2d_mode:
            return Vec3(0.0, 1.0, 0.0)
        return self._right().cross(self._forward()).normalized()
    def get_view_matrix(self) -> Mat4:
        if self._focus_active:
            return Mat4.look_at(self._position, self._orbit_target, Vec3.up())
        return Mat4.look_at(self._position, self._position + self._forward(), Vec3.up())
    def get_projection_matrix(self, aspect: float) -> Mat4:
        if (self._is_2d_mode or self._mode_2d_transition) and self._use_ortho_in_2d and self._is_orthographic:
            if self._ortho_zoom_distance < 0.5:
                dist = (self._position - Vec3.zero()).length()
                self._ortho_zoom_distance = dist * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
                self._stored_ortho_size = self._ortho_zoom_distance
            half_size = self._ortho_zoom_distance * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
            return Mat4.orthographic(-half_size*aspect, half_size*aspect, -half_size, half_size, self._near, self._far)
        if not self._transition_active and not self._is_orthographic:
            return Mat4.perspective(self._fov, aspect, self._near, self._far)
        if self._ortho_zoom_distance < 0.5:
            dist = (self._position - Vec3.zero()).length()
            self._ortho_zoom_distance = dist * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
            self._stored_ortho_size = self._ortho_zoom_distance
        blend = self._ease_in_out_cubic(self._transition_blend)
        anim_fov = math.degrees(2.0 * math.atan(math.tan(math.radians(self.DEFAULT_FOV) * 0.5) ** (1.0 - blend)))
        if blend < 0.5:
            t = blend * 2.0
            persp_mat = Mat4.perspective(anim_fov, aspect, self._near, self._far)
            half_size = self._ortho_zoom_distance * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
            ortho_mat = Mat4.orthographic(-half_size*aspect, half_size*aspect, -half_size, half_size, self._near, self._far)
            return Mat4(persp_mat._d + (ortho_mat._d - persp_mat._d) * t)
        else:
            half_size = self._ortho_zoom_distance * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
            return Mat4.orthographic(-half_size*aspect, half_size*aspect, -half_size, half_size, self._near, self._far)
    def on_mouse_press(self, btn, x, y, alt_pressed=False):
        self._focus_active = False
        self._focus_transition_blend = 1.0
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
            self._release_damping = 0.95
        elif btn == MOUSE_M:
            self._middle_mouse = False
        elif btn == MOUSE_L:
            self._alt_left = False
    def set_viewport_size(self, w: int, h: int):
        self._viewport_w = w
        self._viewport_h = h

    def compute_render_size(self, display_w: int, display_h: int) -> tuple[int, int]:
        if self._resolution_mode == "custom":
            return max(1, int(self._resolution_w)), max(1, int(self._resolution_h))
        scale = max(0.05, min(1.0, self._render_scale))
        return max(1, int(round(display_w * scale))), max(1, int(round(display_h * scale)))

    @staticmethod
    def _ease_in_out_cubic(t: float) -> float:
        if t < 0.5:
            return 4.0 * t * t * t
        else:
            return 1.0 - pow(-2.0 * t + 2.0, 3.0) / 2.0

    def _lerp_angle(self, a, b, t):
        diff = (b - a) % 360.0
        if diff > 180.0:
            diff -= 360.0
        return a + diff * t

    def on_mouse_move(self, x, y):
        dx = float(x - self._last_mx)
        dy = float(y - self._last_my)
        self._last_mx = x
        self._last_my = y
        if dx != 0 or dy != 0:
            self._focus_active = False
            self._focus_transition_blend = 1.0
        if self._is_2d_mode and (self._right_mouse or self._middle_mouse):
            r = self._right()
            u = self._up()
            half_size = self._ortho_zoom_distance * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
            amt = (2.0 * half_size) / self._viewport_h
            self._position = self._position - r * (dx * amt) + u * (dy * amt)
        elif self._right_mouse or self._alt_left:
            self._yaw -= dx * self._rotate_speed
            self._pitch = max(-89.0, min(89.0, self._pitch + dy * self._rotate_speed))
        elif self._middle_mouse:
            r = self._right()
            u = self._up()
            amt = self._move_speed * self._pan_speed
            self._position = self._position - r * (dx * amt) + u * (dy * amt)
    def on_scroll(self, delta, cursor_x=None, cursor_y=None):
        self._focus_active = False
        self._focus_transition_blend = 1.0
        self._scroll_accumulator += delta * 0.1
        accel = max(1.0, pow(1.05, abs(self._scroll_accumulator)) - 1.0)
        scaled_delta = delta * (1.0 + accel)
        if self._is_orthographic or (self._is_2d_mode and self._use_ortho_in_2d):
            scale = 1.0 - scaled_delta * self._zoom_strength
            new_dist = self._ortho_zoom_distance * scale
            new_dist = max(0.1, min(new_dist, 1000.0))
            self._stored_ortho_size = new_dist
            if self._is_2d_mode:
                self._ortho_zoom_distance = new_dist
                self._zoom_target_distance = new_dist
                if cursor_x is not None and cursor_y is not None and self._viewport_w > 0 and self._viewport_h > 0:
                    ndc_x = 2.0 * cursor_x / self._viewport_w - 1.0
                    ndc_y = 1.0 - 2.0 * cursor_y / self._viewport_h
                    self._zoom_cursor_ndc = (ndc_x, ndc_y)
            else:
                self._ortho_zoom_distance = new_dist
        elif self._is_2d_mode:
            forward = self._forward()
            self._position = self._position + forward * scaled_delta * self._zoom_speed
        else:
            self._position = self._position + self._forward() * scaled_delta * self._zoom_speed

    def toggle_2d_mode(self):
        if self._mode_2d_transition:
            return
        if not self._is_2d_mode:
            self._stored_yaw = self._yaw
            self._stored_pitch = self._pitch
            self._stored_fov = self._fov
            self._stored_ortho = self._is_orthographic
            self._mode_2d_transition = True
            self._mode_2d_to_2d = True
            self._mode_2d_blend = 0.0
            if self._use_ortho_in_2d:
                if not self._is_orthographic:
                    self._stored_fov_for_2d = self._fov
                    dist = (self._position - Vec3.zero()).length()
                    self._ortho_zoom_distance = dist * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
                    self._stored_ortho_size = self._ortho_zoom_distance
                    self._is_orthographic = True
                else:
                    if not hasattr(self, '_stored_fov_for_2d'):
                        self._stored_fov_for_2d = self._fov
                    if self._ortho_zoom_distance < 0.5:
                        dist = (self._position - Vec3.zero()).length()
                        self._ortho_zoom_distance = dist * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
                        self._stored_ortho_size = self._ortho_zoom_distance
            else:
                if not hasattr(self, '_stored_fov_for_2d'):
                    self._stored_fov_for_2d = self._fov
            self._zoom_target_distance = self._ortho_zoom_distance
            self._zoom_cursor_ndc = None
        else:
            self._mode_2d_transition = True
            self._mode_2d_to_2d = False
            self._mode_2d_blend = 1.0
            self._yaw = 0.0
            self._pitch = 0.0
            self._is_2d_mode = False
            if not self._stored_ortho:
                self._transition_active = True
                self._transition_blend = 1.0
                self._transition_to_ortho = False
                self._fov = self.PERSPECTIVE_TO_ORTHO_FOV
            elif self._is_orthographic:
                self._transition_active = True
                self._transition_blend = 0.0
                self._transition_to_ortho = False
                self._fov = self._stored_fov_for_2d if hasattr(self, '_stored_fov_for_2d') else self._stored_fov
        if hasattr(self, '_on_projection_changed'):
            self._on_projection_changed()
    def toggle_projection(self):
        self._stored_fov = self._fov
        if self._is_orthographic:
            self._transition_active = True
            self._transition_blend = 1.0
            self._transition_to_ortho = False
            self._fov = self.PERSPECTIVE_TO_ORTHO_FOV
        else:
            if self._ortho_zoom_distance < 0.5:
                dist = (self._position - Vec3.zero()).length()
                self._ortho_zoom_distance = dist * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
                self._stored_ortho_size = self._ortho_zoom_distance
            self._transition_active = True
            self._transition_blend = 0.0
            self._transition_to_ortho = True
            self._fov = self.PERSPECTIVE_TO_ORTHO_FOV
        if hasattr(self, '_on_projection_changed'):
            self._on_projection_changed()
    def update(self, dt):
        if self._transition_active:
            t = min(dt * self._transition_speed, 1.0)
            if self._transition_to_ortho:
                self._transition_blend += t
                if self._transition_blend >= 1.0:
                    self._transition_blend = 1.0
                    self._is_orthographic = True
                    self._transition_active = False
                    self._fov = self._stored_fov_for_2d if hasattr(self, '_stored_fov_for_2d') else self._stored_fov
            else:
                self._transition_blend -= t
                if self._transition_blend <= 0.0:
                    self._transition_blend = 0.0
                    self._transition_active = False
                    self._is_orthographic = False
                    self._fov = self._stored_fov_for_2d if hasattr(self, '_stored_fov_for_2d') else self._stored_fov
        if self._mode_2d_transition:
            adv = min(dt * self._transition_speed, 1.0)
            if self._mode_2d_to_2d:
                self._mode_2d_blend += adv
                if self._mode_2d_blend >= 1.0:
                    self._mode_2d_blend = 1.0
                    self._is_2d_mode = True
                    self._mode_2d_transition = False
                    self._yaw = 0.0
                    self._pitch = 0.0
                    if self._on_2d_mode_changed:
                        self._on_2d_mode_changed()
                else:
                    e = self._ease_in_out_cubic(self._mode_2d_blend)
                    self._yaw = self._lerp_angle(self._stored_yaw, 0.0, e)
                    self._pitch = self._stored_pitch * (1.0 - e)
            else:
                self._mode_2d_blend -= adv
                if self._mode_2d_blend <= 0.0:
                    self._mode_2d_blend = 0.0
                    self._mode_2d_transition = False
                    self._yaw = self._stored_yaw
                    self._pitch = self._stored_pitch
                    self._fov = self._stored_fov
                    if self._on_2d_mode_changed:
                        self._on_2d_mode_changed()
                else:
                    e = self._ease_in_out_cubic(self._mode_2d_blend)
                    self._yaw = self._lerp_angle(0.0, self._stored_yaw, 1.0 - e)
                    self._pitch = self._stored_pitch * (1.0 - e)
        if self._focus_transition_blend < 1.0:
            adv = min(dt * self._transition_speed, 1.0)
            self._focus_transition_blend += adv
            if self._focus_transition_blend >= 1.0:
                self._focus_transition_blend = 1.0
                self._position = self._focus_target_pos
                self._orbit_target = self._focus_target_orbit
                self._yaw = self._focus_target_yaw
                self._pitch = self._focus_target_pitch
            else:
                e = self._ease_in_out_cubic(self._focus_transition_blend)
                self._position = Vec3(
                    self._focus_start_pos.x + (self._focus_target_pos.x - self._focus_start_pos.x) * e,
                    self._focus_start_pos.y + (self._focus_target_pos.y - self._focus_start_pos.y) * e,
                    self._focus_start_pos.z + (self._focus_target_pos.z - self._focus_start_pos.z) * e,
                )
                self._orbit_target = Vec3(
                    self._focus_start_orbit.x + (self._focus_target_orbit.x - self._focus_start_orbit.x) * e,
                    self._focus_start_orbit.y + (self._focus_target_orbit.y - self._focus_start_orbit.y) * e,
                    self._focus_start_orbit.z + (self._focus_target_orbit.z - self._focus_start_orbit.z) * e,
                )
                self._yaw = self._lerp_angle(self._focus_start_yaw, self._focus_target_yaw, e)
                self._pitch = self._focus_start_pitch + (self._focus_target_pitch - self._focus_start_pitch) * e
        if self._is_2d_mode and self._use_ortho_in_2d and abs(self._ortho_zoom_distance - self._zoom_target_distance) > 0.001:
            old_half = self._ortho_zoom_distance * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
            st = min(dt * self._zoom_smooth_speed, 1.0)
            self._ortho_zoom_distance += (self._zoom_target_distance - self._ortho_zoom_distance) * st
            self._stored_ortho_size = self._ortho_zoom_distance
            new_half = self._ortho_zoom_distance * math.tan(math.radians(self.DEFAULT_FOV) * 0.5)
            if old_half > 1e-8 and self._zoom_cursor_ndc is not None:
                per_frame_scale = new_half / old_half
                ndc_x, ndc_y = self._zoom_cursor_ndc
                aspect = self._viewport_w / max(1, self._viewport_h)
                dx = ndc_x * old_half * aspect * (1.0 - per_frame_scale)
                dy = ndc_y * old_half * (1.0 - per_frame_scale)
                self._position = self._position + self._right() * dx + self._up() * dy
                self._zoom_cursor_ndc = None
        if dt > 0.05:
            dt = 0.05
        im = InputManager.instance()
        any_move = im.is_key_pressed(KEY_W) or im.is_key_pressed(KEY_A) or im.is_key_pressed(KEY_S) or im.is_key_pressed(KEY_D) or im.is_key_pressed(KEY_Q) or im.is_key_pressed(KEY_E)
        if any_move or im.is_key_pressed(KEY_SHIFT):
            self._focus_active = False
            self._focus_transition_blend = 1.0
        if self._speed_boost_enabled and any_move:
            self._speed_boost_time += dt
        elif not any_move:
            self._speed_boost_time = 0.0
        if self._speed_boost_enabled:
            boost_t = min(self._speed_boost_time / max(self._speed_boost_ramp_time, 0.001), 1.0)
            boost_factor = 1.0 + (self._speed_boost_mult - 1.0) * boost_t
        else:
            boost_factor = 1.0
        if self._is_2d_mode:
            right = self._right()
            up = self._up()
            accel = Vec3.zero()
            speed = self._move_speed * (self._ortho_zoom_distance / 5.0) * boost_factor
            if im.is_key_pressed(KEY_SHIFT):
                speed *= self._fast_mult
            if im.is_key_pressed(KEY_W):
                accel = accel + up * speed
            if im.is_key_pressed(KEY_S):
                accel = accel - up * speed
            if im.is_key_pressed(KEY_A):
                accel = accel - right * speed
            if im.is_key_pressed(KEY_D):
                accel = accel + right * speed
            if im.is_key_pressed(KEY_E):
                accel = accel + self._forward() * speed
            if im.is_key_pressed(KEY_Q):
                accel = accel - self._forward() * speed
            facc = dt * self._acceleration
            self._vel = self._vel + (accel - self._vel) * min(facc, 1.0)
            self._position = self._position + self._vel * dt
        elif self._right_mouse:
            fwd = self._forward()
            right = self._right()
            up = Vec3.up()
            accel = Vec3.zero()
            speed = self._move_speed * boost_factor
            if im.is_key_pressed(KEY_SHIFT):
                speed *= self._fast_mult
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
            facc = dt * self._acceleration
            self._vel = self._vel + (accel - self._vel) * min(facc, 1.0)
            self._position = self._position + self._vel * dt
        else:
            accel = Vec3.zero()
            self._vel = self._vel * 0.85
            self._position = self._position + self._vel * dt
    def focus_on(self, target: Vec3, distance: float = 5.0):
        dir_to_target = (target - self._position).normalized()
        self._focus_start_pos = self._position
        self._focus_target_pos = target - dir_to_target * distance
        self._focus_start_orbit = self._orbit_target
        self._focus_target_orbit = target
        self._focus_start_yaw = self._yaw
        self._focus_start_pitch = self._pitch
        self._focus_target_yaw = math.degrees(math.atan2(-dir_to_target.x, -dir_to_target.z))
        self._focus_target_pitch = math.degrees(math.asin(max(-1.0, min(1.0, -dir_to_target.y))))
        self._focus_active = True
        self._focus_transition_blend = 0.0
    def frame_bounds(self, center: Vec3, radius: float = 1.0):
        self.focus_on(center, max(radius * 2.5, 1.0))
    def serialize(self) -> dict:
        return {
            "position": self._position.to_list(),
            "yaw": self._yaw, "pitch": self._pitch,
            "fov": self._fov, "near": self._near, "far": self._far,
            "is_orthographic": self._is_orthographic
        }
    def deserialize(self, data: dict):
        p = data.get("position", [0, 3, 10])
        self._position = Vec3(*p)
        self._yaw = data.get("yaw", 0.0)
        self._pitch = data.get("pitch", -15.0)
        self._fov = data.get("fov", self.DEFAULT_FOV)
        self._near = data.get("near", self.DEFAULT_NEAR)
        self._far = data.get("far", self.DEFAULT_FAR)
        self._is_orthographic = data.get("is_orthographic", False)
