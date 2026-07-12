# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import numpy as np
from enum import Enum
from typing import Optional, Any
from core.ecs.ecs import Component, ComponentRegistry
from core.math.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField


class ShapeType(Enum):
    CONE = "cone"
    SPHERE = "sphere"
    HEMISPHERE = "hemisphere"
    BOX = "box"
    CIRCLE = "circle"

class ParticleRenderMode(Enum):
    BILLBOARD = "billboard"
    STRETCHED_BILLBOARD = "stretched_billboard"

class SimulationSpace(Enum):
    LOCAL = "local"
    WORLD = "world"

class ParticleSortMode(Enum):
    NONE = "none"
    BY_DISTANCE = "by_distance"
    BY_AGE = "by_age"

PARTICLE_DTYPE = np.dtype([
    ('position', np.float32, 4),
    ('velocity', np.float32, 4),
    ('color', np.float32, 4),
    ('meta', np.float32, 4),
    ('lifetime', np.float32, 4),
    ('start_color', np.float32, 4),
    ('start_meta', np.float32, 4),
])

def _build_curve_cache(curve_list) -> Optional[tuple]:
    if not curve_list or len(curve_list) < 2:
        return None
    arr = np.array(curve_list, dtype=np.float32)
    return arr[:, 0], arr[:, 1]

def _build_gradient_cache(grad) -> Optional[tuple]:
    if not grad:
        return None
    if isinstance(grad, (list, tuple)):
        stops = sorted(grad, key=lambda s: s[0])
        c_t = np.array([s[0] for s in stops], dtype=np.float32)
        c_v = np.array([s[1][:3] for s in stops], dtype=np.float32).reshape(-1, 3)
        a_t = np.array([s[0] for s in stops], dtype=np.float32)
        a_v = np.array([s[1][3] if len(s[1]) > 3 else 1.0 for s in stops], dtype=np.float32)
        return c_t, c_v, a_t, a_v
    ck = grad.get("color_keys", [(0, [1, 1, 1]), (1, [1, 1, 1])])
    ak = grad.get("alpha_keys", [(0, 1.0), (1, 1.0)])
    c_t = np.array([k[0] for k in ck], dtype=np.float32)
    c_v = np.array([k[1] for k in ck], dtype=np.float32).reshape(-1, 3)
    a_t = np.array([k[0] for k in ak], dtype=np.float32)
    a_v = np.array([k[1] for k in ak], dtype=np.float32)
    return c_t, c_v, a_t, a_v

def _sample_curve(keys_t: np.ndarray, keys_v: np.ndarray, t: float) -> float:
    nk = len(keys_t)
    t_clamped = max(keys_t[0], min(keys_t[-1], t))
    lo, hi = 0, nk - 2
    while lo < hi:
        mid = (lo + hi + 1) >> 1
        if keys_t[mid] <= t_clamped:
            lo = mid
        else:
            hi = mid - 1
    idx = lo if lo < nk - 1 else nk - 2
    dt = keys_t[idx + 1] - keys_t[idx]
    seg = (t_clamped - keys_t[idx]) / (dt if dt > 1e-10 else 1e-10)
    return keys_v[idx] + (keys_v[idx + 1] - keys_v[idx]) * seg

def _presample_curve(curve_cache: Optional[tuple], num_samples: int = 16) -> np.ndarray:
    out = np.ones(num_samples, dtype=np.float32)
    if curve_cache is None:
        return out
    keys_t, keys_v = curve_cache
    for i in range(num_samples):
        t = i / (num_samples - 1)
        out[i] = _sample_curve(keys_t, keys_v, t)
    return out

def _presample_gradient(grad_cache: Optional[tuple], num_samples: int = 16) -> tuple:
    cr = np.ones(num_samples, dtype=np.float32)
    cg = np.ones(num_samples, dtype=np.float32)
    cb = np.ones(num_samples, dtype=np.float32)
    alpha = np.ones(num_samples, dtype=np.float32)
    if grad_cache is None:
        return cr, cg, cb, alpha
    c_t, c_v, a_t, a_v = grad_cache
    nk = len(c_t)
    for i in range(num_samples):
        t = i / (num_samples - 1)
        t_clamped = max(c_t[0], min(c_t[-1], t))
        lo, hi = 0, nk - 2
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if c_t[mid] <= t_clamped:
                lo = mid
            else:
                hi = mid - 1
        idx = lo if lo < nk - 1 else nk - 2
        dt = c_t[idx + 1] - c_t[idx]
        seg = (t_clamped - c_t[idx]) / (dt if dt > 1e-10 else 1e-10)
        cr[i] = c_v[idx, 0] + (c_v[idx + 1, 0] - c_v[idx, 0]) * seg
        cg[i] = c_v[idx, 1] + (c_v[idx + 1, 1] - c_v[idx, 1]) * seg
        cb[i] = c_v[idx, 2] + (c_v[idx + 1, 2] - c_v[idx, 2]) * seg
    for i in range(num_samples):
        t = i / (num_samples - 1)
        t_clamped = max(a_t[0], min(a_t[-1], t))
        lo, hi = 0, len(a_t) - 2
        while lo < hi:
            mid = (lo + hi + 1) >> 1
            if a_t[mid] <= t_clamped:
                lo = mid
            else:
                hi = mid - 1
        idx = lo if lo < len(a_t) - 1 else len(a_t) - 2
        dt = a_t[idx + 1] - a_t[idx]
        seg = (t_clamped - a_t[idx]) / (dt if dt > 1e-10 else 1e-10)
        alpha[i] = a_v[idx] + (a_v[idx + 1] - a_v[idx]) * seg
    return cr, cg, cb, alpha


@ComponentRegistry.register
class ParticleSystem(Component):
    _icon = "ParticleSystem.png"
    _allow_multiple = False
    _gizmo_icon_color = (255, 160, 50)
    _gizmo_icon_label = "P"
    _gizmo_cache_attrs = ("shape_type", "shape_radius", "shape_arc", "shape_angle",
                          "shape_length", "shape_box")
    _gizmo_pass = "particle"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("duration", "Duration", FieldType.FLOAT, min_val=0.01, max_val=300.0, step=0.1, decimals=2),
            InspectorField("looping", "Looping", FieldType.BOOL),
            InspectorField("prewarm", "Prewarm", FieldType.BOOL),
            InspectorField("start_delay", "Start Delay", FieldType.FLOAT, min_val=0.0, max_val=60.0, step=0.1, decimals=2),
            InspectorField("start_lifetime", "Start Lifetime", FieldType.FLOAT, min_val=0.01, max_val=300.0, step=0.1, decimals=2),
            InspectorField("start_lifetime_min", "Lifetime Min", FieldType.FLOAT, min_val=0.0, max_val=300.0, step=0.1, decimals=2),
            InspectorField("start_lifetime_max", "Lifetime Max", FieldType.FLOAT, min_val=0.0, max_val=300.0, step=0.1, decimals=2),
            InspectorField("start_speed", "Start Speed", FieldType.FLOAT, min_val=0.0, max_val=1000.0, step=0.1, decimals=2),
            InspectorField("start_speed_min", "Speed Min", FieldType.FLOAT, min_val=0.0, max_val=1000.0, step=0.1, decimals=2),
            InspectorField("start_speed_max", "Speed Max", FieldType.FLOAT, min_val=0.0, max_val=1000.0, step=0.1, decimals=2),
            InspectorField("start_size", "Start Size", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.01, decimals=3),
            InspectorField("start_size_min", "Size Min", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.01, decimals=3),
            InspectorField("start_size_max", "Size Max", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.01, decimals=3),
            InspectorField("start_rotation", "Start Rotation", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
            InspectorField("flip_rotation", "Flip Rotation", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
            InspectorField("inherit_velocity", "Inherit Velocity", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01, decimals=2),
            InspectorField("gravity_modifier", "Gravity Modifier", FieldType.FLOAT, min_val=0.0, max_val=10.0, step=0.01, decimals=3),
            InspectorField("simulation_space", "Simulation Space", FieldType.ENUM, enum_class=SimulationSpace),
            InspectorField("max_particles", "Max Particles", FieldType.INT, min_val=1, max_val=10000000, step=1, decimals=0),
            InspectorField("rate_over_time", "Rate over Time", FieldType.FLOAT, min_val=0.0, max_val=100000.0, step=1.0, decimals=1),
            InspectorField("shape_type", "Shape", FieldType.ENUM, enum_class=ShapeType),
            InspectorField("shape_angle", "Shape Angle", FieldType.FLOAT, min_val=0.0, max_val=360.0, step=1.0, decimals=1),
            InspectorField("shape_radius", "Shape Radius", FieldType.FLOAT, min_val=0.0, max_val=100.0, step=0.1, decimals=2),
            InspectorField("render_mode", "Render Mode", FieldType.ENUM, enum_class=ParticleRenderMode),
            InspectorField("material_path", "Material", FieldType.RESOURCE_PATH, file_filter="Material (*.mat)"),
            InspectorField("sort_mode", "Sort Mode", FieldType.ENUM, enum_class=ParticleSortMode),
            InspectorField("sorting_fudge", "Sorting Fudge", FieldType.FLOAT, min_val=-100.0, max_val=100.0, step=0.1, decimals=2),
            InspectorField("flipbook_columns", "Flipbook Columns", FieldType.INT, min_val=1, max_val=100, step=1, decimals=0),
            InspectorField("flipbook_rows", "Flipbook Rows", FieldType.INT, min_val=1, max_val=100, step=1, decimals=0),
            InspectorField("flipbook_fps", "Flipbook FPS", FieldType.FLOAT, min_val=0.0, max_val=120.0, step=1.0, decimals=1),
            InspectorField("size_over_lifetime_enabled", "Size over Lifetime", FieldType.BOOL),
            InspectorField("size_over_lifetime_curve", "Size Curve", FieldType.CURVE, toggle_field="size_over_lifetime_enabled"),
            InspectorField("color_over_lifetime_enabled", "Color over Lifetime", FieldType.BOOL),
            InspectorField("color_over_lifetime_gradient", "Color Gradient", FieldType.GRADIENT, toggle_field="color_over_lifetime_enabled"),
            InspectorField("rotation_over_lifetime_enabled", "Rotation over Lifetime", FieldType.BOOL),
            InspectorField("velocity_over_lifetime_enabled", "Velocity over Lifetime", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.duration: float = 5.0
        self.looping: bool = True
        self.prewarm: bool = False
        self.start_delay: float = 0.0
        self.start_lifetime: float = 5.0
        self.start_lifetime_min: float = 0.0
        self.start_lifetime_max: float = 0.0
        self.start_speed: float = 5.0
        self.start_speed_min: float = 0.0
        self.start_speed_max: float = 0.0
        self.start_size: float = 1.0
        self.start_size_min: float = 0.0
        self.start_size_max: float = 0.0
        self.start_color: list[float] = None
        self.start_rotation: float = 0.0
        self.flip_rotation: float = 0.0
        self.inherit_velocity: float = 0.0
        self.gravity_modifier: float = 0.0
        self.simulation_space: SimulationSpace = SimulationSpace.LOCAL
        self.max_particles: int = 1000
        self.rate_over_time: float = 10.0
        self.rate_over_distance: float = 0.0
        self.bursts: list[dict] = None
        self.shape_type: ShapeType = ShapeType.CONE
        self.shape_angle: float = 25.0
        self.shape_radius: float = 1.0
        self.shape_length: float = 5.0
        self.shape_box: list[float] = None
        self.shape_arc: float = 360.0
        self.render_mode: ParticleRenderMode = ParticleRenderMode.BILLBOARD
        self.material_path: str = ""
        self.texture_path: str = ""
        self.sort_mode: ParticleSortMode = ParticleSortMode.NONE
        self.sorting_fudge: float = 0.0
        self.flipbook_columns: int = 1
        self.flipbook_rows: int = 1
        self.flipbook_fps: float = 0.0
        self.size_over_lifetime_enabled: bool = False
        self.size_over_lifetime_curve: list[list[float]] = None
        self.size_over_lifetime_curve_x: list[list[float]] = None
        self.size_over_lifetime_curve_y: list[list[float]] = None
        self.color_over_lifetime_enabled: bool = False
        self.color_over_lifetime_gradient: dict = None
        self.rotation_over_lifetime_enabled: bool = False
        self.rotation_over_lifetime_angular_velocity: float = 0.0
        self.velocity_over_lifetime_enabled: bool = False
        self.velocity_over_lifetime_linear: list[float] = None
        self.velocity_over_lifetime_orbital: list[float] = None
        self._time: float = 0.0
        self._particle_count: int = 0
        self._alive_count: int = 0
        self._emit_accum: float = 0.0
        self._distance_accum: float = 0.0
        self._prewarmed: bool = False
        self._started: bool = False
        self._stopped: bool = False
        self._burst_index: int = 0
        self._prev_position: Vec3 = None
        self._last_delta_pos: Vec3 = Vec3.zero()
        self._emitter_velocity: np.ndarray = None
        self._local_to_world: np.ndarray = None
        self._local_position: np.ndarray = None
        self._particles: Optional[np.ndarray] = None
        self._free_stack: Optional[np.ndarray] = None
        self._free_top: int = 0
        self._last_dt: float = 0.016
        self._alive_mask: Optional[np.ndarray] = None
        self._start_color_arr: Optional[np.ndarray] = None
        self._curve_cache_x: Optional[tuple] = None
        self._curve_cache_y: Optional[tuple] = None
        self._gradient_cache: Optional[tuple] = None

    def _invalidate_caches(self):
        self._curve_cache_x = None
        self._curve_cache_y = None
        self._gradient_cache = None
        self._start_color_arr = None

    _INITIAL_CAPACITY = 1024

    def _ensure_arrays(self):
        if self._particles is not None and len(self._particles) >= 1:
            return
        n = min(self.max_particles, self._INITIAL_CAPACITY)
        self._particles = np.zeros(n, dtype=PARTICLE_DTYPE)
        self._particles['meta'][:, 3] = 0.0
        self._free_stack = np.arange(n, dtype=np.int32)
        self._free_top = n
        self._alive_mask = np.zeros(n, dtype=np.bool_)
        self._particle_count = 0
        self._alive_count = 0

    def _grow_capacity(self):
        old_cap = len(self._particles) if self._particles is not None else 0
        new_cap = min(self.max_particles, max(old_cap * 2, self._INITIAL_CAPACITY))
        if new_cap <= old_cap:
            return
        new_particles = np.zeros(new_cap, dtype=PARTICLE_DTYPE)
        new_particles[:old_cap] = self._particles
        new_particles['meta'][old_cap:, 3] = 0.0
        self._particles = new_particles
        new_free = np.arange(old_cap, new_cap, dtype=np.int32)
        self._free_stack = np.concatenate([self._free_stack, new_free])
        self._free_top += new_cap - old_cap
        new_mask = np.zeros(new_cap, dtype=np.bool_)
        new_mask[:old_cap] = self._alive_mask
        self._alive_mask = new_mask

    def on_start(self):
        self._ensure_arrays()

    def on_enable(self):
        if not self._started:
            self._time = -self.start_delay
            self._started = True
            self._stopped = False
            self._emit_accum = 0.0
            self._burst_index = 0
            self._prewarmed = False
            self._ensure_arrays()

    def on_disable(self):
        self._stop()

    def _stop(self):
        self._stopped = True

    def _ranged(self, lo_attr, hi_attr, base_attr, n):
        lo, hi = getattr(self, lo_attr), getattr(self, hi_attr)
        if lo > 0 or hi > 0:
            lo = lo if lo > 0 else getattr(self, base_attr)
            hi = hi if hi > 0 else getattr(self, base_attr)
            lo, hi = min(lo, hi), max(lo, hi)
            return lo + (hi - lo) * np.random.random(n).astype(np.float32)
        return np.full(n, getattr(self, base_attr), dtype=np.float32)

    def _emit(self, count: int):
        self._ensure_arrays()
        if count > self._free_top:
            self._grow_capacity()
        n = min(count, self._free_top)
        if n == 0:
            return
        self._free_top -= n
        idx = self._free_stack[self._free_top:self._free_top + n].copy()
        idx_arr = idx
        shape = self.shape_type
        radius = self.shape_radius
        half_len = self.shape_length * 0.5
        arc_rad = math.radians(self.shape_arc)
        if self._start_color_arr is None:
            self._start_color_arr = np.array(self.start_color or [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        rng = np.random.random(n).astype(np.float32)
        rng2 = np.random.random(n).astype(np.float32)
        rng3 = np.random.random(n).astype(np.float32)
        if shape == ShapeType.CONE:
            angle_rad = math.radians(self.shape_angle)
            base_r = radius * (1.0 - rng * math.tan(angle_rad))
            ang = rng2 * arc_rad
            r = base_r * np.sqrt(rng3)
            x = r * np.cos(ang)
            z = r * np.sin(ang)
            y = -half_len + np.random.random(n).astype(np.float32) * self.shape_length
            pos = np.column_stack([x, y, z])
            dv = np.column_stack([x, y + half_len, z])
            dlen = np.maximum(np.linalg.norm(dv, axis=1), 1e-8)
            velocity = dv / dlen[:, None]
        elif shape == ShapeType.SPHERE:
            theta = rng * arc_rad
            phi = rng2 * (math.pi * 2.0)
            r = radius * (rng3 ** (1.0/3.0))
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            pos = np.column_stack([x, y, z])
            d = np.maximum(np.linalg.norm(pos, axis=1), 1e-8)
            velocity = pos / d[:, None]
        elif shape == ShapeType.HEMISPHERE:
            theta = rng * min(arc_rad, math.pi * 0.5)
            phi = rng2 * (math.pi * 2.0)
            r = radius * (rng3 ** (1.0/3.0))
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.cos(theta)
            z = r * np.sin(theta) * np.sin(phi)
            pos = np.column_stack([x, y, z])
            d = np.maximum(np.linalg.norm(pos, axis=1), 1e-8)
            velocity = pos / d[:, None]
        elif shape == ShapeType.BOX:
            bx = self.shape_box or [1, 1, 1]
            pos = np.column_stack([
                (rng - 0.5) * bx[0],
                (rng2 - 0.5) * bx[1],
                (rng3 - 0.5) * bx[2],
            ])
            velocity = np.broadcast_to(np.array([[0.0, 1.0, 0.0]], dtype=np.float32), (n, 3)).copy()
        else:
            r = radius * np.sqrt(rng)
            theta = rng2 * arc_rad
            x = r * np.cos(theta)
            z = r * np.sin(theta)
            pos = np.column_stack([x, np.zeros(n, dtype=np.float32), z])
            d = np.maximum(np.sqrt(x*x + z*z), 1e-8)
            velocity = np.column_stack([x/d, np.zeros(n), z/d])
        pos = pos.astype(np.float32)
        velocity = velocity.astype(np.float32)
        if self._local_to_world is not None:
            R = self._local_to_world
            RT = R.T
            pos = pos @ RT
            velocity = velocity @ RT
            if self._local_position is not None:
                pos += self._local_position
        if self.inherit_velocity > 0 and self._emitter_velocity is not None:
            velocity += (self.inherit_velocity * self._emitter_velocity)
        lifetimes = self._ranged('start_lifetime_min', 'start_lifetime_max', 'start_lifetime', n)
        if n > 1:
            phi = 0.6180339887498949
            offsets = np.mod(np.arange(n, dtype=np.float32) * phi, 1.0)
            lifetimes *= (0.925 + 0.15 * offsets)
        size_val = self._ranged('start_size_min', 'start_size_max', 'start_size', n)
        sp_min, sp_max = self.start_speed_min, self.start_speed_max
        if sp_min > 0 or sp_max > 0:
            sp_lo = sp_min if sp_min > 0 else self.start_speed
            sp_hi = sp_max if sp_max > 0 else self.start_speed
            sp_lo, sp_hi = min(sp_lo, sp_hi), max(sp_lo, sp_hi)
            speed = sp_lo + (sp_hi - sp_lo) * np.random.random(n).astype(np.float32)
            velocity *= (speed / max(self.start_speed, 1e-6))[:, None]
        else:
            velocity *= self.start_speed
        rot = math.radians(self.start_rotation) + (np.random.random(n).astype(np.float32) - 0.5) * math.radians(self.flip_rotation)
        sc = self._start_color_arr
        p = self._particles
        p['position'][idx_arr, :3] = pos
        p['velocity'][idx_arr, :3] = velocity
        p['color'][idx_arr] = sc
        p['meta'][idx_arr, 0] = size_val
        p['meta'][idx_arr, 1] = size_val
        p['meta'][idx_arr, 2] = rot
        p['meta'][idx_arr, 3] = 1.0
        p['lifetime'][idx_arr, 0] = lifetimes
        p['lifetime'][idx_arr, 1] = lifetimes
        p['start_color'][idx_arr] = sc
        p['start_meta'][idx_arr, 0] = size_val
        p['start_meta'][idx_arr, 1] = size_val
        p['start_meta'][idx_arr, 2] = rot
        self._alive_mask[idx_arr] = True
        self._particle_count += n
        self._alive_count += n

    def _emit_particles(self, dt: float, delta_pos: Vec3):
        if self._stopped:
            return
        if not self._prewarmed:
            self._prewarmed = True
            if self.rate_over_time > 0 and self.start_lifetime > 0:
                prewarm_n = int(self.rate_over_time * self.start_lifetime * 1.075)
                if prewarm_n > 0:
                    self._grow_capacity()
                    self._emit_accum = max(self._emit_accum, prewarm_n)
        if self.rate_over_time > 0:
            self._emit_accum += dt * self.rate_over_time
            count = int(self._emit_accum)
            if count > 0:
                self._emit(count)
                self._emit_accum -= count
        if self.rate_over_distance > 0:
            dist = delta_pos.length()
            self._distance_accum += dist
            n_units = int(self._distance_accum)
            if n_units > 0:
                self._emit(int(self.rate_over_distance) * n_units)
                self._distance_accum -= n_units
        if self.bursts:
            for burst in self.bursts:
                if abs(self._time - burst.get("time", 0.0)) < dt * 1.5:
                    self._emit(burst.get("count", 1))

    def on_update(self, dt: float):
        if not self.enabled or self._stopped:
            return
        if not self._started:
            self._time = -self.start_delay
            self._started = True
            self._stopped = False
            self._ensure_arrays()
        self._time += dt
        if self._time < 0:
            return
        t = self.transform
        if t is None:
            return
        prev = self._prev_position
        current_pos = t.position
        if prev is None:
            delta_pos = Vec3.zero()
            self._emitter_velocity = None
        else:
            delta_pos = current_pos - prev
            inv_dt = 1.0 / max(dt, 1e-6)
            self._emitter_velocity = np.array([delta_pos.x * inv_dt, delta_pos.y * inv_dt, delta_pos.z * inv_dt], dtype=np.float32)
        self._prev_position = current_pos
        self._last_dt = dt
        self._last_delta_pos = delta_pos
        if self.simulation_space == SimulationSpace.LOCAL:
            wm = t.world_matrix._d
            self._local_to_world = wm[:3, :3].astype(np.float32)
            self._local_position = np.array([current_pos.x, current_pos.y, current_pos.z], dtype=np.float32)
        else:
            self._local_to_world = None
            self._local_position = None
        self._emit_particles(dt, delta_pos)
        if self._time >= self.duration:
            if self.looping:
                self._time = 0.0
                self._burst_index = 0
            else:
                self._stopped = True

    def replenish_free_list(self, dead_indices: np.ndarray):
        n = len(dead_indices)
        if n == 0:
            return
        self._free_stack[self._free_top:self._free_top + n] = dead_indices.astype(np.int32)
        self._free_top += n
        for idx in dead_indices:
            self._alive_mask[idx] = False
        self._particle_count -= n
        self._alive_count -= n

    def get_compute_params(self, dt: float, delta_pos: Vec3) -> dict:
        dx = delta_pos.x if self.simulation_space == SimulationSpace.LOCAL else 0.0
        dy = delta_pos.y if self.simulation_space == SimulationSpace.LOCAL else 0.0
        dz = delta_pos.z if self.simulation_space == SimulationSpace.LOCAL else 0.0
        params = {
            'dt': dt,
            'gravity': -9.81 * self.gravity_modifier,
            'simulation_space': self.simulation_space.value,
            'emitter_delta': (dx, dy, dz),
            'num_particles': len(self._particles) if self._particles is not None else 0,
            'size_enabled': self.size_over_lifetime_enabled,
            'color_enabled': self.color_over_lifetime_enabled,
            'rotation_enabled': self.rotation_over_lifetime_enabled,
            'velocity_enabled': self.velocity_over_lifetime_enabled,
        }
        if self.size_over_lifetime_enabled:
            if self._curve_cache_x is None:
                cx = self.size_over_lifetime_curve_x or self.size_over_lifetime_curve or [(0,1),(1,1)]
                self._curve_cache_x = _build_curve_cache(cx)
            params['size_curve'] = _presample_curve(self._curve_cache_x)
        if self.color_over_lifetime_enabled:
            if self._gradient_cache is None:
                grad = self.color_over_lifetime_gradient or {
                    "alpha_keys": [(0,1),(1,0)],
                    "color_keys": [(0,[1,1,1]),(1,[1,1,1])]
                }
                self._gradient_cache = _build_gradient_cache(grad)
            cr, cg, cb, ca = _presample_gradient(self._gradient_cache)
            params['color_curve_r'] = cr
            params['color_curve_g'] = cg
            params['color_curve_b'] = cb
            params['alpha_curve'] = ca
        if self.rotation_over_lifetime_enabled:
            params['angular_velocity'] = math.radians(self.rotation_over_lifetime_angular_velocity)
        if self.velocity_over_lifetime_enabled:
            params['vel_linear'] = tuple(self.velocity_over_lifetime_linear or [0,0,0])
            params['vel_orbital'] = tuple(self.velocity_over_lifetime_orbital or [0,0,0])
        return params

    def get_particle_count(self) -> int:
        return self._particle_count

    def get_alive_count(self) -> int:
        return self._alive_count

    def is_playing(self) -> bool:
        return not self._stopped and self._started

    def is_paused(self) -> bool:
        return self._stopped and self._started

    def is_stopped(self) -> bool:
        return self._stopped

    def play(self):
        if self._started and not self._stopped:
            return
        self._started = True
        self._stopped = False
        self._time = -self.start_delay
        self._emit_accum = 0.0
        self._burst_index = 0
        self._prewarmed = False

    def pause(self):
        self._stopped = True

    def stop(self):
        self._stopped = True
        self._started = False
        self._time = 0.0
        self._emit_accum = 0.0
        self._burst_index = 0
        self._prewarmed = False
        if self._particles is not None:
            n = len(self._particles)
            self._particles['meta'][:, 3] = 0.0
            self._alive_count = 0
            self._particle_count = 0
            self._free_stack = np.arange(n, dtype=np.int32)
            self._free_top = n
            self._alive_mask[:] = False

    def clear(self):
        if self._particles is not None:
            n = len(self._particles)
            self._particles['meta'][:, 3] = 0.0
            self._alive_count = 0
            self._particle_count = 0
            self._free_stack = np.arange(n, dtype=np.int32)
            self._free_top = n
            self._alive_mask[:] = False
        self._prewarmed = False

    def gizmo_lines(self, color: list[float] | None = None) -> list[tuple[Vec3, Vec3, list[float]]]:
        if color is None:
            color = [0.8, 0.3, 0.8, 0.6]
        tr = self.transform
        if not tr:
            return []
        world_matrix = tr.world_matrix._d
        world_pos = tr.position
        R = world_matrix[:3, :3]
        lines: list[tuple[Vec3, Vec3, list[float]]] = []
        def local_to_world(v: Vec3) -> Vec3:
            vx, vy, vz = v.x, v.y, v.z
            return world_pos + Vec3(
                vx * R[0, 0] + vy * R[1, 0] + vz * R[2, 0],
                vx * R[0, 1] + vy * R[1, 1] + vz * R[2, 1],
                vx * R[0, 2] + vy * R[1, 2] + vz * R[2, 2],
            )
        shape = self.shape_type
        radius = self.shape_radius
        arc_rad = math.radians(self.shape_arc)
        segments = 24
        if shape == ShapeType.CONE:
            angle_rad = math.radians(self.shape_angle)
            half_len = self.shape_length * 0.5
            base_r = radius
            apex = local_to_world(Vec3(0, -half_len, 0))
            pts = []
            for i in range(segments + 1):
                theta = 2.0 * math.pi * i / segments
                pt = local_to_world(Vec3(math.cos(theta) * base_r, half_len, math.sin(theta) * base_r))
                pts.append(pt)
            for i in range(segments):
                lines.append((pts[i], pts[i + 1], color))
            lines.append((apex, pts[0], color))
            lines.append((apex, pts[segments // 4], color))
            lines.append((apex, pts[segments // 2], color))
            lines.append((apex, pts[3 * segments // 4], color))
            inner_r = base_r * (1.0 - math.tan(angle_rad)) if math.tan(angle_rad) < 1.0 else 0.0
            if inner_r > 0.01:
                inner_pts = []
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    pt = local_to_world(Vec3(math.cos(theta) * inner_r, -half_len, math.sin(theta) * inner_r))
                    inner_pts.append(pt)
                for i in range(segments):
                    lines.append((inner_pts[i], inner_pts[i + 1], color))
        elif shape == ShapeType.SPHERE:
            for axis_idx in range(3):
                pts = []
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    if axis_idx == 0:
                        pt = Vec3(0, math.cos(theta) * radius, math.sin(theta) * radius)
                    elif axis_idx == 1:
                        pt = Vec3(math.cos(theta) * radius, 0, math.sin(theta) * radius)
                    else:
                        pt = Vec3(math.cos(theta) * radius, math.sin(theta) * radius, 0)
                    pts.append(local_to_world(pt))
                for i in range(segments):
                    lines.append((pts[i], pts[i + 1], color))
            if arc_rad < math.pi * 2.0 - 0.01:
                arc_pts = []
                for i in range(segments + 1):
                    theta = arc_rad * i / segments
                    pt = Vec3(math.sin(theta) * radius, 0, math.cos(theta) * radius)
                    arc_pts.append(local_to_world(pt))
                for i in range(segments):
                    lines.append((arc_pts[i], arc_pts[i + 1], [0.8, 0.8, 0.2, 0.6]))
        elif shape == ShapeType.HEMISPHERE:
            ring_pts = []
            for i in range(segments + 1):
                theta = 2.0 * math.pi * i / segments
                pt = Vec3(math.cos(theta) * radius, 0, math.sin(theta) * radius)
                ring_pts.append(local_to_world(pt))
            for i in range(segments):
                lines.append((ring_pts[i], ring_pts[i + 1], color))
            for axis_idx in [0, 2]:
                pts = []
                for i in range(segments // 2 + 1):
                    theta = math.pi * i / segments
                    if axis_idx == 0:
                        pt = Vec3(0, math.cos(theta) * radius, math.sin(theta) * radius)
                    else:
                        pt = Vec3(math.cos(theta) * radius, math.sin(theta) * radius, 0)
                    if pt.y < 0:
                        continue
                    pts.append(local_to_world(pt))
                for i in range(len(pts) - 1):
                    lines.append((pts[i], pts[i + 1], color))
            if arc_rad < math.pi * 2.0 - 0.01:
                arc_pts = []
                for i in range(segments + 1):
                    theta = arc_rad * i / segments
                    pt = Vec3(math.sin(theta) * radius, 0, math.cos(theta) * radius)
                    arc_pts.append(local_to_world(pt))
                for i in range(segments):
                    lines.append((arc_pts[i], arc_pts[i + 1], [0.8, 0.8, 0.2, 0.6]))
        elif shape == ShapeType.BOX:
            bx = self.shape_box or [1, 1, 1]
            hx, hy, hz = bx[0] * 0.5, bx[1] * 0.5, bx[2] * 0.5
            corners_local = [
                Vec3(-hx, -hy, -hz), Vec3(hx, -hy, -hz),
                Vec3(hx, hy, -hz), Vec3(-hx, hy, -hz),
                Vec3(-hx, -hy, hz), Vec3(hx, -hy, hz),
                Vec3(hx, hy, hz), Vec3(-hx, hy, hz),
            ]
            corners = [local_to_world(v) for v in corners_local]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for a, b in edges:
                lines.append((corners[a], corners[b], color))
        else:
            pts = []
            for i in range(segments + 1):
                theta = arc_rad * i / segments
                pt = Vec3(math.cos(theta) * radius, 0, math.sin(theta) * radius)
                pts.append(local_to_world(pt))
            for i in range(segments):
                lines.append((pts[i], pts[i + 1], color))
            lines.append((world_pos, local_to_world(Vec3(0, 0, 0)), color))
        return lines

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "duration": self.duration,
            "looping": self.looping,
            "prewarm": self.prewarm,
            "start_delay": self.start_delay,
            "start_lifetime": self.start_lifetime,
            "start_lifetime_min": self.start_lifetime_min,
            "start_lifetime_max": self.start_lifetime_max,
            "start_speed": self.start_speed,
            "start_speed_min": self.start_speed_min,
            "start_speed_max": self.start_speed_max,
            "start_size": self.start_size,
            "start_size_min": self.start_size_min,
            "start_size_max": self.start_size_max,
            "start_color": self.start_color or [1,1,1,1],
            "start_rotation": self.start_rotation,
            "flip_rotation": self.flip_rotation,
            "inherit_velocity": self.inherit_velocity,
            "gravity_modifier": self.gravity_modifier,
            "simulation_space": self.simulation_space.value,
            "max_particles": self.max_particles,
            "rate_over_time": self.rate_over_time,
            "rate_over_distance": self.rate_over_distance,
            "bursts": self.bursts or [],
            "shape_type": self.shape_type.value,
            "shape_angle": self.shape_angle,
            "shape_radius": self.shape_radius,
            "shape_length": self.shape_length,
            "shape_box": self.shape_box or [1,1,1],
            "shape_arc": self.shape_arc,
            "render_mode": self.render_mode.value,
            "material_path": self.material_path,
            "texture_path": self.texture_path,
            "sort_mode": self.sort_mode.value,
            "sorting_fudge": self.sorting_fudge,
            "flipbook_columns": self.flipbook_columns,
            "flipbook_rows": self.flipbook_rows,
            "flipbook_fps": self.flipbook_fps,
            "size_over_lifetime_enabled": self.size_over_lifetime_enabled,
            "size_over_lifetime_curve": self.size_over_lifetime_curve or [[0,1],[1,1]],
            "size_over_lifetime_curve_x": self.size_over_lifetime_curve_x or [[0,1],[1,1]],
            "size_over_lifetime_curve_y": self.size_over_lifetime_curve_y or [[0,1],[1,1]],
            "color_over_lifetime_enabled": self.color_over_lifetime_enabled,
            "color_over_lifetime_gradient": self.color_over_lifetime_gradient if self.color_over_lifetime_gradient is not None else {
                "alpha_keys": [[0, 1], [1, 0]],
                "color_keys": [[0, [1, 1, 1]], [1, [1, 1, 1]]]
            },
            "rotation_over_lifetime_enabled": self.rotation_over_lifetime_enabled,
            "rotation_over_lifetime_angular_velocity": self.rotation_over_lifetime_angular_velocity,
            "velocity_over_lifetime_enabled": self.velocity_over_lifetime_enabled,
            "velocity_over_lifetime_linear": self.velocity_over_lifetime_linear or [0,0,0],
            "velocity_over_lifetime_orbital": self.velocity_over_lifetime_orbital or [0,0,0],
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ParticleSystem:
        ps = cls()
        ps.enabled = data.get("enabled", True)
        ps.duration = data.get("duration", 5.0)
        ps.looping = data.get("looping", True)
        ps.prewarm = data.get("prewarm", False)
        ps.start_delay = data.get("start_delay", 0.0)
        ps.start_lifetime = data.get("start_lifetime", 5.0)
        ps.start_lifetime_min = data.get("start_lifetime_min", 0.0)
        ps.start_lifetime_max = data.get("start_lifetime_max", 0.0)
        ps.start_speed = data.get("start_speed", 5.0)
        ps.start_speed_min = data.get("start_speed_min", 0.0)
        ps.start_speed_max = data.get("start_speed_max", 0.0)
        ps.start_size = data.get("start_size", 1.0)
        ps.start_size_min = data.get("start_size_min", 0.0)
        ps.start_size_max = data.get("start_size_max", 0.0)
        ps.start_color = data.get("start_color", [1,1,1,1])
        ps.start_rotation = data.get("start_rotation", 0.0)
        ps.flip_rotation = data.get("flip_rotation", 0.0)
        ps.inherit_velocity = data.get("inherit_velocity", 0.0)
        ps.gravity_modifier = data.get("gravity_modifier", 0.0)
        ps.simulation_space = SimulationSpace(data.get("simulation_space", "local"))
        ps.max_particles = data.get("max_particles", 1000)
        ps.rate_over_time = data.get("rate_over_time", 10.0)
        ps.rate_over_distance = data.get("rate_over_distance", 0.0)
        ps.bursts = data.get("bursts", [])
        ps.shape_type = ShapeType(data.get("shape_type", "cone"))
        ps.shape_angle = data.get("shape_angle", 25.0)
        ps.shape_radius = data.get("shape_radius", 1.0)
        ps.shape_length = data.get("shape_length", 5.0)
        ps.shape_box = data.get("shape_box", [1,1,1])
        ps.shape_arc = data.get("shape_arc", 360.0)
        ps.render_mode = ParticleRenderMode(data.get("render_mode", "billboard"))
        ps.material_path = data.get("material_path", "") or ""
        ps.texture_path = data.get("texture_path", "") or ""
        ps.sort_mode = ParticleSortMode(data.get("sort_mode", "none"))
        ps.sorting_fudge = data.get("sorting_fudge", 0.0)
        ps.flipbook_columns = data.get("flipbook_columns", 1)
        ps.flipbook_rows = data.get("flipbook_rows", 1)
        ps.flipbook_fps = data.get("flipbook_fps", 0.0)
        ps.size_over_lifetime_enabled = data.get("size_over_lifetime_enabled", False)
        ps.size_over_lifetime_curve = data.get("size_over_lifetime_curve", [[0,1],[1,1]])
        ps.size_over_lifetime_curve_x = data.get("size_over_lifetime_curve_x", None)
        ps.size_over_lifetime_curve_y = data.get("size_over_lifetime_curve_y", None)
        ps.color_over_lifetime_enabled = data.get("color_over_lifetime_enabled", False)
        ps.color_over_lifetime_gradient = data.get("color_over_lifetime_gradient", {
            "alpha_keys": [[0,1],[1,0]],
            "color_keys": [[0,[1,1,1]],[1,[1,1,1]]]
        })
        ps.rotation_over_lifetime_enabled = data.get("rotation_over_lifetime_enabled", False)
        ps.rotation_over_lifetime_angular_velocity = data.get("rotation_over_lifetime_angular_velocity", 0.0)
        ps.velocity_over_lifetime_enabled = data.get("velocity_over_lifetime_enabled", False)
        ps.velocity_over_lifetime_linear = data.get("velocity_over_lifetime_linear", [0,0,0])
        ps.velocity_over_lifetime_orbital = data.get("velocity_over_lifetime_orbital", [0,0,0])
        return ps

    def on_destroy(self):
        self.stop()
