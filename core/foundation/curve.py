# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import numpy as np


class TangentMode(Enum):
    FREE = "free"
    LINEAR = "linear"
    CONSTANT = "constant"
    SMOOTH = "smooth"


@dataclass
class CurveKey:
    time: float
    value: float
    in_tangent: float = 0.0
    out_tangent: float = 0.0
    tangent_mode: TangentMode = TangentMode.SMOOTH


@dataclass
class Curve:
    keys: list[CurveKey] = field(default_factory=list)
    pre_wrap: str = "clamp"
    post_wrap: str = "clamp"

    def add_key(self, time: float, value: float) -> CurveKey:
        k = CurveKey(time=time, value=value)
        self.keys.append(k)
        self.keys.sort(key=lambda x: x.time)
        self._auto_smooth()
        return k

    def remove_key(self, key: CurveKey):
        if key in self.keys:
            self.keys.remove(key)

    def _auto_smooth(self):
        n = len(self.keys)
        if n < 2:
            return
        for i, k in enumerate(self.keys):
            if k.tangent_mode != TangentMode.SMOOTH:
                continue
            if i == 0:
                k.in_tangent = 0
                k.out_tangent = self._compute_slope(i, 1)
            elif i == n - 1:
                k.in_tangent = self._compute_slope(i - 1, i)
                k.out_tangent = 0
            else:
                chord = (self.keys[i + 1].value - self.keys[i - 1].value) / max(self.keys[i + 1].time - self.keys[i - 1].time, 1e-10)
                k.in_tangent = chord
                k.out_tangent = chord

    def _compute_slope(self, i: int, j: int) -> float:
        a, b = self.keys[i], self.keys[j]
        dt = b.time - a.time
        if dt < 1e-10:
            return 0.0
        return (b.value - a.value) / dt

    def evaluate(self, time: float) -> float:
        if not self.keys:
            return 0.0
        if len(self.keys) == 1:
            return self.keys[0].value
        if time <= self.keys[0].time:
            return self.keys[0].value
        if time >= self.keys[-1].time:
            return self.keys[-1].value
        idx = 0
        for i in range(len(self.keys) - 1):
            if self.keys[i].time <= time <= self.keys[i + 1].time:
                idx = i
                break
        k0 = self.keys[idx]
        k1 = self.keys[idx + 1]
        t = (time - k0.time) / max(k1.time - k0.time, 1e-10)
        if k0.tangent_mode == TangentMode.CONSTANT:
            return k0.value
        if k0.tangent_mode == TangentMode.LINEAR:
            return k0.value + (k1.value - k0.value) * t
        dt = k1.time - k0.time
        m0 = k0.out_tangent * dt
        m1 = k1.in_tangent * dt
        t2 = t * t
        t3 = t2 * t
        return (2 * t3 - 3 * t2 + 1) * k0.value + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * k1.value + (t3 - t2) * m1

    def evaluate_array(self, times: np.ndarray) -> np.ndarray:
        if not self.keys:
            return np.zeros_like(times)
        if len(self.keys) == 1:
            return np.full_like(times, self.keys[0].value)
        try:
            from core._curve_batch import evaluate_curve_batch
            nk = len(self.keys)
            key_times = np.empty(nk, dtype=np.float64)
            key_values = np.empty(nk, dtype=np.float64)
            key_in = np.empty(nk, dtype=np.float64)
            key_out = np.empty(nk, dtype=np.float64)
            key_modes = np.empty(nk, dtype=np.int32)
            for i, k in enumerate(self.keys):
                key_times[i] = k.time
                key_values[i] = k.value
                key_in[i] = k.in_tangent
                key_out[i] = k.out_tangent
                key_modes[i] = 0 if k.tangent_mode == TangentMode.FREE else (
                    1 if k.tangent_mode == TangentMode.LINEAR else (
                    2 if k.tangent_mode == TangentMode.CONSTANT else 0))
            return evaluate_curve_batch(
                np.asarray(times, dtype=np.float64),
                key_times, key_values, key_in, key_out, key_modes,
                self.pre_wrap, self.post_wrap,
            )
        except ImportError:
            out = np.zeros_like(times)
            for i, t in enumerate(times):
                out[i] = self.evaluate(t)
            return out

    def to_list(self) -> list[list[float]]:
        return [[k.time, k.value] for k in self.keys]

    @classmethod
    def from_list(cls, data: list[list[float]]) -> Curve:
        c = cls()
        for item in data:
            if len(item) >= 2:
                c.add_key(float(item[0]), float(item[1]))
        return c

    def to_dict(self) -> dict:
        return {
            "keys": [
                {
                    "time": k.time,
                    "value": k.value,
                    "in_tangent": k.in_tangent,
                    "out_tangent": k.out_tangent,
                    "tangent_mode": k.tangent_mode.value
                }
                for k in self.keys
            ],
            "pre_wrap": self.pre_wrap,
            "post_wrap": self.post_wrap,
        }

    def copy(self) -> "Curve":
        return Curve.from_dict(self.to_dict())

    def find_key(self, time: float) -> Optional[CurveKey]:
        for k in self.keys:
            if abs(k.time - float(time)) < 1e-6:
                return k
        return None

    @classmethod
    def from_dict(cls, data: dict) -> Curve:
        c = cls()
        c.pre_wrap = data.get("pre_wrap", "clamp")
        c.post_wrap = data.get("post_wrap", "clamp")
        for kd in data.get("keys", []):
            k = CurveKey(
                time=kd["time"],
                value=kd["value"],
                in_tangent=kd.get("in_tangent", 0.0),
                out_tangent=kd.get("out_tangent", 0.0),
                tangent_mode=TangentMode(kd.get("tangent_mode", "smooth")),
            )
            c.keys.append(k)
        return c


@dataclass
class QuaternionKey:
    time: float
    value: tuple  # (x, y, z, w)


def _quat_slerp(a: tuple, b: tuple, t: float) -> tuple:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    dot = ax * bx + ay * by + az * bz + aw * bw
    if dot < 0.0:
        bx, by, bz, bw = -bx, -by, -bz, -bw
        dot = -dot
    dot = min(max(dot, -1.0), 1.0)
    if dot > 0.9995:
        k = t
        result = (ax + (bx - ax) * k, ay + (by - ay) * k,
                  az + (bz - az) * k, aw + (bw - aw) * k)
        return _quat_normalize(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return _quat_normalize((
        s0 * ax + s1 * bx, s0 * ay + s1 * by,
        s0 * az + s1 * bz, s0 * aw + s1 * bw,
    ))


def _quat_normalize(v: tuple) -> tuple:
    x, y, z, w = v
    length = (x * x + y * y + z * z + w * w) ** 0.5
    if length < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / length, y / length, z / length, w / length)


def _to_quat_tuple(value) -> tuple:
    if isinstance(value, tuple) and len(value) == 4:
        return value
    try:
        from core.maths.math3d import Quat
    except ImportError:
        Quat = None
    if Quat is not None and isinstance(value, Quat):
        return (value.x, value.y, value.z, value.w)
    seq = tuple(value)
    if len(seq) == 4:
        return (float(seq[0]), float(seq[1]), float(seq[2]), float(seq[3]))
    raise ValueError(f"cannot convert to quaternion: {value!r}")


class QuaternionCurve:
    """Unity-style rotation curve: keys hold quaternions, slerp interpolated."""

    def __init__(self):
        self.keys: list[QuaternionKey] = []

    def add_key(self, time: float, value) -> QuaternionKey:
        k = QuaternionKey(float(time), _to_quat_tuple(value))
        self.keys.append(k)
        self.keys.sort(key=lambda x: x.time)
        return k

    def clear(self):
        self.keys = []

    def evaluate(self, time: float) -> tuple:
        if not self.keys:
            return (0.0, 0.0, 0.0, 1.0)
        if time <= self.keys[0].time:
            return self.keys[0].value
        if time >= self.keys[-1].time:
            return self.keys[-1].value
        for i in range(len(self.keys) - 1):
            if self.keys[i].time <= time <= self.keys[i + 1].time:
                a = self.keys[i]
                b = self.keys[i + 1]
                if b.time - a.time < 1e-10:
                    return a.value
                t = (time - a.time) / (b.time - a.time)
                return _quat_slerp(a.value, b.value, t)
        return self.keys[-1].value

    def keys_dict(self) -> list[dict]:
        return [{"time": k.time, "value": list(k.value)} for k in self.keys]

    @classmethod
    def from_keys_dict(cls, items: list[dict]) -> "QuaternionCurve":
        c = cls()
        for item in items:
            c.add_key(float(item["time"]), item["value"])
        return c
