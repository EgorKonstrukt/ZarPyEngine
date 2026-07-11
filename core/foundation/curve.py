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
