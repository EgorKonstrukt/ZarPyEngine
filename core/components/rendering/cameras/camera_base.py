# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from core.math.math3d import Mat4


class CameraBase:
    DEFAULT_FOV = 60.0
    ORTHO_SIZE = 5.0
    DEFAULT_NEAR = 0.01
    DEFAULT_FAR = 1000.0

    def __init__(self):
        self._fov: float = self.DEFAULT_FOV
        self._near: float = self.DEFAULT_NEAR
        self._far: float = self.DEFAULT_FAR
        self._is_orthographic: bool = False
        self._ortho_size: float = self.ORTHO_SIZE
        self._render_scale: float = 1.0
        self._resolution_mode: str = "native"
        self._resolution_w: int = 1920
        self._resolution_h: int = 1080

    @property
    def fov(self) -> float:
        return self._fov

    @fov.setter
    def fov(self, value: float):
        self._fov = max(1.0, min(179.0, value))

    @property
    def near(self) -> float:
        return self._near

    @near.setter
    def near(self, value: float):
        self._near = max(0.001, value)

    @property
    def far(self) -> float:
        return self._far

    @far.setter
    def far(self, value: float):
        self._far = max(0.1, value)

    @property
    def is_orthographic(self) -> bool:
        return self._is_orthographic

    @property
    def ortho_size(self) -> float:
        return self._ortho_size

    @ortho_size.setter
    def ortho_size(self, value: float):
        self._ortho_size = max(0.001, value)

    @property
    def render_scale(self) -> float:
        return self._render_scale

    @render_scale.setter
    def render_scale(self, value: float):
        self._render_scale = max(0.05, min(1.0, value))

    @property
    def resolution_mode(self) -> str:
        return self._resolution_mode

    @resolution_mode.setter
    def resolution_mode(self, value: str):
        self._resolution_mode = value

    @property
    def resolution_w(self) -> int:
        return self._resolution_w

    @resolution_w.setter
    def resolution_w(self, value: int):
        self._resolution_w = max(1, int(value))

    @property
    def resolution_h(self) -> int:
        return self._resolution_h

    @resolution_h.setter
    def resolution_h(self, value: int):
        self._resolution_h = max(1, int(value))

    def get_view_matrix(self) -> Mat4:
        raise NotImplementedError

    def get_projection_matrix(self, aspect: float) -> Mat4:
        if self._is_orthographic:
            hw = self._ortho_size * aspect
            return Mat4.orthographic(-hw, hw, -self._ortho_size, self._ortho_size, self._near, self._far)
        return Mat4.perspective(self._fov, aspect, self._near, self._far)

    def compute_render_size(self, display_w: int, display_h: int) -> tuple[int, int]:
        if self._resolution_mode == "custom":
            return max(1, int(self._resolution_w)), max(1, int(self._resolution_h))
        scale = max(0.05, min(1.0, self._render_scale))
        return max(1, int(round(display_w * scale))), max(1, int(round(display_h * scale)))

    def serialize_base(self) -> dict:
        return {
            "fov": self._fov,
            "near": self._near,
            "far": self._far,
            "is_orthographic": self._is_orthographic,
            "ortho_size": self._ortho_size,
            "render_scale": self._render_scale,
            "resolution_mode": self._resolution_mode,
            "resolution_w": self._resolution_w,
            "resolution_h": self._resolution_h,
        }

    def deserialize_base(self, data: dict):
        self._fov = data.get("fov", self.DEFAULT_FOV)
        self._near = data.get("near", self.DEFAULT_NEAR)
        self._far = data.get("far", self.DEFAULT_FAR)
        self._is_orthographic = data.get("is_orthographic", False)
        self._ortho_size = data.get("ortho_size", self.ORTHO_SIZE)
        self._render_scale = float(data.get("render_scale", 1.0))
        self._resolution_mode = data.get("resolution_mode", "native")
        self._resolution_w = int(data.get("resolution_w", 1920))
        self._resolution_h = int(data.get("resolution_h", 1080))
