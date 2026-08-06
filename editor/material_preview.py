# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPixmap, QPainter, QMouseEvent, QWheelEvent
from PyQt6.QtCore import Qt
from typing import Optional
from editor.gl_offscreen import render_sphere


class MaterialPreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._albedo: list[float] = [1.0, 1.0, 1.0]
        self._metallic: float = 0.0
        self._smoothness: float = 0.5
        self._emission: list[float] = [0.0, 0.0, 0.0]
        self._emit_intensity: float = 0.0
        self._tex_path: str = ""
        self._theta: float = 0.0
        self._phi: float = 0.0
        self._zoom: float = 1.0
        self._last_mouse: Optional[tuple[int, int]] = None
        self._pixmap: Optional[QPixmap] = None
        self._dirty: bool = True
        self.setMinimumSize(120, 120)
        self.setMouseTracking(True)

    @staticmethod
    def _get_pv(kwargs, *keys, default=None):
        for k in keys:
            v = kwargs.get(k)
            if v is not None:
                return v
        return default

    def set_properties(self, *args, **kwargs):
        if args:
            kwargs = args[0]
        albedo = self._get_pv(kwargs, "_BaseColor", "u_albedo_color", default=[1.0, 1.0, 1.0, 1.0])
        metallic = self._get_pv(kwargs, "_Metallic", "u_metallic", default=0.0)
        smoothness = self._get_pv(kwargs, "_Smoothness", "u_smoothness", default=0.5)
        emission = self._get_pv(kwargs, "_EmissionColor", "u_emission", default=[0.0, 0.0, 0.0])
        emit_intensity = self._get_pv(kwargs, "_EmissionIntensity", "u_emission_intensity", default=1.0)
        albedo_tex = self._get_pv(kwargs, "albedo_tex", "_BaseMap", "u_albedo_tex", default="")
        albedo3 = albedo[:3] if len(albedo) >= 3 else [*albedo, 1.0]
        emit3 = emission[:3] if len(emission) >= 3 else [0.0, 0.0, 0.0]
        changed = (
            albedo3 != self._albedo or metallic != self._metallic or
            smoothness != self._smoothness or emit3 != self._emission or
            emit_intensity != self._emit_intensity or albedo_tex != self._tex_path
        )
        if not changed:
            return
        self._albedo = albedo3
        self._metallic = metallic
        self._smoothness = smoothness
        self._emission = emit3
        self._emit_intensity = emit_intensity
        if albedo_tex != self._tex_path:
            self._tex_path = albedo_tex
        self._dirty = True
        self.update()

    def _render(self):
        dpr = self.devicePixelRatio()
        pw = max(2, int(self.width() * dpr))
        ph = max(2, int(self.height() * dpr))
        if pw < 2 or ph < 2:
            self._pixmap = None
            return
        dist = 3.0 / max(self._zoom, 0.01)
        pm = render_sphere(
            pw, ph,
            self._albedo, self._metallic, self._smoothness,
            self._emission, self._emit_intensity,
            self._theta, self._phi, dist,
            self._tex_path,
        )
        self._pixmap = pm
        self._dirty = False

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        if w < 2 or h < 2:
            return
        dpr = self.devicePixelRatio()
        need_resize = self._pixmap is not None and (self._pixmap.width() != int(dpr * w) or self._pixmap.height() != int(dpr * h))
        if self._dirty or self._pixmap is None or need_resize:
            self._render()
        if self._pixmap is not None and not self._pixmap.isNull():
            p.drawPixmap(0, 0, w, h, self._pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._dirty = True

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = (int(event.position().x()), int(event.position().y()))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_mouse = None

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._last_mouse is None:
            return
        dx = event.position().x() - self._last_mouse[0]
        dy = event.position().y() - self._last_mouse[1]
        self._theta += dx * 0.01
        self._phi += dy * 0.01
        self._last_mouse = (int(event.position().x()), int(event.position().y()))
        self._dirty = True
        self.update()

    def wheelEvent(self, event: QWheelEvent):
        self._zoom *= 1.0 + event.angleDelta().y() * 0.001
        self._zoom = max(0.3, min(3.0, self._zoom))
        self._dirty = True
        self.update()
