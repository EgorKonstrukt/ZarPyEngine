# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor

from editor.viewport.overlay import draw_stats_overlay, draw_delta_label
from editor.viewport.navigation_gizmo import draw_navigation_gizmo_overlay
from editor.viewport.collaboration import draw_remote_cursors
from core.config.config import get_global_config


class OverlayWidget(QWidget):
    def __init__(self, viewport, parent=None):
        super().__init__(parent)
        self._vp = viewport
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setStyleSheet("background: transparent;")
        self.setAutoFillBackground(False)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._refresh_timer.timeout.connect(self.update)
        self._apply_fps()

    def _apply_fps(self):
        try:
            cfg = get_global_config()
            fps = int(cfg.get("viewport.overlay_fps", 60))
        except Exception:
            fps = 60
        fps = max(1, min(240, fps))
        if fps >= 240:
            self._refresh_timer.stop()
        else:
            self._refresh_timer.setInterval(int(1000 / fps))
            if not self._refresh_timer.isActive():
                self._refresh_timer.start()

    def set_visible(self, visible: bool):
        self.setVisible(visible)
        if visible:
            self._apply_fps()
        else:
            self._refresh_timer.stop()

    def paintEvent(self, event):
        vp = self._vp
        qp = QPainter(self)
        qp.setRenderHint(QPainter.RenderHint.Antialiasing)
        qp.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        if vp._stats_enabled:
            draw_stats_overlay(vp, qp)
        draw_delta_label(vp, qp)
        draw_navigation_gizmo_overlay(vp, qp)
        draw_remote_cursors(vp, qp)
        if vp._area_selecting:
            x1, y1 = vp._area_start
            x2, y2 = vp._area_end
            qp.save()
            qp.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.PenStyle.DashLine))
            qp.setBrush(QColor(100, 150, 255, 40))
            qp.drawRect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            qp.restore()
        if vp._overlay_canvas and not vp._overlay_canvas.edit_mode:
            vp._overlay_canvas._render_overlay(qp)
        qp.end()
