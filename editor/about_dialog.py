# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
import platform
import random
import sys

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QFrame)
from PyQt6.QtGui import QFont, QPixmap, QPainter, QColor, QPen, QKeyEvent
from PyQt6.QtCore import Qt, QRectF, QTimer

from PyQt6.QtSvg import QSvgRenderer

from core.constants import APP_VERSION, APP_VERSION_DISPLAY

MPL_URL = "https://mozilla.org/MPL/2.0/"
LOGO_W = 256
_EASTER_CLICKS_NEEDED = 10


def _render_logo(target_w: int) -> QPixmap | None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zarin_logo.svg")
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return None
    vb = renderer.viewBoxF()
    if vb.isEmpty():
        vb = QRectF(0, 0, 512, 512)
    target_h = round(target_w * vb.height() / vb.width())
    pm = QPixmap(target_w, target_h)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, target_w, target_h))
    p.end()
    return pm


def _render_icon(size: int) -> QPixmap | None:
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "zarin_icon.svg")
    renderer = QSvgRenderer(path)
    if not renderer.isValid():
        return None
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    return pm


_COLS = 20
_ROWS = 15
_CELL = 22
_TICK_MS = 110

_DIR = {
    Qt.Key.Key_Up: (0, -1),
    Qt.Key.Key_Down: (0, 1),
    Qt.Key.Key_Left: (-1, 0),
    Qt.Key.Key_Right: (1, 0),
}
_OPPOSITE = {
    Qt.Key.Key_Up: Qt.Key.Key_Down,
    Qt.Key.Key_Down: Qt.Key.Key_Up,
    Qt.Key.Key_Left: Qt.Key.Key_Right,
    Qt.Key.Key_Right: Qt.Key.Key_Left,
}

_BG = QColor(24, 24, 28)
_GRID = QColor(30, 30, 36)
_SNAKE_HEAD = QColor(70, 180, 100)
_SNAKE_BODY = QColor(50, 140, 75)
_FOOD = QColor(220, 60, 60)
_TEXT = QColor(200, 200, 210)


class _SnakeGame(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zarin Snake")
        self.setFixedSize(_COLS * _CELL + 2, _ROWS * _CELL + 2)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._head_pm = _render_icon(_CELL)
        self._reset()

    def _reset(self):
        mid_x, mid_y = _COLS // 2, _ROWS // 2
        self._snake = [(mid_x - i, mid_y) for i in range(4)]
        self._dir = (1, 0)
        self._next_dir = (1, 0)
        self._alive = True
        self._score = 0
        self._spawn_food()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_TICK_MS)

    def _spawn_food(self):
        while True:
            pos = (random.randint(0, _COLS - 1), random.randint(0, _ROWS - 1))
            if pos not in self._snake:
                self._food = pos
                return

    def keyPressEvent(self, e: QKeyEvent):
        nd = _DIR.get(e.key())
        if nd is None:
            return
        dx, dy = self._dir
        if nd != (-dx, -dy):
            self._next_dir = nd

    def _tick(self):
        if not self._alive:
            return
        self._dir = self._next_dir
        hx, hy = self._snake[0]
        dx, dy = self._dir
        nx, ny = hx + dx, hy + dy

        if nx < 0 or nx >= _COLS or ny < 0 or ny >= _ROWS:
            return self._die()
        if (nx, ny) in self._snake:
            return self._die()

        self._snake.insert(0, (nx, ny))
        if (nx, ny) == self._food:
            self._score += 1
            self._spawn_food()
        else:
            self._snake.pop()
        self.update()

    def _die(self):
        self._alive = False
        self._timer.stop()
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(0, 0, self.width(), self.height(), _BG)

        for x in range(_COLS + 1):
            p.setPen(_GRID)
            p.drawLine(x * _CELL, 0, x * _CELL, _ROWS * _CELL)
        for y in range(_ROWS + 1):
            p.setPen(_GRID)
            p.drawLine(0, y * _CELL, _COLS * _CELL, y * _CELL)

        fx, fy = self._food
        p.setBrush(_FOOD)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(fx * _CELL + 2, fy * _CELL + 2, _CELL - 4, _CELL - 4)

        for i, (sx, sy) in enumerate(self._snake):
            if i == 0 and self._head_pm is not None:
                p.drawPixmap(sx * _CELL, sy * _CELL, self._head_pm)
            else:
                c = _SNAKE_BODY
                p.setBrush(c)
                p.setPen(Qt.PenStyle.NoPen)
                p.drawRoundedRect(sx * _CELL + 1, sy * _CELL + 1, _CELL - 2, _CELL - 2, 4, 4)

        if not self._alive:
            p.setPen(QPen(_TEXT, 1))
            p.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                       f"Game Over!\nScore: {self._score}\n\nPress Space to restart")
        p.end()

    def keyReleaseEvent(self, e: QKeyEvent):
        if not self._alive and e.key() == Qt.Key.Key_Space:
            self._reset()

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About Zarin Engine")
        self.setFixedWidth(440)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._logo_clicks = 0
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)
        root.setContentsMargins(20, 20, 20, 16)

        logo_pm = _render_logo(LOGO_W)
        if logo_pm is not None:
            logo_lbl = QLabel()
            logo_lbl.setPixmap(logo_pm)
            logo_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_lbl.setCursor(Qt.CursorShape.PointingHandCursor)
            logo_lbl.mousePressEvent = self._on_logo_click
            root.addWidget(logo_lbl)

        ver = QLabel(APP_VERSION_DISPLAY)
        ver.setFont(QFont("Segoe UI", 9))
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(ver)

        root.addWidget(self._hline())

        desc = QLabel(
            "A high-performance 64-bit ECS 3D engine with\n"
            "plugin-based architecture, real-time rendering,\n"
            "and a full-featured editor."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(desc)

        root.addWidget(self._hline())

        gil_enabled = getattr(sys, '_is_gil_enabled', lambda: True)()
        gil_status = "No (nogil)" if not gil_enabled else "Yes"
        info = QLabel(
            f"<b>Tech Stack:</b> Python 3, ModernGL, PyQt6, NumPy, Bullet3 / PhysX<br>"
            f"<b>Plugins:</b> {self._count_plugins()} loaded<br>"
            f"<b>System:</b> {platform.system()} {platform.machine()}<br>"
            f"<b>Python:</b> {sys.version.split()[0]}<br>"
            f"<b>GIL:</b> {gil_status}<br>"
            f"<b>Interpreter:</b> {os.path.basename(sys.executable)}<br>"
            f"<b>Renderer:</b> OpenGL 4.6 Core Profile"
        )
        info.setTextFormat(Qt.TextFormat.RichText)
        root.addWidget(info)

        root.addWidget(self._hline())

        copy_lbl = QLabel("Copyright © 2026 Zarrakun")
        copy_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(copy_lbl)

        lic_lbl = QLabel(
            f'Licensed under the <a href="{MPL_URL}">Mozilla Public License 2.0</a>'
        )
        lic_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lic_lbl.setOpenExternalLinks(True)
        root.addWidget(lic_lbl)

        root.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(90)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _on_logo_click(self, _event):
        self._logo_clicks += 1
        if self._logo_clicks >= _EASTER_CLICKS_NEEDED:
            self._logo_clicks = 0
            dlg = _SnakeGame(self)
            dlg.exec()

    @staticmethod
    def _hline() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    @staticmethod
    def _count_plugins() -> int:
        plugins_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plugins")
        try:
            return len([d for d in os.listdir(plugins_dir)
                        if os.path.isdir(os.path.join(plugins_dir, d)) and not d.startswith("_")])
        except FileNotFoundError:
            return 0
