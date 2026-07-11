# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox, QComboBox, QGridLayout,
    QGroupBox, QScrollArea, QFrame, QSizePolicy, QMenu
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF, QTimer
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QMouseEvent,
    QKeyEvent, QPixmap, QLinearGradient, QAction, QPainterPath,
    QFontMetrics, QTransform
)
import numpy as np
from core.foundation.curve import Curve, CurveKey, TangentMode


class CurvePreview(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curve: Curve = Curve()
        self._bg_color = QColor(37, 37, 37)
        self._line_color = QColor(90, 156, 245)
        self.setMinimumSize(100, 30)
        self.setMaximumHeight(40)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_curve(self, curve: Curve):
        self._curve = curve
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self._bg_color)
        p.setPen(QPen(QColor(60, 60, 60), 1))
        p.drawLine(0, h // 2, w, h // 2)
        if not self._curve.keys:
            p.setPen(QColor(80, 80, 80))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "empty")
            p.end()
            return
        margin = 4
        plot_w = w - margin * 2
        plot_h = h - margin * 2
        times = [k.time for k in self._curve.keys]
        vals = [k.value for k in self._curve.keys]
        t_min, t_max = min(times), max(times)
        v_min, v_max = min(vals), max(vals)
        if t_max - t_min < 1e-10:
            t_max = t_min + 1.0
        if v_max - v_min < 1e-10:
            v_max = v_min + 1.0
        path = QPainterPath()
        samples = 50
        for i in range(samples + 1):
            t = t_min + (t_max - t_min) * i / samples
            v = self._curve.evaluate(t)
            x = margin + (t - t_min) / (t_max - t_min) * plot_w
            y = margin + (1.0 - (v - v_min) / (v_max - v_min)) * plot_h
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        p.setPen(QPen(self._line_color, 2))
        p.drawPath(path)
        p.end()

    def mousePressEvent(self, event):
        self.clicked.emit()


class CurveWidget(QWidget):
    curve_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curve: Curve = Curve()
        self._selected_key: CurveKey = None
        self._dragging: bool = False
        self._drag_tangent: str = None
        self._hovered_key: CurveKey = None
        self._pan_offset = QPointF(0, 0)
        self._zoom = 1.0

        self._grid_color = QColor(42, 42, 42)
        self._major_grid_color = QColor(60, 60, 60)
        self._bg_color = QColor(30, 30, 30)
        self._curve_color = QColor(90, 156, 245)
        self._key_color = QColor(206, 145, 120)
        self._key_selected_color = QColor(220, 220, 220)
        self._tangent_color = QColor(160, 160, 160)
        self._text_color = QColor(140, 140, 140)

        self.setMinimumSize(300, 200)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_curve(self, curve: Curve):
        self._curve = curve
        self._selected_key = None
        self.update()

    def get_curve(self) -> Curve:
        return self._curve

    def _get_key_rect(self, key: CurveKey) -> QRectF:
        x, y = self._key_to_screen(key)
        return QRectF(x - 5, y - 5, 10, 10)

    def _key_to_screen(self, key: CurveKey) -> tuple[float, float]:
        t_min, t_max, v_min, v_max = self._get_bounds()
        m = 30
        pw = self.width() - m * 2
        ph = self.height() - m * 2
        sx = m + (key.time - t_min) / max(t_max - t_min, 1e-10) * pw
        sy = m + (1.0 - (key.value - v_min) / max(v_max - v_min, 1e-10)) * ph
        return sx, sy

    def _screen_to_value(self, sx: float, sy: float) -> tuple[float, float]:
        t_min, t_max, v_min, v_max = self._get_bounds()
        m = 30
        pw = self.width() - m * 2
        ph = self.height() - m * 2
        t = t_min + (sx - m) / max(pw, 1) * (t_max - t_min)
        v = v_min + (1.0 - (sy - m) / max(ph, 1)) * (v_max - v_min)
        return t, v

    def _get_bounds(self) -> tuple[float, float, float, float]:
        if not self._curve.keys:
            return 0.0, 1.0, 0.0, 1.0
        times = [k.time for k in self._curve.keys]
        vals = [k.value for k in self._curve.keys]
        t_min, t_max = min(times), max(times)
        v_min, v_max = min(vals), max(vals)
        if t_max - t_min < 1e-10:
            t_max = t_min + 1.0
        if v_max - v_min < 1e-10:
            v_max = v_min + 1.0
        dt = (t_max - t_min) * 0.15
        dv = (v_max - v_min) * 0.15
        return t_min - dt, t_max + dt, v_min - dv, v_max + dv

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self._bg_color)
        self._draw_grid(p)
        self._draw_curve(p)
        self._draw_tangents(p)
        self._draw_keys(p)
        self._draw_readout(p)

    def _draw_grid(self, p: QPainter):
        t_min, t_max, v_min, v_max = self._get_bounds()
        m = 30
        pw = w = self.width() - m * 2
        ph = self.height() - m * 2
        n_div = 8
        for i in range(n_div + 1):
            frac = i / n_div
            x = m + frac * pw
            p.setPen(QPen(self._grid_color, 1))
            p.drawLine(int(x), m, int(x), m + ph)
            t = t_min + frac * (t_max - t_min)
            p.setPen(self._text_color)
            lbl = f"{t:.2f}"
            fm = QFontMetrics(p.font())
            tw = fm.horizontalAdvance(lbl)
            p.drawText(int(x - tw / 2), m + ph + 14, lbl)
            y = m + frac * ph
            p.setPen(QPen(self._grid_color, 1))
            p.drawLine(m, int(y), m + pw, int(y))
            v = v_max - frac * (v_max - v_min)
            val_lbl = f"{v:.2f}"
            p.setPen(self._text_color)
            p.drawText(2, int(y + 4), val_lbl)
        p.setPen(QPen(self._major_grid_color, 1))
        p.drawRect(m, m, pw, ph)

    def _draw_curve(self, p: QPainter):
        if not self._curve.keys:
            return
        t_min, t_max, v_min, v_max = self._get_bounds()
        m = 30
        pw = self.width() - m * 2
        ph = self.height() - m * 2
        path = QPainterPath()
        samples = max(100, pw)
        for i in range(samples + 1):
            t = t_min + (t_max - t_min) * i / samples
            v = self._curve.evaluate(t)
            x = m + (t - t_min) / max(t_max - t_min, 1e-10) * pw
            y = m + (1.0 - (v - v_min) / max(v_max - v_min, 1e-10)) * ph
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        p.setPen(QPen(self._curve_color, 2))
        p.drawPath(path)

    def _draw_tangents(self, p: QPainter):
        if not self._selected_key:
            return
        k = self._selected_key
        x, y = self._key_to_screen(k)
        t_min, t_max, v_min, v_max = self._get_bounds()
        m = 30
        pw = self.width() - m * 2
        ph = self.height() - m * 2
        def tangent_to_screen(tangent: float) -> tuple[float, float]:
            dt_frac = tangent * (t_max - t_min) / max(pw, 1)
            dv_frac = -tangent * (v_max - v_min) / max(ph, 1)
            return x + dt_frac * pw, y + dv_frac * ph
        if k.in_tangent != 0:
            tx, ty = tangent_to_screen(k.in_tangent * 0.3)
            p.setPen(QPen(self._tangent_color, 1, Qt.PenStyle.DashLine))
            p.drawLine(int(x), int(y), int(tx), int(ty))
            p.setBrush(QBrush(self._tangent_color))
            p.drawEllipse(QPointF(tx, ty), 4, 4)
        if k.out_tangent != 0:
            tx, ty = tangent_to_screen(k.out_tangent * 0.3)
            p.setPen(QPen(self._tangent_color, 1, Qt.PenStyle.DashLine))
            p.drawLine(int(x), int(y), int(tx), int(ty))
            p.setBrush(QBrush(self._tangent_color))
            p.drawEllipse(QPointF(tx, ty), 4, 4)

    def _draw_keys(self, p: QPainter):
        for k in self._curve.keys:
            x, y = self._key_to_screen(k)
            is_selected = k is self._selected_key
            color = self._key_selected_color if is_selected else self._key_color
            p.setBrush(QBrush(color))
            p.setPen(QPen(Qt.GlobalColor.black, 1))
            p.drawEllipse(QPointF(x, y), 6 if is_selected else 4, 6 if is_selected else 4)

    def _draw_readout(self, p: QPainter):
        if self._hovered_key:
            k = self._hovered_key
            text = f"t={k.time:.3f}  v={k.value:.3f}"
            p.setPen(self._text_color)
            p.drawText(self.width() - 180, 15, text)
        elif self._selected_key:
            k = self._selected_key
            text = f"t={k.time:.3f}  v={k.value:.3f}"
            p.setPen(self._text_color)
            p.drawText(self.width() - 180, 15, text)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            for k in self._curve.keys:
                if self._get_key_rect(k).contains(pos):
                    self._selected_key = k
                    self._dragging = True
                    self._drag_offset = QPointF(
                        k.time, k.value
                    )
                    self._drag_start = pos
                    self.update()
                    return
            t, v = self._screen_to_value(pos.x(), pos.y())
            for k in self._curve.keys:
                sx, sy = self._key_to_screen(k)
                t_min, t_max, v_min, v_max = self._get_bounds()
                m = 30
                pw = self.width() - m * 2
                ph = self.height() - m * 2
                def tang_to_screen(tang):
                    return tang * (t_max - t_min) / max(pw, 1), -tang * (v_max - v_min) / max(ph, 1)
                if k.in_tangent != 0:
                    tx, ty = tang_to_screen(k.in_tangent * 0.3)
                    if abs(pos.x() - (sx + tx * pw)) < 8 and abs(pos.y() - (sy + ty * ph)) < 8:
                        self._selected_key = k
                        self._drag_tangent = "in"
                        self._dragging = True
                        self.update()
                        return
                if k.out_tangent != 0:
                    tx, ty = tang_to_screen(k.out_tangent * 0.3)
                    if abs(pos.x() - (sx + tx * pw)) < 8 and abs(pos.y() - (sy + ty * ph)) < 8:
                        self._selected_key = k
                        self._drag_tangent = "out"
                        self._dragging = True
                        self.update()
                        return
            self._selected_key = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        pos = event.position()
        if self._dragging and self._selected_key:
            t, v = self._screen_to_value(pos.x(), pos.y())
            if self._drag_tangent:
                k = self._selected_key
                t_min, t_max, v_min, v_max = self._get_bounds()
                m = 30
                pw = self.width() - m * 2
                ph = self.height() - m * 2
                sx, sy = self._key_to_screen(k)
                dx = (pos.x() - sx) / max(pw, 1) / 0.3 * (t_max - t_min)
                dy = -(pos.y() - sy) / max(ph, 1) / 0.3 * (v_max - v_min)
                if self._drag_tangent == "in":
                    k.in_tangent = dx
                else:
                    k.out_tangent = dx
                self.curve_changed.emit()
                self.update()
                return
            t = max(0.0, min(1.0, t))
            k = self._selected_key
            k.time = round(t, 4)
            k.value = round(v, 4)
            self._curve.keys.sort(key=lambda x: x.time)
            self.curve_changed.emit()
            self.update()
            return
        for k in self._curve.keys:
            if self._get_key_rect(k).contains(pos):
                self._hovered_key = k
                self.update()
                return
        self._hovered_key = None
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            was_tangent_drag = self._drag_tangent is not None
            self._dragging = False
            self._drag_tangent = None
            if was_tangent_drag:
                if self._selected_key:
                    self._selected_key.tangent_mode = TangentMode.FREE
            else:
                self._curve._auto_smooth()
            self.curve_changed.emit()
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            t, v = self._screen_to_value(event.position().x(), event.position().y())
            t = max(0.0, min(1.0, t))
            v = max(0.0, min(1.0, v))
            k = self._curve.add_key(round(t, 4), round(v, 4))
            self._selected_key = k
            self.curve_changed.emit()
            self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            if self._selected_key and len(self._curve.keys) > 1:
                self._curve.remove_key(self._selected_key)
                self._selected_key = None
                self.curve_changed.emit()
                self.update()
        elif event.key() == Qt.Key.Key_Space:
            if self._selected_key:
                k = self._selected_key
                modes = list(TangentMode)
                idx = modes.index(k.tangent_mode)
                k.tangent_mode = modes[(idx + 1) % len(modes)]
                self._curve._auto_smooth()
                self.curve_changed.emit()
                self.update()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        for mode in TangentMode:
            action = menu.addAction(mode.value.capitalize())
            action.triggered.connect(lambda checked, m=mode: self._set_tangent_mode(m))
        menu.exec(event.globalPos())

    def _set_tangent_mode(self, mode: TangentMode):
        if self._selected_key:
            self._selected_key.tangent_mode = mode
            self._curve._auto_smooth()
            self.curve_changed.emit()
            self.update()


class CurveEditorDialog(QDialog):
    def __init__(self, curve: Curve, title: str = "Curve Editor", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(600, 450)
        self.resize(700, 500)
        self._curve = curve

        self.setStyleSheet(f"""
            QDialog {{
                background: #1e1e1e;
            }}
            QGroupBox {{
                color: #cccccc;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 16px;
                font-size: 11px;
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                color: #5a9cf5;
            }}
            QLabel {{
                color: #cccccc;
                font-size: 11px;
                background: transparent;
            }}
            QDoubleSpinBox, QSpinBox {{
                background: #2a2a2a;
                color: #eeeeee;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 1px 2px 1px 4px;
                font-size: 11px;
                min-height: 20px;
                selection-background-color: #5a9cf5;
            }}
            QDoubleSpinBox:hover, QSpinBox:hover {{ border-color: #4a4a4a; }}
            QDoubleSpinBox:focus, QSpinBox:focus {{ border-color: #5a9cf5; }}
            QComboBox {{
                background: #2a2a2a;
                color: #eeeeee;
                border: 1px solid #3c3c3c;
                border-radius: 3px;
                padding: 2px 4px;
                font-size: 11px;
                min-height: 20px;
            }}
            QComboBox:hover {{ border-color: #4a4a4a; }}
            QComboBox::drop-down {{ border: none; width: 16px; }}
            QComboBox::down-arrow {{ width: 8px; height: 8px; }}
            QComboBox QAbstractItemView {{
                background: #252525;
                color: #cccccc;
                border: 1px solid #3c3c3c;
                selection-background-color: #333333;
                selection-color: #eeeeee;
            }}
            QPushButton {{
                color: #cccccc;
                background: #2a2a2a;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 4px 16px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background: #333333;
                color: #eeeeee;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._curve_widget = CurveWidget()
        self._curve_widget.set_curve(curve)
        self._curve_widget.curve_changed.connect(self._on_curve_changed)
        layout.addWidget(self._curve_widget, 1)

        info_group = QGroupBox("Selected Key")
        info_layout = QGridLayout(info_group)
        info_layout.setSpacing(4)
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setRange(0.0, 1.0)
        self._time_spin.setSingleStep(0.01)
        self._time_spin.setDecimals(4)
        self._value_spin = QDoubleSpinBox()
        self._value_spin.setRange(-10.0, 10.0)
        self._value_spin.setSingleStep(0.01)
        self._value_spin.setDecimals(4)
        self._in_tangent_spin = QDoubleSpinBox()
        self._in_tangent_spin.setRange(-100.0, 100.0)
        self._in_tangent_spin.setSingleStep(0.1)
        self._in_tangent_spin.setDecimals(4)
        self._out_tangent_spin = QDoubleSpinBox()
        self._out_tangent_spin.setRange(-100.0, 100.0)
        self._out_tangent_spin.setSingleStep(0.1)
        self._out_tangent_spin.setDecimals(4)
        self._tangent_combo = QComboBox()
        for mode in TangentMode:
            self._tangent_combo.addItem(mode.value.capitalize(), mode)
        self._tangent_combo.currentIndexChanged.connect(self._on_tangent_mode_changed)

        info_layout.addWidget(QLabel("Time:"), 0, 0)
        info_layout.addWidget(self._time_spin, 0, 1)
        info_layout.addWidget(QLabel("Value:"), 0, 2)
        info_layout.addWidget(self._value_spin, 0, 3)
        info_layout.addWidget(QLabel("In Tangent:"), 1, 0)
        info_layout.addWidget(self._in_tangent_spin, 1, 1)
        info_layout.addWidget(QLabel("Out Tangent:"), 1, 2)
        info_layout.addWidget(self._out_tangent_spin, 1, 3)
        info_layout.addWidget(QLabel("Mode:"), 2, 0)
        info_layout.addWidget(self._tangent_combo, 2, 1, 1, 3)

        self._time_spin.valueChanged.connect(self._on_spin_changed)
        self._value_spin.valueChanged.connect(self._on_spin_changed)
        self._in_tangent_spin.valueChanged.connect(self._on_tangent_spin_changed)
        self._out_tangent_spin.valueChanged.connect(self._on_tangent_spin_changed)

        layout.addWidget(info_group)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        btn_layout.addStretch()
        add_btn = QPushButton("Add Key (0,0)")
        add_btn.clicked.connect(lambda: self._add_key_at(0.0, 0.0))
        btn_layout.addWidget(add_btn)
        add_btn2 = QPushButton("Add Key (1,1)")
        add_btn2.clicked.connect(lambda: self._add_key_at(1.0, 1.0))
        btn_layout.addWidget(add_btn2)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        self._updating = False
        self._curve_widget.curve_changed.connect(self._sync_spins)

    def _on_curve_changed(self):
        self._sync_spins()

    def _sync_spins(self):
        if self._updating:
            return
        self._updating = True
        k = self._curve_widget._selected_key
        if k:
            self._time_spin.setValue(k.time)
            self._value_spin.setValue(k.value)
            self._in_tangent_spin.setValue(k.in_tangent)
            self._out_tangent_spin.setValue(k.out_tangent)
            idx = self._tangent_combo.findData(k.tangent_mode)
            if idx >= 0:
                self._tangent_combo.setCurrentIndex(idx)
            self._time_spin.setEnabled(True)
            self._value_spin.setEnabled(True)
            self._in_tangent_spin.setEnabled(True)
            self._out_tangent_spin.setEnabled(True)
            self._tangent_combo.setEnabled(True)
        else:
            self._time_spin.setEnabled(False)
            self._value_spin.setEnabled(False)
            self._in_tangent_spin.setEnabled(False)
            self._out_tangent_spin.setEnabled(False)
            self._tangent_combo.setEnabled(False)
        self._updating = False

    def _on_spin_changed(self):
        if self._updating:
            return
        k = self._curve_widget._selected_key
        if k:
            k.time = round(self._time_spin.value(), 4)
            k.value = round(self._value_spin.value(), 4)
            self._curve._auto_smooth()
            self._curve_widget.update()

    def _on_tangent_spin_changed(self):
        if self._updating:
            return
        k = self._curve_widget._selected_key
        if k:
            k.in_tangent = self._in_tangent_spin.value()
            k.out_tangent = self._out_tangent_spin.value()
            k.tangent_mode = TangentMode.FREE
            self._curve_widget.update()

    def _on_tangent_mode_changed(self, idx):
        if self._updating:
            return
        k = self._curve_widget._selected_key
        if k:
            mode = self._tangent_combo.currentData()
            k.tangent_mode = mode
            self._curve._auto_smooth()
            self._curve_widget.update()

    def _add_key_at(self, t: float, v: float):
        self._curve.add_key(t, v)
        self._curve_widget.update()

    def get_curve(self) -> Curve:
        return self._curve_widget.get_curve()
