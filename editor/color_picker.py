# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from PyQt6.QtWidgets import (
    QDialog, QLineEdit, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSpinBox, QDialogButtonBox, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap, QPainter, QLinearGradient, QConicalGradient, QBrush, QPen, QImage, QPalette


def _hsv_to_rgb(h, s, v):
    h = h % 360
    s = max(0, min(1, s))
    v = max(0, min(1, v))
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x
    return (r + m, g + m, b + m)


def _rgb_to_hsv(r, g, b):
    mx = max(r, g, b)
    mn = min(r, g, b)
    d = mx - mn
    if d == 0:
        h = 0
    elif mx == r:
        h = 60 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60 * (((b - r) / d) + 2)
    else:
        h = 60 * (((r - g) / d) + 4)
    s = 0 if mx == 0 else d / mx
    v = mx
    return (h % 360, s, v)


def _hue_to_screen_angle(hue_deg):
    return math.radians((hue_deg + 90) % 360)


def _screen_angle_to_hue(angle_rad):
    return (math.degrees(angle_rad) - 90 + 360) % 360


class _ColorWheelWidget(QWidget):
    colorChanged = pyqtSignal()
    hueChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 220)
        self._hue = 0.0
        self._sat = 1.0
        self._val = 1.0
        self._wheel_pixmap = None
        self._tri_image = None
        self._dragging = None

    @property
    def _cx(self):
        return self.width() / 2

    @property
    def _cy(self):
        return self.height() / 2

    @property
    def _R_outer(self):
        return min(self.width(), self.height()) / 2 - 2

    @property
    def _R_inner(self):
        return self._R_outer - 22

    @property
    def _R_tri(self):
        return self._R_inner - 6

    def set_hue(self, h):
        h = h % 360
        if h != self._hue:
            self._hue = h
            self._tri_image = None
            self.update()

    def set_sv(self, s, v):
        self._sat = max(0, min(1, s))
        self._val = max(0, min(1, v))
        self.update()

    def get_hue(self):
        return self._hue

    def get_sv(self):
        return (self._sat, self._val)

    def _tri_vertices(self):
        hue_rad = _hue_to_screen_angle(self._hue)
        offset = 2 * math.pi / 3
        r = self._R_tri
        cx, cy = self._cx, self._cy
        return [
            (cx + r * math.cos(hue_rad), cy + r * math.sin(hue_rad)),
            (cx + r * math.cos(hue_rad + offset), cy + r * math.sin(hue_rad + offset)),
            (cx + r * math.cos(hue_rad - offset), cy + r * math.sin(hue_rad - offset)),
        ]

    def paintEvent(self, event):
        if self._wheel_pixmap is None:
            self._render_wheel()
        if self._tri_image is None:
            self._render_triangle()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QApplication.palette().color(QPalette.ColorRole.Window))
        p.drawPixmap(0, 0, self._wheel_pixmap)
        p.drawImage(0, 0, self._tri_image)
        self._draw_tri_indicator(p)
        self._draw_wheel_indicator(p)

    def _render_wheel(self):
        w, h = self.width(), self.height()
        cx, cy = self._cx, self._cy
        R_out = self._R_outer
        R_in = self._R_inner
        pm = QPixmap(w, h)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QConicalGradient(cx, cy, 90)
        for i in range(361):
            pos = i / 360
            hue = i % 360
            grad.setColorAt(pos, QColor.fromHsvF(hue / 360, 1, 1))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(grad))
        p.drawEllipse(int(cx - R_out), int(cy - R_out), int(R_out * 2), int(R_out * 2))
        p.setBrush(QBrush(QApplication.palette().color(QPalette.ColorRole.Window)))
        p.drawEllipse(int(cx - R_in), int(cy - R_in), int(R_in * 2), int(R_in * 2))
        p.end()
        self._wheel_pixmap = pm

    def _render_triangle(self):
        w, h = self.width(), self.height()
        img = QImage(w, h, QImage.Format.Format_ARGB32)
        img.fill(0)
        verts = self._tri_vertices()
        ax, ay = verts[0]
        bx, by = verts[1]
        cx_v, cy_v = verts[2]
        area2 = (bx - ax) * (cy_v - ay) - (cx_v - ax) * (by - ay)
        if abs(area2) < 0.001:
            self._tri_image = img
            return
        min_x = max(0, int(min(ax, bx, cx_v)))
        max_x = min(w, int(max(ax, bx, cx_v)) + 1)
        min_y = max(0, int(min(ay, by, cy_v)))
        max_y = min(h, int(max(ay, by, cy_v)) + 1)
        for py in range(min_y, max_y):
            for px in range(min_x, max_x):
                u = ((bx - px) * (cy_v - py) - (cx_v - px) * (by - py)) / area2
                v = ((cx_v - px) * (ay - py) - (ax - px) * (cy_v - py)) / area2
                wv = 1 - u - v
                if u >= 0 and v >= 0 and wv >= 0:
                    s = u
                    val = 1 - wv
                    r, g, b = _hsv_to_rgb(self._hue, s, val)
                    rgb = (0xFF << 24) | (int(r * 255) << 16) | (int(g * 255) << 8) | int(b * 255)
                    img.setPixel(px, py, rgb)
        self._tri_image = img

    def _draw_wheel_indicator(self, p):
        R_mid = (self._R_outer + self._R_inner) / 2
        hue_rad = _hue_to_screen_angle(self._hue)
        ix = self._cx + R_mid * math.cos(hue_rad)
        iy = self._cy + R_mid * math.sin(hue_rad)
        p.setPen(QPen(Qt.GlobalColor.black, 1))
        p.setBrush(QBrush(Qt.GlobalColor.white))
        p.drawEllipse(int(ix) - 5, int(iy) - 5, 10, 10)
        p.setPen(QPen(Qt.GlobalColor.black, 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(int(ix) - 5, int(iy) - 5, 10, 10)

    def _draw_tri_indicator(self, p):
        verts = self._tri_vertices()
        ax, ay = verts[0]
        bx, by = verts[1]
        cx_v, cy_v = verts[2]
        u = self._sat
        wv = 1 - self._val
        v = 1 - u - wv
        total = u + v + wv
        if total > 0:
            u /= total
            v /= total
            wv /= total
        else:
            u, v, wv = 1, 0, 0
        px = u * ax + v * bx + wv * cx_v
        py = u * ay + v * by + wv * cy_v
        lum = self._val * 0.299 + self._sat * 0.587 + (1 - self._val) * 0.114
        pen = QPen(Qt.GlobalColor.black if lum > 0.5 else Qt.GlobalColor.white)
        pen.setWidth(2)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(int(px) - 5, int(py) - 5, 10, 10)

    def mousePressEvent(self, event):
        px = event.position().x()
        py = event.position().y()
        dx = px - self._cx
        dy = py - self._cy
        dist = math.hypot(dx, dy)
        if self._R_inner <= dist <= self._R_outer:
            self._dragging = "wheel"
            self._update_hue(px, py)
        elif dist < self._R_inner:
            self._dragging = "triangle"
            self._update_sv(px, py)

    def mouseMoveEvent(self, event):
        if self._dragging == "wheel":
            self._update_hue(event.position().x(), event.position().y())
        elif self._dragging == "triangle":
            self._update_sv(event.position().x(), event.position().y())

    def mouseReleaseEvent(self, event):
        if self._dragging == "wheel":
            self._update_hue(event.position().x(), event.position().y())
        elif self._dragging == "triangle":
            self._update_sv(event.position().x(), event.position().y())
        self._dragging = None

    def _update_hue(self, px, py):
        dx = px - self._cx
        dy = py - self._cy
        angle = math.atan2(dy, dx)
        new_hue = _screen_angle_to_hue(angle)
        if new_hue != self._hue:
            self._hue = new_hue
            self._tri_image = None
            self.update()
            self.hueChanged.emit(self._hue)
            self.colorChanged.emit()

    def _update_sv(self, px, py):
        verts = self._tri_vertices()
        ax, ay = verts[0]
        bx, by = verts[1]
        cx_v, cy_v = verts[2]
        area2 = (bx - ax) * (cy_v - ay) - (cx_v - ax) * (by - ay)
        if abs(area2) < 0.001:
            return
        u = ((bx - px) * (cy_v - py) - (cx_v - px) * (by - py)) / area2
        v = ((cx_v - px) * (ay - py) - (ax - px) * (cy_v - py)) / area2
        wv = 1 - u - v
        if u < 0:
            d = v + wv
            if d > 0:
                v /= d
                wv /= d
            u = 0
            v = max(0, min(1, v))
            wv = max(0, min(1, 1 - v))
        elif v < 0:
            d = u + wv
            if d > 0:
                u /= d
                wv /= d
            u = max(0, min(1, u))
            wv = max(0, min(1, 1 - u))
            v = 0
        elif wv < 0:
            d = u + v
            if d > 0:
                u /= d
                v /= d
            u = max(0, min(1, u))
            v = max(0, min(1, 1 - u))
            wv = 0
        new_sat = u
        new_val = 1 - wv
        if new_sat != self._sat or new_val != self._val:
            self._sat = new_sat
            self._val = new_val
            self.update()
            self.colorChanged.emit()


class _AlphaWidget(QWidget):
    alphaChanged = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(220, 20)
        self._alpha = 1.0
        self._color = QColor.fromRgbF(1, 1, 1)
        self._dragging = False

    def set_alpha(self, a):
        self._alpha = max(0, min(1, a))
        self.update()

    def set_color(self, c: QColor):
        self._color = c
        self.update()

    def paintEvent(self, event):
        w, h = self.width(), self.height()
        p = QPainter(self)
        for x in range(0, w, 8):
            for y in range(0, h, 8):
                even = (x // 8 + y // 8) % 2 == 0
                p.fillRect(x, y, 8, 8, Qt.GlobalColor.white if even else Qt.GlobalColor.lightGray)
        r, g, b = self._color.redF(), self._color.greenF(), self._color.blueF()
        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0, QColor.fromRgbF(r, g, b, 0))
        grad.setColorAt(1, QColor.fromRgbF(r, g, b, 1))
        p.fillRect(0, 0, w, h, grad)
        x = int(self._alpha * (w - 1))
        p.setPen(QPen(Qt.GlobalColor.black, 2))
        p.drawLine(x, 0, x, h)

    def mousePressEvent(self, event):
        self._dragging = True
        self._update_alpha(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_alpha(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self._update_alpha(event)

    def _update_alpha(self, event):
        w = self.width()
        if w < 1:
            return
        self._alpha = max(0, min(1, event.position().x() / (w - 1)))
        self.update()
        self.alphaChanged.emit(self._alpha)


class ColorDialog(QDialog):
    colorChanged = pyqtSignal(object)

    def __init__(self, initial: QColor = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Color")
        self.setMinimumWidth(300)

        self.setPalette(QApplication.palette())
        self.setAutoFillBackground(True)
        self.setStyleSheet("""
            QDialog { background: palette(window); }
            QLabel { color: palette(text); background: transparent; }
            QSpinBox {
                background: palette(base); color: palette(text);
                border: 1px solid palette(mid); border-radius: 2px; padding: 2px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background: palette(button); border: 1px solid palette(mid);
                border-radius: 2px; width: 16px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background: palette(light);
            }
            QLineEdit {
                background: palette(base); color: palette(text);
                border: 1px solid palette(mid); border-radius: 2px; padding: 2px;
            }
            QPushButton {
                background: palette(button); color: palette(ButtonText);
                border: 1px solid palette(mid); border-radius: 2px; padding: 4px 16px;
            }
            QPushButton:hover { background: palette(light); }
            QPushButton:pressed { background: palette(dark); color: palette(text); }
            QDialogButtonBox { background: transparent; }
        """)


        self._color = initial.toRgb() if initial and initial.isValid() else QColor.fromRgbF(1, 1, 1)
        h, s, v, a = self._color.hueF(), self._color.saturationF(), self._color.valueF(), self._color.alphaF()
        if h < 0:
            h = 0

        layout = QVBoxLayout(self)

        self._wheel = _ColorWheelWidget()
        self._wheel.set_hue(h * 360)
        self._wheel.set_sv(s, v)
        layout.addWidget(self._wheel, alignment=Qt.AlignmentFlag.AlignCenter)

        self._alpha_slider = _AlphaWidget()
        self._alpha_slider.set_alpha(a)
        self._alpha_slider.set_color(self._color)
        layout.addWidget(self._alpha_slider)

        rgb_layout = QHBoxLayout()
        self._r_spin = QSpinBox()
        self._r_spin.setRange(0, 255)
        self._r_spin.setValue(int(self._color.redF() * 255))
        self._g_spin = QSpinBox()
        self._g_spin.setRange(0, 255)
        self._g_spin.setValue(int(self._color.greenF() * 255))
        self._b_spin = QSpinBox()
        self._b_spin.setRange(0, 255)
        self._b_spin.setValue(int(self._color.blueF() * 255))
        self._a_spin = QSpinBox()
        self._a_spin.setRange(0, 255)
        self._a_spin.setValue(int(a * 255))
        rgb_layout.addWidget(QLabel("R:"))
        rgb_layout.addWidget(self._r_spin)
        rgb_layout.addWidget(QLabel("G:"))
        rgb_layout.addWidget(self._g_spin)
        rgb_layout.addWidget(QLabel("B:"))
        rgb_layout.addWidget(self._b_spin)
        rgb_layout.addWidget(QLabel("A:"))
        rgb_layout.addWidget(self._a_spin)
        layout.addLayout(rgb_layout)

        hex_layout = QHBoxLayout()
        hex_layout.addWidget(QLabel("Hex:"))
        self._hex_edit = QLineEdit()
        self._hex_edit.setText(self._color.name(QColor.NameFormat.HexArgb))
        hex_layout.addWidget(self._hex_edit)
        layout.addLayout(hex_layout)

        self._preview = QLabel()
        self._preview.setFixedHeight(30)
        self._update_preview()
        layout.addWidget(self._preview)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._wheel.colorChanged.connect(self._on_wheel_changed)
        self._wheel.hueChanged.connect(self._on_wheel_hue_changed)
        self._alpha_slider.alphaChanged.connect(self._on_alpha_changed)
        self._r_spin.valueChanged.connect(self._on_rgb_spin)
        self._g_spin.valueChanged.connect(self._on_rgb_spin)
        self._b_spin.valueChanged.connect(self._on_rgb_spin)
        self._a_spin.valueChanged.connect(self._on_a_spin)
        self._hex_edit.editingFinished.connect(self._on_hex_edited)

    def _on_wheel_changed(self):
        s, v = self._wheel.get_sv()
        a = self._color.alphaF()
        self._color.setHsvF(self._wheel.get_hue() / 360.0, s, v, a)
        self._sync_from_color()
        self.colorChanged.emit(self._color)

    def _on_wheel_hue_changed(self, h):
        self._color.setHsvF(h / 360.0, self._wheel._sat, self._wheel._val, self._color.alphaF())
        self._sync_from_color()
        self.colorChanged.emit(self._color)

    def _on_alpha_changed(self, a):
        self._color.setAlphaF(a)
        self._sync_from_color()
        self.colorChanged.emit(self._color)

    def _on_rgb_spin(self):
        self._color.setRgb(self._r_spin.value(), self._g_spin.value(), self._b_spin.value(), self._a_spin.value())
        self._sync_from_color()
        self.colorChanged.emit(self._color)

    def _on_a_spin(self, v):
        self._color.setAlpha(v)
        self._sync_from_color()
        self.colorChanged.emit(self._color)

    def _on_hex_edited(self):
        text = self._hex_edit.text().strip()
        c = QColor(text)
        if c.isValid():
            self._color = c.toRgb()
            self._sync_from_color()
            self.colorChanged.emit(self._color)

    def _sync_from_color(self):
        r, g, b, a = self._color.red(), self._color.green(), self._color.blue(), self._color.alpha()
        h, s, v = self._color.hueF(), self._color.saturationF(), self._color.valueF()
        if h < 0:
            h = 0
        self._r_spin.blockSignals(True)
        self._g_spin.blockSignals(True)
        self._b_spin.blockSignals(True)
        self._a_spin.blockSignals(True)
        self._r_spin.setValue(r)
        self._g_spin.setValue(g)
        self._b_spin.setValue(b)
        self._a_spin.setValue(a)
        self._r_spin.blockSignals(False)
        self._g_spin.blockSignals(False)
        self._b_spin.blockSignals(False)
        self._a_spin.blockSignals(False)
        af = self._color.alphaF()
        self._alpha_slider.set_alpha(af)
        self._alpha_slider.set_color(self._color)
        self._wheel.set_hue(h * 360)
        self._wheel.set_sv(s, v)
        self._hex_edit.setText(self._color.name(QColor.NameFormat.HexArgb))
        self._update_preview()

    def _update_preview(self):
        r, g, b, a = self._color.red(), self._color.green(), self._color.blue(), self._color.alpha()
        self._preview.setStyleSheet(f"background: rgba({r},{g},{b},{a/255});")

    def get_color(self) -> QColor:
        return self._color

    @staticmethod
    def getColor(initial: QColor = None, parent=None, title="") -> QColor:
        from PyQt6.QtGui import QCursor
        dlg = ColorDialog(initial, None)
        if title:
            dlg.setWindowTitle(title)
        dlg.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        dw, dh = dlg.width(), dlg.height()
        cursor_pos = QCursor.pos()
        x = cursor_pos.x() - dw // 2
        y = cursor_pos.y() + 4
        if y + dh > screen.bottom():
            y = cursor_pos.y() - dh - 4
        if y < screen.top():
            y = screen.top()
        if x + dw > screen.right():
            x = screen.right() - dw
        if x < screen.left():
            x = screen.left()
        dlg.move(x, y)
        result = dlg.exec()
        c = dlg.get_color()
        if result == QDialog.DialogCode.Accepted:
            return c
        return initial if initial and initial.isValid() else QColor()


class ColorLineEdit(QWidget):
    colorChanged = pyqtSignal(object)

    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self._color = self._parse_color(color) if color is not None else QColor.fromRgbF(1, 1, 1)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._preview = QPushButton()
        self._preview.setFixedSize(28, 22)
        self._update_swatch()
        self._preview.clicked.connect(self._open_dialog)
        layout.addWidget(self._preview)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText("#rrggbb or rgb(r,g,b)")
        self._edit.setText(self._color.name(QColor.NameFormat.HexArgb))
        self._edit.textChanged.connect(self._on_text_changed)
        layout.addWidget(self._edit, 1)

    @staticmethod
    def _parse_color(color):
        if isinstance(color, QColor):
            return color.toRgb()
        if isinstance(color, (list, tuple)):
            r, g, b, a = (float(v) for v in (color + [1.0, 1.0, 1.0, 1.0])[:4])
            return QColor.fromRgbF(r, g, b, a).toRgb()
        if isinstance(color, str):
            c = QColor(color.strip())
            if c.isValid():
                return c.toRgb()
        return QColor.fromRgbF(1, 1, 1)

    def _update_swatch(self):
        r, g, b, a = self._color.red(), self._color.green(), self._color.blue(), self._color.alpha()
        self._preview.setStyleSheet(
            f"background: rgba({r},{g},{b},{a/255}); border-radius: 3px;"
        )

    def _open_dialog(self):
        from PyQt6.QtGui import QCursor
        dlg = ColorDialog(self._color, None)
        dlg.adjustSize()
        screen = QApplication.primaryScreen().availableGeometry()
        dw, dh = dlg.width(), dlg.height()
        cursor_pos = QCursor.pos()
        x = cursor_pos.x() - dw // 2
        y = cursor_pos.y() + 4
        if y + dh > screen.bottom():
            y = cursor_pos.y() - dh - 4
        if y < screen.top():
            y = screen.top()
        if x + dw > screen.right():
            x = screen.right() - dw
        if x < screen.left():
            x = screen.left()
        dlg.move(x, y)

        def _on_dialog_color(c):
            self._color = c.toRgb()
            self._update_swatch()
            self._edit.blockSignals(True)
            self._edit.setText(self._color.name(QColor.NameFormat.HexArgb))
            self._edit.blockSignals(False)
            self.colorChanged.emit(self._color)

        dlg.colorChanged.connect(_on_dialog_color)
        dlg.exec()
        try:
            dlg.colorChanged.disconnect(_on_dialog_color)
        except TypeError:
            pass
        self._color = dlg.get_color().toRgb()
        self._update_swatch()
        self._edit.blockSignals(True)
        self._edit.setText(self._color.name(QColor.NameFormat.HexArgb))
        self._edit.blockSignals(False)
        self.colorChanged.emit(self._color)

    def _on_text_changed(self, text):
        c = QColor(text.strip())
        if c.isValid():
            self._color = c.toRgb()
            self._update_swatch()
            self.colorChanged.emit(self._color)

    def set_color(self, color):
        if isinstance(color, (list, tuple)):
            fmt = tuple(float(v) for v in color)
            if len(fmt) >= 3:
                r, g, b = fmt[0], fmt[1], fmt[2]
                a = fmt[3] if len(fmt) > 3 else 1.0
                self._color = QColor.fromRgbF(r, g, b, a)
            else:
                self._color = QColor.fromRgbF(1, 1, 1)
        elif isinstance(color, QColor):
            self._color = color.toRgb()
        else:
            c = QColor(str(color))
            self._color = c.toRgb() if c.isValid() else QColor.fromRgbF(1, 1, 1)
        self._update_swatch()
        self._edit.blockSignals(True)
        self._edit.setText(self._color.name(QColor.NameFormat.HexArgb))
        self._edit.blockSignals(False)

    def get_color(self) -> QColor:
        return self._color

    def get_color_rgba(self) -> list[float]:
        return [self._color.redF(), self._color.greenF(), self._color.blueF(), self._color.alphaF()]
