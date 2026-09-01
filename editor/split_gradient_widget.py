# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun
#
# Generic "split gradient" widget. It renders a horizontal gradient bar split
# by N draggable handles. Each handle sits at a normalized fraction of the bar
# (0..1) and is labelled with that fraction multiplied by a scale factor
# (e.g. a distance, a budget of samples, a temperature range, ...). It is used
# by the inspector's SPLIT_GRADIENT field type and is reusable for any concept
# that needs to distribute N boundaries across a fixed range.

from __future__ import annotations

from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QFont, QFontMetrics
from PyQt6.QtWidgets import QWidget, QSizePolicy

from core.config.editor_scale import scale

_MAX_HANDLES = 6


class SplitGradientWidget(QWidget):
    valueChanged = pyqtSignal(list)

    def __init__(self, value: list[float] | None = None, max_value: float = 1.0,
                 decimals: int = 1, parent=None):
        super().__init__(parent)
        self._max_value: float = float(max_value)
        self._decimals: int = int(decimals)
        self.set_value(value if value is not None else [])
        self._drag_index = -1
        self.setMinimumHeight(scale(104))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_max_value(self, max_value: float):
        self._max_value = float(max_value)
        self.update()

    def set_value(self, value):
        if not value:
            value = []
        norm = []
        for x in value:
            try:
                f = float(x)
            except (TypeError, ValueError):
                f = 1.0
            norm.append(min(1.0, max(0.0, f)))
        norm = sorted(norm)
        while len(norm) > _MAX_HANDLES:
            norm.pop()
        self._splits = norm

    def setValue(self, value):
        self.set_value(value)
        self.update()

    def value(self) -> list[float]:
        return list(self._splits)

    def _fraction_x(self, fraction: float) -> float:
        m = 4.0
        w = float(self.width())
        usable = max(1.0, w - m * 2)
        return m + (usable * min(1.0, max(0.0, fraction)))

    def _x_fraction(self, x: float) -> float:
        m = 4.0
        w = float(self.width())
        usable = max(1.0, w - m * 2)
        return (x - m) / usable

    def _section_color(self, index: int, count: int) -> QColor:
        t = index / max(1, count - 1)
        start = (68, 122, 176)
        end = (25, 42, 62)
        r = int(start[0] + (end[0] - start[0]) * t)
        g = int(start[1] + (end[1] - start[1]) * t)
        b = int(start[2] + (end[2] - start[2]) * t)
        return QColor(r, g, b)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        bar_top = scale(8)
        bar_h = scale(34)
        bar_rect = QRectF(4, bar_top, max(1.0, self.width() - 8), bar_h)

        boundaries = [0.0] + list(self._splits) + [1.0]
        for i in range(len(boundaries) - 1):
            x0 = self._fraction_x(boundaries[i])
            x1 = self._fraction_x(boundaries[i + 1])
            seg = QRectF(x0, bar_rect.top(), max(0.5, x1 - x0), bar_rect.height())
            col = self._section_color(i, len(boundaries) - 1)
            grad = QLinearGradient(seg.topLeft(), seg.topRight())
            grad.setColorAt(0.0, col.lighter(125))
            grad.setColorAt(1.0, col)
            p.fillRect(seg, grad)

        p.setPen(QColor(40, 44, 52))
        p.drawRect(bar_rect)

        font = QFont()
        font.setPointSizeF(8)
        p.setFont(font)
        fm = QFontMetrics(font)

        for idx in range(len(self._splits)):
            x = self._fraction_x(self._splits[idx])
            p.setPen(QColor(230, 180, 60))
            p.drawLine(int(x), int(bar_rect.top()), int(x), int(bar_rect.bottom()))
            handle = QRectF(x - 4, bar_rect.top() - 3, 8, 8)
            p.setBrush(QColor(255, 200, 60))
            p.drawEllipse(handle)
            val = self._splits[idx] * self._max_value
            label = f"{val:.{self._decimals}f}"
            tw = fm.horizontalAdvance(label)
            lx = x - tw / 2.0
            ly = int(bar_rect.bottom()) + scale(14)
            p.setPen(QColor(210, 210, 210))
            p.drawText(int(max(0, min(self.width() - tw, lx))), ly, label)

        def tick_label(fraction, txt, align_center=True):
            x = self._fraction_x(fraction)
            tw = fm.horizontalAdvance(txt)
            if align_center:
                lx = x - tw / 2.0
            else:
                lx = x
            lx = max(0.0, min(float(self.width()) - tw, lx))
            p.setPen(QColor(150, 150, 150))
            p.drawText(int(lx), int(bar_rect.top()) - scale(4), txt)

        tick_label(0.0, "0")
        tick_label(1.0, f"{self._max_value:.{self._decimals}f}")
        p.end()

    def _nearest_handle(self, x) -> int:
        best = -1
        best_d = 1e9
        for i, fr in enumerate(self._splits):
            hx = self._fraction_x(fr)
            d = abs(hx - x)
            if d < best_d:
                best_d = d
                best = i
        return best if best_d <= scale(10) else -1

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_index = self._nearest_handle(event.position().x())
            if self._drag_index >= 0:
                self._update_from_x(self._drag_index, event.position().x())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_index >= 0:
            self._update_from_x(self._drag_index, event.position().x())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._drag_index >= 0:
            self._drag_index = -1
            self.valueChanged.emit(list(self._splits))
        super().mouseReleaseEvent(event)

    def _update_from_x(self, index: int, x: float):
        lo = self._splits[index - 1] if index > 0 else 0.0
        hi = self._splits[index + 1] if index < len(self._splits) - 1 else 1.0
        fr = self._x_fraction(x)
        fr = max(lo + 0.001, min(hi - 0.001, fr))
        self._splits[index] = fr
        self.update()
