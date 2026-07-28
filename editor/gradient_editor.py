# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import qtawesome as qta
from PyQt6.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QLinearGradient, QPolygonF, QPalette
)

from editor.color_picker import ColorLineEdit


def _default_gradient():
    return [(0.0, [0.0, 0.0, 0.0, 1.0]), (1.0, [1.0, 1.0, 1.0, 1.0])]


def _sample_gradient(stops, t):
    if not stops:
        return [0, 0, 0, 1]
    t = max(0.0, min(1.0, t))
    stops = sorted(stops, key=lambda s: s[0])
    if t <= stops[0][0]:
        return list(stops[0][1])
    if t >= stops[-1][0]:
        return list(stops[-1][1])
    for i in range(len(stops) - 1):
        t1, c1 = stops[i]
        t2, c2 = stops[i + 1]
        if t1 <= t <= t2:
            if t2 - t1 < 1e-10:
                return list(c1)
            f = (t - t1) / (t2 - t1)
            return [c1[j] + (c2[j] - c1[j]) * f for j in range(4)]
    return list(stops[-1][1])


class GradientPreview(QWidget):
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stops = _default_gradient()
        self.setMinimumSize(100, 24)
        self.setMaximumHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_stops(self, stops):
        self._stops = stops
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if not self._stops:
            p.fillRect(0, 0, w, h, self.palette().color(QPalette.ColorRole.Window))
            p.setPen(self.palette().color(QPalette.ColorRole.PlaceholderText))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "empty")
            p.end()
            return
        grad = QLinearGradient(0, 0, w, 0)
        stops = sorted(self._stops, key=lambda s: s[0])
        for pos, rgba in stops:
            r, g, b, a = rgba
            grad.setColorAt(pos, QColor.fromRgbF(r, g, b, a))
        p.fillRect(0, 0, w, h, grad)
        p.setPen(QPen(self.palette().color(QPalette.ColorRole.Window).darker(130), 1))
        p.drawRect(0, 0, w - 1, h - 1)
        p.end()

    def mousePressEvent(self, event):
        self.clicked.emit()


class _GradientBar(QWidget):
    stopMoved = pyqtSignal()
    stopAdded = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stops = _default_gradient()
        self._selected_idx = -1
        self._dragging = False
        self.setMinimumSize(300, 40)
        self.setMaximumHeight(50)
        self.setMouseTracking(True)

    def set_stops(self, stops):
        self._stops = stops
        self.update()

    def _stop_positions(self):
        w = self.width() - 20
        if w < 1:
            return []
        return [(s[0] * w + 10, s[0]) for s in self._stops]

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        bar_top = 8
        bar_h = h - 24
        bar_rect = QRectF(10, bar_top, w - 20, bar_h)

        grad = QLinearGradient(bar_rect.topLeft(), bar_rect.topRight())
        stops = sorted(self._stops, key=lambda s: s[0])
        for pos, rgba in stops:
            r, g, b, a = rgba
            grad.setColorAt(pos, QColor.fromRgbF(r, g, b, a))
        p.fillRect(bar_rect, grad)
        p.setPen(QPen(self.palette().color(QPalette.ColorRole.Window).darker(130), 1))
        p.drawRect(bar_rect)

        for i, (pos, rgba) in enumerate(self._stops):
            x = pos * (w - 20) + 10
            r, g, b, a = rgba
            is_sel = i == self._selected_idx
            tri_size = 6
            tri_y = bar_top + bar_h
            path = [
                (x, tri_y + tri_size),
                (x - tri_size, tri_y),
                (x + tri_size, tri_y),
            ]
            p.setPen(QPen(Qt.GlobalColor.black if is_sel else self.palette().color(QPalette.ColorRole.Dark), 1))
            col = QColor.fromRgbF(r, g, b)
            p.setBrush(col)
            poly = QPolygonF()
            for a, b in path:
                poly.append(QPointF(a, b))
            p.drawPolygon(poly)

        p.end()

    def mousePressEvent(self, event):
        px = event.position().x()
        w = self.width() - 20
        if w < 1:
            return
        self._selected_idx = -1
        for i, (pos, _) in enumerate(self._stops):
            sx = pos * w + 10
            if abs(px - sx) < 10:
                self._selected_idx = i
                self._dragging = True
                self.update()
                return
        self._selected_idx = -1
        self.update()

    def mouseMoveEvent(self, event):
        if self._dragging and self._selected_idx >= 0:
            w = self.width() - 20
            if w < 1:
                return
            new_pos = max(0, min(1, (event.position().x() - 10) / w))
            self._stops[self._selected_idx] = (new_pos, self._stops[self._selected_idx][1])
            self._stops.sort(key=lambda s: s[0])
            new_idx = next(i for i, s in enumerate(self._stops) if abs(s[0] - new_pos) < 0.001)
            self._selected_idx = new_idx
            self.update()
            self.stopMoved.emit()

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            self.update()
            self.stopMoved.emit()

    def mouseDoubleClickEvent(self, event):
        w = self.width() - 20
        if w < 1:
            return
        pos = max(0, min(1, (event.position().x() - 10) / w))
        self.stopAdded.emit(pos)

    def selected_index(self):
        return self._selected_idx


class _StopRow(QWidget):
    removeRequested = pyqtSignal(int)
    changed = pyqtSignal()

    def __init__(self, index, pos, color, parent=None):
        super().__init__(parent)
        self._index = index
        self._pos = pos
        self._color = color

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(4)

        self._pos_spin = QDoubleSpinBox()
        self._pos_spin.setRange(0.0, 1.0)
        self._pos_spin.setSingleStep(0.01)
        self._pos_spin.setDecimals(3)
        self._pos_spin.setValue(pos)
        self._pos_spin.setFixedWidth(70)
        layout.addWidget(self._pos_spin)

        self._color_edit = ColorLineEdit(color)
        layout.addWidget(self._color_edit, 1)

        self._remove_btn = QPushButton(qta.icon("fa5s.times", color="#d4d4d4"), "")
        self._remove_btn.setFixedSize(24, 24)
        self._remove_btn.setToolTip("Remove stop")
        layout.addWidget(self._remove_btn)

        self._pos_spin.valueChanged.connect(self._on_changed)
        self._color_edit.colorChanged.connect(self._on_changed)
        self._remove_btn.clicked.connect(lambda: self.removeRequested.emit(self._index))

    def _on_changed(self):
        self.changed.emit()

    def get_data(self):
        return (self._pos_spin.value(), self._color_edit.get_color_rgba())

    def set_index(self, idx):
        self._index = idx
        self._remove_btn.setEnabled(idx >= 0)

    def sync_pos(self, pos):
        self._pos_spin.blockSignals(True)
        self._pos_spin.setValue(pos)
        self._pos_spin.blockSignals(False)


class GradientEditorDialog(QDialog):
    def __init__(self, stops=None, title="Gradient Editor", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(450, 350)
        self.resize(500, 400)

        self._stops = [list(s) for s in (stops or _default_gradient())]
        self._updating = False

        pal = self.palette()
        bg = pal.color(QPalette.ColorRole.Window)
        fg = pal.color(QPalette.ColorRole.WindowText)
        mid = pal.color(QPalette.ColorRole.Mid)
        highlight = pal.color(QPalette.ColorRole.Highlight)
        self.setStyleSheet(f"""
            QDialog {{ background: {bg.name()}; }}
            QLabel {{ color: {fg.name()}; font-size: 11px; background: transparent; }}
            QDoubleSpinBox {{
                background: {bg.name()}; color: {fg.name()};
                border: 1px solid {mid.name()}; border-radius: 3px;
                padding: 1px 2px 1px 4px; font-size: 11px; min-height: 20px;
                selection-background-color: {highlight.name()};
            }}
            QDoubleSpinBox:hover {{ border-color: {mid.name()}; }}
            QDoubleSpinBox:focus {{ border-color: {highlight.name()}; }}
            QPushButton {{
                color: {fg.name()}; background: {bg.name()};
                border: 1px solid {mid.name()}; border-radius: 3px;
                padding: 4px 16px; font-size: 11px;
            }}
            QPushButton:hover {{ background: {mid.name()}; color: {fg.name()}; }}
            QScrollArea {{ border: none; background: transparent; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._bar = _GradientBar()
        self._bar.set_stops(self._stops)
        self._bar.stopMoved.connect(self._on_bar_moved)
        self._bar.stopAdded.connect(self._on_bar_add)
        layout.addWidget(self._bar)

        stops_label = QLabel("Stops:")
        layout.addWidget(stops_label)

        self._stop_container = QWidget()
        self._stop_layout = QVBoxLayout(self._stop_container)
        self._stop_layout.setContentsMargins(0, 0, 0, 0)
        self._stop_layout.setSpacing(2)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._stop_container)
        scroll.setMaximumHeight(200)
        layout.addWidget(scroll, 1)

        self._rebuild_stop_rows()

        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add Stop")
        add_btn.clicked.connect(self._add_stop)
        btn_layout.addWidget(add_btn)
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _rebuild_stop_rows(self):
        while self._stop_layout.count():
            item = self._stop_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for i, (pos, rgba) in enumerate(self._stops):
            self._add_stop_row(i, pos, rgba)

    def _add_stop_row(self, idx, pos, rgba):
        row = _StopRow(idx, pos, rgba)
        row.changed.connect(self._on_row_changed)
        row.removeRequested.connect(self._remove_stop)
        self._stop_layout.addWidget(row)

    def _on_row_changed(self):
        if self._updating:
            return
        self._sync_stops_from_rows()

    def _sync_stops_from_rows(self):
        self._stops.clear()
        for i in range(self._stop_layout.count()):
            item = self._stop_layout.itemAt(i)
            if item and item.widget():
                row = item.widget()
                self._stops.append(row.get_data())
        self._stops.sort(key=lambda s: s[0])
        self._bar.set_stops(self._stops)

    def _on_bar_moved(self):
        if self._updating:
            return
        self._stops = self._bar._stops[:]
        self._rebuild_stop_rows()

    def _on_bar_add(self, pos):
        rgba = _sample_gradient(self._stops, pos)
        self._stops.append([pos, rgba])
        self._stops.sort(key=lambda s: s[0])
        self._bar.set_stops(self._stops)
        self._rebuild_stop_rows()

    def _add_stop(self):
        pos = 0.5
        rgba = _sample_gradient(self._stops, pos)
        self._stops.append([pos, rgba])
        self._stops.sort(key=lambda s: s[0])
        self._bar.set_stops(self._stops)
        self._rebuild_stop_rows()

    def _remove_stop(self, idx):
        if len(self._stops) <= 2:
            return
        del self._stops[idx]
        self._bar.set_stops(self._stops)
        self._rebuild_stop_rows()

    def get_stops(self):
        return [tuple(s) for s in self._stops]


class GradientLineEdit(QWidget):
    gradientChanged = pyqtSignal(object)

    def __init__(self, stops=None, parent=None):
        super().__init__(parent)
        self._stops = [list(s) for s in (stops or _default_gradient())]

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._preview = GradientPreview()
        self._preview.set_stops(self._stops)
        self._preview.setToolTip("Click to edit gradient")
        layout.addWidget(self._preview, 1)

        self._edit_btn = QPushButton("Edit")
        layout.addWidget(self._edit_btn)

        self._preview.clicked.connect(self._open_editor)
        self._edit_btn.clicked.connect(self._open_editor)

    def _open_editor(self):
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QCursor
        dlg = GradientEditorDialog(self._stops, "Edit Gradient", None)
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
        if dlg.exec():
            self._stops = [list(s) for s in dlg.get_stops()]
            self._preview.set_stops(self._stops)
            self.gradientChanged.emit(self._stops)

    def set_stops(self, stops):
        self._stops = [list(s) for s in stops]
        self._preview.set_stops(self._stops)

    def get_stops(self):
        return [tuple(s) for s in self._stops]
