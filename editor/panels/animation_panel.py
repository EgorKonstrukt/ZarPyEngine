# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Optional
import qtawesome as qta
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSplitter,
    QFrame, QDoubleSpinBox, QComboBox, QMenu,
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QRectF, QPointF, QLineF
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QMouseEvent, QWheelEvent,
    QKeyEvent, QAction, QPainterPath, QFontMetrics, QPalette,
)
from core.foundation.curve import Curve, CurveKey, TangentMode
from core.components.animation.animation_clip import AnimationClip, AnimationEvent
from core.config.editor_scale import scale, scale_xy

_FPS = 60
_TIMELINE_HEIGHT = 30
_KEYFRAME_SIZE = 8
_ROW_HEIGHT = 22
_LEFT_PANEL_WIDTH = 180
_LABEL_WIDTH = 180



class TimelineRuler(QWidget):
    scrubbed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clip: Optional[AnimationClip] = None
        self._current_time: float = 0.0
        self._pixels_per_second: float = 100.0
        self._dragging: bool = False
        self.setFixedHeight(scale(_TIMELINE_HEIGHT))
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_clip(self, clip: Optional[AnimationClip]):
        self._clip = clip
        self.update()

    def set_time(self, t: float):
        self._current_time = t
        self.update()

    def set_zoom(self, pps: float):
        self._pixels_per_second = max(10.0, pps)
        self.update()

    def _time_to_x(self, t: float) -> float:
        return t * self._pixels_per_second

    def _x_to_time(self, x: float) -> float:
        return x / max(self._pixels_per_second, 1.0)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self.palette().window())
        p.fillRect(0, 0, _LABEL_WIDTH, h, self.palette().window())
        p.setPen(QPen(self.palette().mid(), 1))
        p.drawLine(QPointF(_LABEL_WIDTH, 0), QPointF(_LABEL_WIDTH, h))
        clip_len = self._clip.length if self._clip else 1.0
        total_w = clip_len * self._pixels_per_second
        major_interval = self._get_major_interval()
        minor_interval = major_interval / 5.0
        p.setPen(QPen(self.palette().mid(), 1))
        t = 0.0
        while t <= clip_len:
            x = _LABEL_WIDTH + t * self._pixels_per_second
            is_major = abs(t % major_interval) < 0.001
            is_minor = abs(t % minor_interval) < 0.001
            if is_major:
                p.setPen(QPen(self.palette().mid(), 1))
                p.drawLine(QPointF(x, 8), QPointF(x, h))
                p.setPen(QPen(self.palette().placeholderText().color(), 1))
                font = p.font()
                font.setPointSize(8)
                p.setFont(font)
                p.drawText(QRectF(x + 3, 0, 50, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                           f"{t:.1f}s")
            elif is_minor:
                p.setPen(QPen(self.palette().mid(), 1))
                p.drawLine(QPointF(x, 14), QPointF(x, h))
            t += minor_interval
        cursor_x = _LABEL_WIDTH + self._current_time * self._pixels_per_second
        p.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 2))
        p.drawLine(QPointF(cursor_x, 0), QPointF(cursor_x, h))
        triangle = QPainterPath()
        triangle.moveTo(cursor_x - 5, h)
        triangle.lineTo(cursor_x + 5, h)
        triangle.lineTo(cursor_x, h - 8)
        triangle.closeSubpath()
        p.fillPath(triangle, self.palette().color(QPalette.ColorRole.Highlight))

    def _get_major_interval(self) -> float:
        raw = 50.0 / self._pixels_per_second
        exponents = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        for e in exponents:
            if e >= raw:
                return e
        return 10.0

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._set_time_from_event(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self._set_time_from_event(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._dragging = False

    def _set_time_from_event(self, event):
        t = max(0.0, self._x_to_time(event.position().x() - _LABEL_WIDTH))
        if self._clip:
            t = min(t, self._clip.length)
        self._current_time = t
        self.scrubbed.emit(t)
        self.update()


class Dopesheet(QWidget):
    key_selected = pyqtSignal(str, object)
    selection_changed = pyqtSignal()
    keys_modified = pyqtSignal()
    property_clicked = pyqtSignal(str)
    remove_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._clip: Optional[AnimationClip] = None
        self._current_time: float = 0.0
        self._pixels_per_second: float = 100.0
        self._selected_keys: list[tuple[str, CurveKey]] = []
        self._offset_y: float = 0.0
        self._selected_property: Optional[str] = None
        self._dragging_keys: bool = False
        self._drag_start_x: float = 0.0
        self._drag_orig_times: list[float] = []
        self.setMinimumHeight(100)
        self.setMouseTracking(True)

    def set_clip(self, clip: Optional[AnimationClip]):
        self._clip = clip
        self._selected_keys.clear()
        self.update()

    def set_time(self, t: float):
        self._current_time = t
        self.update()

    def set_zoom(self, pps: float):
        self._pixels_per_second = max(10.0, pps)
        self.update()

    def set_selected_property(self, prop: Optional[str]):
        self._selected_property = prop
        self.update()

    def _time_to_x(self, t: float) -> float:
        return t * self._pixels_per_second

    def _x_to_time(self, x: float) -> float:
        return x / max(self._pixels_per_second, 1.0)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        clip_len = self._clip.length if self._clip else 1.0
        p.fillRect(0, 0, w, h, self.palette().window())
        p.fillRect(0, 0, _LABEL_WIDTH, h, self.palette().window())
        p.setPen(QPen(self.palette().mid(), 1))
        p.drawLine(QPointF(_LABEL_WIDTH, 0), QPointF(_LABEL_WIDTH, h))
        total_w = clip_len * self._pixels_per_second
        for x in range(_LABEL_WIDTH, int(total_w) + _LABEL_WIDTH + 1, int(self._pixels_per_second)):
            p.setPen(QPen(self.palette().mid(), 1))
            p.drawLine(QPointF(x, 0), QPointF(x, h))
        cursor_x = _LABEL_WIDTH + self._current_time * self._pixels_per_second
        p.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 1))
        p.drawLine(QPointF(cursor_x, 0), QPointF(cursor_x, h))
        if not self._clip:
            p.setPen(self.palette().placeholderText().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No clip selected")
            p.end()
            return
        y = 4
        for path, curve in self._clip.curves.items():
            self._draw_curve_row(p, path, curve, y)
            y += _ROW_HEIGHT

    def _draw_curve_row(self, p: QPainter, path: str, curve: Curve, y: float):
        is_sel = path == self._selected_property
        if is_sel:
            p.fillRect(0, y, _LABEL_WIDTH, _ROW_HEIGHT, self.palette().highlight().color())
        p.setPen(QPen(self.palette().text() if is_sel else self.palette().placeholderText().color(), 1))
        fm = QFontMetrics(p.font())
        elided = fm.elidedText(path, Qt.TextElideMode.ElideRight, _LABEL_WIDTH - 8)
        p.drawText(QRectF(4, y, _LABEL_WIDTH - 8, _ROW_HEIGHT), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, elided)
        for k in curve.keys:
            kx = _LABEL_WIDTH + self._time_to_x(k.time)
            is_sel_key = (path, k) in self._selected_keys
            color = self.palette().color(QPalette.ColorRole.Highlight) if is_sel_key else self.palette().color(QPalette.ColorRole.Highlight)
            p.setPen(QPen(color, 1))
            p.setBrush(QBrush(color))
            p.drawRect(QRectF(kx - _KEYFRAME_SIZE / 2, y + _ROW_HEIGHT / 2 - _KEYFRAME_SIZE / 2,
                              _KEYFRAME_SIZE, _KEYFRAME_SIZE))

    def _get_row_path(self, my: float) -> Optional[str]:
        row = int((my - 4) / _ROW_HEIGHT)
        paths = list(self._clip.curves.keys()) if self._clip else []
        if 0 <= row < len(paths):
            return paths[row]
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if not self._clip:
            return
        mx = event.position().x()
        my = event.position().y()
        if event.button() == Qt.MouseButton.LeftButton:
            if mx < _LABEL_WIDTH:
                path = self._get_row_path(my)
                if path:
                    self._selected_property = path
                    self.property_clicked.emit(path)
                    self.update()
                return
            path = self._get_row_path(my)
            if path:
                curve = self._clip.curves[path]
                for k in curve.keys:
                    kx = _LABEL_WIDTH + self._time_to_x(k.time)
                    if abs(mx - kx) < _KEYFRAME_SIZE:
                        self._selected_keys = [(path, k)]
                        self._selected_property = path
                        self._dragging_keys = True
                        self._drag_start_x = mx
                        self._drag_orig_times = [k.time]
                        self.key_selected.emit(path, k)
                        self.property_clicked.emit(path)
                        self.selection_changed.emit()
                        self.update()
                        return

    def contextMenuEvent(self, event):
        if not self._clip:
            return
        my = event.pos().y()
        path = self._get_row_path(my)
        if path:
            menu = QMenu(self)
            act = menu.addAction(f"Remove {path}")
            act.triggered.connect(lambda: self.remove_requested.emit(path))
            menu.exec(event.globalPos())

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging_keys and self._selected_keys and self._clip:
            dx = self._x_to_time(event.position().x() - _LABEL_WIDTH) - self._x_to_time(self._drag_start_x - _LABEL_WIDTH)
            for (path, k), orig_t in zip(self._selected_keys, self._drag_orig_times):
                if path in self._clip.curves:
                    k.time = max(0.0, orig_t + dx)
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_keys:
            self._dragging_keys = False
            self.keys_modified.emit()

    def key_to_time(self, key) -> float:
        return key.time


class CurveEditorWidget(QWidget):
    curve_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._curve: Optional[Curve] = None
        self._path: str = ""
        self._selected_key: Optional[CurveKey] = None
        self._dragging_key: Optional[CurveKey] = None
        self._drag_orig_time: float = 0.0
        self._drag_orig_val: float = 0.0
        self._view_rect = QRectF(-0.5, -1.5, 2.0, 3.0)
        self._margin = 40
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_curve(self, path: str, curve: Optional[Curve]):
        self._path = path
        self._curve = curve
        self._selected_key = None
        self.update()

    def _map_to_curve(self, pos: QPointF) -> QPointF:
        w = self.width() - self._margin * 2
        h = self.height() - self._margin * 2
        if w <= 0 or h <= 0:
            return QPointF(0, 0)
        x = self._view_rect.left() + (pos.x() - self._margin) / w * self._view_rect.width()
        y = self._view_rect.top() + (pos.y() - self._margin) / h * self._view_rect.height()
        return QPointF(x, y)

    def _map_from_curve(self, pt: QPointF) -> QPointF:
        w = self.width() - self._margin * 2
        h = self.height() - self._margin * 2
        if w <= 0 or h <= 0:
            return QPointF(0, 0)
        x = self._margin + (pt.x() - self._view_rect.left()) / self._view_rect.width() * w
        y = self._margin + (pt.y() - self._view_rect.top()) / self._view_rect.height() * h
        return QPointF(x, y)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, self.palette().window())
        p.setPen(QPen(self.palette().mid(), 1))
        p.drawRect(self._margin, self._margin, w - self._margin * 2, h - self._margin * 2)
        grid_x = self._view_rect.width() / 5.0
        grid_y = self._view_rect.height() / 5.0
        for i in range(1, 6):
            gx = self._margin + i * (w - self._margin * 2) / 5.0
            p.setPen(QPen(self.palette().mid(), 1))
            p.drawLine(QPointF(gx, self._margin), QPointF(gx, h - self._margin))
        for i in range(1, 6):
            gy = self._margin + i * (h - self._margin * 2) / 5.0
            p.setPen(QPen(self.palette().mid(), 1))
            p.drawLine(QPointF(self._margin, gy), QPointF(w - self._margin, gy))
        p.setPen(QPen(self.palette().mid(), 1))
        zero_y = self._map_from_curve(QPointF(0, 0)).y()
        p.drawLine(QPointF(self._margin, zero_y), QPointF(w - self._margin, zero_y))
        if not self._curve or len(self._curve.keys) < 1:
            p.setPen(self.palette().placeholderText().color())
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"No keys for {self._path}" if self._path else "Select a property")
            p.end()
            return
        path = QPainterPath()
        samples = 200
        first = True
        for i in range(samples + 1):
            t = self._view_rect.left() + i / samples * self._view_rect.width()
            v = self._curve.evaluate(t)
            pt = self._map_from_curve(QPointF(t, v))
            if first:
                path.moveTo(pt)
                first = False
            else:
                path.lineTo(pt)
        p.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        for k in self._curve.keys:
            pt = self._map_from_curve(QPointF(k.time, k.value))
            is_sel = k is self._selected_key
            color = self.palette().color(QPalette.ColorRole.Highlight) if is_sel else self.palette().color(QPalette.ColorRole.Highlight)
            p.setPen(QPen(color, 1))
            p.setBrush(QBrush(color))
            p.drawEllipse(pt, 5, 5)
            if k.tangent_mode != TangentMode.CONSTANT:
                in_pt = self._map_from_curve(QPointF(k.time - 0.3, k.value - k.in_tangent * 0.3))
                out_pt = self._map_from_curve(QPointF(k.time + 0.3, k.value + k.out_tangent * 0.3))
                p.setPen(QPen(self.palette().text(), 1))
                p.drawLine(in_pt, pt)
                p.drawLine(pt, out_pt)
                p.setPen(QPen(self.palette().text(), 2))
                p.drawEllipse(in_pt, 3, 3)
                p.drawEllipse(out_pt, 3, 3)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._curve:
            mp = event.position()
            for k in self._curve.keys:
                kpt = self._map_from_curve(QPointF(k.time, k.value))
                if (mp - kpt).manhattanLength() < 8:
                    self._selected_key = k
                    self._dragging_key = k
                    self._drag_orig_time = k.time
                    self._drag_orig_val = k.value
                    self.update()
                    return
            self._selected_key = None
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging_key and self._curve:
            mp = event.position()
            curve_pt = self._map_to_curve(mp)
            self._dragging_key.time = max(0.0, curve_pt.x())
            self._dragging_key.value = curve_pt.y()
            self._curve._auto_smooth()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._dragging_key and self._curve:
            self._curve._auto_smooth()
            self.curve_changed.emit()
        self._dragging_key = None

    def wheelEvent(self, event: QWheelEvent):
        zoom = 1.0 + event.angleDelta().y() / 1200.0
        cw = self._view_rect.width() / zoom
        ch = self._view_rect.height() / zoom
        cx = self._view_rect.center().x()
        cy = self._view_rect.center().y()
        self._view_rect = QRectF(cx - cw / 2, cy - ch / 2, cw, ch)
        self.update()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Delete and self._selected_key and self._curve:
            self._curve.remove_key(self._selected_key)
            self._selected_key = None
            self.curve_changed.emit()
            self.update()

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        return QSize(200, 150)


class AnimationPanel(QDockWidget):
    def __init__(self, engine, main_window):
        super().__init__("Animation", main_window)
        self._engine = engine
        self._main_window = main_window
        self._clip: Optional[AnimationClip] = None
        self._clip_path: str = ""
        self._entity: Optional[object] = None
        self._is_playing: bool = False
        self._current_time: float = 0.0
        self._pixels_per_second: float = 100.0
        self._selected_property: Optional[str] = None
        self._dirty: bool = False
        self.setObjectName("AnimationDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setMinimumSize(300, 200)
        self._build_ui()
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._tick)

    def _build_ui(self):
        central = QWidget()
        self.setWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._build_toolbar())
        splitter = QSplitter(Qt.Orientation.Vertical)
        self._timeline_ruler = TimelineRuler()
        self._timeline_ruler.scrubbed.connect(self._on_scrub)
        self._dopesheet = Dopesheet()
        self._dopesheet.key_selected.connect(self._on_key_selected)
        self._dopesheet.keys_modified.connect(self._mark_dirty)
        self._dopesheet.property_clicked.connect(self._on_property_clicked)
        self._dopesheet.remove_requested.connect(self._on_remove_requested)
        self._curve_editor = CurveEditorWidget()
        self._curve_editor.curve_changed.connect(self._mark_dirty)
        timeline_widget = QWidget()
        tl_layout = QVBoxLayout(timeline_widget)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        tl_layout.setSpacing(0)
        tl_layout.addWidget(self._timeline_ruler)
        tl_layout.addWidget(self._dopesheet, 1)
        splitter.addWidget(timeline_widget)
        splitter.addWidget(self._curve_editor)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter, 1)
        self._update_ui()

    def _build_toolbar(self) -> QWidget:
        tb = QWidget()
        layout = QHBoxLayout(tb)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)
        self._play_btn = QPushButton(qta.icon("fa5s.play", color="#d4d4d4"), "")
        self._play_btn.setFixedSize(*scale_xy(26, 22))
        self._play_btn.clicked.connect(self._toggle_play)
        layout.addWidget(self._play_btn)
        self._stop_btn = QPushButton(qta.icon("fa5s.stop", color="#d4d4d4"), "")
        self._stop_btn.setFixedSize(*scale_xy(26, 22))
        self._stop_btn.clicked.connect(self._stop_playback)
        layout.addWidget(self._stop_btn)
        self._record_btn = QPushButton(qta.icon("fa5s.dot-circle", color="#ff4444"), "")
        self._record_btn.setFixedSize(*scale_xy(26, 22))
        layout.addWidget(self._record_btn)
        self._key_btn = QPushButton(qta.icon("fa5s.key", color="#d4d4d4"), "")
        self._key_btn.setFixedSize(*scale_xy(26, 22))
        self._key_btn.setToolTip("Add Keyframe")
        self._key_btn.clicked.connect(self._add_keyframe)
        layout.addWidget(self._key_btn)
        self._time_spin = QDoubleSpinBox()
        self._time_spin.setRange(0.0, 99999.0)
        self._time_spin.setSingleStep(0.01)
        self._time_spin.setDecimals(2)
        self._time_spin.setValue(0.0)
        self._time_spin.valueChanged.connect(self._on_spin_time)
        self._time_spin.setFixedWidth(scale(70))
        self._time_spin.setStyleSheet("border: 1px solid; border-radius: 3px; padding: 1px 4px;")
        layout.addWidget(self._time_spin)
        self._add_prop_btn = QPushButton(qta.icon("fa5s.plus", color="#9ccc65"), " Add Property")
        self._add_prop_btn.setFixedHeight(scale(22))
        self._add_prop_btn.clicked.connect(self._add_property)
        layout.addWidget(self._add_prop_btn)
        self._clip_combo = QComboBox()
        self._clip_combo.setMinimumWidth(150)
        self._clip_combo.setStyleSheet("border: 1px solid; border-radius: 3px; padding: 1px 4px;")
        layout.addWidget(self._clip_combo)
        layout.addStretch()
        return tb

    def _btn_style(self) -> str:
        return ""


    def set_entity(self, entity):
        self._entity = entity
        if entity:
            anim = entity.get_component_by_name("Animation")
            if anim and anim.clip:
                self.load_clip_from_path(anim.clip)
                return
        self.load_clip_from_path("")

    def load_clip_from_path(self, path: str):
        if self._dirty and self._clip and self._clip_path:
            self._clip.save(self._clip_path)
            self._dirty = False
        self._clip_path = path
        if path:
            clip = AnimationClip.load(path)
            self.load_clip(clip)
        else:
            self.load_clip(None)

    def load_clip(self, clip: Optional[AnimationClip]):
        if self._dirty and self._clip and self._clip_path:
            self._clip.save(self._clip_path)
            self._dirty = False
        self._clip = clip
        self._current_time = 0.0
        self._timeline_ruler.set_clip(clip)
        self._dopesheet.set_clip(clip)
        self._curve_editor.set_curve("", None)
        self._selected_property = None
        self._dopesheet.set_selected_property(None)
        self._update_property_list()
        self._update_ui()
        self._timeline_ruler.set_time(0.0)
        self._dopesheet.set_time(0.0)
        self._time_spin.setValue(0.0)

    def _update_property_list(self):
        self._dopesheet.update()

    def _mark_dirty(self):
        self._dirty = True

    def _add_property(self):
        if not self._clip or not self._entity:
            return
        menu = QMenu(self)
        self._build_property_menu(menu)
        btn = self.sender() if self.sender() else self._add_prop_btn
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))

    def _build_property_menu(self, menu: QMenu):
        from core.components.inspector_meta import FieldType
        for comp in self._entity.get_all_components():
            comp_name = type(comp).__name__
            comp_menu = menu.addMenu(comp_name)
            fields = []
            if hasattr(comp, '_inspector_fields'):
                fields = comp._inspector_fields()
            for f in fields:
                if f.field_type == FieldType.FLOAT:
                    self._add_prop_action(comp_menu, f.label, f"{comp_name}/{f.name}")
                elif f.field_type == FieldType.INT:
                    self._add_prop_action(comp_menu, f.label, f"{comp_name}/{f.name}")
                elif f.field_type == FieldType.BOOL:
                    self._add_prop_action(comp_menu, f.label, f"{comp_name}/{f.name}")
                elif f.field_type == FieldType.VEC3:
                    sub = comp_menu.addMenu(f.label)
                    self._add_prop_action(sub, "x", f"{comp_name}/{f.name}.x")
                    self._add_prop_action(sub, "y", f"{comp_name}/{f.name}.y")
                    self._add_prop_action(sub, "z", f"{comp_name}/{f.name}.z")
                elif f.field_type == FieldType.VEC2:
                    sub = comp_menu.addMenu(f.label)
                    self._add_prop_action(sub, "x", f"{comp_name}/{f.name}.x")
                    self._add_prop_action(sub, "y", f"{comp_name}/{f.name}.y")
                elif f.field_type == FieldType.COLOR:
                    sub = comp_menu.addMenu(f.label)
                    self._add_prop_action(sub, "r", f"{comp_name}/{f.name}.r")
                    self._add_prop_action(sub, "g", f"{comp_name}/{f.name}.g")
                    self._add_prop_action(sub, "b", f"{comp_name}/{f.name}.b")
                    self._add_prop_action(sub, "a", f"{comp_name}/{f.name}.a")

    def _add_prop_action(self, menu: QMenu, label: str, path: str):
        action = menu.addAction(label)
        action.setData(path)
        action.triggered.connect(lambda checked=False, p=path: self._on_add_property(p))

    def _on_add_property(self, path: str):
        if self._clip:
            self._clip.add_curve(path)
            self._mark_dirty()
            self._update_property_list()

    def _on_property_clicked(self, path: str):
        self._selected_property = path
        if self._clip and path in self._clip.curves:
            self._curve_editor.set_curve(path, self._clip.curves[path])
            self._dopesheet.set_selected_property(path)

    def _on_remove_requested(self, path: str):
        if self._clip:
            self._clip.remove_curve(path)
            self._mark_dirty()
            self._update_property_list()
            if self._selected_property == path:
                self._selected_property = None
                self._curve_editor.set_curve("", None)

    def _on_key_selected(self, path, key):
        if self._clip and path in self._clip.curves:
            self._selected_property = path
            self._dopesheet.set_selected_property(path)
            self._curve_editor.set_curve(path, self._clip.curves[path])

    def _on_scrub(self, t: float):
        self._current_time = t
        self._time_spin.blockSignals(True)
        self._time_spin.setValue(t)
        self._time_spin.blockSignals(False)
        self._dopesheet.set_time(t)

    def _on_spin_time(self, t: float):
        self._current_time = t
        self._timeline_ruler.set_time(t)
        self._dopesheet.set_time(t)

    def _toggle_play(self):
        if self._is_playing:
            self._is_playing = False
            self._play_btn.setIcon(qta.icon("fa5s.play", color="#d4d4d4"))
            self._play_timer.stop()
        else:
            self._is_playing = True
            self._play_btn.setIcon(qta.icon("fa5s.pause", color="#d4d4d4"))
            self._play_timer.start(int(1000 / _FPS))

    def _stop_playback(self):
        self._is_playing = False
        self._play_btn.setIcon(qta.icon("fa5s.play", color="#d4d4d4"))
        self._play_timer.stop()
        self._current_time = 0.0
        self._timeline_ruler.set_time(0.0)
        self._dopesheet.set_time(0.0)
        self._time_spin.setValue(0.0)

    def _tick(self):
        if not self._clip:
            return
        dt = 1.0 / _FPS
        self._current_time += dt
        if self._current_time >= self._clip.length:
            if self._clip.loop:
                self._current_time %= self._clip.length
            else:
                self._current_time = self._clip.length
                self._toggle_play()
        self._timeline_ruler.set_time(self._current_time)
        self._dopesheet.set_time(self._current_time)
        self._time_spin.blockSignals(True)
        self._time_spin.setValue(self._current_time)
        self._time_spin.blockSignals(False)

    def _add_keyframe(self):
        if not self._clip or not self._selected_property:
            return
        curve = self._clip.add_curve(self._selected_property)
        curve.add_key(self._current_time, 0.0)
        self._mark_dirty()
        self._update_property_list()
        self._curve_editor.set_curve(self._selected_property, curve)

    def _update_ui(self):
        has_clip = self._clip is not None
        self._play_btn.setEnabled(has_clip)
        self._stop_btn.setEnabled(has_clip)
        self._key_btn.setEnabled(has_clip)
        self._add_prop_btn.setEnabled(has_clip)

    def load_config(self, config):
        pass

