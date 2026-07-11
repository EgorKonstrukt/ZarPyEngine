# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
from typing import Optional, Any
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QMouseEvent, QKeyEvent, QWheelEvent, QEnterEvent
from PyQt6.QtWidgets import QWidget, QSizePolicy, QApplication, QMdiArea, QMdiSubWindow
from core.gui.widgets import (
    Panel, Button, Label, ANCHOR_STRETCH_ALL,
    WIDGET_REGISTRY, apply_fusion_style,
)


NATIVE_WIDGET_TYPES = {"Button", "Label", "Slider", "TextInput", "Toggle",
                       "ProgressBar", "Dropdown", "ScrollPanel", "Image", "Panel",
                       "RadioButton", "ListWidget", "TableWidget", "TreeWidget",
                       "TabWidget", "GroupBox", "SpinBox", "DoubleSpinBox",
                       "TextEdit", "Dial", "HtmlView",
                       "Splitter", "StackedWidget", "ToolBox", "Calendar",
                       "LCDNumber", "PlainText", "ScrollBar", "ToolButton", "FontCombo"}


class GuiCanvas(QWidget):
    widget_selected = pyqtSignal(object)
    widget_changed = pyqtSignal(object)
    scene_modified = pyqtSignal()
    copy_requested = pyqtSignal()
    paste_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._screen_w = 1920
        self._screen_h = 1080
        self._root = Panel(0, 0, 100, 100, self)
        self._root.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._root.setAutoFillBackground(False)
        self._root._anchor = ANCHOR_STRETCH_ALL
        self._selected_widget: Optional[QWidget] = None
        self._selected_widgets: list[QWidget] = []
        self._hovered_widget: Optional[QWidget] = None
        self._edit_mode = False
        self._drag_widget: Optional[QWidget] = None
        self._drag_handle = ""
        self._drag_start_x = 0.0
        self._drag_start_y = 0.0
        self._drag_start_w = 0.0
        self._drag_start_h = 0.0
        self._drag_start_mx = 0
        self._drag_start_my = 0
        self._drag_start_positions: dict[int, tuple[int, int]] = {}
        self._selection_box_start: Optional[tuple[float, float]] = None
        self._selection_rect: Optional[tuple[float, float, float, float]] = None
        self._is_selecting = False
        self._snap_to_grid = True
        self._grid_size = 8
        self._show_grid = True
        self._show_selection = True
        self._auto_align = True
        self._guide_lines: list[tuple[float, float, float, float]] = []
        self._background_color = "#1a1a1a"
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._is_panning = False
        self._pan_start_mx = 0
        self._pan_start_my = 0
        self._pan_start_px = 0.0
        self._pan_start_py = 0.0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(100, 100)
        apply_fusion_style(self)

    @property
    def root(self) -> Panel:
        return self._root

    @property
    def edit_mode(self) -> bool:
        return self._edit_mode

    @edit_mode.setter
    def edit_mode(self, v: bool):
        self._edit_mode = v
        for child in self._root.children():
            if isinstance(child, QWidget):
                child.setEnabled(not v if hasattr(child, 'setEnabled') else True)
        self._update_all_anchors(self.width(), self.height())
        self._apply_z_order()
        self.update()

    @property
    def selected_widget(self) -> Optional[QWidget]:
        return self._selected_widget

    @selected_widget.setter
    def selected_widget(self, w: Optional[QWidget]):
        if self._selected_widget != w:
            self._selected_widget = w
            if w is None:
                self._selected_widgets.clear()
            elif w not in self._selected_widgets:
                self._selected_widgets[:] = [w]
            self.widget_selected.emit(w)
            self.update()

    def toggle_selection(self, widget: QWidget):
        if widget in self._selected_widgets:
            self._selected_widgets.remove(widget)
            self._selected_widget = self._selected_widgets[-1] if self._selected_widgets else None
        else:
            self._selected_widgets.append(widget)
            self._selected_widget = widget
        self.widget_selected.emit(self._selected_widget)
        self.update()

    def _select_widgets_in_rect(self, x1, y1, x2, y2):
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        selected = []
        for child in self._root.children():
            if not isinstance(child, QWidget) or not child.isVisible():
                continue
            ox, oy = self._get_widget_root_offset(child)
            cx, cy = child.x() + ox, child.y() + oy
            cw, ch = child.width(), child.height()
            if cx < x2 and cx + cw > x1 and cy < y2 and cy + ch > y1:
                selected.append(child)
        self._selected_widgets = selected[:]
        self._selected_widget = selected[-1] if selected else None
        self.widget_selected.emit(self._selected_widget)

    def _store_drag_start_positions(self):
        self._drag_start_positions.clear()
        for w in self._selected_widgets:
            try:
                self._drag_start_positions[id(w)] = (w.x(), w.y())
            except RuntimeError:
                pass

    @property
    def auto_align(self) -> bool:
        return self._auto_align

    @auto_align.setter
    def auto_align(self, v: bool):
        self._auto_align = v
        if not v:
            self._guide_lines = []
            self.update()

    @property
    def snap_to_grid(self) -> bool:
        return self._snap_to_grid

    @snap_to_grid.setter
    def snap_to_grid(self, v: bool):
        self._snap_to_grid = v

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @grid_size.setter
    def grid_size(self, v: int):
        self._grid_size = max(1, v)

    @property
    def zoom(self) -> float:
        return self._zoom

    @zoom.setter
    def zoom(self, v: float):
        self._zoom = max(0.1, min(10.0, v))
        self._update_display()
        self.zoom_changed.emit(self._zoom)

    zoom_changed = pyqtSignal(float)

    @property
    def screen_width(self) -> int:
        return self._screen_w

    @screen_width.setter
    def screen_width(self, v: int):
        self._screen_w = max(320, v)
        self._update_root_geometry()
        self.update()

    @property
    def screen_height(self) -> int:
        return self._screen_h

    @screen_height.setter
    def screen_height(self, v: int):
        self._screen_h = max(240, v)
        self._update_root_geometry()
        self.update()

    def _to_root(self, sx: float, sy: float) -> tuple[float, float]:
        return sx + self._pan_x, sy + self._pan_y

    def find_by_id(self, wid: str) -> Optional[QWidget]:
        for child in self._root.children():
            if isinstance(child, QWidget) and getattr(child, '_widget_id', None) == wid:
                return child
        return None

    def add_widget(self, widget: QWidget, parent_widget: Optional[QWidget] = None) -> QWidget:
        parent = parent_widget or self._root
        widget.setParent(parent)
        widget.setVisible(True)
        if hasattr(widget, 'update_anchor'):
            widget.update_anchor(self.width(), self.height())
        self._apply_z_order()
        self.update()
        return widget

    def remove_widget(self, widget: QWidget):
        if self._selected_widget is widget:
            self._selected_widget = None
        if widget in self._selected_widgets:
            self._selected_widgets.remove(widget)
        widget.setParent(None)
        widget.deleteLater()
        self.update()

    def clear(self):
        for child in list(self._root.children()):
            if isinstance(child, QWidget):
                child.setParent(None)
                child.deleteLater()
        self._selected_widget = None
        self._selected_widgets.clear()
        self.update()

    def get_all_widgets(self) -> list[QWidget]:
        return [c for c in self._root.children() if isinstance(c, QWidget)]

    def _apply_z_order(self):
        ordered: list[QWidget] = []
        def _collect(parent):
            for child in parent.children():
                if isinstance(child, QWidget):
                    ordered.append(child)
                    _collect(child)
        _collect(self._root)
        ordered.sort(key=lambda w: getattr(w, '_z_index', 0))
        for w in ordered:
            try:
                w.raise_()
            except RuntimeError:
                pass

    def _update_all_anchors(self, vw: float, vh: float):
        def _walk(parent):
            for child in parent.children():
                if not isinstance(child, QWidget):
                    continue
                if hasattr(child, 'update_anchor'):
                    try:
                        child.update_anchor(vw, vh)
                    except RuntimeError:
                        pass
                _walk(child)
        _walk(self._root)

    def is_point_over_widget(self, px: float, py: float) -> bool:
        hit = self.hit_test_widget(px, py)
        return hit is not None and hit is not self._root

    def resize_canvas(self, w: float, h: float):
        pass

    def _update_display(self):
        self._update_root_geometry()
        self._update_all_anchors(self.width(), self.height())
        self.update()

    def _update_root_geometry(self):
        self._root.setGeometry(int(-self._pan_x), int(-self._pan_y), self.width(), self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_root_geometry()
        self._update_all_anchors(self.width(), self.height())

    def _snap_value(self, v: float) -> float:
        if self._snap_to_grid and self._grid_size > 0:
            return round(v / self._grid_size) * self._grid_size
        return v

    def _get_all_widget_edges(self, exclude_widget) -> list[tuple[str, float]]:
        edges: list[tuple[str, float]] = []
        for child in self._root.children():
            if not isinstance(child, QWidget) or child is exclude_widget or child is self._root:
                continue
            if not child.isVisible():
                continue
            ox, oy = self._get_widget_root_offset(child)
            cl = child.x() + ox
            ct = child.y() + oy
            cr = cl + child.width()
            cb = ct + child.height()
            cx = (cl + cr) / 2
            cy = (ct + cb) / 2
            edges += [('left', cl), ('right', cr), ('cx', cx),
                      ('top', ct), ('bottom', cb), ('cy', cy)]
        edges += [('left', 0), ('right', self._screen_w),
                  ('top', 0), ('bottom', self._screen_h),
                  ('cx', self._screen_w / 2), ('cy', self._screen_h / 2)]
        return edges

    def _snap_axis(self, moving: dict[str, float], others: list[tuple[str, float]],
                   threshold: float) -> tuple[float | None, str | None, float | None, tuple | None]:
        best_dist = threshold
        best_my_name: str | None = None
        best_other_val: float | None = None
        best_guide: tuple | None = None
        for my_name, my_val in moving.items():
            for other_name, other_val in others:
                dist = abs(my_val - other_val)
                if dist < best_dist:
                    best_dist = dist
                    best_my_name = my_name
                    best_other_val = other_val
                    is_vert = my_name in ('left', 'right', 'cx')
                    v = other_val - self._pan_x if is_vert else other_val - self._pan_y
                    if is_vert:
                        best_guide = (v, 0, v, self.height())
                    else:
                        best_guide = (0, v, self.width(), v)
        return best_dist, best_my_name, best_other_val, best_guide

    def _snap_move_widget(self, widget: QWidget, new_x: float, new_y: float) -> tuple[float, float]:
        w_off_x, w_off_y = self._get_widget_root_offset(widget)
        rw = widget.width()
        rh = widget.height()
        vert = {'left': new_x + w_off_x, 'right': new_x + w_off_x + rw, 'cx': new_x + w_off_x + rw / 2}
        horiz = {'top': new_y + w_off_y, 'bottom': new_y + w_off_y + rh, 'cy': new_y + w_off_y + rh / 2}
        others = self._get_all_widget_edges(widget)
        t = 6.0
        _, sx_name, sx_val, gx = self._snap_axis(vert, others, t)
        _, sy_name, sy_val, gy = self._snap_axis(horiz, others, t)
        self._guide_lines = []
        if gx: self._guide_lines.append(gx)
        if gy: self._guide_lines.append(gy)
        nrx = new_x + w_off_x
        nry = new_y + w_off_y
        if sx_name and sx_val is not None:
            nrx = (sx_val if sx_name == 'left' else
                   sx_val - rw if sx_name == 'right' else
                   sx_val - rw / 2)
        if sy_name and sy_val is not None:
            nry = (sy_val if sy_name == 'top' else
                   sy_val - rh if sy_name == 'bottom' else
                   sy_val - rh / 2)
        return nrx - w_off_x, nry - w_off_y

    def _snap_resize_edges(self, widget: QWidget, left: float, top: float,
                           right: float, bottom: float, handle: str
                           ) -> tuple[float, float, float, float, list]:
        w_off_x, w_off_y = self._get_widget_root_offset(widget)
        moving_vert: dict[str, float] = {}
        moving_horiz: dict[str, float] = {}
        if "w" in handle: moving_vert['left'] = left + w_off_x
        if "e" in handle: moving_vert['right'] = right + w_off_x
        if "n" in handle: moving_horiz['top'] = top + w_off_y
        if "s" in handle: moving_horiz['bottom'] = bottom + w_off_y
        if not moving_vert and not moving_horiz:
            return left, top, right, bottom, []
        others = self._get_all_widget_edges(widget)
        t = 6.0
        _, sx_name, sx_val, gx = self._snap_axis(moving_vert, others, t)
        _, sy_name, sy_val, gy = self._snap_axis(moving_horiz, others, t)
        guides = []
        if gx: guides.append(gx)
        if gy: guides.append(gy)
        if sx_val is not None and sx_name == 'left':
            left = sx_val - w_off_x
        elif sx_val is not None and sx_name == 'right':
            right = sx_val - w_off_x
        if sy_val is not None and sy_name == 'top':
            top = sy_val - w_off_y
        elif sy_val is not None and sy_name == 'bottom':
            bottom = sy_val - w_off_y
        return left, top, right, bottom, guides

    def _get_widget_root_offset(self, w: QWidget) -> tuple[int, int]:
        try:
            ox, oy = 0, 0
            p = w.parent()
            while p and p is not self._root:
                ox += p.x()
                oy += p.y()
                p = p.parent()
            return ox, oy
        except RuntimeError:
            return 0, 0

    def _get_handle_at(self, wx: float, wy: float, w: QWidget) -> str:
        try:
            _ = w.parent()
        except RuntimeError:
            return ""
        ox, oy = self._get_widget_root_offset(w)
        lx = wx - (w.x() + ox)
        ly = wy - (w.y() + oy)
        gw, gh = w.width(), w.height()
        hs = 6
        if abs(lx) < hs and abs(ly) < hs: return "nw"
        if abs(lx - gw) < hs and abs(ly) < hs: return "ne"
        if abs(lx) < hs and abs(ly - gh) < hs: return "sw"
        if abs(lx - gw) < hs and abs(ly - gh) < hs: return "se"
        if abs(lx - gw / 2) < hs and abs(ly) < hs: return "n"
        if abs(lx - gw / 2) < hs and abs(ly - gh) < hs: return "s"
        if abs(lx) < hs and abs(ly - gh / 2) < hs: return "w"
        if abs(lx - gw) < hs and abs(ly - gh / 2) < hs: return "e"
        return ""

    def _resize_from_handle(self, w: QWidget, handle: str, dx: float, dy: float):
        if dx == 0 and dy == 0:
            return
        ox = self._drag_start_x
        oy = self._drag_start_y
        ow = self._drag_start_w
        oh = self._drag_start_h

        left, right = ox, ox + ow
        top, bottom = oy, oy + oh

        if "n" in handle: top = oy + dy
        if "s" in handle: bottom = oy + oh + dy
        if "w" in handle: left = ox + dx
        if "e" in handle: right = ox + ow + dx

        # Snap only moving edges to grid
        if "n" in handle: top = self._snap_value(top)
        if "s" in handle: bottom = self._snap_value(bottom)
        if "w" in handle: left = self._snap_value(left)
        if "e" in handle: right = self._snap_value(right)

        # Shift aspect-ratio lock (corner handles only)
        is_corner = len(handle) == 2
        if is_corner and ow > 0 and oh > 0:
            mods = QApplication.keyboardModifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                aspect = ow / oh
                rw, rh = right - left, bottom - top
                if abs(dx) >= abs(dy):
                    rh = round(rw / aspect)
                else:
                    rw = round(rh * aspect)
                if "n" in handle: top = bottom - rh
                if "w" in handle: left = right - rw
                if "e" in handle: right = left + rw
                if "s" in handle: bottom = top + rh

        # Widget auto-align snap (moving edges only)
        if self._auto_align:
            left, top, right, bottom, gl = self._snap_resize_edges(w, left, top, right, bottom, handle)
            self._guide_lines = gl
        else:
            self._guide_lines = []

        nw = right - left
        nh = bottom - top

        if nw >= 8 and nh >= 8:
            w.setGeometry(int(left), int(top), int(nw), int(nh))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        vw, vh = self.width(), self.height()
        painter.fillRect(0, 0, vw, vh, QColor(self._background_color))
        if self._show_grid and self._edit_mode:
            gs = int(self._grid_size * self._zoom)
            if gs > 0:
                painter.setPen(QPen(QColor("#333333"), 1))
                start_x = int((-self._pan_x) % gs) if gs > 0 else 0
                for gx in range(start_x, vw, gs):
                    painter.drawLine(gx, 0, gx, vh)
                start_y = int((-self._pan_y) % gs) if gs > 0 else 0
                for gy in range(start_y, vh, gs):
                    painter.drawLine(0, gy, vw, gy)
        if self._edit_mode:
            self._draw_screen_overlay(painter)
            if self._show_selection and self._selected_widgets:
                self._draw_selection(painter)
        if self._edit_mode and self._selection_rect:
            sx, sy, sw, sh = self._selection_rect
            painter.setPen(QPen(QColor("#4a7ab5"), 1, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(QColor(74, 122, 181, 40)))
            painter.drawRect(int(sx), int(sy), int(sw), int(sh))
        if self._auto_align and self._guide_lines:
            painter.setPen(QPen(QColor("#ff4444"), 1, Qt.PenStyle.DashLine))
            for x1, y1, x2, y2 in self._guide_lines:
                painter.drawLine(int(x1), int(y1), int(x2), int(y2))
        painter.end()

    def _find_sub_window(self, widget: QWidget) -> Optional[QWidget]:
        p = widget.parent()
        while p and p is not self._root:
            if isinstance(p, QMdiSubWindow):
                return p
            p = p.parent()
        return None

    def hit_test_widget(self, px: float, py: float) -> Optional[QWidget]:
        rx, ry = self._to_root(px, py)
        px, py = int(rx), int(ry)
        def _deepest(parent, off_x=0, off_y=0):
            for child in reversed(list(parent.children())):
                if not isinstance(child, QWidget) or not child.isVisible():
                    continue
                cg = child.geometry()
                cx, cy = off_x + cg.x(), off_y + cg.y()
                if cx <= px <= cx + cg.width() and cy <= py <= cy + cg.height():
                    deeper = _deepest(child, cx, cy)
                    if deeper:
                        return deeper
                    if isinstance(child, QMdiArea):
                        continue
                    return child
            return None
        return _deepest(self._root)

    def _render_overlay(self, painter: QPainter):
        if self._edit_mode:
            self._draw_screen_overlay(painter)

    def _draw_screen_overlay(self, painter: QPainter):
        sw = int(self._screen_w * self._zoom)
        sh = int(self._screen_h * self._zoom)
        px = int(-self._pan_x)
        py = int(-self._pan_y)
        painter.setPen(QPen(QColor("#6a8ab5"), 1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(px, py, sw, sh)
        painter.setPen(QColor("#6a8ab5"))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        label = f"{self._screen_w} Г— {self._screen_h}"
        painter.drawText(px + 6, py + sh - 6, label)

    def _draw_selection(self, painter: QPainter):
        if not self._selected_widgets:
            return
        hs = 5
        for w in list(self._selected_widgets):
            try:
                _ = w.parent()
            except RuntimeError:
                self._selected_widgets.remove(w)
                continue
            ox, oy = self._get_widget_root_offset(w)
            gx, gy = w.x() + ox, w.y() + oy
            gw, gh = w.width(), w.height()
            sx = int(gx - self._pan_x)
            sy = int(gy - self._pan_y)
            painter.setPen(QPen(QColor("#4a7ab5"), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(sx, sy, gw, gh)
            handles = [
                (sx, sy), (int(sx + gw / 2), sy), (sx + gw, sy),
                (sx, int(sy + gh / 2)), (sx + gw, int(sy + gh / 2)),
                (sx, sy + gh), (int(sx + gw / 2), sy + gh), (sx + gw, sy + gh),
            ]
            painter.setBrush(QBrush(QColor("#4a7ab5")))
            painter.setPen(QPen(QColor("#ffffff"), 1))
            for hx, hy in handles:
                painter.drawRect(int(hx - hs / 2), int(hy - hs / 2), hs, hs)
        sel = self._selected_widget
        if not sel:
            return
        try:
            _ = sel.parent()
        except RuntimeError:
            self._selected_widget = None
            return
        ox, oy = self._get_widget_root_offset(sel)
        gx, gy = sel.x() + ox, sel.y() + oy
        gw, gh = sel.width(), sel.height()
        sx = int(gx - self._pan_x)
        sy = int(gy - self._pan_y)
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        if self._drag_widget is sel:
            if self._drag_handle == "move":
                dd = f" О”x:{sel.x() - self._drag_start_x:+d} О”y:{sel.y() - self._drag_start_y:+d}"
            else:
                dd = f" О”w:{sel.width() - self._drag_start_w:+d} О”h:{sel.height() - self._drag_start_h:+d}"
        else:
            dd = ""
        info = f"x:{gx} y:{gy}  {gw}Г—{gh}{dd}"
        iw = painter.fontMetrics().horizontalAdvance(info) + 8
        ih = painter.fontMetrics().height() + 2
        ix = sx
        iy = sy - ih - 2
        if iy < 0:
            iy = sy + gh + 2
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
        painter.drawRect(ix, iy, iw, ih)
        painter.setPen(QColor("#cccccc"))
        painter.drawText(ix + 4, iy + ih - 3, info)

    def mousePressEvent(self, event: QMouseEvent):
        if not self._edit_mode:
            super().mousePressEvent(event)
            return
        mx, my = event.position().x(), event.position().y()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._is_panning = True
            self._pan_start_mx = int(mx)
            self._pan_start_my = int(my)
            self._pan_start_px = self._pan_x
            self._pan_start_py = self._pan_y
            return
        rx, ry = self._to_root(mx, my)
        for w in list(self._selected_widgets):
            handle = self._get_handle_at(rx, ry, w)
            if handle:
                self._drag_widget = w
                self._drag_handle = handle
                self._drag_start_x = w.x()
                self._drag_start_y = w.y()
                self._drag_start_w = w.width()
                self._drag_start_h = w.height()
                self._drag_start_mx = mx
                self._drag_start_my = my
                self._store_drag_start_positions()
                return
        hit = self.hit_test_widget(mx, my)
        if hit:
            sub = self._find_sub_window(hit)
            target = sub or hit
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.toggle_selection(target)
            else:
                self.selected_widget = target
            self._drag_widget = target
            self._drag_handle = "move"
            self._drag_start_x = target.x()
            self._drag_start_y = target.y()
            self._drag_start_w = target.width()
            self._drag_start_h = target.height()
            self._drag_start_mx = mx
            self._drag_start_my = my
            self._store_drag_start_positions()
        else:
            if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
                self.selected_widget = None
            self._selection_box_start = (mx, my)
            self._is_selecting = True

    def mouseMoveEvent(self, event):
        if not self._edit_mode:
            super().mouseMoveEvent(event)
            return
        mx, my = event.position().x(), event.position().y()
        if self._is_panning:
            dx = mx - self._pan_start_mx
            dy = my - self._pan_start_my
            self._pan_x = self._pan_start_px - dx
            self._pan_y = self._pan_start_py - dy
            self._update_root_geometry()
            self.update()
            return
        if self._is_selecting and self._selection_box_start:
            sx, sy = self._selection_box_start
            self._selection_rect = (min(sx, mx), min(sy, my), abs(mx - sx), abs(my - sy))
            self.update()
            return
        if self._drag_widget and self._drag_handle:
            dx = mx - self._drag_start_mx
            dy = my - self._drag_start_my
            if self._drag_handle == "move":
                if dx == 0 and dy == 0:
                    self._guide_lines = []
                    self.update()
                    return
                for w in list(self._selected_widgets):
                    try:
                        sx, sy = self._drag_start_positions[id(w)]
                    except (KeyError, RuntimeError):
                        continue
                    w.move(int(self._snap_value(sx + dx)), int(self._snap_value(sy + dy)))
                if self._auto_align and len(self._selected_widgets) == 1:
                    new_x = self._snap_value(self._drag_start_x + dx)
                    new_y = self._snap_value(self._drag_start_y + dy)
                    nx, ny = self._snap_move_widget(self._drag_widget, new_x, new_y)
                    self._drag_widget.move(int(nx), int(ny))
            else:
                self._resize_from_handle(self._drag_widget, self._drag_handle, dx, dy)
            self.update()
        elif self._selected_widget:
            rx, ry = self._to_root(mx, my)
            handle = self._get_handle_at(rx, ry, self._selected_widget)
            cursor_map = {
                "nw": Qt.CursorShape.SizeFDiagCursor, "ne": Qt.CursorShape.SizeBDiagCursor,
                "sw": Qt.CursorShape.SizeBDiagCursor, "se": Qt.CursorShape.SizeFDiagCursor,
                "n": Qt.CursorShape.SizeVerCursor, "s": Qt.CursorShape.SizeVerCursor,
                "w": Qt.CursorShape.SizeHorCursor, "e": Qt.CursorShape.SizeHorCursor,
            }
            self.setCursor(cursor_map.get(handle, Qt.CursorShape.ArrowCursor))

    def enterEvent(self, event: QEnterEvent):
        if self._edit_mode:
            self.setFocus()
        super().enterEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if self._edit_mode:
            delta = event.angleDelta().y()
            if delta == 0:
                return
            factor = 1.1 if delta > 0 else 1 / 1.1
            mx, my = event.position().x(), event.position().y()
            rx, ry = self._to_root(mx, my)
            old_zoom = self._zoom
            self._zoom = max(0.1, min(10.0, self._zoom * factor))
            zoom_factor = self._zoom / old_zoom
            self._pan_x = rx * zoom_factor - mx
            self._pan_y = ry * zoom_factor - my
            self._update_display()
            self.zoom_changed.emit(self._zoom)
        else:
            super().wheelEvent(event)

    def mouseReleaseEvent(self, event):
        if not self._edit_mode:
            super().mouseReleaseEvent(event)
            return
        if self._is_panning:
            self._is_panning = False
            return
        if self._is_selecting and self._selection_box_start:
            self._is_selecting = False
            if self._selection_rect:
                sx, sy, sw, sh = self._selection_rect
                rx1, ry1 = self._to_root(sx, sy)
                rx2, ry2 = self._to_root(sx + sw, sy + sh)
                self._select_widgets_in_rect(rx1, ry1, rx2, ry2)
            self._selection_box_start = None
            self._selection_rect = None
            self.update()
            return
        widget = self._drag_widget
        self._drag_widget = None
        self._drag_handle = ""
        self._drag_start_positions.clear()
        if self._guide_lines:
            self._guide_lines.clear()
            self.update()
        if widget:
            self.widget_changed.emit(widget)

    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)

    def keyReleaseEvent(self, event):
        if not self._edit_mode:
            super().keyReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if not self._edit_mode:
            super().keyPressEvent(event)
            return
        key = event.key()
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            if key == Qt.Key.Key_C and self._selected_widget:
                self.copy_requested.emit()
                return
            elif key == Qt.Key.Key_V:
                self.paste_requested.emit()
                return
        if not self._selected_widgets:
            return
        step = self._grid_size if self._snap_to_grid else 1
        if key in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            for w in list(self._selected_widgets):
                self.remove_widget(w)
            self._selected_widgets.clear()
            self._selected_widget = None
            self.widget_selected.emit(None)
            self.update()
        elif key == Qt.Key.Key_Up:
            for w in self._selected_widgets:
                w.move(w.x(), int(self._snap_value(w.y() - step)))
            self.update()
            self.widget_changed.emit(self._selected_widget)
        elif key == Qt.Key.Key_Down:
            for w in self._selected_widgets:
                w.move(w.x(), int(self._snap_value(w.y() + step)))
            self.update()
            self.widget_changed.emit(self._selected_widget)
        elif key == Qt.Key.Key_Left:
            for w in self._selected_widgets:
                w.move(int(self._snap_value(w.x() - step)), w.y())
            self.update()
            self.widget_changed.emit(self._selected_widget)
        elif key == Qt.Key.Key_Right:
            for w in self._selected_widgets:
                w.move(int(self._snap_value(w.x() + step)), w.y())
            self.update()
            self.widget_changed.emit(self._selected_widget)

    def serialize(self) -> dict:
        children_data = [self._serialize_widget(c) for c in self._root.children()
                         if isinstance(c, QWidget) and hasattr(c, 'serialize')]
        return {
            "canvas_w": self.width(),
            "canvas_h": self.height(),
            "children": children_data,
        }

    def _serialize_widget(self, widget: QWidget) -> dict:
        if not hasattr(widget, '_widget_id') or widget._widget_id is None:
            widget._widget_id = f"w{id(widget)}"
        data = widget.serialize()
        data["id"] = widget._widget_id
        data["children"] = [self._serialize_widget(c) for c in widget.children()
                             if isinstance(c, QWidget) and hasattr(c, 'serialize')]
        return data

    def deserialize(self, data: dict):
        self.clear()
        vw, vh = self.width(), self.height()
        for cd in data.get("children", []):
            self._deserialize_widget(cd, self._root, vw, vh)
        self._apply_z_order()
        self._update_root_geometry()
        self.update()

    def _deserialize_widget(self, cd: dict, parent_widget: QWidget, vw: float, vh: float):
        wtype = cd.get("type", "Button")
        cls = WIDGET_REGISTRY.get(wtype)
        if not cls or not hasattr(cls, 'deserialize'):
            return
        widget = cls.deserialize(cd)
        widget._widget_id = cd.get("id")
        widget.setParent(parent_widget)
        widget.setVisible(True)
        if hasattr(widget, 'update_anchor'):
            widget.update_anchor(vw, vh)
        for child_cd in cd.get("children", []):
            self._deserialize_widget(child_cd, widget, vw, vh)

    def save_to_file(self, path: str):
        with open(path, "w") as f:
            json.dump(self.serialize(), f, indent=2)

    def load_from_file(self, path: str):
        with open(path) as f:
            self.deserialize(json.load(f))
