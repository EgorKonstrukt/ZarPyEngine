# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import hashlib
from collections import deque
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QScrollArea,
                              QTabWidget, QTreeWidget, QTreeWidgetItem,
                              QHeaderView, QFrame, QStyledItemDelegate)
from PyQt6.QtCore import QTimer, Qt, QRectF, QPointF
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QFontMetrics
from core.config.editor_scale import scale, scale_xy

def _name_to_color(name: str) -> QColor:
    raw = name.replace("_ms", "").replace("_", "").replace(" ", "").strip().lower()
    if not raw:
        return QColor(140, 140, 140)
    h = int(hashlib.md5(raw.encode()).hexdigest()[:6], 16)
    hue = (h % 360)
    sat = 60 + (h % 30)
    lig = 60 + (h % 20)
    return QColor.fromHsl(hue, sat, lig)

def _name_to_hex(name: str) -> str:
    return _name_to_color(name).name()

_FONT = QFont("Segoe UI", 9)
_FONT_BOLD = QFont("Segoe UI", 9, QFont.Weight.Bold)
_FONT_SMALL = QFont("Segoe UI", 8)
_FONT_TINY = QFont("Segoe UI", 7)

class TimelineOverview(QWidget):
    HEIGHT = 60
    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames_data: deque[dict] = deque(maxlen=300)
        self._selected_index = -1
        self._hover_index = -1
        self._budget_ms = 16.67
        self.setMouseTracking(True)
        self.setFixedHeight(scale(self.HEIGHT))
        self.setMinimumWidth(100)
    def add_frame(self, flat_data: dict, frame_time_ms: float):
        self._frames_data.append({"flat": dict(flat_data), "total": frame_time_ms})
        self.update()
    def clear_data(self):
        self._frames_data.clear()
        self._selected_index = -1
        self.update()
    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.HEIGHT
        p.fillRect(0, 0, w, h, QColor(18, 18, 18))
        n = len(self._frames_data)
        if n == 0:
            p.setPen(QColor(80, 80, 80))
            p.setFont(_FONT_SMALL)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No frame data")
            p.end()
            return
        bar_w = max(2, w // max(n, 1) - 1)
        self._bar_w = bar_w
        total_w = n * (bar_w + 1)
        offset = max(0, total_w - w)
        budget_y = h - (h * min(self._budget_ms / 50.0, 1.0))
        p.setPen(QPen(QColor(60, 60, 60), 1, Qt.PenStyle.DashLine))
        p.drawLine(0, int(budget_y), w, int(budget_y))
        p.setPen(QColor(80, 80, 80))
        p.setFont(_FONT_TINY)
        p.drawText(w - 50, int(budget_y) - 2, f"{self._budget_ms:.0f}ms")
        y_max = max(max(d["total"] for d in self._frames_data), self._budget_ms * 1.5)
        for i, d in enumerate(self._frames_data):
            x = i * (bar_w + 1) - offset
            if x < -bar_w or x > w:
                continue
            total = d["total"]
            bar_h = max(1, int((total / y_max) * h))
            bar_color = self._pick_bar_color(d["flat"])
            if i == self._selected_index:
                p.fillRect(x, h - bar_h, bar_w, bar_h, bar_color.lighter(130))
                p.setPen(QPen(QColor(255, 255, 255), 1))
                p.drawRect(x, h - bar_h, bar_w, bar_h)
            elif i == self._hover_index:
                p.fillRect(x, h - bar_h, bar_w, bar_h, bar_color.lighter(115))
                p.setPen(QPen(QColor(180, 180, 180), 1))
                p.drawRect(x, h - bar_h, bar_w, bar_h)
            else:
                p.fillRect(x, h - bar_h, bar_w, bar_h, bar_color)
        if self._hover_index >= 0 and self._hover_index < n:
            d = self._frames_data[self._hover_index]
            txt = f"Frame {self._hover_index}: {d['total']:.2f}ms"
            fm = QFontMetrics(_FONT_SMALL)
            tw = fm.horizontalAdvance(txt) + 10
            tx = min(max(0, self._hover_index * (bar_w + 1) - offset - tw // 2), w - tw)
            p.fillRect(tx, 0, tw, 18, QColor(0, 0, 0, 200))
            p.setPen(QColor(255, 255, 255))
            p.setFont(_FONT_SMALL)
            p.drawText(tx + 5, 13, txt)
        p.end()
    def _pick_bar_color(self, flat: dict) -> QColor:
        main_key = max(flat, key=lambda k: flat[k]) if flat else ""
        return _name_to_color(main_key)
    def mouseMoveEvent(self, event):
        x = event.position().x()
        n = len(self._frames_data)
        bw = getattr(self, '_bar_w', 3)
        total_w = n * (bw + 1)
        w = self.width()
        offset = max(0, total_w - w)
        idx = (int(x) + offset) // (bw + 1) if bw > 0 else -1
        if 0 <= idx < n:
            self._hover_index = idx
        else:
            self._hover_index = -1
        self.update()
    def mousePressEvent(self, event):
        x = event.position().x()
        n = len(self._frames_data)
        bw = getattr(self, '_bar_w', 3)
        total_w = n * (bw + 1)
        w = self.width()
        offset = max(0, total_w - w)
        idx = (int(x) + offset) // (bw + 1) if bw > 0 else -1
        if 0 <= idx < n:
            self._selected_index = idx
            parent_panel = self.parent()
            while parent_panel and not hasattr(parent_panel, '_select_frame'):
                parent_panel = parent_panel.parent()
            if parent_panel:
                parent_panel._select_frame(idx)
        self.update()
    def leaveEvent(self, event):
        self._hover_index = -1
        self.update()

class BarDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        if index.column() != 0:
            super().paint(painter, option, index)
            return
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = option.rect
        painter.fillRect(rect, QColor(30, 30, 30))
        frac_data = index.data(Qt.ItemDataRole.UserRole)
        if frac_data is not None:
            frac = float(frac_data) / 100.0
            bar_w = max(2, int(rect.width() * min(frac, 1.0) * 0.85))
            if bar_w > 2:
                c = index.data(Qt.ItemDataRole.UserRole + 1)
                if isinstance(c, str):
                    c = QColor(c)
                pct = frac * 100.0
                if pct > 50:
                    bar_color = QColor(255, 60, 60)
                elif pct > 20:
                    bar_color = QColor(255, 160, 40)
                elif pct > 10:
                    bar_color = QColor(255, 220, 60)
                else:
                    bar_color = c
                painter.fillRect(rect.x() + 2, rect.y() + 3, bar_w, rect.height() - 6, bar_color)
                painter.setPen(QPen(bar_color.darker(130), 1))
                painter.drawRect(rect.x() + 2, rect.y() + 3, bar_w, rect.height() - 6)
        painter.restore()

class HierarchyView(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["", "Name", "Total", "Self", "Calls", "% Frame"])
        self.header().setStretchLastSection(False)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(0, 60)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.setAlternatingRowColors(True)
        self.setIndentation(16)
        self.setAnimated(True)
        self.setItemDelegate(BarDelegate(self))
        self.setStyleSheet("""
            QTreeWidget { border: none; font: 9pt 'Segoe UI'; }
            QTreeWidget::item { padding: 1px 2px; min-height: 22px; }
            QHeaderView::section { border: 1px solid;
                padding: 3px 6px; font: 8pt 'Segoe UI'; }
        """)
        self.setColumnCount(6)

        self._items: dict[str, QTreeWidgetItem] = {}
        self._scope_order: list[str] = []
        self._parent_map: dict[str, str | None] = {}
        self._children_map: dict[str, list[str]] = {}
        self._built = False

    def set_frame_data(self, samples: list, frame_time_ms: float, flat_data: dict | None = None):
        # Add Unaccounted вЂ” time between frames not covered by "frame" scope
        frame_dur = frame_time_ms
        for s in samples:
            if s.name == "frame" and s.depth == 0:
                frame_dur = s.duration_ms
                break
        unaccounted = max(0.0, frame_time_ms - frame_dur)
        if unaccounted > frame_time_ms * 0.03:
            from core.engine.engine import ProfileSample
            samples = list(samples) + [ProfileSample("Unaccounted", 0, 0.0, unaccounted, "#555555")]

        self._base_frame_ms = frame_time_ms

        agg: dict[str, dict] = {}
        for s in samples:
            agg.setdefault(s.name, {"total": 0.0, "count": 0})
            agg[s.name]["total"] += s.duration_ms
            agg[s.name]["count"] += 1

        parent_map: dict[str, str | None] = {}
        depth_stack: list[tuple[str, int]] = []
        for s in sorted(samples, key=lambda x: (x.start_ms, x.depth)):
            while depth_stack and depth_stack[-1][1] >= s.depth:
                depth_stack.pop()
            parent_map[s.name] = depth_stack[-1][0] if depth_stack else None
            depth_stack.append((s.name, s.depth))

        children_map: dict[str, list[str]] = {}
        for name, parent in parent_map.items():
            children_map.setdefault(parent if parent else "", []).append(name)

        def dfs(name: str, out: list[str]):
            if name not in out:
                out.append(name)
            for c in children_map.get(name, []):
                dfs(c, out)

        this_order: list[str] = []
        for s in samples:
            if parent_map.get(s.name) is None:
                dfs(s.name, this_order)

        for name in this_order:
            if name not in self._scope_order:
                self._scope_order.append(name)

        self._parent_map.update(parent_map)
        for parent, children in children_map.items():
            for child in children:
                if child not in self._children_map.setdefault(parent, []):
                    self._children_map[parent].append(child)

        if not self._built:
            self._build_tree(agg, self._base_frame_ms)
            self._built = True
        else:
            self._update_tree(agg, self._base_frame_ms)

    def _make_pct_color(self, pct: float) -> QColor:
        if pct > 50:
            return QColor(255, 80, 80)
        elif pct > 20:
            return QColor(255, 170, 50)
        elif pct > 10:
            return QColor(255, 220, 80)
        return QColor(160, 160, 160)

    def _setup_item(self, item: QTreeWidgetItem, name: str, total: float, self_ms: float,
                    count: int, frame_time_ms: float):
        pct = total / max(frame_time_ms, 0.001) * 100.0
        frac = min(pct, 100.0)
        item.setData(0, Qt.ItemDataRole.UserRole, frac)
        item.setData(0, Qt.ItemDataRole.UserRole + 1, _name_to_hex(name))

        color = _name_to_color(name)
        pct_color = self._make_pct_color(pct)
        is_bottleneck = pct > 10

        font = item.font(1)
        font.setBold(is_bottleneck)
        item.setFont(1, font)

        item.setText(1, f"  {name}")
        item.setForeground(1, QBrush(color))
        item.setText(2, f"{total:.3f}")
        item.setForeground(2, QBrush(pct_color if is_bottleneck else QColor(140, 200, 255)))
        item.setText(3, f"{self_ms:.3f}")
        item.setForeground(3, QBrush(pct_color if is_bottleneck else QColor(180, 180, 180)))
        item.setText(4, str(count))
        item.setForeground(4, QBrush(pct_color.lighter(130) if is_bottleneck else QColor(160, 160, 160)))
        item.setText(5, f"{pct:.1f}%")
        item.setForeground(5, QBrush(pct_color))

        item.setSizeHint(0, QRectF(0, 0, 60, 22).size().toSize())
        item.setSizeHint(1, QRectF(0, 0, 100, 22).size().toSize())

    def _build_tree(self, agg: dict, frame_time_ms: float):
        self.clear()
        self._items.clear()

        def add(parent_widget, names: list[str]):
            for name in names:
                if name not in self._scope_order:
                    continue
                info = agg.get(name, {"total": 0.0, "count": 0})
                total = info["total"]
                children_total = sum(
                    agg.get(c, {}).get("total", 0.0)
                    for c in self._children_map.get(name, [])
                )
                self_ms = max(0.0, total - children_total)

                item = QTreeWidgetItem(parent_widget)
                self._items[name] = item
                self._setup_item(item, name, total, self_ms, info["count"], frame_time_ms)

                if name in self._children_map:
                    add(item, self._children_map[name])

        roots = [n for n in self._scope_order if self._parent_map.get(n) is None]
        add(self, roots)
        self.expandAll()

    def _update_tree(self, agg: dict, frame_time_ms: float):
        for name, item in list(self._items.items()):
            info = agg.get(name, {"total": 0.0, "count": 0})
            total = info["total"]
            children_total = sum(
                agg.get(c, {}).get("total", 0.0)
                for c in self._children_map.get(name, [])
            )
            self_ms = max(0.0, total - children_total)
            self._setup_item(item, name, total, self_ms, info["count"], frame_time_ms)

        for name in self._scope_order:
            if name in self._items:
                continue
            parent_name = self._parent_map.get(name)
            if parent_name and parent_name in self._items:
                parent_item = self._items[parent_name]
            else:
                parent_item = self

            info = agg.get(name, {"total": 0.0, "count": 0})
            total = info["total"]
            children_total = sum(
                agg.get(c, {}).get("total", 0.0)
                for c in self._children_map.get(name, [])
            )
            self_ms = max(0.0, total - children_total)

            item = QTreeWidgetItem(parent_item)
            self._items[name] = item
            self._setup_item(item, name, total, self_ms, info["count"], frame_time_ms)

        self.expandAll()

class TimelineWidget(QWidget):
    MAX_SAMPLES = 300
    def __init__(self, parent=None):
        super().__init__(parent)
        self._series: dict[str, deque] = {}
        self._colors: dict[str, QColor] = {}
        self._total_frames = 0
        self.setMinimumHeight(120)
    def add_frame(self, flat_data: dict[str, float]):
        self._total_frames += 1
        if not flat_data:
            self.update()
            return
        for key, val in flat_data.items():
            if key not in self._series:
                self._colors[key] = _name_to_color(key)
                self._series[key] = deque(maxlen=self.MAX_SAMPLES)
            self._series[key].append(val)
        for key in list(self._series.keys()):
            if key not in flat_data:
                self._series[key].append(0.0)
        self.update()
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(24, 24, 24))
        margin = 45
        plot_w = w - margin * 2
        plot_h = h - margin * 2
        if plot_w < 10 or plot_h < 10:
            p.end()
            return
        if not self._series:
            p.setPen(QColor(100, 100, 100))
            p.setFont(_FONT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No timeline data")
            p.end()
            return
        p.setPen(QPen(QColor(45, 45, 45), 1))
        for i in range(5):
            y = margin + plot_h * (1.0 - i * 0.25)
            p.drawLine(margin, int(y), w - margin, int(y))
        p.setPen(QColor(80, 80, 80))
        p.setFont(_FONT_SMALL)
        for i in range(5):
            val = i * 0.25 * 33.33
            p.drawText(2, int(margin + plot_h * (1.0 - i * 0.25) + 3), f"{val:.0f}ms")
        budget_y = margin + plot_h * (1.0 - 16.67 / 33.33)
        p.setPen(QPen(QColor(80, 200, 80, 80), 1, Qt.PenStyle.DashLine))
        p.drawLine(margin, int(budget_y), w - margin, int(budget_y))
        p.setPen(QColor(80, 200, 80, 100))
        p.setFont(_FONT_TINY)
        p.drawText(w - margin - 30, int(budget_y) - 2, "16.67ms")
        global_max = 0.1
        for key, data in self._series.items():
            if data:
                global_max = max(global_max, max(data))
        y_max = max(global_max * 1.2, 16.67)
        for key, data in self._series.items():
            color = self._colors.get(key, QColor("#aaaaaa"))
            if len(data) < 2:
                continue
            n = len(data)
            pts = []
            for i, val in enumerate(data):
                px = margin + (plot_w * i / max(n - 1, 1))
                py = margin + plot_h * (1.0 - val / y_max)
                pts.append(QPointF(px, py))
            pen = QPen(color, 1.5)
            p.setPen(pen)
            for i in range(1, len(pts)):
                p.drawLine(pts[i-1], pts[i])
        legend_x = margin
        legend_y = h - margin + 4
        for key in self._series.keys():
            if legend_x + 120 < w:
                color = self._colors.get(key, QColor("#aaaaaa"))
                p.fillRect(legend_x, legend_y, 8, 8, color)
                p.setPen(QColor(180, 180, 180))
                p.setFont(_FONT_SMALL)
                label = key.replace("_ms", "").replace("_", " ").title()
                p.drawText(legend_x + 11, legend_y + 8, label)
                fm = QFontMetrics(_FONT_SMALL)
                legend_x += fm.horizontalAdvance(label) + 28
        p.end()

class FlameGraphWidget(QWidget):
    ROW_H = 22
    def __init__(self, parent=None):
        super().__init__(parent)
        self._samples: list = []
        self._frame_time_ms: float = 1.0
        self._hovered: tuple | None = None
        self.setMouseTracking(True)
        self.setMinimumHeight(120)
    def set_data(self, samples: list, frame_time_ms: float):
        self._samples = samples
        self._frame_time_ms = max(frame_time_ms, 0.001)
        work_ms = self._frame_time_ms
        for s in samples:
            if s.name == "frame" and s.depth == 0:
                work_ms = max(s.duration_ms, 0.001)
                break
        self._work_time_ms = work_ms
        h = (max((s.depth for s in samples), default=0) + 2) * self.ROW_H
        self.setMinimumHeight(max(h, 120))
        self.update()
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor(24, 24, 24))
        if not self._samples:
            p.setPen(QColor(100, 100, 100))
            p.setFont(_FONT)
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No profile data")
            p.end()
            return
        margin = 2
        plot_w = w - margin * 2
        if plot_w < 10:
            p.end()
            return
        bft = self._work_time_ms  # bar widths auto-scaled to fill plot
        pft = self._frame_time_ms  # percentages match hierarchy
        # Draw idle gap label (right of "frame" end)
        frame_end_ms = 0.0
        for s in self._samples:
            if s.name == "frame" and s.depth == 0:
                frame_end_ms = s.start_ms + s.duration_ms
                break
        if frame_end_ms > 0 and frame_end_ms < self._frame_time_ms:
            idle_x = margin + (frame_end_ms / bft) * plot_w
            if idle_x < w - margin:
                p.setPen(QColor(60, 60, 70))
                p.setFont(_FONT_TINY)
                idle_pct = (self._frame_time_ms - frame_end_ms) / self._frame_time_ms * 100
                p.drawText(QRectF(idle_x, 4, w - idle_x - 4, 16),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
                           f"  Idle {idle_pct:.0f}%")
        for s in self._samples:
            x = margin + (s.start_ms / bft) * plot_w
            bw = max(1, (s.duration_ms / bft) * plot_w)
            y = s.depth * self.ROW_H + 2
            bh = self.ROW_H - 3
            if x + bw < 0 or x > w:
                continue
            color = _name_to_color(s.name)
            p.fillRect(QRectF(x, y, bw, bh), color)
            if bw > 2:
                p.setPen(QPen(QColor(0, 0, 0, 40), 1))
                p.drawRect(QRectF(x, y, bw, bh))
            if bw > 40:
                p.setPen(QColor(220, 220, 220))
                p.setFont(_FONT_SMALL)
                pct = s.duration_ms / pft * 100.0
                label = f"{s.name} ({pct:.1f}%)"
                tw = QFontMetrics(_FONT_SMALL).horizontalAdvance(label)
                if tw < bw - 6:
                    p.drawText(QRectF(x + 3, y, bw - 6, bh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
            elif bw > 16 and s.depth > 0:
                p.setPen(QColor(220, 220, 220))
                p.setFont(_FONT_TINY)
                pct = s.duration_ms / pft * 100.0
                label = f"{pct:.1f}%"
                tw = QFontMetrics(_FONT_TINY).horizontalAdvance(label)
                if tw < bw - 4:
                    p.drawText(QRectF(x + 2, y, bw - 4, bh), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label)
        if self._hovered:
            sx, sy, sample = self._hovered
            pct = sample.duration_ms / pft * 100.0
            txt = f"{sample.name}  {sample.duration_ms:.3f}ms ({pct:.1f}%)"
            fm = QFontMetrics(_FONT)
            tw = fm.horizontalAdvance(txt) + 10
            tx = min(sx, max(0, w - tw))
            p.fillRect(tx, sy - 22, tw, 20, QColor(0, 0, 0, 210))
            p.setPen(QColor(255, 255, 255))
            p.setFont(_FONT)
            p.drawText(tx + 5, sy - 22 + 14, txt)
        p.end()
    def mouseMoveEvent(self, event):
        x, y = event.position().x(), event.position().y()
        margin = 2
        w = self.width()
        plot_w = w - margin * 2
        ft = self._work_time_ms
        self._hovered = None
        if plot_w > 0 and self._samples:
            for s in self._samples:
                sx = margin + (s.start_ms / ft) * plot_w
                bw = max(1, (s.duration_ms / ft) * plot_w)
                sy = s.depth * self.ROW_H + 2
                bh = self.ROW_H - 3
                if sx <= x <= sx + bw and sy <= y <= sy + bh:
                    self._hovered = (int(sx), int(sy), s)
                    break
        self.update()
    def leaveEvent(self, event):
        self._hovered = None
        self.update()

class ProfilerHeader(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(12)
        bf = _FONT_BOLD
        self._fps_lbl = QLabel("FPS: 0")
        self._fps_lbl.setFont(bf)
        layout.addWidget(self._fps_lbl)
        self._frame_lbl = QLabel("Frame: 0")
        self._frame_lbl.setFont(bf)
        layout.addWidget(self._frame_lbl)
        self._time_lbl = QLabel("0.00ms")
        self._time_lbl.setFont(bf)
        layout.addWidget(self._time_lbl)
        self._min_lbl = QLabel("Min: 0.00")
        self._min_lbl.setFont(_FONT)
        layout.addWidget(self._min_lbl)
        self._avg_lbl = QLabel("Avg: 0.00")
        self._avg_lbl.setFont(_FONT)
        layout.addWidget(self._avg_lbl)
        self._max_lbl = QLabel("Max: 0.00")
        self._max_lbl.setFont(_FONT)
        layout.addWidget(self._max_lbl)
        layout.addStretch()
        self._budget_indicator = QLabel("")
        self._budget_indicator.setFixedSize(*scale_xy(10, 10))
        layout.addWidget(self._budget_indicator)
    def update_stats(self, fps: float, frame_count: int, frame_time_ms: float,
                     min_ms: float, avg_ms: float, max_ms: float):
        self._fps_lbl.setText(f"FPS: {fps:.1f}")
        self._frame_lbl.setText(f"Frame: {frame_count}")
        self._time_lbl.setText(f"{frame_time_ms:.2f}ms")
        self._min_lbl.setText(f"Min: {min_ms:.2f}")
        self._avg_lbl.setText(f"Avg: {avg_ms:.2f}")
        self._max_lbl.setText(f"Max: {max_ms:.2f}")
        budget_exceeded = frame_time_ms > 16.67
        self._budget_indicator.setStyleSheet(
            f"background: {'#FF6B6B' if budget_exceeded else '#81C784'}; border-radius: 5px;"
        )
        self._budget_indicator.setToolTip(
            "Frame budget exceeded!" if budget_exceeded else "Within frame budget"
        )

class ProfilerPanel(QDockWidget):
    def __init__(self, engine, parent=None):
        super().__init__("Profiler", parent)
        self._engine = engine
        self._selected_frame_idx = -1
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(200)
        prof = getattr(engine, '_profiler', None)
        if prof:
            prof.capture_frames = True
    def showEvent(self, ev):
        super().showEvent(ev)
        prof = getattr(self._engine, '_profiler', None)
        if prof:
            prof.capture_frames = True
    def hideEvent(self, ev):
        super().hideEvent(ev)
        prof = getattr(self._engine, '_profiler', None)
        if prof:
            prof.capture_frames = False
    def load_config(self, config) -> None:
        refresh_interval = config.get("profiler.refresh_interval", 200)
        self._timer.setInterval(refresh_interval)
    def _setup_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._header = ProfilerHeader()
        layout.addWidget(self._header)
        self._timeline_overview = TimelineOverview()
        layout.addWidget(self._timeline_overview)
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet("""
            QTabWidget::pane { border: none; }
            QTabBar::tab { padding: 5px 14px; border: none;
                font: 9pt 'Segoe UI'; }
            QTabBar::tab:selected { border-bottom: 2px solid; }
        """)
        self._hierarchy_tree = HierarchyView()
        self._tabs.addTab(self._hierarchy_tree, "Hierarchy")
        self._timeline_widget = TimelineWidget()
        self._tabs.addTab(self._timeline_widget, "Timeline")
        self._flame_graph = FlameGraphWidget()
        scroll_fg = QScrollArea()
        scroll_fg.setWidgetResizable(True)
        scroll_fg.setStyleSheet("QScrollArea { border: none; }")
        scroll_fg.setWidget(self._flame_graph)
        self._tabs.addTab(scroll_fg, "Flame Graph")
        layout.addWidget(self._tabs, 1)
        self.setWidget(container)
    def _select_frame(self, idx: int):
        self._selected_frame_idx = idx
    def _get_fps(self):
        eng = self._engine
        vp = eng.viewport if hasattr(eng, 'viewport') else None
        if vp and hasattr(vp, '_fps') and vp._fps > 0:
            return vp._fps
        if hasattr(eng, 'fps') and eng.fps > 0:
            return eng.fps
        prof = getattr(eng, '_profiler', None)
        if prof:
            frames = prof.frames
            if len(frames) >= 2:
                recent = frames[-min(len(frames), 30):]
                total_ms = sum(f.frame_time_ms for f in recent)
                if total_ms > 0:
                    return len(recent) / (total_ms / 1000.0)
        return 0.0
    def _refresh(self):
        eng = self._engine
        fps = self._get_fps()
        prof = eng._profiler if hasattr(eng, '_profiler') else None
        if not prof:
            return
        frames = prof.frames
        if not frames:
            return

        num_avg = min(15, len(frames))
        recent = frames[-num_avg:]
        last = frames[-1]
        flat = last.flat_data

        avg_frame_time = sum(f.frame_time_ms for f in recent) / num_avg

        agg = {}
        for f in recent:
            for s in f.samples:
                if s.name not in agg:
                    agg[s.name] = {"total": 0.0, "count": 0, "depth": s.depth}
                agg[s.name]["total"] += s.duration_ms
                agg[s.name]["count"] += 1

        from core.engine.engine import ProfileSample
        start_map: dict[str, float] = {}
        for s in last.samples:
            if s.name not in start_map:
                start_map[s.name] = s.start_ms
        averaged_samples = []
        for name, data in agg.items():
            st = start_map.get(name, 0.0)
            averaged_samples.append(
                ProfileSample(name, data["depth"], st, data["total"] / num_avg, "#aaaaaa")
            )

        avg_flat = {}
        for f in recent:
            for key, val in f.flat_data.items():
                avg_flat[key] = avg_flat.get(key, 0.0) + val
        for key in avg_flat:
            avg_flat[key] /= num_avg

        self._timeline_overview.add_frame(flat, last.frame_time_ms)
        self._timeline_widget.add_frame(flat)

        fc = last.frame_number
        all_times = [f.frame_time_ms for f in frames]
        min_ms = min(all_times) if all_times else 0.0
        overall_avg = sum(all_times) / len(all_times) if all_times else 0.0
        max_ms = max(all_times) if all_times else 0.0
        self._header.update_stats(fps, fc, avg_frame_time, min_ms, overall_avg, max_ms)

        tab_idx = self._tabs.currentIndex()
        if tab_idx == 0:
            self._hierarchy_tree.set_frame_data(averaged_samples, avg_frame_time, avg_flat)
        elif tab_idx == 2:
            self._flame_graph.set_data(last.samples, last.frame_time_ms)
