# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
                              QLineEdit, QTabWidget, QTreeWidget, QTreeWidgetItem,
                              QFileDialog, QCheckBox, QAbstractItemView)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import (QFont, QPainter, QColor, QPen, QPainterPath, QFontMetrics)

from core.engine.engine import Engine
from core.foundation.time_travel import (SnapshotRecorder, FrameSnapshot, diff_scenes,
                              find_entity_frame_changes)
from core.config.editor_scale import scale

_C_BG = QColor("#1e1e2e")
_C_RULER = QColor("#252535")
_C_RULER_TICK = QColor("#555566")
_C_RULER_LABEL = QColor("#888899")
_C_RULER_LINE = QColor("#3a3a4a")
_C_PLAYHEAD = QColor("#ffcc00")
_C_HOVER = QColor("#7ec8e3")
_C_BOOKMARK = QColor("#ff6688")
_C_BREAKPOINT = QColor("#ff4444")
_C_TEXT = QColor("#cccccc")
_C_DIM = QColor("#888888")
_C_DIFF_ADD = QColor("#44cc44")
_C_DIFF_DEL = QColor("#cc4444")
_C_DIFF_CHG = QColor("#cccc44")


def _heat_color(t: float) -> QColor:
    r = min(255, int(255 * t * 2))
    g = min(255, int(255 * (1 - t) * 2))
    return QColor(r, g, 80)


class _TimelineScrubber(QWidget):
    MIN_PPF = 3
    MAX_PPF = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list[FrameSnapshot] = []
        self._current: int = -1
        self._hover: int = -1
        self._ppf: float = 8.0
        self._scroll: float = 0.0
        self._dragging: bool = False
        self._show_heat: bool = False
        self._max_time_ms: float = 33.0
        self.setMinimumHeight(scale(80))
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._font = QFont("Segoe UI", 8)
        self._small = QFont("Segoe UI", 7)

    def ruler_h(self) -> int: return scale(28)
    def bar_h(self) -> int: return scale(44)
    def total_h(self) -> int: return self.ruler_h() + self.bar_h() + scale(4)

    def set_frames(self, frames: list[FrameSnapshot]):
        self._frames = frames
        if frames:
            times = [f.frame_time_ms for f in frames if f.frame_time_ms > 0]
            if times:
                self._max_time_ms = max(times)
        self._current = min(self._current, len(frames) - 1) if frames else -1
        self.update()

    def set_current(self, idx: int):
        self._current = max(-1, min(idx, len(self._frames) - 1))
        self._ensure_visible()
        self.update()

    def current_index(self) -> int: return self._current
    def frame_count(self) -> int: return len(self._frames)

    def set_heatmap(self, on: bool):
        self._show_heat = on
        self.update()

    def _frame_at(self, x: float) -> int:
        if not self._frames:
            return -1
        i = int((x + self._scroll) / self._ppf)
        return max(0, min(len(self._frames) - 1, i))

    def _x_of(self, i: int) -> float:
        return i * self._ppf - self._scroll

    def _ensure_visible(self):
        if self._current < 0:
            return
        cx = self._current * self._ppf
        vw = self.width()
        if cx < self._scroll:
            self._scroll = cx - vw * 0.1
        elif cx > self._scroll + vw - 20:
            self._scroll = cx - vw * 0.8
        self._scroll = max(0, self._scroll)

    def _call(self, attr: str, *args):
        p = self.parent()
        while p and not hasattr(p, attr):
            p = p.parent()
        if p:
            getattr(p, attr)(*args)

    def mousePressEvent(self, event):
        if not self._frames:
            super().mousePressEvent(event)
            return
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            idx = self._frame_at(event.position().x())
            self._call("_on_bookmark_toggle", idx)
            self.update()
        else:
            self._dragging = True
            self._current = self._frame_at(event.position().x())
            self._call("_on_frame_restore", self._current)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        old_h = self._hover
        self._hover = self._frame_at(event.position().x())
        if self._dragging and self._frames:
            i = self._frame_at(event.position().x())
            if 0 <= i < len(self._frames) and i != self._current:
                self._current = i
                self._call("_on_frame_peek", i)
                self.update()
        if old_h != self._hover:
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._dragging:
            self._dragging = False
            if self._current >= 0:
                self._call("_on_frame_restore", self._current)
            self.update()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._hover = -1
        self._dragging = False
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ControlModifier:
            old = self._ppf
            self._ppf *= 1.15 if event.angleDelta().y() > 0 else 0.87
            self._ppf = max(self.MIN_PPF, min(self.MAX_PPF, self._ppf))
            cx = event.position().x() + self._scroll
            self._scroll = max(0, cx - event.position().x())
        else:
            n = len(self._frames)
            if n == 0:
                super().wheelEvent(event)
                return
            step = -1 if event.angleDelta().y() > 0 else 1
            idx = max(0, min(n - 1, self._current + step))
            if idx != self._current:
                self._call("_on_frame_restore", idx)
        self.update()
        super().wheelEvent(event)

    def keyPressEvent(self, event):
        if not self._frames:
            super().keyPressEvent(event)
            return
        k = event.key()
        n = len(self._frames) - 1
        if k == Qt.Key.Key_Left:
            self._call("_on_frame_restore", max(0, self._current - 1))
        elif k == Qt.Key.Key_Right:
            self._call("_on_frame_restore", min(n, self._current + 1))
        elif k == Qt.Key.Key_Home:
            self._call("_on_frame_restore", 0)
        elif k == Qt.Key.Key_End:
            self._call("_on_frame_restore", n)
        elif k == Qt.Key.Key_PageUp:
            step = max(1, int(self.width() / max(self._ppf, 1)))
            self._call("_on_frame_restore", max(0, self._current - step))
        elif k == Qt.Key.Key_PageDown:
            step = max(1, int(self.width() / max(self._ppf, 1)))
            self._call("_on_frame_restore", min(n, self._current + step))
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        w = self.width()
        rh = self.ruler_h()
        bh = self.bar_h()
        ppf = self._ppf
        n = len(self._frames)

        painter.fillRect(0, 0, w, self.height(), _C_BG)

        if n == 0:
            painter.setPen(_C_DIM)
            painter.setFont(self._font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "Record to capture snapshots")
            painter.end()
            return

        bar_y = rh + scale(2)
        vs = max(0, int(self._scroll / ppf))
        ve = min(n, int((self._scroll + w) / ppf) + 1)

        painter.fillRect(0, 0, w, rh, _C_RULER)
        step = max(1, int(30 / ppf))
        for i in range(vs, ve, step):
            x = self._x_of(i)
            if x < -20 or x > w + 20:
                continue
            painter.setPen(_C_RULER_TICK)
            painter.drawLine(int(x), rh - scale(6), int(x), rh)
            painter.setPen(_C_RULER_LABEL)
            painter.setFont(self._small)
            painter.drawText(int(x) + scale(3), rh - scale(8), str(i))

        for i in range(vs, ve):
            if self._frames[i].bookmarked:
                x = self._x_of(i)
                painter.fillRect(int(x), 0, max(1, int(ppf)), scale(4), _C_BOOKMARK)
        for i in range(vs, ve):
            if self._frames[i].breakpoint_hit:
                x = self._x_of(i)
                painter.fillRect(int(x), scale(4), max(1, int(ppf)), scale(4), _C_BREAKPOINT)

        painter.setPen(_C_RULER_LINE)
        painter.drawLine(0, rh, w, rh)

        max_ent = max(f.entity_count for f in self._frames) or 1
        for i in range(vs, ve):
            x = self._x_of(i)
            f = self._frames[i]
            bw = max(1, ppf - 1)
            ent_h = max(2, int((f.entity_count / max_ent) * (bh - scale(4))))
            if self._show_heat and f.frame_time_ms > 0:
                t = min(1.0, f.frame_time_ms / max(self._max_time_ms, 1))
                color = _heat_color(t)
            elif i == self._current:
                color = _C_PLAYHEAD
            elif i == self._hover:
                color = _C_HOVER
            else:
                v = int(80 + (f.entity_count / max_ent) * 120)
                color = QColor(v, v, v)
            painter.fillRect(QRectF(x, bar_y + bh - ent_h, bw, ent_h), color)

        painter.setPen(_C_RULER_LINE)
        painter.drawLine(0, bar_y + bh, w, bar_y + bh)

        if self._current >= 0:
            px = self._x_of(self._current)
            painter.setPen(QPen(_C_PLAYHEAD, 2))
            painter.drawLine(int(px), 0, int(px), bar_y + bh)
            path = QPainterPath()
            path.moveTo(int(px), scale(6))
            path.lineTo(int(px) - scale(5), scale(6) + scale(7))
            path.lineTo(int(px) + scale(5), scale(6) + scale(7))
            path.closeSubpath()
            painter.fillPath(path, _C_PLAYHEAD)

        if self._current >= 0 and self._current < n:
            f = self._frames[self._current]
            painter.setPen(_C_PLAYHEAD)
            painter.setFont(self._font)
            parts = [f"#{f.frame_number}", f"{f.entity_count}e"]
            if f.frame_time_ms > 0:
                parts.append(f"{f.frame_time_ms:.1f}ms")
            info = "  ".join(parts)
            tw = painter.fontMetrics().horizontalAdvance(info) + scale(10)
            ir = QRectF(w - tw - scale(4), bar_y + bh + scale(2), tw, scale(18))
            painter.fillRect(ir, _C_BG)
            painter.drawText(ir, Qt.AlignmentFlag.AlignCenter, info)

        painter.end()


class _DiffTree(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Change", "Detail"])
        self.setColumnCount(2)
        self.setRootIsDecorated(True)
        self.setAlternatingRowColors(True)
        self.setStyleSheet("""
            QTreeWidget { background: #1e1e2e; color: #ccc; border: none;
                          font-size: 11px; }
            QTreeWidget::item { padding: 2px 4px; }
            QTreeWidget::item:alternate { background: #252535; }
        """)
        self.header().setStretchLastSection(True)

    def show_diff(self, diff: dict, names: Optional[dict] = None):
        self.clear()
        names = names or {}
        for eid in diff.get("created", []):
            name = names.get(eid, eid)
            item = QTreeWidgetItem([f"+ {name}", "created"])
            item.setForeground(0, _C_DIFF_ADD)
            self.addTopLevelItem(item)
        for eid in diff.get("deleted", []):
            name = names.get(eid, eid)
            item = QTreeWidgetItem([f"  {name}", "deleted"])
            item.setForeground(0, _C_DIFF_DEL)
            self.addTopLevelItem(item)
        for eid, changes in diff.get("changed", {}).items():
            name = names.get(eid, eid)
            parent = QTreeWidgetItem([f"~ {name}", ""])
            parent.setForeground(0, _C_DIFF_CHG)
            self.addTopLevelItem(parent)
            for c in changes:
                tag = c[0]
                if tag == "name":
                    child = QTreeWidgetItem([f"  name: {c[1]} -> {c[2]}", ""])
                elif tag == "+comp":
                    child = QTreeWidgetItem([f"  + {c[1]}", ""])
                    child.setForeground(0, _C_DIFF_ADD)
                elif tag == "-comp":
                    child = QTreeWidgetItem([f"  - {c[1]}", ""])
                    child.setForeground(0, _C_DIFF_DEL)
                elif tag == "~comp":
                    cparent = QTreeWidgetItem([f"  ~ {c[1]}", ""])
                    cparent.setForeground(0, _C_DIFF_CHG)
                    parent.addChild(cparent)
                    for field, ov, nv in c[2]:
                        QTreeWidgetItem(cparent, [f"    {field}: {ov} -> {nv}", ""])
                    continue
                parent.addChild(child)
            parent.setExpanded(True)

    def show_entity_changes(self, frames: list[FrameSnapshot],
                            entity_id: str, entity_name: str):
        self.clear()
        indices = find_entity_frame_changes(frames, entity_id)
        item = QTreeWidgetItem([f"Entity: {entity_name} ({entity_id})", ""])
        self.addTopLevelItem(item)
        for idx in indices:
            snap = frames[idx]
            child = QTreeWidgetItem([f"  Frame {snap.frame_number}",
                                     f"{snap.entity_count}e"])
            item.addChild(child)
        item.setExpanded(True)


class TimeTravelPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Time Travel", parent)
        self._engine = engine
        self._recorder: Optional[SnapshotRecorder] = None
        self._restored: bool = False
        self._live_data: Optional[dict] = None
        self._playing: bool = False
        self._prev_frame_data: Optional[dict] = None
        self._diff_cache: Optional[dict] = None
        self._diff_names: Optional[dict] = None

        if not getattr(engine, '_time_travel_recorder', None):
            engine._time_travel_recorder = SnapshotRecorder()
        self._recorder = engine._time_travel_recorder

        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_step)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.setInterval(200)

        self._setup_ui()

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        r1 = QHBoxLayout()
        r1.setSpacing(2)

        def btn(text, tip, cb, w=26):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.setFixedSize(scale(w), scale(24))
            b.clicked.connect(cb)
            r1.addWidget(b)
            return b

        self._first_btn = btn(chr(0x23EE), "First", self._go_first)
        self._b10_btn = btn(chr(0x23EA), "Back 10", self._go_back10)
        self._b1_btn = btn(chr(0x25C0), "Back 1", self._go_back1)
        self._play_btn = btn(chr(0x25B6), "Play (Space)", self._toggle_play)
        self._f1_btn = btn(chr(0x25B6), "Forward 1", self._go_forward1)
        self._f10_btn = btn(chr(0x23E9), "Forward 10", self._go_forward10)
        self._last_btn = btn(chr(0x23ED), "Last", self._go_last)

        r1.addSpacing(6)

        self._record_btn = btn("R", "Record snapshots", self._toggle_record, w=28)
        self._record_btn.setCheckable(True)
        self._record_btn.setStyleSheet(
            "QPushButton { color: #ff4444; }"
            "QPushButton:checked { color: #fff; background: #cc2222; }")

        r1.addSpacing(6)

        self._live_btn = btn("L", "Return to live scene", self._go_live, w=24)
        self._live_btn.setEnabled(False)

        r1.addSpacing(4)

        r1.addWidget(QLabel("Spd:"))
        self._speed_sb = QDoubleSpinBox()
        self._speed_sb.setRange(0.1, 10.0)
        self._speed_sb.setValue(1.0)
        self._speed_sb.setSingleStep(0.25)
        self._speed_sb.setFixedWidth(scale(55))
        self._speed_sb.valueChanged.connect(self._on_speed_changed)
        r1.addWidget(self._speed_sb)

        r1.addStretch()

        export_btn = btn("Save", "Export current frame as .zpes", self._export_frame, w=40)
        layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.setSpacing(4)
        r2.addWidget(QLabel("BP:"))
        self._bp_input = QLineEdit()
        self._bp_input.setPlaceholderText("entity_count > 5 and frame > 10")
        self._bp_input.setFixedHeight(scale(22))
        self._bp_input.returnPressed.connect(self._set_breakpoint)
        r2.addWidget(self._bp_input)
        bp_clear = QPushButton("X")
        bp_clear.setFixedSize(scale(20), scale(22))
        bp_clear.clicked.connect(self._clear_breakpoint)
        r2.addWidget(bp_clear)
        layout.addLayout(r2)

        r3 = QHBoxLayout()
        r3.setSpacing(8)
        self._heat_cb = QCheckBox("Heatmap")
        self._heat_cb.toggled.connect(self._on_heatmap_toggled)
        r3.addWidget(self._heat_cb)

        self._dedup_cb = QCheckBox("Skip unchanged")
        self._dedup_cb.toggled.connect(self._on_dedup_toggled)
        r3.addWidget(self._dedup_cb)

        r3.addWidget(QLabel("Every:"))
        self._interval_sb = QSpinBox()
        self._interval_sb.setRange(1, 60)
        self._interval_sb.setValue(1)
        self._interval_sb.setFixedWidth(scale(45))
        self._interval_sb.valueChanged.connect(self._on_interval_changed)
        r3.addWidget(self._interval_sb)
        r3.addWidget(QLabel("f"))

        r3.addStretch()

        clear_btn = btn("Clear", "Clear all frames", self._clear, w=50)
        layout.addLayout(r3)

        self._timeline = _TimelineScrubber()
        layout.addWidget(self._timeline)

        info = QHBoxLayout()
        self._frame_lbl = QLabel("No frames")
        self._frame_lbl.setStyleSheet("color: #ccc; font-size: 11px;")
        info.addWidget(self._frame_lbl)
        info.addStretch()
        self._status_lbl = QLabel("Idle")
        self._status_lbl.setStyleSheet("color: #888; font-size: 11px;")
        info.addWidget(self._status_lbl)
        layout.addLayout(info)

        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.setStyleSheet("""
            QTabWidget::pane { background: #1e1e2e; border: 1px solid #3a3a4a; }
            QTabBar::tab { background: #252535; color: #888; padding: 2px 8px; }
            QTabBar::tab:selected { background: #1e1e2e; color: #ffcc00; }
        """)

        self._diff_tree = _DiffTree()
        self._bottom_tabs.addTab(self._diff_tree, "Diff")
        self._bottom_tabs.currentChanged.connect(self._on_tab_changed)

        self._bk_tree = QTreeWidget()
        self._bk_tree.setHeaderLabels(["Frame", "Entities", "Time"])
        self._bk_tree.setAlternatingRowColors(True)
        self._bk_tree.setStyleSheet(self._diff_tree.styleSheet())
        self._bk_tree.itemDoubleClicked.connect(self._on_bookmark_clicked)
        self._bottom_tabs.addTab(self._bk_tree, "Bookmarks")

        search_container = QWidget()
        search_layout = QVBoxLayout(search_container)
        search_layout.setContentsMargins(0, 0, 0, 0)
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Entity ID or name...")
        self._search_input.setFixedHeight(scale(22))
        self._search_input.returnPressed.connect(self._do_entity_search)
        self._search_tree = QTreeWidget()
        self._search_tree.setHeaderLabels(["Frames where entity changed"])
        self._search_tree.setAlternatingRowColors(True)
        self._search_tree.setStyleSheet(self._diff_tree.styleSheet())
        self._search_tree.itemDoubleClicked.connect(self._on_search_result_clicked)
        search_layout.addWidget(self._search_input)
        search_layout.addWidget(self._search_tree)
        self._bottom_tabs.addTab(search_container, "Find Entity")

        self._bottom_tabs.setMaximumHeight(scale(180))
        layout.addWidget(self._bottom_tabs)

        self.setWidget(w)

    def _go_first(self):
        self._on_frame_restore(0)

    def _go_back10(self):
        self._on_frame_restore(max(0, self._timeline.current_index() - 10))

    def _go_back1(self):
        self._on_frame_restore(max(0, self._timeline.current_index() - 1))

    def _toggle_play(self):
        if self._playing:
            self._playing = False
            self._play_timer.stop()
            self._play_btn.setText(chr(0x25B6))
        else:
            if self._timeline.current_index() < 0 and self._recorder and self._recorder.num_frames > 0:
                self._on_frame_restore(0)
            if self._timeline.current_index() >= 0:
                self._playing = True
                speed = max(0.1, self._speed_sb.value())
                self._play_timer.setInterval(int(33 / speed))
                self._play_timer.start()
                self._play_btn.setText(chr(0x23F8))

    def _go_forward1(self):
        n = self._recorder.num_frames if self._recorder else 0
        self._on_frame_restore(min(n - 1, self._timeline.current_index() + 1))

    def _go_forward10(self):
        n = self._recorder.num_frames if self._recorder else 0
        self._on_frame_restore(min(n - 1, self._timeline.current_index() + 10))

    def _go_last(self):
        if self._recorder and self._recorder.num_frames > 0:
            self._on_frame_restore(self._recorder.num_frames - 1)

    def _play_step(self):
        idx = self._timeline.current_index()
        n = self._recorder.num_frames if self._recorder else 0
        if idx < 0 or idx >= n - 1:
            self._toggle_play()
            return
        speed = max(0.1, self._speed_sb.value())
        self._play_timer.setInterval(int(33 / speed))
        self._on_frame_restore(idx + 1)

    def _on_speed_changed(self, val: float):
        if self._playing:
            self._play_timer.setInterval(int(33 / max(0.1, val)))

    def _toggle_record(self):
        if self._recorder.is_recording:
            self._recorder.stop()
            self._record_btn.setChecked(False)
            self._status_lbl.setText("Stopped")
            self._refresh_timer.stop()
        else:
            self._recorder.clear()
            self._recorder.capture_interval = self._interval_sb.value()
            self._recorder.start()
            self._record_btn.setChecked(True)
            self._status_lbl.setText("Recording...")
            self._restored = False
            self._live_data = None
            self._live_btn.setEnabled(False)
            self._prev_frame_data = None
            self._diff_cache = None
            self._refresh_timer.start()

    def _on_interval_changed(self, v):
        if self._recorder:
            self._recorder.capture_interval = v

    def _set_breakpoint(self):
        if self._recorder:
            self._recorder.breakpoint_expr = self._bp_input.text().strip()

    def _clear_breakpoint(self):
        if self._recorder:
            self._recorder.breakpoint_expr = ""
        self._bp_input.clear()

    def _on_heatmap_toggled(self, on):
        self._timeline.set_heatmap(on)

    def _on_dedup_toggled(self, on):
        if self._recorder:
            self._recorder.filter_unchanged = on

    def _on_frame_peek(self, idx: int):
        if not self._recorder:
            return
        snap = self._recorder.get_frame(idx)
        if snap:
            self._update_labels(snap)

    def _on_frame_restore(self, idx: int):
        if not self._recorder or not self._recorder.num_frames:
            return
        idx = max(0, min(idx, self._recorder.num_frames - 1))
        snap = self._recorder.get_frame(idx)
        if snap is None:
            return

        if not self._restored and self._engine.scene:
            self._live_data = self._engine.scene.serialize()
            self._prev_frame_data = None

        if self._engine.play_mode:
            self._engine.stop_play()

        from core.ecs.ecs import Scene as SceneCls
        restored = SceneCls.deserialize(snap.data, self._engine._component_registry)
        old_path = self._engine.scene.path if self._engine.scene else None
        if old_path:
            restored.path = old_path
        self._engine._scene = restored
        self._engine._emit_event("scene_loaded", restored)
        self._restored = True
        self._live_btn.setEnabled(True)

        self._diff_cache = None
        if self._prev_frame_data is not None and idx > 0:
            self._diff_cache = diff_scenes(self._prev_frame_data, snap.data)
            self._diff_names = {}
            for eid in set(self._diff_cache.get("created", [])) | set(self._diff_cache.get("deleted", [])) | set(self._diff_cache.get("changed", {}).keys()):
                self._diff_names[eid] = snap.data.get("entities", {}).get(eid, {}).get("name", eid)
        self._prev_frame_data = snap.data

        if self._bottom_tabs.currentIndex() == 0:
            self._refresh_diff_tab()

        self._timeline.set_current(idx)
        self._update_labels(snap)

    def _on_tab_changed(self, tab_index: int):
        if tab_index == 0:
            self._refresh_diff_tab()

    def _refresh_diff_tab(self):
        self._diff_tree.clear()
        if self._diff_cache:
            self._diff_tree.show_diff(self._diff_cache, self._diff_names)
        else:
            self._diff_tree.addTopLevelItem(
                QTreeWidgetItem(["Select two frames to see changes", ""]))

    def _on_bookmark_toggle(self, idx: int):
        if self._recorder:
            self._recorder.toggle_bookmark(idx)
            self._refresh_bookmarks()

    def _on_bookmark_clicked(self, item, col):
        text = item.text(0)
        if text.isdigit():
            self._on_frame_restore(int(text))

    def _on_search_result_clicked(self, item, col):
        text = item.text(0)
        if text.startswith("Frame "):
            try:
                self._on_frame_restore(int(text.split()[1]))
            except (ValueError, IndexError):
                pass

    def _do_entity_search(self):
        if not self._recorder:
            return
        query = self._search_input.text().strip().lower()
        if not query:
            return
        frames = self._recorder.frames()
        self._search_tree.clear()
        found = False
        for snap in reversed(frames):
            for eid, edata in snap.data.get("entities", {}).items():
                name = edata.get("name", "")
                if query in eid.lower() or query in name.lower():
                    item = QTreeWidgetItem([f"Frame {snap.frame_number}  {name} ({eid})"])
                    self._search_tree.addTopLevelItem(item)
                    found = True
                    break
        if not found:
            self._search_tree.addTopLevelItem(QTreeWidgetItem(["No matches"]))

    def _go_live(self):
        if self._live_data is not None and self._engine.scene is not None:
            from core.ecs.ecs import Scene as SceneCls
            live = SceneCls.deserialize(self._live_data, self._engine._component_registry)
            old_path = self._engine.scene.path if self._engine.scene else None
            if old_path:
                live.path = old_path
            self._engine._scene = live
            self._engine._emit_event("scene_loaded", live)
            self._live_data = None
            self._restored = False
            self._prev_frame_data = None
            self._diff_cache = None
            self._live_btn.setEnabled(False)
            self._status_lbl.setText("Live")
            self._timeline.set_current(-1)
            self._frame_lbl.setText("Live scene")
            self._diff_tree.clear()

    def _export_frame(self):
        if not self._recorder:
            return
        idx = self._timeline.current_index()
        snap = self._recorder.get_frame(idx)
        if not snap:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Frame as Scene", f"frame_{snap.frame_number}.zpes",
            "Scene (*.zpes)")
        if path:
            self._recorder.export_frame(idx, path)

    def _clear(self):
        if self._playing:
            self._toggle_play()
        if self._recorder:
            self._recorder.stop()
            self._recorder.clear()
        self._record_btn.setChecked(False)
        self._refresh_timer.stop()
        self._timeline.set_frames([])
        self._frame_lbl.setText("No frames")
        self._status_lbl.setText("Idle")
        self._live_btn.setEnabled(False)
        self._live_data = None
        self._restored = False
        self._prev_frame_data = None
        self._diff_cache = None
        self._diff_tree.clear()
        self._bk_tree.clear()
        self._search_tree.clear()

    def _refresh(self):
        if not self._recorder:
            return
        frames = self._recorder.frames()
        self._timeline.set_frames(frames)
        if self._restored:
            idx = self._timeline.current_index()
            if idx >= 0:
                snap = self._recorder.get_frame(idx)
                if snap:
                    self._update_labels(snap)
        else:
            n = self._recorder.num_frames
            if n > 0:
                snap = self._recorder.get_frame(n - 1)
                if snap:
                    self._frame_lbl.setText(
                        f"{n} frames  latest #{snap.frame_number}")
                    self._status_lbl.setText("Recording")
        self._refresh_bookmarks()

    def _refresh_bookmarks(self):
        if not self._recorder:
            self._bk_tree.clear()
            return
        self._bk_tree.clear()
        for i in self._recorder.bookmarked_indices():
            snap = self._recorder.get_frame(i)
            if snap:
                item = QTreeWidgetItem([
                    str(snap.frame_number),
                    str(snap.entity_count),
                    f"{snap.frame_time_ms:.1f}ms" if snap.frame_time_ms > 0 else ""
                ])
                self._bk_tree.addTopLevelItem(item)

    def _update_labels(self, snap: FrameSnapshot):
        parts = [f"Frame {snap.frame_number}"]
        if snap.entity_count >= 0:
            parts.append(f"{snap.entity_count} entities")
        if snap.frame_time_ms > 0:
            parts.append(f"{snap.frame_time_ms:.1f}ms")
        if snap.bookmarked:
            parts.append("BM")
        if snap.breakpoint_hit:
            parts.append("BP")
        self._frame_lbl.setText("  ".join(parts))
        self._status_lbl.setText(f"Viewing #{snap.frame_number}")

    def load_config(self, config):
        pass

    def save_config(self, config):
        pass
