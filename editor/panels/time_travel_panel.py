# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QSlider, QSpinBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush

from core.engine import Engine
from core.time_travel import SnapshotRecorder, FrameSnapshot
from core.editor_scale import scale


class _TimelineWidget(QWidget):
    """Custom timeline widget that shows recorded frames as a bar chart.

    Clicking on a position triggers frame selection.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list[FrameSnapshot] = []
        self._selected_index: int = -1
        self._hover_index: int = -1
        self.setMinimumHeight(scale(60))
        self.setMouseTracking(True)

    def set_frames(self, frames: list[FrameSnapshot], selected: int = -1):
        self._frames = frames
        self._selected_index = selected
        self.update()

    def set_selected(self, index: int):
        self._selected_index = index
        self.update()

    def _frame_at_pos(self, x: int) -> int:
        if not self._frames:
            return -1
        w = self.width()
        if w <= 0:
            return -1
        idx = int(x / w * len(self._frames))
        return max(0, min(len(self._frames) - 1, idx))

    def mousePressEvent(self, event):
        if self._frames:
            idx = self._frame_at_pos(event.position().x())
            self._selected_index = idx
            parent = self.parent()
            while parent and not hasattr(parent, '_on_frame_selected'):
                parent = parent.parent()
            if parent:
                parent._on_frame_selected(idx)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        old = self._hover_index
        self._hover_index = self._frame_at_pos(event.position().x())
        if old != self._hover_index:
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hover_index = -1
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        h = self.height()
        n = len(self._frames)
        bar_h = h - scale(12)

        painter.fillRect(0, 0, w, h, QColor("#1e1e2e"))

        if n == 0:
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No snapshots recorded")
            painter.end()
            return

        bar_w = max(2, w / n - 1)
        max_entities = max(f.entity_count for f in self._frames) if self._frames else 1
        max_entities = max(max_entities, 1)

        for i, f in enumerate(self._frames):
            x = int(i * (w / n))
            bh = max(2, int((f.entity_count / max_entities) * (bar_h - 4)))

            if i == self._selected_index:
                color = QColor("#ffcc00")
            elif i == self._hover_index:
                color = QColor("#7ec8e3")
            else:
                intensity = int(80 + (f.entity_count / max_entities) * 120)
                color = QColor(intensity, intensity, intensity)

            painter.fillRect(int(x), bar_h - bh + scale(6), int(bar_w), bh, color)

        painter.setPen(QColor("#444444"))
        painter.drawLine(0, bar_h + scale(6), w, bar_h + scale(6))

        if self._selected_index >= 0 and self._selected_index < n:
            f = self._frames[self._selected_index]
            painter.setPen(QColor("#ffcc00"))
            info = f"Frame {f.frame_number}  entities={f.entity_count}"
            painter.drawText(scale(4), scale(10), info)

        painter.end()


class TimeTravelPanel(QDockWidget):
    """Panel for time-travel debugging: record, scrub, and inspect past frames."""

    def __init__(self, engine: Engine, parent=None):
        super().__init__("Time Travel", parent)
        self._engine = engine
        self._recorder: Optional[SnapshotRecorder] = None
        self._restored_scene: bool = False
        self._live_scene_data: Optional[dict] = None

        if not getattr(engine, '_time_travel_recorder', None):
            engine._time_travel_recorder = SnapshotRecorder()
        self._recorder = engine._time_travel_recorder

        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.setInterval(200)

        self._setup_ui()

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        self._record_btn = QPushButton("Record")
        self._record_btn.setCheckable(True)
        self._record_btn.setFixedHeight(scale(24))
        self._record_btn.clicked.connect(self._toggle_record)
        toolbar.addWidget(self._record_btn)

        self._live_btn = QPushButton("Live")
        self._live_btn.setEnabled(False)
        self._live_btn.setFixedHeight(scale(24))
        self._live_btn.clicked.connect(self._go_live)
        toolbar.addWidget(self._live_btn)

        toolbar.addStretch()

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(scale(24))
        clear_btn.clicked.connect(self._clear)
        toolbar.addWidget(clear_btn)

        toolbar.addWidget(QLabel("Every:"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(1, 60)
        self._interval_spin.setValue(1)
        self._interval_spin.setFixedWidth(scale(50))
        self._interval_spin.setToolTip("Capture every N frames")
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        toolbar.addWidget(self._interval_spin)
        toolbar.addWidget(QLabel("frame(s)"))

        layout.addLayout(toolbar)

        self._timeline = _TimelineWidget()
        layout.addWidget(self._timeline)

        info_bar = QHBoxLayout()
        self._frame_label = QLabel("No frames")
        self._frame_label.setStyleSheet("color: #cccccc;")
        info_bar.addWidget(self._frame_label)
        info_bar.addStretch()
        self._status_label = QLabel("Idle")
        self._status_label.setStyleSheet("color: #888888;")
        info_bar.addWidget(self._status_label)
        layout.addLayout(info_bar)

        self.setWidget(w)

    def _toggle_record(self):
        if self._recorder.is_recording:
            self._recorder.stop()
            self._record_btn.setChecked(False)
            self._record_btn.setText("Record")
            self._status_label.setText("Stopped")
            self._refresh_timer.stop()
        else:
            self._recorder.clear()
            self._recorder.capture_interval = self._interval_spin.value()
            self._recorder.start()
            self._record_btn.setChecked(True)
            self._record_btn.setText("Recording...")
            self._status_label.setText("Recording")
            self._restored_scene = False
            self._live_scene_data = None
            self._live_btn.setEnabled(False)
            self._refresh_timer.start()

    def _on_interval_changed(self, val: int):
        if self._recorder:
            self._recorder.capture_interval = val

    def _on_frame_selected(self, index: int):
        if not self._recorder or not self._recorder.num_frames:
            return
        index = max(0, min(index, self._recorder.num_frames - 1))
        snap = self._recorder.get_frame(index)
        if snap is None:
            return

        if not self._restored_scene and self._engine.scene:
            self._live_scene_data = self._engine.scene.serialize()

        if self._engine.play_mode:
            self._engine.stop_play()

        from core.ecs import Scene as SceneCls
        restored = SceneCls.deserialize(snap.data,
                                        self._engine._component_registry)
        old_path = self._engine.scene.path if self._engine.scene else None
        if old_path:
            restored.path = old_path
        self._engine._scene = restored
        self._engine._emit_event("scene_loaded", restored)
        self._restored_scene = True
        self._live_btn.setEnabled(True)
        self._status_label.setText(f"Viewing frame {snap.frame_number}")

        self._timeline.set_selected(index)
        self._update_info()

    def _go_live(self):
        if self._live_scene_data is not None and self._engine.scene is not None:
            from core.ecs import Scene as SceneCls
            live = SceneCls.deserialize(self._live_scene_data,
                                        self._engine._component_registry)
            old_path = self._engine.scene.path if self._engine.scene else None
            if old_path:
                live.path = old_path
            self._engine._scene = live
            self._engine._emit_event("scene_loaded", live)
            self._live_scene_data = None
            self._restored_scene = False
            self._live_btn.setEnabled(False)
            self._status_label.setText("Live")
            self._timeline.set_selected(-1)
            self._update_info()

    def _clear(self):
        if self._recorder:
            self._recorder.stop()
            self._recorder.clear()
        self._record_btn.setChecked(False)
        self._record_btn.setText("Record")
        self._refresh_timer.stop()
        self._timeline.set_frames([])
        self._frame_label.setText("No frames")
        self._status_label.setText("Idle")
        self._live_btn.setEnabled(False)
        self._live_scene_data = None
        self._restored_scene = False

    def _refresh(self):
        if not self._recorder:
            return
        frames = self._recorder.frames()
        selected = self._timeline._selected_index
        if selected >= len(frames):
            selected = len(frames) - 1
        self._timeline.set_frames(frames, selected)
        self._update_info()

    def _update_info(self):
        if not self._recorder:
            return
        n = self._recorder.num_frames
        total = self._recorder.total_frames
        if n > 0:
            snap = self._recorder.get_frame(n - 1)
            if snap:
                self._frame_label.setText(
                    f"Frames: {n} captured / {total} total  "
                    f"Latest: #{snap.frame_number} ({snap.entity_count} entities)")
        else:
            self._frame_label.setText("No frames")

    def load_config(self, config):
        pass

    def save_config(self, config):
        pass
