# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import time
from collections import deque

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.components.properties import make_prop_reader
from core.foundation.logger import Logger
from editor.inspector.entity_property_picker import EntityPropertyPicker

from .plotter import ChartWidget


class PlotterPanel(QWidget):
    def __init__(self, engine, parent=None, plugin=None, persist: bool = True):
        super().__init__(parent)
        self._engine = engine
        self._plugin = plugin
        self._persist_shared = persist and plugin is not None
        self._config: dict = {}
        self._tracked: dict[str, dict] = {}
        self._frame = 0
        interval = 100
        limit = 2000
        if plugin is not None:
            interval = int(plugin.get_config("refresh_interval", interval))
            limit = int(plugin.get_config("history_limit", limit))
        self._config["refresh_interval"] = interval
        self._config["history_limit"] = limit
        self._history_limit = limit
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(max(8, interval))
        sps = max(1, round(1000 / max(8, interval)))
        self._sps_box.setValue(sps)
        self._pts_box.setValue(self._history_limit)
        self._timer.start()
        self._sync_scene()

    def set_active(self, active: bool):
        if active:
            self._timer.start()
        else:
            self._timer.stop()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        xtop = QHBoxLayout()
        xtop.setSpacing(4)
        xtop.addWidget(QLabel("X:"))
        self._x_time = QCheckBox("Time")
        self._x_time.setChecked(True)
        self._x_time.toggled.connect(self._on_x_time_toggled)
        xtop.addWidget(self._x_time)
        xtop.addStretch(1)
        layout.addLayout(xtop)
        self._x_picker = EntityPropertyPicker(self._engine.scene)
        self._x_picker.setEnabled(False)
        layout.addWidget(self._x_picker)
        ytop = QHBoxLayout()
        ytop.setSpacing(4)
        ytop.addWidget(QLabel("Y:"))
        ytop.addStretch(1)
        layout.addLayout(ytop)
        self._y_picker = EntityPropertyPicker(self._engine.scene)
        layout.addWidget(self._y_picker)
        btns = QHBoxLayout()
        btns.setSpacing(4)
        btns.addStretch(1)
        self._btn_track = QPushButton("Track")
        self._btn_track.clicked.connect(self._add_tracked)
        btns.addWidget(self._btn_track)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self._clear_all)
        btns.addWidget(self._btn_clear)
        layout.addLayout(btns)
        self._chart = ChartWidget(show_legend=True, show_toolbar=True)
        self._chart.setLabel("left", "")
        self._chart.setLabel("bottom", "time")
        layout.addWidget(self._chart, 1)
        opts = QHBoxLayout()
        opts.setSpacing(4)
        opts.addWidget(QLabel("Samples/s:"))
        self._sps_box = QSpinBox()
        self._sps_box.setRange(1, 120)
        self._sps_box.setValue(10)
        self._sps_box.valueChanged.connect(self._on_sps_changed)
        opts.addWidget(self._sps_box)
        opts.addSpacing(8)
        opts.addWidget(QLabel("Max points:"))
        self._pts_box = QSpinBox()
        self._pts_box.setRange(50, 1000000)
        self._pts_box.setValue(2000)
        self._pts_box.setSingleStep(100)
        self._pts_box.valueChanged.connect(self._on_points_changed)
        opts.addWidget(self._pts_box)
        opts.addStretch(1)
        layout.addLayout(opts)
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status)

    def _on_sps_changed(self, sample_rate: int):
        self._timer.setInterval(max(8, round(1000 / max(1, sample_rate))))
        self._persist()

    def _on_points_changed(self, limit: int):
        self._history_limit = max(50, int(limit))
        for entry in self._tracked.values():
            entry["x"].maxlen = self._history_limit
            entry["y"].maxlen = self._history_limit
        self._persist()

    def _persist(self):
        self._config["refresh_interval"] = self._timer.interval()
        self._config["history_limit"] = self._history_limit
        if not self._persist_shared:
            return
        try:
            self._plugin.set_config("refresh_interval", self._config["refresh_interval"])
            self._plugin.set_config("history_limit", self._config["history_limit"])
        except Exception as e:
            Logger.error(f"Plotter config persist failed: {e}", e)

    def _on_x_time_toggled(self, checked: bool):
        self._x_picker.setEnabled(not checked)
        if checked:
            self._chart.setLabel("bottom", "time")
        else:
            label = self._x_picker.property_label()
            self._chart.setLabel("bottom", label or "")

    def _add_tracked(self):
        y_eid = self._y_picker.current_entity_id()
        y_path = self._y_picker.current_property()
        if not y_eid or not y_path:
            return
        scene = self._engine.scene
        entity = scene.get_entity(y_eid) if scene else None
        name = entity.name if entity is not None else y_eid[:8]
        ylabel = self._y_picker.property_label()
        if self._x_time.isChecked():
            start = time.monotonic()
            def x_reader():
                return time.monotonic() - start
            kind = "time"
            label = f"{name}.{ylabel}"
        else:
            x_eid = self._x_picker.current_entity_id()
            x_path = self._x_picker.current_property()
            if not x_eid or not x_path:
                return
            kind = "prop"
            xlabel = self._x_picker.property_label()
            x_reader = make_prop_reader(lambda: self._engine.scene, x_eid, x_path)
            label = f"{xlabel} vs {name}.{ylabel}"
        key = f"#prop:{y_eid}:{y_path}:{kind}"
        if key in self._tracked:
            return
        line = self._chart.plot(label=label)
        entry = {
            "x": deque(maxlen=self._history_limit),
            "y": deque(maxlen=self._history_limit),
            "line": line,
            "x_reader": x_reader,
            "y_reader": make_prop_reader(lambda: self._engine.scene, y_eid, y_path),
            "label": label,
        }
        self._tracked[key] = entry
        self._update_status()

    def _clear_all(self):
        self._tracked.clear()
        self._chart.clearAll()
        self._update_status()

    def _tick(self):
        self._frame += 1
        self._sync_scene()
        if not self._tracked:
            return
        for entry in self._tracked.values():
            yv = entry["y_reader"]()
            if yv is None:
                continue
            xv = entry["x_reader"]()
            entry["x"].append(xv)
            entry["y"].append(yv)
            entry["line"].setData(list(entry["x"]), list(entry["y"]))
        self._update_status()

    def _sync_scene(self):
        scene = self._engine.scene
        if scene is None:
            return
        for picker in (self._x_picker, self._y_picker):
            if picker.scene() is not scene:
                picker.set_scene(scene)

    def _update_status(self):
        if not self._tracked:
            self._status.setText("No series tracked. Pick X/Y properties and press Track.")
            return
        parts = []
        for entry in self._tracked.values():
            data = entry["y"]
            if data:
                parts.append(f"{entry['label']}: {data[-1]:.4g}")
        self._status.setText("  |  ".join(parts))