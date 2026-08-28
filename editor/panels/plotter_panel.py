# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from collections import deque
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QComboBox, QPushButton, QLabel)
from PyQt6.QtCore import QTimer, Qt
from editor.plotter import ChartWidget


class PlotterPanel(QDockWidget):
    def __init__(self, engine, parent=None):
        super().__init__("Plotter", parent)
        self._engine = engine
        self._tracked: dict[str, tuple[deque, object]] = {}
        self._frame = 0
        self._known_keys: tuple = ()
        self._history_limit = 2000
        self._setup_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(100)
        self._refresh_keys(force=True)

    def load_config(self, config) -> None:
        interval = config.get("plotter.refresh_interval", 100)
        limit = config.get("plotter.history_limit", 2000)
        self._timer.setInterval(max(30, int(interval)))
        self._history_limit = max(50, int(limit))
        for data, _ in self._tracked.values():
            data.maxlen = self._history_limit

    def _setup_ui(self):
        self.setObjectName("PlotterDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        bar.addWidget(QLabel("Metric:"))
        self._combo = QComboBox()
        bar.addWidget(self._combo, 1)
        self._btn_add = QPushButton("Track")
        self._btn_add.clicked.connect(self._add_tracked)
        bar.addWidget(self._btn_add)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(self._clear_all)
        bar.addWidget(self._btn_clear)
        layout.addLayout(bar)
        self._chart = ChartWidget(show_legend=True, show_toolbar=True)
        self._chart.setLabel("left", "ms")
        layout.addWidget(self._chart, 1)
        self._status = QLabel("")
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._status)
        self.setWidget(root)

    def _refresh_keys(self, force: bool = False):
        data = self._engine.profiler_data
        keys = tuple(sorted(data.keys()))
        if not force and keys == self._known_keys:
            return
        self._known_keys = keys
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem("")
        self._combo.addItems(keys)
        idx = self._combo.findText(current)
        if idx >= 0:
            self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)

    def _add_tracked(self):
        key = self._combo.currentText()
        if not key or key in self._tracked:
            return
        data = deque(maxlen=self._history_limit)
        line = self._chart.plot(label=key)
        self._tracked[key] = (data, line)
        self._update_status()

    def _clear_all(self):
        self._tracked.clear()
        self._chart.clearAll()
        self._update_status()

    def _tick(self):
        self._frame += 1
        if not self._tracked:
            self._refresh_keys()
            return
        self._refresh_keys()
        for key, (data, line) in list(self._tracked.items()):
            val = self._engine.get_profiler_data(key, 0.0)
            data.append(val)
            n = len(data)
            line.setData(list(range(n)), list(data))
        self._update_status()

    def _update_status(self):
        if not self._tracked:
            self._status.setText("No series tracked. Select a metric and press Track.")
            return
        parts = []
        for key, (data, _) in self._tracked.items():
            if data:
                parts.append(f"{key}: {data[-1]:.3f}ms")
        self._status.setText("  |  ".join(parts))