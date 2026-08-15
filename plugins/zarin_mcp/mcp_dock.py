# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import time as _time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QSpinBox, QPlainTextEdit, QApplication,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


class ZarinMCPPanel(QWidget):
    def __init__(self, engine, plugin):
        super().__init__()
        self._engine = engine
        self._plugin = plugin
        self._pending = []
        self._lines = []
        self._max_lines = 400
        self._build_ui()
        self._attach_server()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._flush)
        self._refresh_timer.start(400)

    def _mcp_server(self):
        return getattr(self._plugin, "_server", None)

    def _attach_server(self):
        srv = self._mcp_server()
        if srv is not None:
            srv.add_activity_listener(self._on_activity)
            for rec in srv.recent_actions(50):
                self._append_line(rec)
        self._refresh_status()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("ZarinMCP")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        header.addWidget(title)
        header.addStretch()
        self._status_lbl = QLabel("Stopped")
        self._status_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        header.addWidget(self._status_lbl)
        layout.addLayout(header)

        srv_group = QGroupBox("Server")
        srv_form = QFormLayout(srv_group)
        srv_form.setContentsMargins(6, 12, 6, 6)

        self._endpoint_lbl = QLabel("-")
        self._endpoint_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        srv_form.addRow("Endpoint:", self._endpoint_lbl)

        port_row = QHBoxLayout()
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(int(self._plugin.get_config("port", 9100)))
        self._port_spin.setFixedWidth(90)
        port_row.addWidget(self._port_spin)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._on_apply_port)
        port_row.addWidget(self._apply_btn)
        port_row.addStretch()
        srv_form.addRow("Port:", port_row)

        ctrl_row = QHBoxLayout()
        self._start_btn = QPushButton("Start")
        self._start_btn.clicked.connect(self._on_start)
        ctrl_row.addWidget(self._start_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self._on_stop)
        ctrl_row.addWidget(self._stop_btn)
        ctrl_row.addStretch()
        srv_form.addRow("Control:", ctrl_row)
        layout.addWidget(srv_group)

        info_group = QGroupBox("Registry")
        info_layout = QVBoxLayout(info_group)
        info_layout.setContentsMargins(6, 12, 6, 6)
        self._tools_lbl = QLabel("Tools: -")
        self._resources_lbl = QLabel("Resources: -")
        self._templates_lbl = QLabel("Templates: -")
        self._prompts_lbl = QLabel("Prompts: -")
        info_layout.addWidget(self._tools_lbl)
        info_layout.addWidget(self._resources_lbl)
        info_layout.addWidget(self._templates_lbl)
        info_layout.addWidget(self._prompts_lbl)
        layout.addWidget(info_group)

        act_group = QGroupBox("Agent Activity")
        act_layout = QVBoxLayout(act_group)
        act_layout.setContentsMargins(6, 12, 6, 6)

        act_bar = QHBoxLayout()
        self._act_count_lbl = QLabel("0")
        act_bar.addWidget(self._act_count_lbl)
        act_bar.addStretch()
        copy_btn = QPushButton("Copy Endpoint")
        copy_btn.clicked.connect(self._on_copy)
        act_bar.addWidget(copy_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._on_clear)
        act_bar.addWidget(clear_btn)
        act_layout.addLayout(act_bar)

        self._activity = QPlainTextEdit()
        self._activity.setReadOnly(True)
        font = QFont("Consolas", 9)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._activity.setFont(font)
        act_layout.addWidget(self._activity)
        layout.addWidget(act_group)

    def _on_activity(self, rec):
        self._pending.append(rec)

    def _flush(self):
        if self._pending:
            recs = self._pending[:]
            self._pending.clear()
            for rec in recs:
                self._append_line(rec)
        self._refresh_status()

    def _line_for(self, rec):
        ts = _time.strftime("%H:%M:%S", _time.localtime(rec.get("timestamp", _time.time())))
        kind = rec.get("kind", "?")
        name = rec.get("name", "?")
        status = rec.get("status", "ok")
        ms = rec.get("ms")
        ms_part = f" ({ms:.0f} ms)" if isinstance(ms, (int, float)) else ""
        if kind == "resource":
            head = f"[{ts}] read  {name}{ms_part}"
        elif kind == "prompt":
            head = f"[{ts}] prompt {name}{ms_part}"
        else:
            head = f"[{ts}] tool  {name}{ms_part}"
        if status == "error":
            return f"{head} -> ERROR: {rec.get('error', '?')}"
        if kind == "tool" or kind == "prompt":
            args = rec.get("args")
            if args:
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                if len(args_str) > 160:
                    args_str = args_str[:160] + "..."
                head = f"{head} {args_str}"
        return head

    def _append_line(self, rec):
        self._lines.append(self._line_for(rec))
        if len(self._lines) > self._max_lines:
            self._lines = self._lines[-self._max_lines:]
        self._activity.setPlainText("\n".join(self._lines))
        sb = self._activity.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._act_count_lbl.setText(str(len(self._lines)))

    def _refresh_status(self):
        srv = self._mcp_server()
        if srv is None:
            self._status_lbl.setText("No server")
            self._status_lbl.setStyleSheet("color: #888888;")
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(False)
            return
        running = srv.is_running
        if running:
            self._status_lbl.setText("Running")
            self._status_lbl.setStyleSheet("color: #44cc44;")
        else:
            self._status_lbl.setText("Stopped")
            self._status_lbl.setStyleSheet("color: #ff4444;")
        self._start_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._endpoint_lbl.setText(f"http://{srv.host}:{srv.port}/sse")
        reg = self._plugin._registry
        self._tools_lbl.setText(f"Tools: {len(reg.tools)}")
        self._resources_lbl.setText(f"Resources: {len(reg.resources)}")
        self._templates_lbl.setText(f"Templates: {len(reg.resource_templates)}")
        self._prompts_lbl.setText(f"Prompts: {len(reg.prompts)}")

    def _on_start(self):
        srv = self._mcp_server()
        if srv and not srv.is_running:
            srv.start_sse()

    def _on_stop(self):
        srv = self._mcp_server()
        if srv and srv.is_running:
            srv.stop()

    def _on_apply_port(self):
        srv = self._mcp_server()
        if srv is None:
            return
        self._plugin.set_config("port", self._port_spin.value())
        srv.restart()

    def _on_clear(self):
        srv = self._mcp_server()
        if srv:
            srv.clear_activity()
        self._lines.clear()
        self._activity.clear()
        self._act_count_lbl.setText("0")

    def _on_copy(self):
        srv = self._mcp_server()
        if srv:
            QApplication.clipboard().setText(f"http://{srv.host}:{srv.port}/sse")
