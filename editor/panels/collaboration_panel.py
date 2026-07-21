# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QLabel, QPushButton, QLineEdit, QSpinBox,
                              QListWidget, QListWidgetItem, QFrame,
                              QProgressBar, QMessageBox, QFileDialog)
from PyQt6.QtCore import Qt, QTimer
import qtawesome as qta
from PyQt6.QtGui import QColor, QFont


class CollaborationPanel(QDockWidget):
    def __init__(self, engine, parent=None):
        super().__init__("Collaboration", parent)
        self._engine = engine
        self._collab: Optional = None
        self._setup_ui()
        self._update_timer = QTimer(self)
        self._update_timer.timeout.connect(self._refresh_peers)
        self._update_timer.start(500)

    def set_collaboration_manager(self, mgr):
        self._collab = mgr
        self._collab.set_on_scene_sync(self._on_scene_sync)

    def _on_scene_sync(self, scene_data: dict) -> bool:
        scene = self._engine.scene
        if scene and scene.dirty:
            btn = QMessageBox.question(
                self, "Scene Not Saved",
                "The current scene has unsaved changes.\n"
                "Save before syncing with the collaborator's scene?",
                QMessageBox.StandardButton.Save |
                QMessageBox.StandardButton.Discard |
                QMessageBox.StandardButton.Cancel
            )
            if btn == QMessageBox.StandardButton.Cancel:
                return False
            if btn == QMessageBox.StandardButton.Save:
                if scene.path:
                    self._engine.save_scene()
                else:
                    path, _ = QFileDialog.getSaveFileName(
                        self, "Save Scene", "scenes/", "Scenes (*.zpes)"
                    )
                    if path:
                        if not path.endswith(".zpes"):
                            path += ".zpes"
                        self._engine.save_scene(path)
                    else:
                        return False
        return True

    def _setup_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        title = QLabel("Collaboration")
        title.setStyleSheet("font-size: 11px; font-weight: bold; padding: 2px;")
        layout.addWidget(title)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep)

        self._status_label = QLabel("Disconnected")
        self._status_label.setStyleSheet("font-size: 10px;")
        layout.addWidget(self._status_label)

        self._name_input = QLineEdit()
        self._name_input.setPlaceholderText("Your name")
        self._name_input.setText("User")
        layout.addWidget(self._name_input)

        host_row = QHBoxLayout()
        self._host_input = QLineEdit()
        self._host_input.setPlaceholderText("Host")
        self._host_input.setText("0.0.0.0")
        host_row.addWidget(self._host_input)

        self._port_input = QSpinBox()
        self._port_input.setRange(1024, 65535)
        self._port_input.setValue(9876)
        host_row.addWidget(self._port_input)
        layout.addLayout(host_row)

        self._host_btn = QPushButton(qta.icon("fa5s.server", color="#fff"), " Host Session")
        self._host_btn.setStyleSheet("QPushButton { background: #2471a3; }")
        self._host_btn.clicked.connect(self._on_host)
        layout.addWidget(self._host_btn)

        connect_row = QHBoxLayout()
        self._connect_host_input = QLineEdit()
        self._connect_host_input.setPlaceholderText("Server IP")
        self._connect_host_input.setText("127.0.0.1")
        connect_row.addWidget(self._connect_host_input)

        self._connect_port_input = QSpinBox()
        self._connect_port_input.setRange(1024, 65535)
        self._connect_port_input.setValue(9876)
        connect_row.addWidget(self._connect_port_input)
        layout.addLayout(connect_row)

        self._connect_btn = QPushButton(qta.icon("fa5s.plug", color="#fff"), " Connect")
        self._connect_btn.setStyleSheet("QPushButton { background: #2e7d32; }")
        self._connect_btn.clicked.connect(self._on_connect)
        layout.addWidget(self._connect_btn)

        self._disconnect_btn = QPushButton(qta.icon("fa5s.unlink", color="#fff"), " Disconnect")
        self._disconnect_btn.setStyleSheet("QPushButton { background: #c0392b; }")
        self._disconnect_btn.clicked.connect(self._on_disconnect)
        self._disconnect_btn.setEnabled(False)
        layout.addWidget(self._disconnect_btn)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep2)

        self._play_btn = QPushButton(qta.icon("fa5s.play", color="#d4d4d4"), " Play Mode (Lock)")
        self._play_btn.setCheckable(True)
        self._play_btn.clicked.connect(self._on_play_toggle)
        layout.addWidget(self._play_btn)

        bw_row = QHBoxLayout()
        self._bw_sent_label = QLabel("Up: 0 B/s")
        self._bw_sent_label.setStyleSheet("font-size: 9px;")
        bw_row.addWidget(self._bw_sent_label)
        self._bw_recv_label = QLabel("Down: 0 B/s")
        self._bw_recv_label.setStyleSheet("font-size: 9px;")
        bw_row.addWidget(self._bw_recv_label)
        layout.addLayout(bw_row)

        self._latency_label = QLabel("Latency: -- ms")
        self._latency_label.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._latency_label)

        sep_assets = QFrame()
        sep_assets.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep_assets)

        asset_title = QLabel("Asset Sync")
        asset_title.setStyleSheet("font-size: 10px; font-weight: bold;")
        layout.addWidget(asset_title)

        self._asset_status = QLabel("Idle")
        self._asset_status.setStyleSheet("font-size: 9px;")
        layout.addWidget(self._asset_status)

        self._asset_progress = QProgressBar()
        self._asset_progress.setRange(0, 100)
        self._asset_progress.setValue(0)
        self._asset_progress.setFixedHeight(12)
        self._asset_progress.setVisible(False)
        layout.addWidget(self._asset_progress)

        self._sync_btn = QPushButton(qta.icon("fa5s.sync-alt", color="#d4d4d4"), " Sync Assets")
        self._sync_btn.clicked.connect(self._on_sync_assets)
        layout.addWidget(self._sync_btn)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(sep3)

        peer_title = QLabel("Connected Peers")
        peer_title.setStyleSheet("font-size: 10px; font-weight: bold;")
        layout.addWidget(peer_title)

        self._peer_list = QListWidget()
        layout.addWidget(self._peer_list)

        layout.addStretch()

        self.setWidget(container)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )

    def _on_host(self):
        if not self._collab:
            return
        name = self._name_input.text() or "Host"
        host = self._host_input.text() or "0.0.0.0"
        port = self._port_input.value()
        self._collab.start_server(host, port)
        self._collab.connect("127.0.0.1", port, name)
        self._update_status()

    def _on_connect(self):
        if not self._collab:
            return
        name = self._name_input.text() or "User"
        host = self._connect_host_input.text() or "127.0.0.1"
        port = self._connect_port_input.value()
        self._collab.connect(host, port, name)
        self._update_status()

    def _on_disconnect(self):
        if self._collab:
            self._collab.stop()
        self._update_status()

    def _update_status(self):
        connected = self._collab and self._collab.connected
        if connected:
            self._status_label.setText("Connected")
            self._status_label.setStyleSheet("font-size: 10px;")
        else:
            self._status_label.setText("Disconnected")
            self._status_label.setStyleSheet("font-size: 10px;")
        self._host_btn.setEnabled(not connected)
        self._connect_btn.setEnabled(not connected)
        self._disconnect_btn.setEnabled(connected)

    def _on_play_toggle(self, checked: bool):
        if self._collab:
            self._collab.send_play_mode(checked)

    def _refresh_peers(self):
        if not self._collab:
            return
        self._update_status()
        self._peer_list.clear()
        collab = self._collab
        for pid, peer in collab.peers.items():
            c = peer.color
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            ping_text = f"  [{peer.ping_ms:.0f}ms]" if peer.ping_ms > 0 else "  [--ms]"
            item = QListWidgetItem(f"{peer.name}{ping_text}")
            item.setForeground(QColor(r, g, b))
            f = QFont("Segoe UI", 9)
            item.setFont(f)
            self._peer_list.addItem(item)
        bw_sent = collab.bytes_sent
        bw_recv = collab.bytes_received
        self._bw_sent_label.setText(f"Up: {self._format_bytes(bw_sent)}")
        self._bw_recv_label.setText(f"Down: {self._format_bytes(bw_recv)}")
        latency = collab.latency_ms
        self._latency_label.setText(f"Latency: {latency:.0f} ms" if latency > 0 else "Latency: -- ms")
        self._play_btn.setEnabled(collab.is_host)
        self._play_btn.setChecked(collab.play_mode_active)
        self._update_asset_sync()

    def _update_asset_sync(self):
        if not self._collab:
            return
        progress = self._collab.asset_sync_progress
        total = progress["total"]
        current = progress["current"]
        file_name = progress["current_file"]
        failed = progress["failed"]
        syncing = self._collab.asset_syncing
        if syncing and total > 0:
            pct = int(current / total * 100) if total > 0 else 0
            self._asset_progress.setValue(pct)
            self._asset_progress.setVisible(True)
            self._asset_status.setText(f"Syncing: {file_name} ({current}/{total})")
            self._sync_btn.setEnabled(False)
        else:
            self._asset_progress.setVisible(total > 0 and not syncing)
            if total > 0 and not syncing:
                self._asset_progress.setValue(100)
            if failed:
                self._asset_status.setText(f"Synced ({failed} failed)")
                self._asset_status.setStyleSheet("font-size: 9px;")
            elif total > 0 and not syncing:
                self._asset_status.setText("Synced")
                self._asset_status.setStyleSheet("font-size: 9px;")
            else:
                self._asset_status.setText("Idle")
                self._asset_status.setStyleSheet("font-size: 9px;")
            self._sync_btn.setEnabled(self._collab.connected)

    def _on_sync_assets(self):
        if not self._collab or not self._collab.connected:
            return
        self._collab.request_asset_list()

    @staticmethod
    def _format_bytes(b: int) -> str:
        if b < 1024:
            return f"{b} B"
        elif b < 1024 * 1024:
            return f"{b / 1024:.1f} KB"
        else:
            return f"{b / (1024 * 1024):.1f} MB"
