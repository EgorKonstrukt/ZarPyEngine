# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QGroupBox, QFormLayout, QFrame,
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


class VRControlWidget(QWidget):
    def __init__(self, engine, vr_plugin):
        super().__init__()
        self._engine = engine
        self._plugin = vr_plugin
        self._build_ui()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh)
        self._refresh_timer.start(333)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._status_lbl = QLabel("VR: Not Available")
        self._status_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        layout.addWidget(self._status_lbl)

        self._toggle_btn = QPushButton("Initialize VR")
        self._toggle_btn.setEnabled(False)
        self._toggle_btn.clicked.connect(self._on_toggle)
        layout.addWidget(self._toggle_btn)

        ipd_group = QGroupBox("IPD")
        ipd_layout = QFormLayout(ipd_group)
        ipd_layout.setContentsMargins(6, 12, 6, 6)

        ipd_row = QHBoxLayout()
        self._ipd_slider = QSlider(Qt.Orientation.Horizontal)
        self._ipd_slider.setRange(10, 200)
        self._ipd_slider.setValue(63)
        self._ipd_slider.valueChanged.connect(self._on_ipd_changed)
        ipd_row.addWidget(self._ipd_slider)
        self._ipd_lbl = QLabel("0.063 m")
        self._ipd_lbl.setFixedWidth(60)
        ipd_row.addWidget(self._ipd_lbl)
        ipd_layout.addRow(ipd_row)
        ipd_btn_row = QHBoxLayout()
        self._ipd_reset_btn = QPushButton("Reset IPD")
        self._ipd_reset_btn.clicked.connect(self._on_reset_ipd)
        ipd_btn_row.addWidget(self._ipd_reset_btn)
        ipd_btn_row.addStretch()
        ipd_layout.addRow(ipd_btn_row)
        layout.addWidget(ipd_group)

        origin_group = QGroupBox("Tracking")
        origin_layout = QVBoxLayout(origin_group)
        origin_layout.setContentsMargins(6, 12, 6, 6)
        self._reset_origin_btn = QPushButton("Reset HMD Origin")
        self._reset_origin_btn.clicked.connect(self._on_reset_origin)
        origin_layout.addWidget(self._reset_origin_btn)
        self._offset_lbl = QLabel("Offset: 0.00, 0.00, 0.00")
        origin_layout.addWidget(self._offset_lbl)
        layout.addWidget(origin_group)

        view_group = QGroupBox("Viewport")
        view_layout = QVBoxLayout(view_group)
        view_layout.setContentsMargins(6, 12, 6, 6)
        self._eye_view_btn = QPushButton("View: Eye")
        self._eye_view_btn.clicked.connect(self._on_toggle_view)
        view_layout.addWidget(self._eye_view_btn)
        layout.addWidget(view_group)

        ctrl_group = QGroupBox("Controllers")
        ctrl_layout = QVBoxLayout(ctrl_group)
        ctrl_layout.setContentsMargins(6, 12, 6, 6)
        self._ctrl_left_lbl = QLabel("Left:  --")
        self._ctrl_right_lbl = QLabel("Right: --")
        ctrl_layout.addWidget(self._ctrl_left_lbl)
        ctrl_layout.addWidget(self._ctrl_right_lbl)
        layout.addWidget(ctrl_group)

        self._session_lbl = QLabel("Session: --")
        layout.addWidget(self._session_lbl)

        ar_group = QGroupBox("AR / XR")
        ar_layout = QVBoxLayout(ar_group)
        ar_layout.setContentsMargins(6, 12, 6, 6)
        self._ar_lbl = QLabel("AR: Off")
        ar_layout.addWidget(self._ar_lbl)
        self._spawn_rig_btn = QPushButton("Spawn XR Rig")
        self._spawn_rig_btn.clicked.connect(self._on_spawn_rig)
        ar_layout.addWidget(self._spawn_rig_btn)
        layout.addWidget(ar_group)

        layout.addStretch()

    def _on_toggle(self):
        self._plugin.toggle_vr()

    def _on_ipd_changed(self, value):
        from plugins.vr_plugin.vr_core import set_ipd
        ipd = value / 1000.0
        set_ipd(ipd)
        self._ipd_lbl.setText(f"{ipd:.3f} m")

    def _on_reset_ipd(self):
        from plugins.vr_plugin.vr_core import reset_ipd
        reset_ipd()
        self._refresh()

    def _on_reset_origin(self):
        from plugins.vr_plugin.vr_core import reset_hmd_origin
        reset_hmd_origin()

    def _on_toggle_view(self):
        from plugins.vr_plugin.vr_core import toggle_eye_view
        v = toggle_eye_view()
        self._eye_view_btn.setText("View: Eye" if v else "View: Desktop")
        if hasattr(self._plugin, '_eye_view'):
            self._plugin._eye_view = v
            if self._plugin._vr_view_btn is not None:
                try:
                    import qtawesome as qta
                    if v:
                        self._plugin._vr_view_btn.setText(" Eye")
                        self._plugin._vr_view_btn.setIcon(qta.icon("fa5s.eye", color="#4fc3f7"))
                    else:
                        self._plugin._vr_view_btn.setText(" Desktop")
                        self._plugin._vr_view_btn.setIcon(qta.icon("fa5s.desktop", color="#aed581"))
                except Exception:
                    pass
        try:
            if self._engine.viewport:
                self._engine.viewport.update()
        except Exception:
            pass

    def _on_spawn_rig(self):
        try:
            rig = self._plugin.spawn_xr_rig(self._engine)
            if rig is not None:
                self._ar_lbl.setText("XR Rig: Spawned")
                try:
                    if self._engine.viewport:
                        self._engine.viewport.update()
                except Exception:
                    pass
        except Exception:
            pass

    def _refresh(self):
        from plugins.vr_plugin.vr_core import (
            is_available, vr_enabled, session_running, is_active,
            get_ipd, get_hmd_pos_offset, get_controllers, is_eye_view,
        )
        avail = is_available()
        enabled = vr_enabled() or is_active()
        running = session_running()

        self._toggle_btn.setEnabled(avail)
        if enabled:
            if running:
                self._status_lbl.setText("VR: Active")
                self._status_lbl.setStyleSheet("color: #44cc44;")
            else:
                self._status_lbl.setText("VR: Initializing...")
                self._status_lbl.setStyleSheet("color: #cccc44;")
            self._toggle_btn.setText("Shutdown VR")
        else:
            self._status_lbl.setText("VR: Off")
            self._status_lbl.setStyleSheet("color: #888888;")
            self._toggle_btn.setText("Initialize VR")

        ipd = get_ipd()
        self._ipd_slider.blockSignals(True)
        self._ipd_slider.setValue(int(ipd * 1000))
        self._ipd_slider.blockSignals(False)
        self._ipd_lbl.setText(f"{ipd:.3f} m")
        try:
            self._eye_view_btn.setText("View: Eye" if is_eye_view() else "View: Desktop")
            self._eye_view_btn.setEnabled(enabled)
        except Exception:
            pass

        off = get_hmd_pos_offset()
        self._offset_lbl.setText(f"Offset: {off[0]:.2f}, {off[1]:.2f}, {off[2]:.2f}")

        self._session_lbl.setText(f"Session: {'Running' if running else 'Idle'}")

        try:
            from plugins.vr_plugin.vr_core import is_ar_mode, supports_passthrough_ar
            if is_ar_mode():
                self._ar_lbl.setText("AR: Active (passthrough)" if supports_passthrough_ar() else "AR: Active (alpha-blend)")
                self._ar_lbl.setStyleSheet("color: #44ccaa;")
            else:
                self._ar_lbl.setText("AR: Off (add ARSession component)")
                self._ar_lbl.setStyleSheet("color: #888888;")
        except Exception:
            pass

        ctrls = get_controllers()
        for lbl, ctrl, name in [
            (self._ctrl_left_lbl, ctrls[0], "Left"),
            (self._ctrl_right_lbl, ctrls[1], "Right"),
        ]:
            if ctrl.valid:
                trig = getattr(ctrl, 'trigger', 0.0)
                ts = getattr(ctrl, 'thumbstick', (0.0, 0.0))
                lbl.setText(f"{name}: OK grip={ctrl.grip:.2f} trig={trig:.2f} stick={ts[0]:.2f},{ts[1]:.2f}")
                lbl.setStyleSheet("color: #44cc44;")
            else:
                lbl.setText(f"{name}:  --")
                lbl.setStyleSheet("color: #888888;")
