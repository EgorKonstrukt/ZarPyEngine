# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import qtawesome as qta
from core.config.editor_scale import scale, scale_xy
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QPushButton, QLabel,
                             QSpinBox, QCheckBox, QFileDialog, QMessageBox, QComboBox)
from PyQt6.QtCore import Qt, pyqtSignal


def _separator():
    s = QFrame()
    s.setFrameShape(QFrame.Shape.VLine)
    s.setStyleSheet("QFrame { color: #555; margin: 2px 0; }")
    s.setFixedWidth(scale(2))
    return s


class GuiEditorToolbar(QFrame):
    toggle_grid = pyqtSignal(bool)
    toggle_snap = pyqtSignal(bool)
    toggle_auto_align = pyqtSignal(bool)
    grid_size_changed = pyqtSignal(int)
    clear_requested = pyqtSignal()
    save_requested = pyqtSignal()
    load_requested = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    screen_w_changed = pyqtSignal(int)
    screen_h_changed = pyqtSignal(int)
    resolution_preset = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame#guiToolbar { background-color: #2d2d2d; border-bottom: 1px solid #444; }
            QPushButton {
                background-color: #3a3a3a; color: #ddd; border: 1px solid #555;
                border-radius: 3px; padding: 4px 10px; font-size: 11px;
            }
            QPushButton:hover { background-color: #4a4a4a; }
            QPushButton:checked { background-color: #4a7ab5; color: #fff; }
            QSpinBox { background-color: #1e1e1e; color: #ddd; border: 1px solid #555;
                       border-radius: 2px; padding: 2px; font-size: 11px; max-width: 60px; }
            QLabel { color: #aaa; font-size: 11px; }
            QCheckBox { color: #aaa; font-size: 11px; }
        """)
        self.setObjectName("guiToolbar")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 6, 3)
        layout.setSpacing(6)
        self._mode_btn = QPushButton("Edit Mode")
        self._mode_btn.setCheckable(True)
        self._mode_btn.setChecked(True)
        layout.addWidget(self._mode_btn)
        layout.addWidget(_separator())
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self.save_requested.emit)
        layout.addWidget(self._save_btn)
        self._load_btn = QPushButton("Load")
        self._load_btn.clicked.connect(self.load_requested.emit)
        layout.addWidget(self._load_btn)
        layout.addWidget(_separator())
        self._grid_cb = QCheckBox("Grid")
        self._grid_cb.setChecked(True)
        self._grid_cb.stateChanged.connect(lambda s: self.toggle_grid.emit(s == Qt.CheckState.Checked.value))
        layout.addWidget(self._grid_cb)
        self._snap_cb = QCheckBox("Snap")
        self._snap_cb.setChecked(True)
        self._snap_cb.stateChanged.connect(lambda s: self.toggle_snap.emit(s == Qt.CheckState.Checked.value))
        layout.addWidget(self._snap_cb)
        self._align_cb = QCheckBox("Align")
        self._align_cb.setChecked(True)
        self._align_cb.stateChanged.connect(lambda s: self.toggle_auto_align.emit(s == Qt.CheckState.Checked.value))
        layout.addWidget(self._align_cb)
        layout.addWidget(QLabel("Size:"))
        self._grid_size_sb = QSpinBox()
        self._grid_size_sb.setRange(1, 64)
        self._grid_size_sb.setValue(8)
        self._grid_size_sb.valueChanged.connect(self.grid_size_changed.emit)
        layout.addWidget(self._grid_size_sb)
        layout.addWidget(_separator())
        layout.addWidget(QLabel("Screen:"))
        self._screen_preset = QComboBox()
        self._screen_preset.setStyleSheet("QComboBox { background-color:#1e1e1e; color:#ddd; border:1px solid #555; border-radius:2px; padding:2px; font-size:10px; min-width:100px; } QComboBox::drop-down { border:none; } QComboBox::down-arrow { image:none; }")
        presets = [
            ("1920Г—1080 (1080p)", 1920, 1080),
            ("2560Г—1440 (1440p)", 2560, 1440),
            ("3840Г—2160 (4K)", 3840, 2160),
            ("1366Г—768 (HD)", 1366, 768),
            ("1280Г—720 (720p)", 1280, 720),
            ("1024Г—768 (XGA)", 1024, 768),
            ("800Г—600 (SVGA)", 800, 600),
        ]
        self._preset_data = presets
        for name, w, h in presets:
            self._screen_preset.addItem(name, (w, h))
        self._screen_preset.setCurrentIndex(0)
        self._screen_preset.currentIndexChanged.connect(self._on_preset_idx)
        layout.addWidget(self._screen_preset)
        self._screen_w_sb = QSpinBox()
        self._screen_w_sb.setRange(320, 7680)
        self._screen_w_sb.setValue(1920)
        self._screen_w_sb.setSingleStep(160)
        layout.addWidget(self._screen_w_sb)
        self._screen_h_sb = QSpinBox()
        self._screen_h_sb.setRange(240, 4320)
        self._screen_h_sb.setValue(1080)
        self._screen_h_sb.setSingleStep(90)
        layout.addWidget(self._screen_h_sb)
        layout.addWidget(_separator())
        self._zoom_out_btn = QPushButton("в€’")
        self._zoom_out_btn.setFixedWidth(scale(24))
        self._zoom_out_btn.clicked.connect(lambda: self.zoom_changed.emit(-0.1))
        layout.addWidget(self._zoom_out_btn)
        self._zoom_label = QLabel("100%")
        self._zoom_label.setStyleSheet("color: #ddd; font-size: 11px; min-width: 36px; text-align: center;")
        layout.addWidget(self._zoom_label)
        self._zoom_in_btn = QPushButton(qta.icon("fa5s.search-plus", color="#d4d4d4"), "")
        self._zoom_in_btn.setFixedWidth(scale(24))
        self._zoom_in_btn.clicked.connect(lambda: self.zoom_changed.emit(0.1))
        layout.addWidget(self._zoom_in_btn)
        layout.addStretch()
        self._clear_btn = QPushButton(qta.icon("fa5s.times-circle", color="#ff6b6b"), " Clear")
        
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        layout.addWidget(self._clear_btn)

    def set_zoom_label(self, percent: int):
        self._zoom_label.setText(f"{percent}%")

    def _on_preset_idx(self, idx: int):
        data = self._screen_preset.currentData()
        if data:
            w, h = data
            self._screen_w_sb.setValue(w)
            self._screen_h_sb.setValue(h)
            self.screen_w_changed.emit(w)
            self.screen_h_changed.emit(h)

    def set_screen_size(self, w: int, h: int):
        for i, (name, pw, ph) in enumerate(self._preset_data):
            if pw == w and ph == h:
                self._screen_preset.setCurrentIndex(i)
                return
        self._screen_w_sb.setValue(w)
        self._screen_h_sb.setValue(h)
        self.screen_w_changed.emit(w)
        self.screen_h_changed.emit(h)
