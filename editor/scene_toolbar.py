# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.config.editor_scale import scale, scale_xy
from PyQt6.QtWidgets import (QToolBar, QLabel, QPushButton, QCheckBox,
                             QDoubleSpinBox, QComboBox, QFrame, QWidget,
                             QHBoxLayout, QSizePolicy)
from PyQt6.QtCore import pyqtSignal, Qt
from core.gizmo.gizmo import GizmoMode, GizmoSpace
from core.renderer.types import RenderMode


def _make_sep() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.VLine)
    f.setFrameShadow(QFrame.Shadow.Sunken)
    return f


class SceneToolbar(QToolBar):
    """Left section: Gizmo mode (Q/W/E/R) + Space (World/Local)."""
    gizmo_mode_changed = pyqtSignal(object)
    gizmo_space_changed = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__("Scene Tools", parent)
        self.setMovable(False)
        self._setup()

    def _setup(self):
        gizmo_label = QLabel("Gizmo: ")
        self.addWidget(gizmo_label)
        self._btn_none = QPushButton("Q")
        self._btn_none.setFixedWidth(scale(28))
        self._btn_none.setToolTip("No Gizmo (Q)")
        self._btn_none.clicked.connect(lambda: self.gizmo_mode_changed.emit(GizmoMode.NONE))
        self.addWidget(self._btn_none)
        self._btn_translate = QPushButton("W")
        self._btn_translate.setFixedWidth(scale(28))
        self._btn_translate.setToolTip("Translate (W)")
        self._btn_translate.clicked.connect(lambda: self.gizmo_mode_changed.emit(GizmoMode.TRANSLATE))
        self.addWidget(self._btn_translate)
        self._btn_rotate = QPushButton("E")
        self._btn_rotate.setFixedWidth(scale(28))
        self._btn_rotate.setToolTip("Rotate (E)")
        self._btn_rotate.clicked.connect(lambda: self.gizmo_mode_changed.emit(GizmoMode.ROTATE))
        self.addWidget(self._btn_rotate)
        self._btn_scale = QPushButton("R")
        self._btn_scale.setFixedWidth(scale(28))
        self._btn_scale.setToolTip("Scale (R)")
        self._btn_scale.clicked.connect(lambda: self.gizmo_mode_changed.emit(GizmoMode.SCALE))
        self.addWidget(self._btn_scale)
        self.addWidget(_make_sep())
        space_label = QLabel("Space: ")
        self.addWidget(space_label)
        self._space_cb = QComboBox()
        self._space_cb.addItems(["World", "Local"])
        self._space_cb.currentTextChanged.connect(self._on_space_changed)
        self.addWidget(self._space_cb)

    def _on_space_changed(self, text: str):
        space = GizmoSpace.WORLD if text == "World" else GizmoSpace.LOCAL
        self.gizmo_space_changed.emit(space)


class RenderToolbar(QToolBar):
    """Right section: Render mode, Camera, Skybox, FX, Grid, Snap."""
    render_mode_changed = pyqtSignal(object)
    grid_toggled = pyqtSignal(bool)
    snap_toggled = pyqtSignal(bool)
    snap_translate_changed = pyqtSignal(float)
    snap_rotate_changed = pyqtSignal(float)
    snap_scale_changed = pyqtSignal(float)
    skybox_toggled = pyqtSignal(bool)
    effects_toggled = pyqtSignal(bool)
    camera_projection_changed = pyqtSignal()
    mode_2d_toggled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Render Tools", parent)
        self.setMovable(False)
        self._setup()

    def _setup(self):
        self.addWidget(QLabel("Render: "))
        self._shaded_btn = QPushButton("S")
        self._shaded_btn.setFixedWidth(scale(24))
        self._shaded_btn.setToolTip("Shaded")
        self._shaded_btn.clicked.connect(lambda: self.render_mode_changed.emit(RenderMode.SHADED))
        self.addWidget(self._shaded_btn)
        self._sh_wire_btn = QPushButton("S+W")
        self._sh_wire_btn.setFixedWidth(scale(32))
        self._sh_wire_btn.setToolTip("Shaded + Wireframe")
        self._sh_wire_btn.clicked.connect(lambda: self.render_mode_changed.emit(RenderMode.SHADED_WIREFRAME))
        self.addWidget(self._sh_wire_btn)
        self._flat_btn = QPushButton("F")
        self._flat_btn.setFixedWidth(scale(24))
        self._flat_btn.setToolTip("Flat (no lighting)")
        self._flat_btn.clicked.connect(lambda: self.render_mode_changed.emit(RenderMode.FLAT))
        self.addWidget(self._flat_btn)
        self.addWidget(_make_sep())
        self._cam_persp_btn = QPushButton("Perspective")
        self._cam_persp_btn.setFixedWidth(scale(80))
        self._cam_persp_btn.setToolTip("Toggle Perspective/Orthographic Camera")
        self._cam_persp_btn.setCheckable(True)
        self._cam_persp_btn.setChecked(True)
        self._cam_persp_btn.clicked.connect(self._on_camera_projection_changed)
        self.addWidget(self._cam_persp_btn)
        self._cam_2d_btn = QPushButton("2D")
        self._cam_2d_btn.setFixedWidth(scale(32))
        self._cam_2d_btn.setToolTip("Toggle 2D Mode")
        self._cam_2d_btn.setCheckable(True)
        self._cam_2d_btn.clicked.connect(self.mode_2d_toggled)
        self.addWidget(self._cam_2d_btn)
        self.addWidget(_make_sep())
        self._skybox_cb = QCheckBox("Skybox")
        self._skybox_cb.setChecked(True)
        self._skybox_cb.toggled.connect(self.skybox_toggled)
        self.addWidget(self._skybox_cb)
        self._effects_cb = QCheckBox("FX")
        self._effects_cb.setChecked(True)
        self._effects_cb.setToolTip("Toggle post-processing effects")
        self._effects_cb.toggled.connect(self.effects_toggled)
        self.addWidget(self._effects_cb)
        self.addWidget(_make_sep())
        self._grid_cb = QCheckBox("Grid")
        self._grid_cb.setChecked(True)
        self._grid_cb.toggled.connect(self.grid_toggled)
        self.addWidget(self._grid_cb)
        self.addWidget(_make_sep())
        self._snap_cb = QCheckBox("Snap")
        self._snap_cb.setChecked(True)
        self._snap_cb.toggled.connect(self.snap_toggled)
        self.addWidget(self._snap_cb)
        snap_t_label = QLabel(" T:")
        self.addWidget(snap_t_label)
        self._snap_t_sb = QDoubleSpinBox()
        self._snap_t_sb.setRange(0.0, 100.0)
        self._snap_t_sb.setSingleStep(0.1)
        self._snap_t_sb.setDecimals(3)
        self._snap_t_sb.setValue(0.25)
        self._snap_t_sb.setFixedWidth(scale(65))
        self._snap_t_sb.setToolTip("Translate snap")
        self._snap_t_sb.valueChanged.connect(self.snap_translate_changed)
        self.addWidget(self._snap_t_sb)
        snap_r_label = QLabel(" R:")
        self.addWidget(snap_r_label)
        self._snap_r_sb = QDoubleSpinBox()
        self._snap_r_sb.setRange(0.0, 360.0)
        self._snap_r_sb.setSingleStep(5.0)
        self._snap_r_sb.setDecimals(1)
        self._snap_r_sb.setValue(15.0)
        self._snap_r_sb.setFixedWidth(scale(60))
        self._snap_r_sb.setToolTip("Rotate snap (degrees)")
        self._snap_r_sb.valueChanged.connect(self.snap_rotate_changed)
        self.addWidget(self._snap_r_sb)
        snap_s_label = QLabel(" S:")
        self.addWidget(snap_s_label)
        self._snap_s_sb = QDoubleSpinBox()
        self._snap_s_sb.setRange(0.0, 100.0)
        self._snap_s_sb.setSingleStep(0.1)
        self._snap_s_sb.setDecimals(3)
        self._snap_s_sb.setValue(0.25)
        self._snap_s_sb.setFixedWidth(scale(60))
        self._snap_s_sb.setToolTip("Scale snap")
        self._snap_s_sb.valueChanged.connect(self.snap_scale_changed)
        self.addWidget(self._snap_s_sb)

    def _on_camera_projection_changed(self):
        self.camera_projection_changed.emit()

    def save_state(self):
        from core.config.config import get_global_config
        cfg = get_global_config()
        cfg.set("toolbar.grid", self._grid_cb.isChecked())
        cfg.set("toolbar.snap", self._snap_cb.isChecked())
        cfg.set("toolbar.snap_translate", self._snap_t_sb.value())
        cfg.set("toolbar.snap_rotate", self._snap_r_sb.value())
        cfg.set("toolbar.snap_scale", self._snap_s_sb.value())
        cfg.set("toolbar.skybox", self._skybox_cb.isChecked())
        cfg.set("toolbar.effects", self._effects_cb.isChecked())
        cfg.save()

    def load_state(self):
        from core.config.config import get_global_config
        cfg = get_global_config()
        if cfg.has("toolbar.grid"):
            self._grid_cb.setChecked(cfg.get("toolbar.grid", True))
        if cfg.has("toolbar.snap"):
            self._snap_cb.setChecked(cfg.get("toolbar.snap", True))
        if cfg.has("toolbar.snap_translate"):
            self._snap_t_sb.setValue(cfg.get("toolbar.snap_translate", 0.25))
        if cfg.has("toolbar.snap_rotate"):
            self._snap_r_sb.setValue(cfg.get("toolbar.snap_rotate", 15.0))
        if cfg.has("toolbar.snap_scale"):
            self._snap_s_sb.setValue(cfg.get("toolbar.snap_scale", 0.25))
        if cfg.has("toolbar.skybox"):
            self._skybox_cb.setChecked(cfg.get("toolbar.skybox", False))
        if cfg.has("toolbar.effects"):
            self._effects_cb.setChecked(cfg.get("toolbar.effects", True))
