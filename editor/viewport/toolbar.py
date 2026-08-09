# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import qtawesome as qta
from core.config.editor_scale import scale, scale_xy
from PyQt6.QtWidgets import QDoubleSpinBox, QFrame, QHBoxLayout, QLabel, QMenu, QPushButton, QSpinBox, QVBoxLayout, QWidget, QWidgetAction
from PyQt6.QtGui import QPalette

from core.maths.math3d import Vec3


def _pal(role: QPalette.ColorRole) -> str:
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app:
        return app.palette().color(role).name()
    return "#cccccc"


def _darken(color_name: str, factor: float = 0.8) -> str:
    from PyQt6.QtGui import QColor
    c = QColor(color_name)
    return QColor(int(c.red() * factor), int(c.green() * factor), int(c.blue() * factor)).name()


def setup_toolbar(vp):
    vp._toolbar = QFrame(vp)
    vp._toolbar.setStyleSheet("""
        QFrame {
            border-bottom: 1px solid #444;
            border-top-left-radius: 4px;
            border-top-right-radius: 4px;
        }
    """)
    toolbar_layout = QVBoxLayout(vp._toolbar)
    toolbar_layout.setContentsMargins(6, 3, 6, 3)
    toolbar_layout.setSpacing(3)

    cam_row = QHBoxLayout()
    vp._cam_menu_btn = QPushButton(qta.icon("fa5s.camera", color="#d4d4d4"), " Camera")
    vp._cam_menu_btn.setMenu(create_camera_menu(vp))
    vp._cam_menu_btn.setMinimumWidth(60)
    cam_row.addWidget(vp._cam_menu_btn)

    vp._stats_btn = QPushButton(qta.icon("fa5s.chart-bar", color="#d4d4d4"), " Stats")
    vp._stats_btn.setCheckable(True)
    vp._stats_btn.setMinimumWidth(50)
    vp._stats_btn.clicked.connect(vp._toggle_stats)
    cam_row.addWidget(vp._stats_btn)

    vp._audio_btn = QPushButton(qta.icon("fa5s.music", color="#d4d4d4"), " Audio")
    vp._audio_btn.setCheckable(True)
    vp._audio_btn.setMinimumWidth(50)
    vp._audio_btn.clicked.connect(vp._toggle_audio_viz)
    cam_row.addWidget(vp._audio_btn)

    vp._audio_viz_btn = QPushButton(qta.icon("fa5s.sliders-h", color="#d4d4d4"), " Viz")
    vp._audio_viz_btn.setMenu(create_audio_viz_menu(vp))
    vp._audio_viz_btn.setMinimumWidth(40)
    cam_row.addWidget(vp._audio_viz_btn)

    vp._bvh_btn = QPushButton(qta.icon("fa5s.sitemap", color="#d4d4d4"), " BVH")
    vp._bvh_btn.setCheckable(True)
    vp._bvh_btn.setMinimumWidth(40)
    vp._bvh_btn.clicked.connect(vp._toggle_bvh_debug)
    cam_row.addWidget(vp._bvh_btn)

    vp._depth_spin = QSpinBox()
    vp._depth_spin.setRange(0, 32)
    vp._depth_spin.setValue(24)
    vp._depth_spin.setMinimumWidth(50)
    cam_row.addWidget(vp._depth_spin)

    depth_label = QLabel("Depth")
    cam_row.addWidget(depth_label)
    cam_row.addStretch()

    vp._cam_pos_label = QLabel("Cam: 0.0, 0.0, 0.0")
    vp._cam_pos_label.setStyleSheet("font-size: 10px;")
    cam_row.addWidget(vp._cam_pos_label)

    vp._cursor_x_label = QLabel("X: -")
    vp._cursor_x_label.setStyleSheet("color: #ff4444; font-size: 10px;")
    cam_row.addWidget(vp._cursor_x_label)

    vp._cursor_y_label = QLabel("Y: -")
    vp._cursor_y_label.setStyleSheet("color: #44cc44; font-size: 10px;")
    cam_row.addWidget(vp._cursor_y_label)

    vp._cursor_z_label = QLabel("Z: -")
    vp._cursor_z_label.setStyleSheet("color: #4488ff; font-size: 10px;")
    cam_row.addWidget(vp._cursor_z_label)

    vp._depth_spin.valueChanged.connect(vp._on_depth_changed)
    vp._depth_spin.installEventFilter(vp)

    toolbar_layout.addLayout(cam_row)

    prefab_row = QHBoxLayout()
    prefab_row.setContentsMargins(0, 0, 0, 0)
    prefab_row.setSpacing(4)
    mw = vp.parent()
    from editor.main_window.handlers import on_return_to_scene
    vp._return_btn = QPushButton("To Scene")
    vp._return_btn.setFixedHeight(scale(22))
    vp._return_btn.clicked.connect(lambda: on_return_to_scene(mw))
    prefab_row.addWidget(vp._return_btn)
    prefab_row.addStretch()
    vp._prefab_btns = QWidget()
    vp._prefab_btns.setLayout(prefab_row)
    vp._prefab_btns.hide()
    toolbar_layout.addWidget(vp._prefab_btns)

    vp._toolbar.setFixedHeight(scale(30))


def create_camera_menu(vp):
    menu = QMenu(vp._toolbar)
    fov_label_w = QWidget()
    fov_label_layout = QHBoxLayout(fov_label_w)
    fov_label_layout.setContentsMargins(4, 1, 4, 1)
    fov_label_layout.addWidget(QLabel("FOV"))
    fov_label_layout.addStretch()
    fov_label_action = QWidgetAction(menu)
    fov_label_action.setDefaultWidget(fov_label_w)
    menu.addAction(fov_label_action)

    vp._fov_spin = QDoubleSpinBox()
    vp._fov_spin.setRange(1.0, 179.0)
    vp._fov_spin.setValue(vp._cam._fov if hasattr(vp._cam, '_fov') else vp._cam.DEFAULT_FOV)
    vp._fov_spin.setSingleStep(5.0)
    vp._fov_spin.setDecimals(1)
    vp._fov_spin.setMinimumWidth(60)
    fov_widget = QWidget()
    fov_layout = QHBoxLayout(fov_widget)
    fov_layout.setContentsMargins(4, 1, 4, 1)
    fov_layout.addWidget(vp._fov_spin)
    vp._fov_spin.valueChanged.connect(vp._on_fov_changed)
    fov_spin_action = QWidgetAction(menu)
    fov_spin_action.setDefaultWidget(fov_widget)
    menu.addAction(fov_spin_action)

    near_label_w = QWidget()
    near_label_layout = QHBoxLayout(near_label_w)
    near_label_layout.setContentsMargins(4, 1, 4, 1)
    near_label_layout.addWidget(QLabel("Near"))
    near_label_layout.addStretch()
    near_label_action = QWidgetAction(menu)
    near_label_action.setDefaultWidget(near_label_w)
    menu.addAction(near_label_action)

    vp._near_spin = QDoubleSpinBox()
    vp._near_spin.setRange(0.001, 10.0)
    vp._near_spin.setValue(vp._cam._near if hasattr(vp._cam, '_near') else vp._cam.DEFAULT_NEAR)
    vp._near_spin.setSingleStep(0.01)
    vp._near_spin.setDecimals(3)
    vp._near_spin.setMinimumWidth(60)
    near_widget = QWidget()
    near_layout = QHBoxLayout(near_widget)
    near_layout.setContentsMargins(4, 1, 4, 1)
    near_layout.addWidget(vp._near_spin)
    vp._near_spin.valueChanged.connect(vp._on_near_changed)
    near_spin_action = QWidgetAction(menu)
    near_spin_action.setDefaultWidget(near_widget)
    menu.addAction(near_spin_action)

    far_label_w = QWidget()
    far_label_layout = QHBoxLayout(far_label_w)
    far_label_layout.setContentsMargins(4, 1, 4, 1)
    far_label_layout.addWidget(QLabel("Far"))
    far_label_layout.addStretch()
    far_label_action = QWidgetAction(menu)
    far_label_action.setDefaultWidget(far_label_w)
    menu.addAction(far_label_action)

    vp._far_spin = QDoubleSpinBox()
    vp._far_spin.setRange(10.0, 10000.0)
    vp._far_spin.setValue(vp._cam._far if hasattr(vp._cam, '_far') else vp._cam.DEFAULT_FAR)
    vp._far_spin.setSingleStep(100.0)
    vp._far_spin.setDecimals(0)
    vp._far_spin.setMinimumWidth(60)
    far_widget = QWidget()
    far_layout = QHBoxLayout(far_widget)
    far_layout.setContentsMargins(4, 1, 4, 1)
    far_layout.addWidget(vp._far_spin)
    vp._far_spin.valueChanged.connect(vp._on_far_changed)
    far_spin_action = QWidgetAction(menu)
    far_spin_action.setDefaultWidget(far_widget)
    menu.addAction(far_spin_action)

    move_label_w = QWidget()
    move_label_layout = QHBoxLayout(move_label_w)
    move_label_layout.setContentsMargins(4, 1, 4, 1)
    move_label_layout.addWidget(QLabel("Move Speed"))
    move_label_layout.addStretch()
    move_label_action = QWidgetAction(menu)
    move_label_action.setDefaultWidget(move_label_w)
    menu.addAction(move_label_action)

    vp._move_speed_spin = QDoubleSpinBox()
    vp._move_speed_spin.setRange(1.0, 50.0)
    vp._move_speed_spin.setValue(vp._cam._move_speed if hasattr(vp._cam, '_move_speed') else vp._cam.MOVE_SPEED)
    vp._move_speed_spin.setSingleStep(1.0)
    vp._move_speed_spin.setDecimals(1)
    vp._move_speed_spin.setMinimumWidth(60)
    move_widget = QWidget()
    move_layout = QHBoxLayout(move_widget)
    move_layout.setContentsMargins(4, 1, 4, 1)
    move_layout.addWidget(vp._move_speed_spin)
    vp._move_speed_spin.valueChanged.connect(vp._on_move_speed_changed)
    move_spin_action = QWidgetAction(menu)
    move_spin_action.setDefaultWidget(move_widget)
    menu.addAction(move_spin_action)

    rot_label_w = QWidget()
    rot_label_layout = QHBoxLayout(rot_label_w)
    rot_label_layout.setContentsMargins(4, 1, 4, 1)
    rot_label_layout.addWidget(QLabel("Rotate Speed"))
    rot_label_layout.addStretch()
    rot_label_action = QWidgetAction(menu)
    rot_label_action.setDefaultWidget(rot_label_w)
    menu.addAction(rot_label_action)

    vp._rotate_speed_spin = QDoubleSpinBox()
    vp._rotate_speed_spin.setRange(0.05, 2.0)
    vp._rotate_speed_spin.setValue(vp._cam._rotate_speed if hasattr(vp._cam, '_rotate_speed') else vp._cam.ROTATE_SPEED)
    vp._rotate_speed_spin.setSingleStep(0.05)
    vp._rotate_speed_spin.setDecimals(2)
    vp._rotate_speed_spin.setMinimumWidth(60)
    rot_widget = QWidget()
    rot_layout = QHBoxLayout(rot_widget)
    rot_layout.setContentsMargins(4, 1, 4, 1)
    rot_layout.addWidget(vp._rotate_speed_spin)
    vp._rotate_speed_spin.valueChanged.connect(vp._on_rotate_speed_changed)
    rot_spin_action = QWidgetAction(menu)
    rot_spin_action.setDefaultWidget(rot_widget)
    menu.addAction(rot_spin_action)

    res_label_w = QWidget()
    res_label_layout = QHBoxLayout(res_label_w)
    res_label_layout.setContentsMargins(4, 1, 4, 1)
    res_label_layout.addWidget(QLabel("Render Resolution"))
    res_label_layout.addStretch()
    res_label_action = QWidgetAction(menu)
    res_label_action.setDefaultWidget(res_label_w)
    menu.addAction(res_label_action)

    vp._render_scale_spin = QSpinBox()
    vp._render_scale_spin.setRange(5, 100)
    vp._render_scale_spin.setValue(int(round((vp._cam._render_scale if hasattr(vp._cam, '_render_scale') else 1.0) * 100)))
    vp._render_scale_spin.setSingleStep(5)
    vp._render_scale_spin.setSuffix("%")
    vp._render_scale_spin.setMinimumWidth(60)
    res_widget = QWidget()
    res_layout = QHBoxLayout(res_widget)
    res_layout.setContentsMargins(4, 1, 4, 1)
    res_layout.addWidget(vp._render_scale_spin)
    vp._render_scale_spin.valueChanged.connect(vp._on_render_scale_changed)
    res_spin_action = QWidgetAction(menu)
    res_spin_action.setDefaultWidget(res_widget)
    menu.addAction(res_spin_action)

    return menu


def _viz_row(menu, label: str):
    label_w = QWidget()
    label_layout = QHBoxLayout(label_w)
    label_layout.setContentsMargins(4, 1, 4, 1)
    label_layout.addWidget(QLabel(label))
    label_layout.addStretch()
    label_action = QWidgetAction(menu)
    label_action.setDefaultWidget(label_w)
    menu.addAction(label_action)


def create_audio_viz_menu(vp):
    opts = getattr(vp, "_audio_viz_opts", {}) or {}
    menu = QMenu(vp._toolbar)

    _viz_row(menu, "Sensitivity")

    vp._scope_gain_spin = QDoubleSpinBox()
    vp._scope_gain_spin.setRange(0.2, 8.0)
    vp._scope_gain_spin.setValue(float(opts.get("scope_gain", 0.82)))
    vp._scope_gain_spin.setSingleStep(0.1)
    vp._scope_gain_spin.setDecimals(2)
    vp._scope_gain_spin.setMinimumWidth(60)
    scope_widget = QWidget()
    scope_layout = QHBoxLayout(scope_widget)
    scope_layout.setContentsMargins(4, 1, 4, 1)
    scope_layout.addWidget(QLabel("Scope"))
    scope_layout.addStretch()
    scope_layout.addWidget(vp._scope_gain_spin)
    vp._scope_gain_spin.valueChanged.connect(
        lambda v: vp._set_audio_viz_opt("scope_gain", float(v)))
    scope_spin_action = QWidgetAction(menu)
    scope_spin_action.setDefaultWidget(scope_widget)
    menu.addAction(scope_spin_action)

    vp._wave_gain_spin = QDoubleSpinBox()
    vp._wave_gain_spin.setRange(0.5, 20.0)
    vp._wave_gain_spin.setValue(float(opts.get("wave_gain", 1.0) or 1.0))
    vp._wave_gain_spin.setSingleStep(0.5)
    vp._wave_gain_spin.setDecimals(2)
    vp._wave_gain_spin.setMinimumWidth(60)
    wave_widget = QWidget()
    wave_layout = QHBoxLayout(wave_widget)
    wave_layout.setContentsMargins(4, 1, 4, 1)
    wave_layout.addWidget(QLabel("Wave"))
    wave_layout.addStretch()
    wave_layout.addWidget(vp._wave_gain_spin)
    vp._wave_gain_spin.valueChanged.connect(
        lambda v: vp._set_audio_viz_opt("wave_gain", float(v)))
    wave_spin_action = QWidgetAction(menu)
    wave_spin_action.setDefaultWidget(wave_widget)
    menu.addAction(wave_spin_action)

    vp._hold_decay_spin = QDoubleSpinBox()
    vp._hold_decay_spin.setRange(0.05, 8.0)
    vp._hold_decay_spin.setValue(float(opts.get("hold_decay", 0.8)))
    vp._hold_decay_spin.setSingleStep(0.1)
    vp._hold_decay_spin.setDecimals(2)
    vp._hold_decay_spin.setMinimumWidth(60)
    decay_widget = QWidget()
    decay_layout = QHBoxLayout(decay_widget)
    decay_layout.setContentsMargins(4, 1, 4, 1)
    decay_layout.addWidget(QLabel("Hold Decay"))
    decay_layout.addStretch()
    decay_layout.addWidget(vp._hold_decay_spin)
    vp._hold_decay_spin.valueChanged.connect(
        lambda v: vp._set_audio_viz_opt("hold_decay", float(v)))
    decay_spin_action = QWidgetAction(menu)
    decay_spin_action.setDefaultWidget(decay_widget)
    menu.addAction(decay_spin_action)

    return menu
