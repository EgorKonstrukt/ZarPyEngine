# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from core.config.editor_scale import scale, scale_xy
from PyQt6.QtWidgets import (QToolBar, QPushButton, QLabel, QDoubleSpinBox,
                             QWidget, QSizePolicy, QHBoxLayout, QFrame,
                             QToolButton)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QAction, QIcon

from editor.scene_toolbar import SceneToolbar, RenderToolbar
from editor.main_window.handlers import toggle_play_stop, reset_camera, on_gizmo_vis_toggled


def _set_play_btn_style(btn, text):
    if text == "Play":
        btn.setStyleSheet("QPushButton { background: #2e7d32; color: #fff; }")
    elif text == "Stop":
        btn.setStyleSheet("QPushButton { background: #c0392b; color: #fff; }")
    btn.setText(text)


def _action_to_toolbutton(action, parent=None) -> QToolButton:
    btn = QToolButton(parent)
    btn.setDefaultAction(action)
    btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    btn.setAutoRaise(True)
    return btn


def setup_toolbar(mw):
    mw._main_toolbar = QToolBar("Main", mw)
    mw._main_toolbar.setObjectName("MainToolbar")
    mw._main_toolbar.setMovable(False)
    s = scale_xy(20, 20)
    mw._main_toolbar.setIconSize(QSize(*s))
    mw.addToolBar(Qt.ToolBarArea.TopToolBarArea, mw._main_toolbar)

    # Container widget with QHBoxLayout for proper stretch centering
    container = QWidget()
    lay = QHBoxLayout(container)
    lay.setContentsMargins(2, 2, 2, 2)
    lay.setSpacing(0)

    # ── Left section ──
    mw._gizmo_vis_act = QAction(QIcon.fromTheme("transform-both", QIcon("")), "Gizmo", mw)
    mw._gizmo_vis_act.setCheckable(True)
    mw._gizmo_vis_act.setChecked(True)
    mw._gizmo_vis_act.setToolTip("Toggle Gizmo Visibility")
    mw._gizmo_vis_act.triggered.connect(lambda c: on_gizmo_vis_toggled(mw, c))
    lay.addWidget(_action_to_toolbutton(mw._gizmo_vis_act, mw))

    reset_cam_btn = QToolButton(mw)
    reset_cam_btn.setText("Reset Camera")
    reset_cam_btn.setToolTip("Reset Camera Position")
    reset_cam_btn.setAutoRaise(True)
    reset_cam_btn.clicked.connect(lambda: reset_camera(mw))
    lay.addWidget(reset_cam_btn)

    mw._scene_toolbar = SceneToolbar(mw)
    mw._scene_toolbar.setObjectName("SceneToolbar")
    lay.addWidget(mw._scene_toolbar)

    sep_left = QFrame()
    sep_left.setFrameShape(QFrame.Shape.VLine)
    sep_left.setFrameShadow(QFrame.Shadow.Sunken)
    lay.addWidget(sep_left)

    # ── Stretch: push center toward middle ──
    lay.addStretch(1)

    # ── Center section: Play/Stop, Pause, Time Scale ──
    mw._play_btn = QPushButton("Play")
    _set_play_btn_style(mw._play_btn, "Play")
    mw._play_btn.setFixedWidth(scale(90))
    mw._play_btn.clicked.connect(lambda: toggle_play_stop(mw))
    lay.addWidget(mw._play_btn)

    mw._pause_btn = QPushButton("Pause")
    mw._pause_btn.setStyleSheet(
        "QPushButton { background: #b7950b; color: #fff; }"
        "QPushButton:disabled { background: #5a5a5a; color: #888; }"
    )
    mw._pause_btn.setFixedWidth(scale(90))
    mw._pause_btn.setEnabled(False)
    lay.addWidget(mw._pause_btn)

    ts_label = QLabel(" TS:")
    lay.addWidget(ts_label)
    mw._ts_sb = QDoubleSpinBox()
    mw._ts_sb.setRange(0.0, 10.0)
    mw._ts_sb.setSingleStep(0.1)
    mw._ts_sb.setDecimals(2)
    mw._ts_sb.setValue(1.0)
    mw._ts_sb.setFixedWidth(scale(70))
    mw._ts_sb.valueChanged.connect(lambda v: setattr(mw._engine, "time_scale", v))
    lay.addWidget(mw._ts_sb)

    # ── Stretch: push center toward middle ──
    lay.addStretch(1)

    sep_right = QFrame()
    sep_right.setFrameShape(QFrame.Shape.VLine)
    sep_right.setFrameShadow(QFrame.Shadow.Sunken)
    lay.addWidget(sep_right)

    # ── Right section: render/camera/snap + plugins ──
    mw._render_toolbar = RenderToolbar(mw)
    mw._render_toolbar.setObjectName("RenderToolbar")
    lay.addWidget(mw._render_toolbar)

    add_plugin_toolbar_actions(mw, lay)

    mw._main_toolbar.addWidget(container)


def add_plugin_toolbar_actions(mw, layout):
    registry = mw._engine.plugin_ui_registry
    for info in registry["toolbar_actions"]:
        try:
            text = info["text"]
            callback = info["callback"]
            tooltip = info.get("tooltip", text)
            icon_path = info.get("icon")
            if icon_path:
                act = QAction(QIcon(icon_path), text, mw)
            else:
                act = QAction(text, mw)
            act.setToolTip(tooltip)
            act.triggered.connect(callback)
            layout.addWidget(_action_to_toolbutton(act, mw))
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"Failed to add plugin toolbar action '{info.get('text', '?')}': {e}")
