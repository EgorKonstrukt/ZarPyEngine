# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import json
import os
from typing import Optional

from PyQt6.QtWidgets import QMainWindow, QDockWidget, QWidget
from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QCloseEvent

from core.engine import Engine
from core.logger import Logger

from editor.main_window.docks import setup_docks
from editor.main_window.menu import setup_menu
from editor.main_window.toolbar import setup_toolbar
from editor.main_window.statusbar import setup_statusbar
from editor.main_window.connections import connect_signals
from editor.main_window.state import restore_camera, save_state
from editor.main_window.postinit import post_init, initial_dock_sizes
from editor.main_window.project import switch_project, open_project_manager, open_project_browse
from editor.main_window.handlers import (
    new_scene, open_scene, save_scene, save_scene_as,
    toggle_play_stop, toggle_pause,
    undo, redo,
    on_gizmo_vis_toggled, reset_camera,
    on_global_config_changed, open_global_settings, open_project_settings,
    show_build_dialog, show_about,
    on_scene_loaded,
)


class EditorMainWindow(QMainWindow):
    def __init__(self, engine: Engine):
        super().__init__()
        self._engine = engine
        self._settings = QSettings("Zarin", "Editor")
        self._play_dock: Optional[QDockWidget] = None
        self._scene_snapshot: Optional[dict] = None
        self._prefab_mode: bool = False
        self._saved_scene = None
        self._prefab_path: Optional[str] = None
        self._global_settings_dlg = None
        self._project_settings_dlg = None
        self.setWindowTitle("Zarin Engine Editor")
        self.setMinimumSize(1280, 720)
        self.setContentsMargins(0, 0, 0, 0)
        self.setDockOptions(
            QMainWindow.DockOption.AllowNestedDocks |
            QMainWindow.DockOption.AnimatedDocks |
            QMainWindow.DockOption.AllowTabbedDocks)

        self._dummy_central = QWidget()
        self._dummy_central.setMinimumSize(0, 0)
        self._dummy_central.setMaximumSize(0, 0)
        self.setCentralWidget(self._dummy_central)
        self._restored_geometry_once = False

        setup_docks(self)
        setup_menu(self)
        setup_toolbar(self)
        setup_statusbar(self)
        connect_signals(self)
        restore_camera(self)
        if not self._layout_restored:
            save_state(self)
        engine.on("scene_loaded", lambda s: on_scene_loaded(self, s))
        self._setup_engine_events()
        QTimer.singleShot(0, lambda: post_init(self))

    def _setup_engine_events(self):
        from editor.main_window.handlers import on_play_start, on_play_stop
        self._engine.on("play_start", lambda _: on_play_start(self, _))
        self._engine.on("play_stop", lambda _: on_play_stop(self, _))

    def _update_status(self):
        if self._status_fps_lbl:
            vp = getattr(self._engine, 'viewport', None)
            render_fps = vp._fps if vp and hasattr(vp, '_fps') else 0.0
            tps = self._engine.tps
            self._status_fps_lbl.setText(f"FPS: {render_fps:.0f} | TPS: {tps:.0f}")
        if self._engine.scene and self._engine.scene.dirty:
            name = self._engine.scene.name
            self.setWindowTitle(f"Zarin Engine Editor - {name}*")
        from core.commands import get_history
        h = get_history()
        self._undo_act.setEnabled(h.can_undo)
        self._undo_act.setText(f"Undo ({h.undo_text.split()[-1] if h.can_undo else ''})" if h.can_undo else "Undo")
        self._redo_act.setEnabled(h.can_redo)
        self._redo_act.setText(f"Redo ({h.redo_text.split()[-1] if h.can_redo else ''})" if h.can_redo else "Redo")

    def closeEvent(self, event: QCloseEvent):
        if self._engine.play_mode:
            self._engine.stop_play()
        from core.config import get_global_config
        cfg = get_global_config()
        if hasattr(self, '_terminal') and hasattr(self._terminal, 'save_config'):
            self._terminal.save_config(cfg)
        if hasattr(self, '_project'):
            self._project.save_config(cfg)
        cfg.save()
        if self._engine.scene and self._engine.scene.dirty:
            from PyQt6.QtWidgets import QMessageBox
            reply = QMessageBox.question(self, "Unsaved Changes",
                                         "Scene has unsaved changes. Save before closing?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
            if reply == QMessageBox.StandardButton.Yes:
                save_scene(self)
        save_state(self)
        self._engine.shutdown()
        event.accept()
