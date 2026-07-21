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

from core.engine.engine import Engine
from core.foundation.logger import Logger

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
    on_gizmo_vis_toggled, reset_camera,
    on_global_config_changed, open_global_settings, open_project_settings,
    show_build_dialog, show_about,
    on_scene_loaded,
)
from editor.main_window.scene_tabs import SceneTabManager
from core.network.protocol import MessageType


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
        self._scene_tab_manager = SceneTabManager(self._engine, self._scene_tab_bar, self)
        self._init_collab_scene_tabs()
        setup_statusbar(self)
        connect_signals(self)
        restore_camera(self)
        if not self._layout_restored:
            save_state(self)
        engine.on("scene_loaded", lambda s: on_scene_loaded(self, s))
        self._setup_engine_events()
        QTimer.singleShot(0, lambda: post_init(self))

    def _init_collab_scene_tabs(self):
        collab = self._engine.collab_manager
        if collab:
            mgr = self._scene_tab_manager
            collab.set_on_remote_scene_open(lambda data: QTimer.singleShot(0, lambda: self._on_remote_scene_open(data)))
            collab.set_on_remote_tab_switch(lambda name: QTimer.singleShot(0, lambda: mgr.switch_to_tab(name)))
            collab.set_on_remote_tab_close(lambda name: QTimer.singleShot(0, lambda: mgr.remove_tab(name)))
            mgr.tab_switched.connect(lambda name: self._on_scene_tab_switched(name))
            mgr.tab_added.connect(lambda name: self._on_scene_tab_added(name))

    def _on_remote_scene_open(self, data: dict):
        name = data.get("name", "RemoteScene")
        path = data.get("path", "")
        scene_data = data.get("data")
        if scene_data is None:
            return
        self._tab_add_lock = True
        self._scene_tab_manager.add_tab(name, path, scene_data)
        self._tab_add_lock = False

    def _on_scene_tab_added(self, name: str):
        self._tab_add_lock = getattr(self, '_tab_add_lock', False)
        if self._tab_add_lock:
            return
        collab = self._engine.collab_manager
        if collab and collab.connected:
            info = self._scene_tab_manager.get_tab_info(name)
            if info and info.data:
                collab.send_scene_open(info.name, info.path or "", info.data)

    def _on_scene_tab_switched(self, name: str):
        collab = self._engine.collab_manager
        if collab and collab.connected:
            collab.send_scene_tab_switch(name)
            if self._engine.scene:
                collab.update_server_scene(self._engine.scene)

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
        if self._status_gpu_name_lbl:
            try:
                vp = getattr(self._engine, 'viewport', None)
                info = getattr(vp, '_gl_info_cache', None) if vp else None
                if info:
                    name = info.get("GL_RENDERER", "")
                    if name and self._status_gpu_name_lbl.text() != name:
                        self._status_gpu_name_lbl.setText(name)
            except Exception:
                pass
        self._update_hw_monitors()
        if self._engine.scene and self._engine.scene.dirty:
            name = self._engine.scene.name
            self.setWindowTitle(f"Zarin Engine Editor - {name}*")
        from core.foundation.commands import get_history
        h = get_history()
        self._undo_act.setEnabled(h.can_undo)
        self._undo_act.setText(f"Undo ({h.undo_text.split()[-1] if h.can_undo else ''})" if h.can_undo else "Undo")
        self._redo_act.setEnabled(h.can_redo)
        self._redo_act.setText(f"Redo ({h.redo_text.split()[-1] if h.can_redo else ''})" if h.can_redo else "Redo")

    def _update_hw_monitors(self):
        try:
            import psutil
            if self._status_cpu_lbl:
                self._status_cpu_lbl.setText(f"CPU: {psutil.cpu_percent(interval=None):.0f}%")
            if self._status_ram_lbl:
                mem = psutil.virtual_memory()
                self._status_ram_lbl.setText(f"RAM: {mem.used / (1024*1024):.0f} / {mem.total / (1024*1024):.0f} MB")
        except Exception:
            pass
        if self._status_gpu_lbl or self._status_vram_lbl:
            self._update_gpu_stats()

    _NVML_HANDLE = None

    def _update_gpu_stats(self):
        gpu_used = None
        vram_used = None
        vram_total = None
        try:
            if EditorMainWindow._NVML_HANDLE is None:
                import pynvml
                pynvml.nvmlInit()
                EditorMainWindow._NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(EditorMainWindow._NVML_HANDLE)
            gpu_used = util.gpu
            info = pynvml.nvmlDeviceGetMemoryInfo(EditorMainWindow._NVML_HANDLE)
            vram_used = info.used // (1024*1024)
            vram_total = info.total // (1024*1024)
        except Exception:
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        gpu_used = float(parts[0])
                        vram_used = int(parts[1])
                        vram_total = int(parts[2])
            except Exception:
                pass
        if self._status_gpu_lbl:
            self._status_gpu_lbl.setText(f"GPU: {gpu_used:.0f}%" if gpu_used is not None else "GPU: N/A")
        if self._status_vram_lbl:
            self._status_vram_lbl.setText(
                f"VRAM: {vram_used} / {vram_total} MB" if vram_used is not None else "VRAM: N/A"
            )

    def closeEvent(self, event: QCloseEvent):
        if self._engine.play_mode:
            self._engine.stop_play()
        from core.config.config import get_global_config
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
