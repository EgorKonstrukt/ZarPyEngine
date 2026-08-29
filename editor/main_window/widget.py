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
from PyQt6.QtCore import Qt, QSettings, QTimer, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QCloseEvent


class _CollabProxy(QObject):
    scene_open = pyqtSignal(object)
    tab_switch = pyqtSignal(str, str)
    tab_close = pyqtSignal(str)
    peers_changed = pyqtSignal()
    script_open = pyqtSignal(str, str)
    script_change = pyqtSignal(str, str)
    script_cursor = pyqtSignal(str, object)


class _HwMonitorWorker(QObject):
    hw_updated = pyqtSignal(float, float, float, float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll)

    def start(self):
        import psutil
        psutil.cpu_percent(interval=None)
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _poll(self):
        cpu = 0.0
        ram_used = 0.0
        ram_total = 0.0
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            ram_used = mem.used / (1024 * 1024)
            ram_total = mem.total / (1024 * 1024)
        except Exception:
            pass
        self.hw_updated.emit(cpu, ram_used, ram_total, 0.0, 0.0, 0.0)

from core.engine.engine import Engine
from core.foundation.logger import Logger

from editor.main_window.docks import setup_docks
from editor.main_window.menu import setup_menu
from editor.main_window.toolbar import setup_toolbar
from editor.main_window.statusbar import setup_statusbar
from editor.main_window.connections import connect_signals
from editor.main_window.state import restore_camera, save_state, restore_script_tabs
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
        from editor.scene_async import call_on_main
        engine._gui_poster = call_on_main
        self._settings = QSettings("Zarin", "Editor")
        self._play_dock: Optional[QDockWidget] = None
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
        self._init_script_tabs()
        setup_statusbar(self)
        self._hw_worker = _HwMonitorWorker()
        self._hw_thread = QThread(self)
        self._hw_worker.moveToThread(self._hw_thread)
        self._hw_thread.started.connect(self._hw_worker.start)
        self._hw_worker.hw_updated.connect(self._on_hw_updated)
        self._hw_thread.start()
        connect_signals(self)
        restore_camera(self)
        if not self._layout_restored:
            save_state(self)
        engine.on("scene_loaded", lambda s: on_scene_loaded(self, s))
        self._setup_engine_events()
        QTimer.singleShot(0, lambda: post_init(self))

    def _init_collab_scene_tabs(self):
        mgr = self._scene_tab_manager
        mgr.tab_switched.connect(lambda name: self._on_scene_tab_switched(name))
        mgr.tab_added.connect(lambda name: self._on_scene_tab_added(name))
        collab = self._engine.collab_manager
        if collab:
            self._collab_proxy = _CollabProxy()
            self._collab_proxy.scene_open.connect(self._on_remote_scene_open)
            self._collab_proxy.tab_switch.connect(self._on_remote_tab_switch)
            self._collab_proxy.tab_close.connect(mgr.remove_tab)
            self._collab_proxy.peers_changed.connect(self._update_tab_peer_indicators)
            collab.set_on_remote_scene_open(lambda data: self._collab_proxy.scene_open.emit(data))
            collab.set_on_remote_tab_switch(lambda pid, name: self._collab_proxy.tab_switch.emit(pid, name))
            collab.set_on_remote_tab_close(lambda name: self._collab_proxy.tab_close.emit(name))
            collab.set_peer_joined_callback(lambda _: self._collab_proxy.peers_changed.emit())
            collab.set_peer_left_callback(lambda _: self._collab_proxy.peers_changed.emit())
            self._collab_proxy.script_open.connect(self._on_remote_script_open)
            self._collab_proxy.script_change.connect(self._on_remote_script_change)
            self._collab_proxy.script_cursor.connect(self._on_remote_script_cursor)
            collab.set_on_remote_script_open(lambda data: self._collab_proxy.script_open.emit(data.get("path", ""), data.get("content", "")))
            collab.set_on_remote_script_change(lambda data: self._collab_proxy.script_change.emit(data.get("path", ""), data.get("content", "")))
            collab.set_on_remote_script_cursor(lambda pid, data: self._collab_proxy.script_cursor.emit(pid, data))
            print("[COLLAB] Tab callbacks registered", flush=True)

    def _init_script_tabs(self):
        sw = self._script_editor._script_widget
        mgr = self._scene_tab_manager
        self._syncing_script = False
        self._script_peer_editors: dict[str, list[list[float]]] = {}

        for i in range(sw._tabs.count()):
            tab = sw._tabs.widget(i)
            path = tab._file_path or ""
            mgr.add_script_tab(path, tab._tab_title())

        sw.tab_opened.connect(self._on_script_tab_opened)
        sw.tab_closed.connect(self._on_script_tab_closed_internal)
        sw.tab_switched.connect(self._on_script_internal_tab_switched)
        mgr.script_tab_selected.connect(self._on_script_tab_in_bar_selected)
        mgr.script_tab_closed.connect(self._on_script_tab_closed_from_bar)
        collab = self._engine.collab_manager
        if collab:
            sw.collab_file_opened.connect(lambda path, content: collab.send_script_open(path, content))
            sw.collab_file_saved.connect(lambda path, content: collab.send_script_change(path, content))
            sw.collab_cursor_changed.connect(lambda path, pos, anchor, end: collab.send_script_cursor(path, pos, anchor, end))
            sw.collab_ops_ready.connect(lambda path, ops: collab.send_script_ops(path, ops))
            collab.set_on_remote_script_ops(lambda pid, data: sw.apply_remote_ops(data.get("path", ""), data.get("ops", [])))

        saved_paths = restore_script_tabs(self)
        if saved_paths:
            for path in saved_paths:
                sw.open_script(path)
            if sw._tabs.count() > 0:
                first = sw._tabs.widget(0)
                if first and not first._file_path:
                    sw._tabs.removeTab(0)

    def _on_script_tab_opened(self, path: str):
        sw = self._script_editor._script_widget
        for i in range(sw._tabs.count()):
            tab = sw._tabs.widget(i)
            if tab._file_path == path:
                self._scene_tab_manager.add_script_tab(path, tab._tab_title())
                return

    def _on_script_tab_closed_internal(self, path: str):
        self._scene_tab_manager.remove_script_tab_by_path(path)

    def _on_script_tab_in_bar_selected(self, path: str):
        sw = self._script_editor._script_widget
        tab = sw._current_tab()
        if tab and (tab._file_path or "") == path:
            self._script_editor.show()
            self._script_editor.raise_()
            return
        if self._syncing_script:
            return
        self._syncing_script = True
        try:
            sw.open_script(path)
            self._script_editor.show()
            self._script_editor.raise_()
        finally:
            self._syncing_script = False

    def _on_script_tab_closed_from_bar(self, path: str):
        sw = self._script_editor._script_widget
        for i in range(sw._tabs.count()):
            tab = sw._tabs.widget(i)
            if tab._file_path == path:
                tab._close_self()
                return

    def _on_script_internal_tab_switched(self, path: str):
        if self._syncing_script:
            return
        mgr = self._scene_tab_manager
        for i in range(self._scene_tab_bar.count()):
            if mgr.script_path_at(i) == path:
                if self._scene_tab_bar.currentIndex() != i:
                    self._scene_tab_bar.blockSignals(True)
                    self._scene_tab_bar.setCurrentIndex(i)
                    self._scene_tab_bar.blockSignals(False)
                return

    def _on_remote_script_open(self, path: str, content: str):
        if not path:
            return
        sw = self._script_editor._script_widget
        for i in range(sw._tabs.count()):
            tab = sw._tabs.widget(i)
            if tab._file_path == path:
                return
        sw.open_script(path)

    def _on_remote_script_change(self, path: str, content: str):
        if not path:
            return
        sw = self._script_editor._script_widget
        for i in range(sw._tabs.count()):
            tab = sw._tabs.widget(i)
            if tab._file_path == path:
                if tab._dirty:
                    return
                tab.set_content(content, from_remote=True)
                return
        sw.open_script(path)

    def _on_remote_script_cursor(self, peer_id: str, data: dict):
        path = data.get("path", "")
        pos = data.get("pos", 0)
        sel_anchor = data.get("sel_anchor", pos)
        sel_end = data.get("sel_end", pos)
        collab = self._engine.collab_manager
        peer = collab.get_peer(peer_id) if collab else None
        if peer and path:
            self._script_editor._script_widget.update_remote_cursor(
                peer_id, path, pos, sel_anchor, sel_end, peer.color, peer.name
            )
            self._script_peer_editors[path] = [
                p.color for p in collab.peers.values()
                if p.peer_id in self._script_editor._script_widget._remote_cursors.get(path, {})
            ]
            self._scene_tab_manager.update_script_peer_indicators(self._script_peer_editors)

    def _on_remote_scene_open(self, data: dict):
        try:
            name = data.get("name", "RemoteScene")
            path = data.get("path", "")
            scene_data = data.get("data")
            print(f"[COLLAB] _on_remote_scene_open called: name={name}, path={path}, has_data={scene_data is not None}", flush=True)
            if scene_data is None:
                return
            from core.ecs.ecs import Scene, ComponentRegistry
            scene = Scene.deserialize(scene_data, ComponentRegistry)
            if path:
                scene.path = path
            self._tab_add_lock = True
            tab_name = self._scene_tab_manager.add_tab(name, path=path, scene=scene)
            print(f"[COLLAB] Tab created: {tab_name}", flush=True)
            collab = self._engine.collab_manager
            if collab and tab_name:
                collab.set_current_tab(tab_name)
        except Exception as e:
            import traceback
            traceback.print_exc()
        finally:
            self._tab_add_lock = False

    def _on_scene_tab_added(self, name: str):
        locked = getattr(self, '_tab_add_lock', False)
        if locked:
            return
        collab = self._engine.collab_manager
        if collab and collab.connected:
            info = self._scene_tab_manager.get_tab_info(name)
            if info and info.scene:
                try:
                    data = info.scene.serialize()
                    collab.send_scene_open(info.name, info.path or "", data)
                    if collab.is_host:
                        collab.update_server_scene(info.scene)
                    print(f"[COLLAB] Sent SCENE_OPEN for tab '{info.name}' to peers (host={collab.is_host})", flush=True)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        else:
            print(f"[COLLAB] _on_scene_tab_added '{name}' skipped: collab={collab is not None}, connected={collab.connected if collab else False}", flush=True)

    def _on_remote_tab_switch(self, peer_id: str, tab_name: str):
        self._update_tab_peer_indicators()

    def _update_tab_peer_indicators(self):
        collab = self._engine.collab_manager
        peers = collab.peers if collab and collab.connected else {}
        tab_peers: dict[str, list[list[float]]] = {}
        for p in peers.values():
            if p.current_tab:
                tab_peers.setdefault(p.current_tab, []).append(p.color)
        self._scene_tab_manager.update_peer_indicators(tab_peers)
        self._scene_tab_manager.update_script_peer_indicators(self._script_peer_editors)

    def _on_scene_tab_switched(self, name: str):
        self._viewport_dock.show()
        self._viewport_dock.raise_()
        collab = self._engine.collab_manager
        if collab:
            collab.set_current_tab(name)
            if collab.connected:
                collab.send_scene_tab_switch(name)

    def _setup_engine_events(self):
        from editor.main_window.handlers import on_play_start, on_play_stop
        self._engine.on("play_start", lambda _: on_play_start(self, _))
        self._engine.on("play_stop", lambda _: on_play_stop(self, _))

    def _update_status(self):
        if self._status_fps_lbl:
            vp = getattr(self._engine, 'viewport', None)
            render_fps = vp._fps if vp and hasattr(vp, '_fps') else 0.0
            tps = self._engine.tps
            s = f"FPS: {render_fps:.0f} | TPS: {tps:.0f}"
            if s != self._status_fps_lbl.text():
                self._status_fps_lbl.setText(s)
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
        if self._engine.scene and self._engine.scene.dirty:
            name = self._engine.scene.name
            self.setWindowTitle(f"Zarin Engine Editor - {name}*")
        from core.foundation.commands import get_history
        h = get_history()
        self._undo_act.setEnabled(h.can_undo)
        self._undo_act.setText(f"Undo ({h.undo_text.split()[-1] if h.can_undo else ''})" if h.can_undo else "Undo")
        self._redo_act.setEnabled(h.can_redo)
        self._redo_act.setText(f"Redo ({h.redo_text.split()[-1] if h.can_redo else ''})" if h.can_redo else "Redo")

    def _on_hw_updated(self, cpu, ram_used, ram_total, gpu, vram_used, vram_total):
        if self._status_cpu_lbl:
            s = f"CPU: {cpu:.0f}%"
            if s != self._status_cpu_lbl.text():
                self._status_cpu_lbl.setText(s)
        if self._status_ram_lbl:
            s = f"RAM: {ram_used:.0f} / {ram_total:.0f} MB"
            if s != self._status_ram_lbl.text():
                self._status_ram_lbl.setText(s)

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
                self._engine.save_scene()
        save_state(self)
        self._engine.shutdown()
        event.accept()
