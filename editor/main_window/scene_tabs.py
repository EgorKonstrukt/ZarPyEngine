# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QPainter, QColor, QBrush, QIcon
from PyQt6.QtWidgets import QTabBar, QMessageBox

from core.ecs.ecs import Scene
from core.math.math3d import Vec3

try:
    import qtawesome as qta
except ImportError:
    qta = None

_QTA_COLORS = {"#d4d4d4": "#d4d4d4"}

def _qta_icon(name: str) -> QIcon:
    if qta is None:
        return QIcon()
    return qta.icon(name, color="#d4d4d4")

def _icon_for_script(path: str) -> QIcon:
    ext = os.path.splitext(path)[1].lower()
    mapping = {
        ".py": "fa5b.python",
        ".txt": "fa5s.file-alt",
        ".md": "fa5s.file-alt",
        ".json": "fa5s.file-code",
        ".xml": "fa5s.file-code",
        ".yaml": "fa5s.file-code",
        ".yml": "fa5s.file-code",
        ".toml": "fa5s.file-code",
        ".cfg": "fa5s.file-code",
        ".ini": "fa5s.file-code",
        ".csv": "fa5s.file-csv",
        ".html": "fa5s.file-code",
        ".css": "fa5s.file-code",
        ".js": "fa5s.file-code",
        ".ts": "fa5s.file-code",
        ".sh": "fa5s.terminal",
        ".bat": "fa5s.terminal",
        ".ps1": "fa5s.terminal",
        ".cpp": "fa5s.file-code",
        ".h": "fa5s.file-code",
        ".rs": "fa5s.file-code",
        ".go": "fa5s.file-code",
    }
    name = mapping.get(ext, "fa5s.file")
    return _qta_icon(name)


class SceneTabInfo:
    __slots__ = ("name", "path", "scene", "dirty", "camera_state",
                 "selected_entity_ids", "play_mode", "scene_snapshot",
                 "prefab_path", "origin_tab")
    def __init__(self, name: str = "Scene", path: Optional[str] = None, scene: Optional[Scene] = None):
        self.name = name
        self.path = path
        self.scene = scene
        self.dirty = False
        self.camera_state: Optional[dict] = None
        self.selected_entity_ids: list[str] = []
        self.play_mode: bool = False
        self.scene_snapshot: Optional[dict] = None
        self.prefab_path: Optional[str] = None
        self.origin_tab: Optional[str] = None


class SceneTabManager(QObject):
    tab_switched = pyqtSignal(str)
    tab_added = pyqtSignal(str)
    script_tab_selected = pyqtSignal(str)
    script_tab_closed = pyqtSignal(str)

    SCRIPT_TAB_PREFIX = "__script__"

    def __init__(self, engine, tab_bar: QTabBar, mw):
        super().__init__(mw)
        self._engine = engine
        self._tab_bar = tab_bar
        self._mw = mw
        self._tabs: dict[str, SceneTabInfo] = {}
        self._tab_names: list[str] = []
        self._active_tab: Optional[str] = None
        self._switching = False

        tab_bar.currentChanged.connect(self._on_tab_changed)
        tab_bar.tabCloseRequested.connect(self._on_tab_close_requested)
        tab_bar.tabMoved.connect(self._on_tab_moved)
        tab_bar.setTabsClosable(True)
        tab_bar.setMovable(True)
        tab_bar.setDrawBase(False)
        tab_bar.setExpanding(False)

    def add_script_tab(self, path: str, title: str) -> int:
        pos = len(self._tab_names)
        idx = self._tab_bar.insertTab(pos, title)
        self._tab_bar.setTabData(idx, self.SCRIPT_TAB_PREFIX + path)
        self._tab_bar.setTabIcon(idx, _icon_for_script(path))
        return idx

    def is_script_tab(self, idx: int) -> bool:
        data = self._tab_bar.tabData(idx)
        return isinstance(data, str) and data.startswith(self.SCRIPT_TAB_PREFIX)

    def script_path_at(self, idx: int) -> str:
        data = self._tab_bar.tabData(idx)
        if isinstance(data, str) and data.startswith(self.SCRIPT_TAB_PREFIX):
            return data[len(self.SCRIPT_TAB_PREFIX):]
        return ""

    def update_script_tab_title(self, path: str, title: str):
        for i in range(self._tab_bar.count()):
            if self.script_path_at(i) == path:
                self._tab_bar.setTabText(i, title)
                break

    def remove_script_tab_by_path(self, path: str):
        for i in range(self._tab_bar.count()):
            if self.script_path_at(i) == path:
                self._tab_bar.removeTab(i)
                break

    def _on_tab_moved(self, from_idx: int, to_idx: int):
        self._tab_names = []
        for i in range(self._tab_bar.count()):
            data = self._tab_bar.tabData(i)
            if isinstance(data, str) and not data.startswith(self.SCRIPT_TAB_PREFIX):
                self._tab_names.append(data)

    _ZARIN_ICON = None

    @classmethod
    def _get_zarin_icon(cls) -> QIcon:
        if cls._ZARIN_ICON is None:
            svg = os.path.join(os.path.dirname(__file__), "..", "..", "zarin_icon.svg")
            if os.path.exists(svg):
                cls._ZARIN_ICON = QIcon(svg)
            else:
                cls._ZARIN_ICON = QIcon()
        return cls._ZARIN_ICON

    def _scene_tab_idx(self, tab_name: str) -> int:
        for i in range(self._tab_bar.count()):
            if self._tab_bar.tabData(i) == tab_name:
                return i
        return -1

    def add_tab(self, name: str, path: Optional[str] = None, scene: Optional[Scene] = None) -> str:
        tab_name = self._unique_name(name)
        if scene is None:
            self._save_current_to_tab()
            self._active_tab = tab_name
            self._engine.new_scene(tab_name)
            if path:
                self._engine.scene.path = path
            scene = self._engine.scene
        info = SceneTabInfo(tab_name, path, scene)
        self._tabs[tab_name] = info
        pos = len(self._tab_names)
        self._tab_names.append(tab_name)
        idx = self._tab_bar.insertTab(pos, tab_name)
        self._tab_bar.setTabData(idx, tab_name)
        self._tab_bar.setTabIcon(idx, self._get_zarin_icon())
        self._tab_bar.setCurrentIndex(idx)
        self.tab_added.emit(tab_name)
        return tab_name

    def remove_tab(self, tab_name: str):
        if tab_name not in self._tabs:
            return
        idx = self._scene_tab_idx(tab_name)
        self._tab_names.remove(tab_name)
        self._tabs.pop(tab_name, None)
        if idx >= 0:
            self._tab_bar.removeTab(idx)

    def switch_to_tab(self, tab_name: str):
        if tab_name not in self._tabs:
            return
        idx = self._scene_tab_idx(tab_name)
        if idx >= 0:
            self._tab_bar.setCurrentIndex(idx)

    def tab_name_at(self, idx: int) -> Optional[str]:
        data = self._tab_bar.tabData(idx)
        if isinstance(data, str) and not data.startswith(self.SCRIPT_TAB_PREFIX):
            return data
        return None

    @property
    def tab_names(self) -> list[str]:
        return list(self._tab_names)

    @property
    def active_tab(self) -> Optional[str]:
        return self._active_tab

    def get_tab_info(self, tab_name: str) -> Optional[SceneTabInfo]:
        return self._tabs.get(tab_name)

    def _unique_name(self, base: str) -> str:
        if base not in self._tabs:
            return base
        i = 2
        while f"{base} {i}" in self._tabs:
            i += 1
        return f"{base} {i}"

    def _save_current_to_tab(self):
        if self._active_tab and self._active_tab in self._tabs and self._engine.scene:
            info = self._tabs[self._active_tab]
            info.scene = self._engine.scene
            info.dirty = self._engine.scene.dirty
            scene_path = getattr(self._engine.scene, 'path', None)
            if scene_path:
                info.path = scene_path
            vp = getattr(self._mw, '_viewport', None)
            if vp and hasattr(vp, '_cam'):
                cam = vp._cam
                info.camera_state = {
                    "position": cam._position.to_list(),
                    "yaw": cam._yaw,
                    "pitch": cam._pitch,
                    "orbit_target": cam._orbit_target.to_list(),
                    "orbit_dist": cam._orbit_dist,
                    "is_2d_mode": cam._is_2d_mode,
                    "ortho_zoom_distance": cam._ortho_zoom_distance,
                    "stored_ortho_size": cam._stored_ortho_size,
                    "fov": cam._fov,
                    "near": cam._near,
                    "far": cam._far,
                }
            if vp:
                info.selected_entity_ids = [e.id for e in vp._selected_entities]
            info.play_mode = self._engine.play_mode

    def _on_tab_changed(self, idx: int):
        if self._switching:
            return
        if self.is_script_tab(idx):
            self._save_current_to_tab()
            self._active_tab = None
            self.script_tab_selected.emit(self.script_path_at(idx))
            return
        self._switching = True
        try:
            target = self.tab_name_at(idx)
            if target is None or target == self._active_tab:
                return

            self._save_current_to_tab()

            if self._engine.play_mode:
                self._engine.stop_play()
                from core.ecs.ecs import Scene as S
                from core.engine.engine import Engine as Eng
                active_info = self._tabs.get(self._active_tab)
                if active_info and active_info.scene_snapshot:
                    restored = S.deserialize(active_info.scene_snapshot, Eng.instance()._component_registry)
                    restored.path = self._engine.scene.path if self._engine.scene else active_info.path
                    self._engine._scene = restored
                    self._engine._plugin_manager.notify_scene_loaded(restored)
                    active_info.scene = restored
                    active_info.scene_snapshot = None
                    active_info.play_mode = False

            info = self._tabs.get(target)
            if info is None:
                return

            self._active_tab = target
            self._load_scene(info)
            self.tab_switched.emit(target)
        finally:
            self._switching = False

    def _on_tab_close_requested(self, idx: int):
        if self.is_script_tab(idx):
            path = self.script_path_at(idx)
            self.script_tab_closed.emit(path)
            self._tab_bar.removeTab(idx)
            return
        tab_name = self.tab_name_at(idx)
        if tab_name is None:
            return
        info = self._tabs.get(tab_name)
        scene = info.scene if info else None

        if info and info.prefab_path:
            from core.ecs.prefab import PrefabLibrary
            pref = PrefabLibrary.load(info.prefab_path)
            prefab_guid = pref.guid if pref else None
            scene = info.scene
            if scene:
                from editor.main_window.handlers import _save_prefab_direct
                _save_prefab_direct(info.prefab_path, scene)
            origin = info.origin_tab
            if origin and origin in self._tabs:
                origin_info = self._tabs.get(origin)
                if origin_info and origin_info.scene and prefab_guid:
                    from editor.main_window.handlers import _refresh_prefab_instances
                    _refresh_prefab_instances(origin_info.scene, prefab_guid, self._mw._engine._component_registry)
            if tab_name == self._active_tab:
                self._active_tab = None
            self.remove_tab(tab_name)
            self._mw._prefab_mode = False
            self._mw._prefab_path = None
            self._mw._viewport._prefab_btns.hide()
            from core.config.editor_scale import scale
            self._mw._viewport._toolbar.setFixedHeight(scale(30))
            return

        is_dirty = scene.dirty if scene else False
        if is_dirty and scene:
            reply = QMessageBox.question(
                self._mw, "Unsaved Changes",
                f"Scene '{info.name}' has unsaved changes. Save before closing?",
                QMessageBox.StandardButton.Yes |
                QMessageBox.StandardButton.No |
                QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
            if reply == QMessageBox.StandardButton.Yes:
                self._save_scene_tab(info)

        if tab_name == self._active_tab and self._engine.play_mode:
            self._engine.stop_play()
            if info and info.scene_snapshot:
                from core.ecs.ecs import Scene as S
                from core.engine.engine import Engine as Eng
                restored = S.deserialize(info.scene_snapshot, Eng.instance()._component_registry)
                restored.path = self._engine.scene.path if self._engine.scene else info.path
                self._engine._scene = restored
                self._engine._plugin_manager.notify_scene_loaded(restored)

        if tab_name == self._active_tab:
            self._active_tab = None

        self.remove_tab(tab_name)

    def _load_scene(self, info: SceneTabInfo):
        if info.scene is None:
            self._engine.new_scene(info.name)
            if info.path:
                self._engine.scene.path = info.path
            info.scene = self._engine.scene
        else:
            if self._engine.scene:
                self._engine._plugin_manager.notify_scene_unloaded(self._engine.scene)
            from core.components.rendering.postfx.graphics_effect import GraphicsEffect
            GraphicsEffect.cleanup_registry()
            self._engine._scene = info.scene
            self._engine._plugin_manager.notify_scene_loaded(info.scene)
        self._engine._emit_event("scene_loaded", info.scene)

        vp = getattr(self._mw, '_viewport', None)
        if vp and hasattr(vp, '_cam') and info.camera_state:
            cs = info.camera_state
            cam = vp._cam
            cam._position = Vec3(*cs["position"])
            cam._yaw = cs["yaw"]
            cam._pitch = cs["pitch"]
            cam._orbit_target = Vec3(*cs["orbit_target"])
            cam._orbit_dist = cs["orbit_dist"]
            cam._is_2d_mode = cs["is_2d_mode"]
            cam._ortho_zoom_distance = cs["ortho_zoom_distance"]
            cam._stored_ortho_size = cs["stored_ortho_size"]
            cam._fov = cs.get("fov", cam.DEFAULT_FOV)
            cam._near = cs.get("near", cam.DEFAULT_NEAR)
            cam._far = cs.get("far", cam.DEFAULT_FAR)

        if vp:
            resolved = [info.scene.get_entity(eid) for eid in info.selected_entity_ids if info.scene.get_entity(eid)]
            vp._selected_entities = resolved
            vp._set_gizmo_entity(resolved[0] if resolved else None)
            vp.entity_selected.emit(resolved[0] if resolved else None)
            vp.entities_selected.emit(resolved)
            from editor.viewport.collaboration import send_collab_selection
            send_collab_selection(vp)

        if hasattr(self._mw, '_hierarchy'):
            self._mw._hierarchy.refresh()
        idx = self._scene_tab_idx(info.name)
        if idx >= 0:
            self._tab_bar.setTabText(idx, info.name)

    def _save_scene_tab(self, info: SceneTabInfo):
        if info.scene is None:
            return
        if info.path:
            from core.engine.engine import Engine
            Engine.instance().save_scene(info.path)
        else:
            from PyQt6.QtWidgets import QFileDialog
            path, _ = QFileDialog.getSaveFileName(self._mw, "Save Scene", "scenes/", "Scenes (*.zpes)")
            if path:
                if not path.endswith(".zpes"):
                    path += ".zpes"
                info.path = path
                Engine.instance().save_scene(path)

    def find_prefab_tab(self) -> Optional[str]:
        for name, info in self._tabs.items():
            if info.prefab_path:
                return name
        return None

    def update_peer_indicators(self, tab_peers: dict[str, list[list[float]]]):
        DOT_SIZE = 8
        PAD = 2
        for tab_name in self._tab_names:
            colors = tab_peers.get(tab_name, [])
            idx = self._scene_tab_idx(tab_name)
            if idx < 0:
                continue
            self._set_tab_icon(idx, colors)

    def update_script_peer_indicators(self, script_peers: dict[str, list[list[float]]]):
        DOT_SIZE = 8
        PAD = 2
        for i in range(self._tab_bar.count()):
            if not self.is_script_tab(i):
                continue
            path = self.script_path_at(i)
            colors = script_peers.get(path, [])
            self._set_tab_icon(i, colors)

    def _set_tab_icon(self, idx: int, colors: list[list[float]]):
        DOT_SIZE = 8
        PAD = 2
        if colors:
            n = len(colors)
            pw = DOT_SIZE * n + PAD * (n - 1) + PAD * 2
            ph = DOT_SIZE + PAD * 2
            pix = QPixmap(pw, ph)
            pix.fill(Qt.GlobalColor.transparent)
            with QPainter(pix) as p:
                p.setRenderHint(QPainter.RenderHint.Antialiasing)
                for i, rgb in enumerate(colors):
                    r, g, b = (int(c * 255) for c in rgb[:3])
                    p.setBrush(QBrush(QColor(r, g, b)))
                    p.setPen(Qt.PenStyle.NoPen)
                    cx = PAD + i * (DOT_SIZE + PAD) + DOT_SIZE // 2
                    cy = PAD + DOT_SIZE // 2
                    p.drawEllipse(cx - DOT_SIZE // 2, cy - DOT_SIZE // 2, DOT_SIZE, DOT_SIZE)
            self._tab_bar.setTabIcon(idx, QIcon(pix))
        else:
            self._tab_bar.setTabIcon(idx, QIcon())

    def update_tab_name(self, old_name: str, new_name: str):
        if old_name not in self._tabs:
            return
        info = self._tabs.pop(old_name)
        info.name = new_name
        self._tabs[new_name] = info
        idx = self._scene_tab_idx(old_name)
        if idx >= 0:
            self._tab_bar.setTabData(idx, new_name)
            self._tab_bar.setTabText(idx, new_name)
        name_idx = self._tab_names.index(old_name)
        self._tab_names[name_idx] = new_name
        if self._active_tab == old_name:
            self._active_tab = new_name
