# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QTabBar, QMessageBox

from core.ecs.ecs import Scene, ComponentRegistry


class SceneTabInfo:
    __slots__ = ("name", "path", "data", "dirty")
    def __init__(self, name: str = "Scene", path: Optional[str] = None, data: Optional[dict] = None):
        self.name = name
        self.path = path
        self.data = data
        self.dirty = False


class SceneTabManager(QObject):
    tab_switched = pyqtSignal(str)
    tab_added = pyqtSignal(str)

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
        tab_bar.setTabsClosable(True)
        tab_bar.setMovable(True)
        tab_bar.setDrawBase(False)
        tab_bar.setExpanding(False)

    def add_tab(self, name: str, path: Optional[str] = None, data: Optional[dict] = None) -> str:
        tab_name = self._unique_name(name)
        if data is None:
            self._save_current_to_tab()
            self._engine.new_scene(tab_name)
            if path:
                self._engine.scene.path = path
            data = self._engine.scene.serialize()
            if hasattr(self._mw, '_hierarchy'):
                self._mw._hierarchy.refresh()
            self._active_tab = tab_name
        info = SceneTabInfo(tab_name, path, data)
        self._tabs[tab_name] = info
        self._tab_names.append(tab_name)
        self._tab_bar.addTab(tab_name)
        self._tab_bar.setCurrentIndex(self._tab_bar.count() - 1)
        self.tab_added.emit(tab_name)
        return tab_name

    def remove_tab(self, tab_name: str):
        if tab_name not in self._tabs:
            return
        idx = self._tab_names.index(tab_name)
        self._tab_names.remove(tab_name)
        self._tabs.pop(tab_name, None)
        self._tab_bar.removeTab(idx)

    def switch_to_tab(self, tab_name: str):
        if tab_name not in self._tabs:
            return
        idx = self._tab_names.index(tab_name)
        self._tab_bar.setCurrentIndex(idx)

    def tab_name_at(self, idx: int) -> Optional[str]:
        if 0 <= idx < len(self._tab_names):
            return self._tab_names[idx]
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
            info.data = self._engine.scene.serialize()
            info.dirty = self._engine.scene.dirty
            scene_path = getattr(self._engine.scene, 'path', None)
            if scene_path:
                info.path = scene_path

    def _on_tab_changed(self, idx: int):
        if self._switching:
            return
        self._switching = True
        try:
            target = self.tab_name_at(idx)
            if target is None or target == self._active_tab:
                return

            self._save_current_to_tab()

            info = self._tabs.get(target)
            if info is None:
                return

            self._active_tab = target
            self._load_scene(info)
            self.tab_switched.emit(target)
        finally:
            self._switching = False

    def _on_tab_close_requested(self, idx: int):
        tab_name = self.tab_name_at(idx)
        if tab_name is None:
            return
        info = self._tabs.get(tab_name)
        is_dirty = info.dirty if info else False
        if tab_name == self._active_tab and self._engine.scene and self._engine.scene.dirty:
            is_dirty = True
        if is_dirty and info and info.data:
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

        if tab_name == self._active_tab:
            self._save_current_to_tab()
            self._active_tab = None

        self.remove_tab(tab_name)

    def _load_scene(self, info: SceneTabInfo):
        if info.data is None:
            self._engine.new_scene(info.name)
            if info.path:
                self._engine.scene.path = info.path
        else:
            from core.components.rendering.postfx.graphics_effect import GraphicsEffect
            GraphicsEffect.cleanup_registry()
            scene = Scene.deserialize(info.data, ComponentRegistry)
            scene.path = info.path or ""
            self._engine._scene = scene
            self._engine._plugin_manager.notify_scene_loaded(scene)
            self._engine._emit_event("scene_loaded", scene)

        if hasattr(self._mw, '_viewport') and self._mw._viewport and hasattr(self._mw._viewport, 'renderer') and self._mw._viewport.renderer:
            self._mw._viewport.renderer.release_all_caches()
        if hasattr(self._mw, '_hierarchy'):
            self._mw._hierarchy.refresh()
        self._tab_bar.setTabText(self._tab_names.index(info.name) if info.name in self._tab_names else 0, info.name)

    def _save_scene_tab(self, info: SceneTabInfo):
        if info.data is None:
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

    def update_tab_name(self, old_name: str, new_name: str):
        if old_name not in self._tabs:
            return
        info = self._tabs.pop(old_name)
        info.name = new_name
        self._tabs[new_name] = info
        idx = self._tab_names.index(old_name)
        self._tab_names[idx] = new_name
        self._tab_bar.setTabText(idx, new_name)
        if self._active_tab == old_name:
            self._active_tab = new_name
