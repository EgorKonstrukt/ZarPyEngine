# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""
Build Settings dialog вЂ” Unity-style scene list and plugin management.
Allows configuring which scenes and plugins are included in the build.
"""
from __future__ import annotations
import os
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QFileDialog, QMessageBox, QAbstractItemView,
    QGroupBox, QCheckBox, QScrollArea, QWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from core.config.build_settings import BuildSettings
from core.foundation.logger import Logger


class BuildSettingsDialog(QDialog):
    build_settings_changed = pyqtSignal()

    def __init__(self, project_root: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Build Settings")
        self.setMinimumSize(540, 600)
        self._project_root = project_root
        self._settings = BuildSettings.instance() or BuildSettings()
        self._settings.load(os.path.join(project_root, "BuildSettings.json"))
        self._setup_ui()
        self._refresh_scene_list()
        self._refresh_plugin_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # в”Ђв”Ђ Scenes section в”Ђв”Ђ
        header = QLabel("Scenes In Build")
        header.setFont(QFont("", 12, QFont.Weight.Bold))
        layout.addWidget(header)

        info = QLabel("Drag to reorder. First scene is loaded on startup.")
        info.setStyleSheet("color: #888;")
        layout.addWidget(info)

        self._scene_list = QListWidget()
        self._scene_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self._scene_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._scene_list.model().rowsMoved.connect(self._on_reorder)
        layout.addWidget(self._scene_list)

        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton("+ Add Scene")
        self._add_btn.clicked.connect(self._add_scene)
        btn_layout.addWidget(self._add_btn)
        self._remove_btn = QPushButton("- Remove")
        self._remove_btn.clicked.connect(self._remove_scene)
        btn_layout.addWidget(self._remove_btn)
        btn_layout.addStretch()
        self._up_btn = QPushButton("в–І")
        self._up_btn.setFixedWidth(30)
        self._up_btn.clicked.connect(self._move_up)
        btn_layout.addWidget(self._up_btn)
        self._down_btn = QPushButton("в–ј")
        self._down_btn.setFixedWidth(30)
        self._down_btn.clicked.connect(self._move_down)
        btn_layout.addWidget(self._down_btn)
        layout.addLayout(btn_layout)

        # в”Ђв”Ђ Plugins section в”Ђв”Ђ
        plugin_header = QLabel("Plugins In Build")
        plugin_header.setFont(QFont("", 12, QFont.Weight.Bold))
        layout.addWidget(plugin_header)

        plugin_info = QLabel("Check plugins to include. Unchecked plugins are excluded from the build.")
        plugin_info.setStyleSheet("color: #888;")
        layout.addWidget(plugin_info)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self._plugin_container = QWidget()
        self._plugin_layout = QVBoxLayout(self._plugin_container)
        self._plugin_layout.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(self._plugin_container)
        layout.addWidget(scroll)

        # в”Ђв”Ђ Build Options в”Ђв”Ђ
        options_group = QGroupBox("Build Options")
        options_layout = QVBoxLayout()
        self._strip_check = QCheckBox("Strip unused assets (scan scenes for references)")
        self._strip_check.setChecked(self._settings._build_options.get("strip_unused_assets", True))
        options_layout.addWidget(self._strip_check)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # в”Ђв”Ђ Bottom buttons в”Ђв”Ђ
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_and_close)
        bottom_layout.addWidget(save_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        bottom_layout.addWidget(cancel_btn)
        layout.addLayout(bottom_layout)

    def _refresh_scene_list(self):
        self._scene_list.clear()
        scenes = self._settings.scenes
        for i, scene in enumerate(scenes):
            item = QListWidgetItem()
            name = Path(scene).stem
            text = f"  {scene}"
            if i == 0:
                text += "  (Startup)"
            item.setText(text)
            item.setData(Qt.ItemDataRole.UserRole, scene)
            self._scene_list.addItem(item)

    def _add_scene(self):
        scenes_dir = os.path.join(self._project_root, "scenes")
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add Scenes", scenes_dir,
            "Zarin Scenes (*.zpes);;All Files (*)"
        )
        if files:
            for f in files:
                rel = os.path.relpath(f, self._project_root)
                self._settings.add_scene(rel)
            self._refresh_scene_list()

    def _remove_scene(self):
        item = self._scene_list.currentItem()
        if item:
            scene = item.data(Qt.ItemDataRole.UserRole)
            self._settings.remove_scene(scene)
            self._refresh_scene_list()

    def _move_up(self):
        row = self._scene_list.currentRow()
        if row > 0:
            self._settings.move_scene(row, row - 1)
            self._refresh_scene_list()
            self._scene_list.setCurrentRow(row - 1)

    def _move_down(self):
        row = self._scene_list.currentRow()
        if row < self._scene_list.count() - 1:
            self._settings.move_scene(row, row + 1)
            self._refresh_scene_list()
            self._scene_list.setCurrentRow(row + 1)

    def _refresh_plugin_list(self):
        # Clear existing checkboxes
        while self._plugin_layout.count():
            w = self._plugin_layout.takeAt(0).widget()
            if w:
                w.deleteLater()

        selected = set(self._settings.build_plugins)
        self._plugin_checkboxes = {}
        plugins_dir = os.path.join(self._project_root, "plugins")

        # Scan plugins/ root
        if os.path.isdir(plugins_dir):
            for fname in sorted(os.listdir(plugins_dir)):
                fpath = os.path.join(plugins_dir, fname)
                if fname.endswith(".py") and not fname.startswith("_"):
                    name = fname[:-3]
                    self._add_plugin_checkbox(name, f"plugins/{fname}", selected)
                elif os.path.isdir(fpath) and not fname.startswith("_"):
                    init_path = os.path.join(fpath, "__init__.py")
                    if os.path.isfile(init_path):
                        # Package-style plugin (e.g. plugins/user/bsp_import_plugin)
                        self._add_package_plugins(fpath, fname, selected)

        self._plugin_layout.addStretch()

    def _add_plugin_checkbox(self, name: str, label: str, selected: set):
        cb = QCheckBox(name)
        if name in selected:
            cb.setChecked(True)
        cb.setToolTip(label)
        self._plugin_checkboxes[name] = cb
        self._plugin_layout.addWidget(cb)

    def _add_package_plugins(self, pkg_dir: str, prefix: str, selected: set):
        for fname in sorted(os.listdir(pkg_dir)):
            if fname.endswith(".py") and not fname.startswith("_"):
                name = prefix + "." + fname[:-3]
                self._add_plugin_checkbox(name, f"plugins/{prefix}/{fname}", selected)

    def _on_reorder(self):
        """Handle drag-drop reorder."""
        scenes = []
        for i in range(self._scene_list.count()):
            item = self._scene_list.item(i)
            scenes.append(item.data(Qt.ItemDataRole.UserRole))
        self._settings.set_scenes(scenes)

    def _save_and_close(self):
        self._settings._build_options["strip_unused_assets"] = self._strip_check.isChecked()
        selected = [name for name, cb in self._plugin_checkboxes.items() if cb.isChecked()]
        self._settings.set_plugins(selected)
        save_path = os.path.join(self._project_root, "BuildSettings.json")
        self._settings.save(save_path)
        self.build_settings_changed.emit()
        self.accept()
