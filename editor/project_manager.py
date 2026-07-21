# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
import qtawesome as qta
from pathlib import Path
from datetime import datetime
from typing import Optional
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget,
                             QListWidgetItem, QPushButton, QLabel, QLineEdit,
                             QFileDialog, QMessageBox, QWidget, QMenu)
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QIcon, QAction

from editor.constants import PROJECTS_DB_PATH
from core.config.editor_scale import scale, scale_xy


def _load_projects_db() -> list[dict]:
    try:
        if os.path.exists(PROJECTS_DB_PATH):
            with open(PROJECTS_DB_PATH, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return []


def _save_projects_db(projects: list[dict]):
    try:
        Path(PROJECTS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(PROJECTS_DB_PATH, "w") as f:
            json.dump(projects, f, indent=2)
    except Exception as e:
        from core.foundation.logger import Logger
        Logger.error(f"Failed to save projects db: {e}")


def _get_recent_projects(max_count: int = 10) -> list[dict]:
    projects = _load_projects_db()
    projects.sort(key=lambda p: p.get("last_opened", ""), reverse=True)
    return projects[:max_count]


def _add_recent_project(name: str, path: str):
    projects = _load_projects_db()
    entry = {"name": name, "path": os.path.abspath(path), "last_opened": datetime.now().isoformat()}
    projects = [p for p in projects if p.get("path") != os.path.abspath(path)]
    projects.append(entry)
    _save_projects_db(projects)


def _create_project_directory(path: str, name: str) -> bool:
    try:
        os.makedirs(os.path.join(path, "assets"), exist_ok=True)
        os.makedirs(os.path.join(path, "scenes"), exist_ok=True)
        settings = {
            "project": {"name": name, "version": "1.0.0", "default_scene": ""},
            "input": {"horizontal": "a,d", "vertical": "w,s", "mouse_sensitivity": 1.0},
            "rendering": {"render_pipeline": "forward", "anti_aliasing": "none", "shadow_distance": 50.0},
        }
        with open(os.path.join(path, "ProjectSettings.json"), "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        from core.foundation.logger import Logger
        Logger.error(f"Failed to create project: {e}")
        return False


class ProjectManagerDialog(QDialog):
    project_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Project Manager")
        self.setMinimumSize(600, 400)
        self.resize(700, 500)
        self._setup_ui()
        self._refresh_list()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("Project Manager")
        header_font = QFont("Segoe UI", 16, QFont.Weight.Bold)
        header.setFont(header_font)
        header.setStyleSheet("color: #ddd; padding: 8px 0;")
        layout.addWidget(header)

        subtitle = QLabel("Select a project to open, or create a new one.")
        subtitle.setStyleSheet("color: #888; padding-bottom: 8px;")
        layout.addWidget(subtitle)

        self._project_list = QListWidget()
        self._project_list.setStyleSheet("""
            QListWidget {
                background: #1e1e1e;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 12px 8px;
                border-bottom: 1px solid #2a2a2a;
            }
            QListWidget::item:selected {
                background: #264f78;
            }
            QListWidget::item:hover {
                background: #2a2d2e;
            }
        """)
        self._project_list.itemDoubleClicked.connect(self._on_open_project)
        layout.addWidget(self._project_list, 1)

        btn_layout = QHBoxLayout()

        self._new_btn = QPushButton("New Project")
        self._new_btn.setMinimumHeight(32)
        self._new_btn.clicked.connect(self._on_new_project)
        btn_layout.addWidget(self._new_btn)

        self._open_btn = QPushButton("Open Project")
        self._open_btn.setMinimumHeight(32)
        self._open_btn.clicked.connect(self._on_open_project)
        btn_layout.addWidget(self._open_btn)

        self._browse_btn = QPushButton("Browse...")
        self._browse_btn.setMinimumHeight(32)
        self._browse_btn.clicked.connect(self._on_browse)
        btn_layout.addWidget(self._browse_btn)

        btn_layout.addStretch()

        self._remove_btn = QPushButton("Remove")
        self._remove_btn.setMinimumHeight(32)
        self._remove_btn.clicked.connect(self._on_remove_project)
        btn_layout.addWidget(self._remove_btn)

        layout.addLayout(btn_layout)

    def _refresh_list(self):
        self._project_list.clear()
        projects = _load_projects_db()
        projects.sort(key=lambda p: p.get("last_opened", ""), reverse=True)

        if not projects:
            item = QListWidgetItem("  No projects yet. Create a new project to get started.")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            item.setForeground(QColor("#666"))
            self._project_list.addItem(item)
            return

        for p in projects:
            name = p.get("name", "Untitled")
            path = p.get("path", "")
            last = p.get("last_opened", "")
            try:
                dt = datetime.fromisoformat(last)
                last_str = dt.strftime("%Y-%m-%d %H:%M")
            except Exception:
                last_str = ""
            display_name = f"  {name}"
            if os.path.isdir(path):
                display_path = f"    {path}"
            else:
                display_path = f"    {path}  (missing)"
            text = f"{display_name}\n{display_path}"
            if last_str:
                text += f"\n    Last opened: {last_str}"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setData(Qt.ItemDataRole.UserRole + 1, name)
            if not os.path.isdir(path):
                item.setForeground(QColor("#666"))
            self._project_list.addItem(item)

    def _on_new_project(self):
        dlg = _NewProjectDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            name = dlg._name_edit.text().strip()
            path = dlg._path_edit.text().strip()
            if not name or not path:
                return
            if _create_project_directory(path, name):
                _add_recent_project(name, path)
                self._refresh_list()
                self.project_selected.emit(path)
                self.accept()
            else:
                QMessageBox.warning(self, "Error", f"Failed to create project at:\n{path}")

    def _on_open_project(self):
        items = self._project_list.selectedItems()
        if not items:
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        if not path or not os.path.isdir(path):
            QMessageBox.warning(self, "Error", "Project directory not found.")
            return
        _add_recent_project(items[0].data(Qt.ItemDataRole.UserRole + 1), path)
        self.project_selected.emit(path)
        self.accept()

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if path:
            settings_path = os.path.join(path, "ProjectSettings.json")
            name = os.path.basename(path)
            if os.path.exists(settings_path):
                try:
                    with open(settings_path) as f:
                        data = json.load(f)
                    name = data.get("project", {}).get("name", name)
                except Exception:
                    pass
            _add_recent_project(name, path)
            self._refresh_list()
            self.project_selected.emit(path)
            self.accept()

    def _on_remove_project(self):
        items = self._project_list.selectedItems()
        if not items:
            return
        path = items[0].data(Qt.ItemDataRole.UserRole)
        projects = _load_projects_db()
        projects = [p for p in projects if p.get("path") != path]
        _save_projects_db(projects)
        self._refresh_list()


class _NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(450)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Project Name:"))
        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("My Project")
        self._name_edit.textChanged.connect(self._on_name_changed)
        layout.addWidget(self._name_edit)

        layout.addWidget(QLabel("Project Location:"))
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText(os.path.expanduser("~"))
        path_layout.addWidget(self._path_edit, 1)
        browse_btn = QPushButton(qta.icon("fa5s.folder-open", color="#d4d4d4"), "")
        browse_btn.setFixedWidth(scale(32))
        browse_btn.clicked.connect(self._on_browse)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self._create_btn = QPushButton("Create")
        self._create_btn.clicked.connect(self.accept)
        self._create_btn.setEnabled(False)
        btn_layout.addWidget(self._create_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _on_name_changed(self, text):
        self._create_btn.setEnabled(bool(text.strip()))
        if text.strip() and not self._path_edit.text().strip():
            default = os.path.join(str(Path.home()), text.strip())
            self._path_edit.setText(default)

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Location")
        if path:
            name = self._name_edit.text().strip()
            if name:
                path = os.path.join(path, name)
            self._path_edit.setText(path)
