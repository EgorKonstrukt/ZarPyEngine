# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from typing import Optional
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QStackedWidget,
    QWidget, QAbstractItemView)
from PyQt6.QtCore import Qt, QSize, QEvent
from PyQt6.QtGui import QPixmap, QIcon
import qtawesome as qta
from core.config.editor_scale import scale
from editor.inspector.constants import _accent
from editor.inspector.helpers import get_component_icon_pixmap


_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ICONS_DIR = os.path.join(os.path.dirname(__file__), "icons")

_CATEGORY_ICON_MAP = {
    "Transform": "transform.png",
    "Rendering": "rendering.png",
    "Physics": "physics.png",
    "Physics 2D": "physics2d.png",
    "Lighting": "lighting.png",
    "Audio": "audio.png",
    "Constraints": "constraints.png",
    "Network": "network.png",
    "Scripting": "scripting.png",
    "Animation": "animation.png",
    "Environment": "environment.png",
    "GUI": "gui.png",
    "Navigation": "navigation.png",
    "Scripts": "scripts.png",
}


def _category_icon(name: str) -> QIcon:
    filename = _CATEGORY_ICON_MAP.get(name, "other.png")
    path = os.path.join(_ICONS_DIR, filename)
    if os.path.exists(path):
        return QIcon(QPixmap(path))
    return QIcon()


class ComponentPickerDialog(QDialog):
    def __init__(self, entity, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Component")
        self.setMinimumSize(380, 480)
        self.resize(420, 520)
        self._entity = entity
        self._selected: Optional[dict] = None
        self._all_components: list[tuple] = []
        self._all_scripts: list[dict] = []
        self._categories: dict[str, list[tuple]] = {}
        self._search_text = ""
        self._setup_ui()
        self._load_data()
        self._show_categories()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            target = self._comp_list if self._stack.currentIndex() == 1 else self._cat_list
            delta = event.angleDelta().y()
            row = target.currentRow()
            new_row = row - 1 if delta > 0 else row + 1
            new_row = max(0, min(new_row, target.count() - 1))
            target.setCurrentRow(new_row)
            return True
        if event.type() == QEvent.Type.KeyPress and event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._stack.currentIndex() == 0:
                items = self._cat_list.selectedItems()
                if items:
                    self._on_category_clicked(items[0])
            else:
                items = self._comp_list.selectedItems()
                if items:
                    self._on_comp_selected()
                    self._accept_selection()
            return True
        return super().eventFilter(obj, event)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search components...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_search)
        layout.addWidget(self._search)

        self._stack = QStackedWidget()
        self.installEventFilter(self)
        self._stack.installEventFilter(self)
        self._search.installEventFilter(self)

        self._cat_list = QListWidget()
        self._cat_list.setSpacing(0)
        self._cat_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._cat_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._cat_list.itemClicked.connect(self._on_category_clicked)
        self._cat_list.installEventFilter(self)
        self._stack.addWidget(self._cat_list)

        comp_page = QWidget()
        comp_layout = QVBoxLayout(comp_page)
        comp_layout.setContentsMargins(0, 0, 0, 0)
        comp_layout.setSpacing(0)

        self._back_btn = QPushButton("  Back")
        self._back_btn.setIcon(qta.icon("fa5s.arrow-left", color="#ccc"))
        self._back_btn.setCheckable(False)
        self._back_btn.clicked.connect(self._show_categories)
        self._back_btn.setFixedHeight(scale(32))
        comp_layout.addWidget(self._back_btn)

        self._comp_list = QListWidget()
        self._comp_list.setSpacing(0)
        self._comp_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._comp_list.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self._comp_list.itemClicked.connect(self._on_comp_selected)
        self._comp_list.itemDoubleClicked.connect(self._accept_selection)
        self._comp_list.installEventFilter(self)
        comp_layout.addWidget(self._comp_list, 1)

        self._stack.addWidget(comp_page)
        layout.addWidget(self._stack, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)
        self._add_btn = QPushButton("Add")
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._accept_selection)
        btn_row.addWidget(self._add_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

    def _load_data(self):
        from core.ecs.ecs import ComponentRegistry
        from core.engine.engine import Engine
        all_reg = ComponentRegistry.all()
        self._all_components = []
        self._categories.clear()
        for name, cls in all_reg.items():
            if getattr(cls, "_editor_hidden", False):
                continue
            cats = ComponentRegistry.get_categories(name)
            display_cat = cats[0] if cats else "Other"
            can_multiple = getattr(cls, '_allow_multiple', False)
            already = not can_multiple and self._entity.has_component(cls)
            entry = (name, cls, already)
            self._all_components.append(entry)
            if display_cat not in self._categories:
                self._categories[display_cat] = []
            self._categories[display_cat].append(entry)
        self._all_scripts = []
        eng = Engine.instance()
        project_root = eng.project_root if eng is not None else _PROJECT_ROOT
        bases = [
            os.path.join(project_root, "assets"),
            os.path.join(project_root, "scripts"),
        ]
        seen_paths: set[str] = set()
        for base in bases:
            base = os.path.normpath(base)
            if not os.path.isdir(base):
                continue
            for root, dirs, files in os.walk(base):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
                for f in files:
                    if not f.endswith(".py") or f.startswith("__"):
                        continue
                    full = os.path.normpath(os.path.join(root, f))
                    if full in seen_paths:
                        continue
                    seen_paths.add(full)
                    rel = os.path.relpath(full, base)
                    self._all_scripts.append({"name": rel, "path": full, "type": "script"})
        if self._all_scripts:
            self._categories.setdefault("Scripts", [])

    def _show_categories(self):
        self._search.setText("")
        self._add_btn.setEnabled(False)
        self._cat_list.clear()
        self._stack.setCurrentIndex(0)
        sorted_cats = sorted(self._categories.keys(), key=str.lower)
        for cat in sorted_cats:
            count = len(self._all_scripts) if cat == "Scripts" else len(self._categories[cat])
            item = QListWidgetItem(_category_icon(cat), f"  {cat}  ({count})")
            item.setData(Qt.ItemDataRole.UserRole, cat)
            item.setSizeHint(QSize(0, scale(32)))
            self._cat_list.addItem(item)

    def _on_category_clicked(self, item):
        cat = item.data(Qt.ItemDataRole.UserRole)
        self._show_components(cat)

    def _show_components(self, category: str):
        self._add_btn.setEnabled(False)
        self._comp_list.clear()
        for name, cls, already in self._categories.get(category, []):
            pix = get_component_icon_pixmap(cls, 16)
            label = f"  {name}"
            if already:
                label += "  (*)"
            item = QListWidgetItem(QIcon(pix), label)
            item.setData(Qt.ItemDataRole.UserRole, {"type": "component", "name": name})
            item.setSizeHint(QSize(0, scale(28)))
            if already:
                item.setToolTip("Already added")
            else:
                item.setToolTip(f"Add {name}")
            self._comp_list.addItem(item)
        if category == "Scripts":
            scripts_in_cat = self._all_scripts
        else:
            scripts_in_cat = [s for s in self._all_scripts if category.lower() in s["name"].lower()]
        if scripts_in_cat:
            for s in scripts_in_cat:
                label_name = os.path.splitext(s["name"])[0]
                item = QListWidgetItem(f"  {label_name}")
                item.setData(Qt.ItemDataRole.UserRole, s)
                item.setSizeHint(QSize(0, scale(28)))
                item.setToolTip(f"Add script: {s['name']}")
                self._comp_list.addItem(item)
        self._stack.setCurrentIndex(1)

    def _on_search(self, text: str):
        self._search_text = text
        if not text:
            self._show_categories()
            return
        self._add_btn.setEnabled(False)
        self._comp_list.clear()
        lower = text.lower()
        for name, cls, already in self._all_components:
            if getattr(cls, "_editor_hidden", False):
                continue
            if lower not in name.lower():
                continue
            pix = get_component_icon_pixmap(cls, 16)
            label = f"  {name}"
            if already:
                label += "  (*)"
            item = QListWidgetItem(QIcon(pix), label)
            item.setData(Qt.ItemDataRole.UserRole, {"type": "component", "name": name})
            item.setSizeHint(QSize(0, scale(28)))
            if already:
                item.setToolTip("Already added")
            else:
                item.setToolTip(f"Add {name}")
            self._comp_list.addItem(item)
        for s in self._all_scripts:
            if lower not in s["name"].lower():
                continue
            item = QListWidgetItem(f"  {s['name']}")
            item.setData(Qt.ItemDataRole.UserRole, s)
            item.setSizeHint(QSize(0, scale(28)))
            item.setToolTip(f"Add script: {s['name']}")
            self._comp_list.addItem(item)
        self._stack.setCurrentIndex(1)

    def _on_comp_selected(self):
        items = self._comp_list.selectedItems()
        if items:
            data = items[0].data(Qt.ItemDataRole.UserRole)
            self._add_btn.setEnabled(data is not None)
        else:
            self._add_btn.setEnabled(False)

    def _accept_selection(self):
        items = self._comp_list.selectedItems()
        if items:
            data = items[0].data(Qt.ItemDataRole.UserRole)
            if data:
                self._selected = data
                self.accept()

    def selected_result(self) -> Optional[dict]:
        return self._selected
