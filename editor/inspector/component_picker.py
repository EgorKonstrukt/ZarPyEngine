# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from typing import Optional
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, QListWidgetItem, QPushButton, \
    QLabel, QStackedWidget, QSizePolicy, QWidget, QListView, QAbstractItemView
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QIcon, QColor, QFont, QPainter, QBrush, QPen
from core.editor_scale import scale, scale_xy
from editor.inspector.constants import _accent
from editor.inspector.helpers import get_component_icon_pixmap

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

class _CategoryIconWidget(QLabel):
    _cache: dict[str, QPixmap] = {}
    @classmethod
    def _get_pixmap(cls, label: str, size: int) -> QPixmap:
        if label in cls._cache:
            return cls._cache[label]
        from PyQt6.QtGui import QPainter, QBrush, QPen
        from PyQt6.QtCore import QRect
        colors = [
            (80, 160, 220), (140, 200, 80), (220, 160, 70), (180, 100, 200),
            (200, 120, 80), (100, 200, 180), (200, 180, 100), (160, 140, 220),
            (100, 180, 200), (220, 140, 140), (140, 200, 160), (180, 180, 180),
        ]
        idx = hash(label) % len(colors)
        r, g, b = colors[idx]
        pm = QPixmap(size, size)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setBrush(QBrush(QColor(r, g, b)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(0, 0, size, size, 6, 6)
        p.setPen(QColor(255, 255, 255))
        f = QFont("Segoe UI", size // 3, QFont.Weight.Bold)
        p.setFont(f)
        p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, label[0].upper())
        p.end()
        cls._cache[label] = pm
        return pm
    def __init__(self, label: str, size: int = 48):
        super().__init__()
        self._label = label
        pm = self._get_pixmap(label, size)
        self.setPixmap(pm)
        self.setFixedSize(size + 8, size + 8)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setToolTip(label)

class ComponentPickerDialog(QDialog):
    def __init__(self, entity, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Component")
        self.setMinimumSize(420, 520)
        self.resize(460, 560)
        self._entity = entity
        self._selected: Optional[dict] = None
        self._all_components: list[tuple] = []
        self._all_scripts: list[dict] = []
        self._categories: dict[str, list[tuple]] = {}
        self._search_text = ""
        self._setup_ui()
        self._load_data()
        self._show_categories()
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search components and scripts...")
        self._search.textChanged.connect(self._on_search)
        layout.addWidget(self._search)
        self._nav_bar = QWidget()
        self._nav_layout = QHBoxLayout(self._nav_bar)
        self._nav_layout.setContentsMargins(0, 0, 0, 0)
        self._nav_layout.setSpacing(4)
        self._back_btn = QPushButton("\u25C0  Back")
        self._back_btn.setStyleSheet(f"color: {_accent()}; background: transparent; border: none; font-size: 11px; padding: 2px 4px; text-align: left;")
        self._back_btn.clicked.connect(self._show_categories)
        self._back_btn.hide()
        self._nav_title = QLabel()
        self._nav_layout.addWidget(self._back_btn)
        self._nav_layout.addWidget(self._nav_title)
        self._nav_layout.addStretch()
        layout.addWidget(self._nav_bar)
        self._stack = QStackedWidget()
        self._cat_list = QListWidget()
        self._cat_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._cat_list.setIconSize(QSize(48, 48))
        self._cat_list.setGridSize(QSize(120, 100))
        self._cat_list.setWordWrap(True)
        self._cat_list.setSpacing(4)
        self._cat_list.setUniformItemSizes(True)
        self._cat_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._cat_list.setLayoutMode(QListView.LayoutMode.Batched)
        self._cat_list.setBatchSize(20)
        self._cat_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._cat_list.itemClicked.connect(self._on_category_clicked)
        self._stack.addWidget(self._cat_list)
        self._comp_list = QListWidget()
        self._comp_list.setSpacing(1)
        self._comp_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._comp_list.itemClicked.connect(self._on_comp_selected)
        self._comp_list.itemDoubleClicked.connect(self._accept_selection)
        self._stack.addWidget(self._comp_list)
        layout.addWidget(self._stack, 1)
        btn_row = QHBoxLayout()
        self._add_btn = QPushButton("Add")
        self._add_btn.setEnabled(False)
        self._add_btn.clicked.connect(self._accept_selection)
        btn_row.addWidget(self._add_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
    def _load_data(self):
        from core.ecs import ComponentRegistry
        from core.engine import Engine
        all_reg = ComponentRegistry.all()
        self._all_components = []
        self._categories.clear()
        for name, cls in all_reg.items():
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
        for candidate in (
            os.path.join(project_root, "assets", "scripts"),
            os.path.join(project_root, "scripts"),
        ):
            scripts_dir = os.path.normpath(candidate)
            if os.path.isdir(scripts_dir):
                break
        else:
            return
        for root, dirs, files in os.walk(scripts_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
            for f in files:
                if f.endswith(".py") and not f.startswith("__"):
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, scripts_dir)
                    self._all_scripts.append({"name": rel, "path": full, "type": "script"})
    def _show_categories(self):
        self._search.setText("")
        self._back_btn.hide()
        self._nav_title.setText("")
        self._add_btn.setEnabled(False)
        self._cat_list.clear()
        self._stack.setCurrentIndex(0)
        sorted_cats = sorted(self._categories.keys(), key=str.lower)
        for cat in sorted_cats:
            icon_w = _CategoryIconWidget(cat, 48)
            item = QListWidgetItem(QIcon(icon_w.pixmap()), cat)
            item.setData(Qt.ItemDataRole.UserRole, cat)
            item.setToolTip(f"{len(self._categories[cat])} components")
            self._cat_list.addItem(item)
    def _on_category_clicked(self, item):
        cat = item.data(Qt.ItemDataRole.UserRole)
        self._show_components(cat)
    def _show_components(self, category: str):
        self._back_btn.show()
        self._nav_title.setText(category)
        self._add_btn.setEnabled(False)
        self._comp_list.clear()
        for name, cls, already in self._categories.get(category, []):
            pix = get_component_icon_pixmap(cls, 16)
            item = QListWidgetItem(QIcon(pix), name)
            item.setData(Qt.ItemDataRole.UserRole, {"type": "component", "name": name})
            if already:
                item.setToolTip("Already added")
            else:
                item.setToolTip(f"Add {name} component")
            self._comp_list.addItem(item)
        scripts_in_cat = [s for s in self._all_scripts if category.lower() in s["name"].lower()]
        if scripts_in_cat:
            for s in scripts_in_cat:
                item = QListWidgetItem(f"  {s['name']}")
                item.setData(Qt.ItemDataRole.UserRole, s)
                item.setToolTip(f"Add script: {s['name']}")
                self._comp_list.addItem(item)
        self._stack.setCurrentIndex(1)
    def _on_search(self, text: str):
        self._search_text = text
        if not text:
            self._show_categories()
            return
        self._back_btn.hide()
        self._nav_title.setText(f'Search: "{text}"')
        self._comp_list.clear()
        self._add_btn.setEnabled(False)
        lower = text.lower()
        from core.ecs import ComponentRegistry
        for name, cls, already in self._all_components:
            if lower not in name.lower():
                continue
            pix = get_component_icon_pixmap(cls, 16)
            item = QListWidgetItem(QIcon(pix), name)
            item.setData(Qt.ItemDataRole.UserRole, {"type": "component", "name": name})
            if already:
                item.setToolTip("Already added")
            else:
                item.setToolTip(f"Add {name} component")
            self._comp_list.addItem(item)
        for s in self._all_scripts:
            if lower not in s["name"].lower():
                continue
            item = QListWidgetItem(s["name"])
            item.setData(Qt.ItemDataRole.UserRole, s)
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
