# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtWidgets import (QWidget, QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                              QTreeWidget, QTreeWidgetItem, QLineEdit, QPushButton)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QCursor, QIcon

from editor.inspector.helpers import make_gameobject_picker, get_component_icon_pixmap
from core.components.properties import iter_entity_prop_groups, iter_entity_props

_TREE_STYLE = """
QTreeWidget {
    background: palette(base); color: palette(text);
    border: 1px solid palette(mid); border-radius: 4px;
    padding: 2px; outline: none;
}
QTreeWidget::item { padding: 3px 2px; }
QTreeWidget::item:selected {
    background: palette(highlight); color: palette(highlighted-text);
    border-radius: 3px;
}
QTreeWidget::item:hover { background: palette(light); }
QTreeWidget::branch { background: transparent; }
"""

_FIELD_STYLE = """
QLineEdit {
    background: palette(base); color: palette(text);
    border: 1px solid palette(mid); border-radius: 3px;
    padding: 3px 6px; font-size: 11px;
}
"""


class _PropertyLineEdit(QLineEdit):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class EntityPropertyTree(QWidget):
    entity_changed = pyqtSignal(str)
    property_changed = pyqtSignal(str)

    def __init__(self, scene=None, entity_id: str = "", parent=None, show_entity_row: bool = True):
        self._show_entity_row = show_entity_row
        super().__init__(parent)
        self._scene = scene
        self._entity_id = ""
        self._props: list[tuple[str, str]] = []
        self._path_items: dict[str, QTreeWidgetItem] = {}
        self._setup_ui(scene, entity_id)

    def _setup_ui(self, scene, entity_id: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        if self._show_entity_row:
            self._entity_row = QHBoxLayout()
            self._entity_row.setContentsMargins(0, 0, 0, 0)
            self._entity_row.setSpacing(4)
            lbl = QLabel("Entity")
            lbl.setFixedWidth(48)
            self._entity_row.addWidget(lbl)
            self._entity_picker = make_gameobject_picker(entity_id, scene, self._on_entity_picked)
            self._entity_row.addWidget(self._entity_picker, 1)
            layout.addLayout(self._entity_row)
        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setIndentation(14)
        self._tree.setUniformRowHeights(True)
        self._tree.setStyleSheet(_TREE_STYLE)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._tree, 1)
        self._entity_id = entity_id
        self._reload_props()

    def set_scene(self, scene):
        if scene is self._scene:
            return
        self._scene = scene
        self._rebuild_entity_picker()
        self._reload_props()

    def set_entity_id(self, entity_id: str):
        self._entity_id = entity_id
        self._reload_props()

    def scene(self):
        return self._scene

    def _rebuild_entity_picker(self):
        if not self._show_entity_row:
            return
        old = self._entity_picker
        entity_id = self._entity_id
        self._entity_picker = make_gameobject_picker(entity_id, self._scene, self._on_entity_picked)
        self._entity_row.replaceWidget(old, self._entity_picker)
        old.deleteLater()

    def _on_entity_picked(self, eid: str):
        changed = eid != self._entity_id
        self._entity_id = eid
        self._reload_props()
        if changed:
            self.entity_changed.emit(self._entity_id)

    def _reload_props(self):
        self._tree.clear()
        self._props.clear()
        self._path_items.clear()
        entity = None
        if self._scene and self._entity_id:
            entity = self._scene.get_entity(self._entity_id)
        if entity is not None:
            for cname, comp, entries in iter_entity_prop_groups(entity):
                comp_item = QTreeWidgetItem([cname])
                comp_item.setFlags(comp_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                pix = get_component_icon_pixmap(type(comp), 16)
                if pix is not None and not pix.isNull():
                    comp_item.setIcon(0, QIcon(pix))
                for label, base, leaf in entries:
                    if leaf:
                        parent = QTreeWidgetItem([label])
                        parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                        comp_item.addChild(parent)
                        for item in leaf:
                            cell = QTreeWidgetItem([item])
                            cell.setData(0, Qt.ItemDataRole.UserRole, f"{base}.{item}")
                            parent.addChild(cell)
                            self._path_items[f"{base}.{item}"] = cell
                            self._props.append((f"{label}.{item}", f"{base}.{item}"))
                    else:
                        cell = QTreeWidgetItem([label])
                        cell.setData(0, Qt.ItemDataRole.UserRole, base)
                        comp_item.addChild(cell)
                        self._path_items[base] = cell
                        self._props.append((label, base))
                self._tree.addTopLevelItem(comp_item)
                comp_item.setExpanded(True)

    def _on_selection_changed(self):
        items = self._tree.selectedItems()
        if not items:
            return
        path = items[0].data(0, Qt.ItemDataRole.UserRole) or ""
        self.property_changed.emit(path)

    def set_property(self, path: str):
        item = self._path_items.get(path)
        if item is not None:
            self._tree.setCurrentItem(item)
            self.property_changed.emit(path)

    def current_entity_id(self) -> str:
        return self._entity_id

    def current_property(self) -> str:
        if not self._props:
            return ""
        items = self._tree.selectedItems()
        if not items:
            return ""
        return items[0].data(0, Qt.ItemDataRole.UserRole) or ""

    def property_label(self) -> str:
        items = self._tree.selectedItems()
        if not items:
            return ""
        path = items[0].data(0, Qt.ItemDataRole.UserRole) or ""
        return self._label_for(path) if path else ""

    def _label_for(self, path: str) -> str:
        for label, prop_path in self._props:
            if prop_path == path:
                return label
        return path


class EntityPropertyDialog(QDialog):
    def __init__(self, scene=None, entity_id: str = "", path: str = "", parent=None, locked: bool = False):
        super().__init__(parent)
        self.setWindowTitle("Select Property")
        self.setMinimumSize(360, 460)
        self.setModal(True)
        self._locked = bool(locked)
        self._locked_id = entity_id or ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._picker = EntityPropertyTree(scene, entity_id, self, show_entity_row=not locked)
        layout.addWidget(self._picker, 1)
        btns = QHBoxLayout()
        btns.addStretch(1)
        self._ok = QPushButton("OK")
        self._ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(self._ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)
        if path:
            self._picker.set_property(path)

    def _position_near_cursor(self):
        self.adjustSize()
        screen = self.screen().availableGeometry() if self.screen() else None
        cursor = QCursor.pos()
        x = cursor.x() - 40
        y = cursor.y() + 4
        if screen is not None:
            if y + self.height() > screen.bottom():
                y = cursor.y() - self.height() - 4
            if x + self.width() > screen.right():
                x = screen.right() - self.width()
            x = max(screen.left(), x)
            y = max(screen.top(), y)
        self.move(x, y)

    def exec_near_cursor(self) -> int:
        self._position_near_cursor()
        return self.exec()

    def result_entity_id(self) -> str:
        return self._locked_id if self._locked else self._picker.current_entity_id()

    def result_property(self) -> str:
        return self._picker.current_property()

    def result_label(self) -> str:
        return self._picker.property_label()


class EntityPropertyPicker(QWidget):
    entity_changed = pyqtSignal(str)
    property_changed = pyqtSignal(str)

    def __init__(self, scene=None, entity_id: str = "", path: str = "", parent=None):
        super().__init__(parent)
        self._scene = scene
        self._entity_id = entity_id
        self._current_path = path
        self._labels: dict[str, str] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self._entity_row = QHBoxLayout()
        self._entity_row.setContentsMargins(0, 0, 0, 0)
        self._entity_row.setSpacing(4)
        lbl = QLabel("Entity")
        lbl.setFixedWidth(48)
        self._entity_row.addWidget(lbl)
        self._entity_picker = make_gameobject_picker(self._entity_id, self._scene, self._on_entity_picked)
        self._entity_row.addWidget(self._entity_picker, 1)
        layout.addLayout(self._entity_row)
        prop_row = QHBoxLayout()
        prop_row.setContentsMargins(0, 0, 0, 0)
        prop_row.setSpacing(4)
        lbl2 = QLabel("Property")
        lbl2.setFixedWidth(48)
        prop_row.addWidget(lbl2)
        self._line = _PropertyLineEdit()
        self._line.setReadOnly(True)
        self._line.setPlaceholderText("(none)")
        self._line.setStyleSheet(_FIELD_STYLE)
        self._line.setToolTip("")
        self._line.clicked.connect(self._open_dialog)
        prop_row.addWidget(self._line, 1)
        btn = QPushButton("...")
        btn.setFixedWidth(28)
        btn.setToolTip("Pick property")
        btn.clicked.connect(self._open_dialog)
        prop_row.addWidget(btn)
        layout.addLayout(prop_row)
        self._refresh_labels()
        self._update_line()

    def _refresh_labels(self):
        self._labels.clear()
        if self._scene and self._entity_id:
            entity = self._scene.get_entity(self._entity_id)
            if entity is not None:
                for label, path in iter_entity_props(entity):
                    self._labels[path] = label

    def _update_line(self):
        label = self._labels.get(self._current_path, "")
        self._line.setText(label)
        self._line.setToolTip(self._current_path if self._current_path else "")

    def set_scene(self, scene):
        if scene is self._scene:
            return
        self._scene = scene
        self._rebuild_entity_picker()
        self._refresh_labels()
        self._update_line()

    def scene(self):
        return self._scene

    def _rebuild_entity_picker(self):
        old = self._entity_picker
        entity_id = self._entity_id
        self._entity_picker = make_gameobject_picker(entity_id, self._scene, self._on_entity_picked)
        self._entity_row.replaceWidget(old, self._entity_picker)
        old.deleteLater()

    def _on_entity_picked(self, eid: str):
        changed = eid != self._entity_id
        self._entity_id = eid
        if not self._labels or changed:
            self._refresh_labels()
        if changed:
            self.entity_changed.emit(self._entity_id)
        self._update_line()

    def _open_dialog(self):
        dlg = EntityPropertyDialog(self._scene, self._entity_id, self._current_path, self)
        if dlg.exec_near_cursor() == QDialog.DialogCode.Accepted:
            eid = dlg.result_entity_id()
            path = dlg.result_property()
            label = dlg.result_label()
            if eid and eid != self._entity_id:
                self._entity_id = eid
                self._rebuild_entity_picker()
                self._refresh_labels()
                self.entity_changed.emit(eid)
            if path:
                self._current_path = path
                self._labels[path] = label
                self._update_line()
                self.property_changed.emit(path)

    def set_entity_id(self, entity_id: str):
        self._entity_id = entity_id
        self._refresh_labels()
        self._update_line()

    def set_property(self, path: str):
        self._current_path = path
        self._refresh_labels()
        self._update_line()
        if path:
            self.property_changed.emit(path)

    def current_entity_id(self) -> str:
        return self._entity_id

    def current_property(self) -> str:
        return self._current_path

    def property_label(self) -> str:
        return self._labels.get(self._current_path, "") or self._current_path