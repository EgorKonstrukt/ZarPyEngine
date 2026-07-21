# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import time
from typing import Optional, TYPE_CHECKING
import qtawesome as qta
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QTreeWidget, QTreeWidgetItem, QPushButton,
                              QMenu, QLineEdit, QLabel, QInputDialog, QAbstractItemView,
                              QStyledItemDelegate, QApplication, QHeaderView)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QMimeData
from PyQt6.QtGui import QKeySequence, QAction, QDrag, QColor, QKeyEvent, QBrush, QPixmap, QIcon
from editor.inspector.helpers import _flash_overlay
if TYPE_CHECKING:
    from core.ecs.ecs import Entity, Scene
    from core.engine.engine import Engine
_ENTITY_MIME = "application/x-zpe-entity"
_COMPONENT_MIME = "application/x-zpe-component"

import os
from core.config.editor_scale import scale, scale_xy

def _get_component_icon_pixmap(cls, size: int = 16) -> QPixmap:
    icon_name = getattr(cls, '_icon', None)
    if icon_name:
        icons_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'core', 'components', 'icons')
        icon_path = os.path.join(icons_dir, icon_name)
        if os.path.exists(icon_path):
            return QPixmap(icon_path).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    gizmo_dir = os.path.join(os.path.dirname(__file__), '..', 'gizmo_icons')
    icon_path = os.path.join(gizmo_dir, f'{cls.__name__}.png')
    if os.path.exists(icon_path):
        return QPixmap(icon_path).scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
    r, g, b = getattr(cls, '_gizmo_icon_color', (140, 60, 200))
    label = getattr(cls, '_gizmo_icon_label', '?')
    pix = QPixmap(size, size)
    pix.fill(Qt.GlobalColor.transparent)
    from PyQt6.QtGui import QPainter, QColor as QC, QFont as QF, QBrush as QB
    from PyQt6.QtCore import QRect
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QB(QC(r, g, b)))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawRoundedRect(0, 0, size, size, 3, 3)
    if label:
        p.setPen(QC(255, 255, 255))
        f = QF("Segoe UI", size // 2, QF.Weight.Bold)
        p.setFont(f)
        p.drawText(QRect(0, 0, size, size), Qt.AlignmentFlag.AlignCenter, label[0].upper())
    p.end()
    return pix
class HierarchyTree(QTreeWidget):
    entity_reparented = pyqtSignal(object, object)
    delete_requested = pyqtSignal(list)
    copy_requested = pyqtSignal()
    paste_requested = pyqtSignal()
    component_drop_requested = pyqtSignal(object, object, object)
    def __init__(self, panel: HierarchyPanel, parent=None):
        super().__init__(parent)
        self._panel = panel
        self.setDragDropMode(QTreeWidget.DragDropMode.DragDrop)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._press_item = None
        self._press_pos = None
        self._drag_started = False
    def supportedDropActions(self):
        return Qt.DropAction.MoveAction | Qt.DropAction.CopyAction
    def mimeTypes(self):
        return [_ENTITY_MIME]
    def mimeData(self, items):
        mime = QMimeData()
        eids = []
        for item in items:
            eid = item.data(0, Qt.ItemDataRole.UserRole)
            if eid:
                eids.append(eid)
        if eids:
            mime.setData(_ENTITY_MIME, ",".join(eids).encode("utf-8"))
        return mime
    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._press_item = self.itemAt(self._press_pos)
            if self._press_item:
                self._drag_started = False
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self._press_item and self._press_pos:
            delta = event.position().toPoint() - self._press_pos
            if delta.manhattanLength() >= QApplication.startDragDistance():
                self._drag_started = True
                items = self.selectedItems()
                if not items:
                    items = [self._press_item]
                elif self._press_item not in items:
                    items.append(self._press_item)
                mime = self.mimeData(items)
                drag = QDrag(self)
                drag.setMimeData(mime)
                drag.exec(Qt.DropAction.MoveAction)
                self._press_item = None
                self._press_pos = None
                return
        super().mouseMoveEvent(event)
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._drag_started:
            self._drag_started = False
            self._press_item = None
            self._press_pos = None
            return
        super().mouseReleaseEvent(event)
    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat(_COMPONENT_MIME):
            event.acceptProposedAction()
            return
        if not event.mimeData().hasFormat(_ENTITY_MIME):
            event.ignore()
            return
        target_item = self.itemAt(event.position().toPoint())
        dragged_ids = bytes(event.mimeData().data(_ENTITY_MIME)).decode("utf-8").split(",")
        if target_item is not None:
            target_eid = target_item.data(0, Qt.ItemDataRole.UserRole)
            if not target_eid or not dragged_ids:
                event.ignore()
                return
            if target_eid in dragged_ids:
                event.ignore()
                return
            for did in dragged_ids:
                dragged_item = self._find_item_by_id(did)
                if dragged_item and self._is_descendant(target_item, dragged_item):
                    event.ignore()
                    return
        event.acceptProposedAction()
    def _find_item_by_id(self, eid: str):
        return self._find_recursive(eid, self.invisibleRootItem())
    def _find_recursive(self, eid: str, parent_item):
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if child.data(0, Qt.ItemDataRole.UserRole) == eid:
                return child
            found = self._find_recursive(eid, child)
            if found:
                return found
        return None
    def _is_descendant(self, ancestor_item, check_item):
        if not check_item or not ancestor_item:
            return False
        parent = check_item.parent()
        while parent:
            if parent == ancestor_item:
                return True
            parent = parent.parent()
        return False
    def dropEvent(self, event):
        if event.mimeData().hasFormat(_COMPONENT_MIME):
            raw = bytes(event.mimeData().data(_COMPONENT_MIME)).decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                event.ignore()
                return
            target_item = self.itemAt(event.position().toPoint())
            if target_item is not None:
                target_eid = target_item.data(0, Qt.ItemDataRole.UserRole)
                if target_eid:
                    is_copy = event.proposedAction() == Qt.DropAction.CopyAction
                    self.component_drop_requested.emit(target_eid, data, is_copy)
                    event.acceptProposedAction()
                    return
            event.ignore()
            return
        if not event.mimeData().hasFormat(_ENTITY_MIME):
            event.ignore()
            return
        target_item = self.itemAt(event.position().toPoint())
        dragged_ids = bytes(event.mimeData().data(_ENTITY_MIME)).decode("utf-8").split(",")
        if not dragged_ids:
            event.ignore()
            return
        if target_item is None:
            for did in dragged_ids:
                self.entity_reparented.emit(did, None)
            event.acceptProposedAction()
            return
        target_eid = target_item.data(0, Qt.ItemDataRole.UserRole)
        if not target_eid:
            event.ignore()
            return
        for did in dragged_ids:
            if did == target_eid:
                event.ignore()
                return
        pos = self.dropIndicatorPosition()
        if pos in (QAbstractItemView.DropIndicatorPosition.AboveItem, QAbstractItemView.DropIndicatorPosition.BelowItem):
            target_parent = target_item.parent()
            if target_parent:
                parent_eid = target_parent.data(0, Qt.ItemDataRole.UserRole)
            else:
                parent_eid = None
            for did in dragged_ids:
                self.entity_reparented.emit(did, parent_eid)
        else:
            for did in dragged_ids:
                self.entity_reparented.emit(did, target_eid)
        event.acceptProposedAction()
    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()
        nvk = event.nativeVirtualKey()
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_requested.emit()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_requested.emit()
            event.accept()
            return
        if mods & Qt.KeyboardModifier.ControlModifier and mods & Qt.KeyboardModifier.ShiftModifier and key == Qt.Key.Key_A:
            items = self.selectedItems()
            if items:
                eids = []
                for item in items:
                    eid = item.data(0, Qt.ItemDataRole.UserRole)
                    if eid:
                        eids.append(eid)
                if eids:
                    self._panel._toggle_active_by_ids(eids)
            event.accept()
            return
        if key == Qt.Key.Key_F2:
            items = self.selectedItems()
            if items:
                self.editItem(items[0], 0)
        elif key == Qt.Key.Key_Delete:
            items = self.selectedItems()
            eids = []
            for item in items:
                eid = item.data(0, Qt.ItemDataRole.UserRole)
                if eid:
                    eids.append(eid)
            if eids:
                self.delete_requested.emit(eids)
        else:
            super().keyPressEvent(event)
class HierarchyPanel(QDockWidget):
    entity_selected = pyqtSignal(object)
    entities_selected = pyqtSignal(list)
    entity_double_clicked = pyqtSignal(str)
    select_prefab_asset = pyqtSignal(str)
    open_prefab_editor = pyqtSignal(str)
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Hierarchy", parent)
        self._engine = engine
        self._scene: Optional[Scene] = None
        self._selected_entity: Optional[Entity] = None
        self._editing_item: Optional[QTreeWidgetItem] = None
        self._last_render_version: int = -1
        self._setup_ui()
        engine.on("scene_loaded", self._on_scene_loaded)
        engine.on("scene_saved", lambda _: self._refresh())
        engine.on("history_changed", lambda _: self._on_history_changed())
    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 4, 4, 2)
        toolbar.setSpacing(2)
        add_btn = QPushButton(qta.icon("fa5s.plus", color="#fff"), "")
        add_btn.setFixedSize(*scale_xy(24, 24))
        add_btn.setToolTip("Create Entity")
        add_btn.setStyleSheet("QPushButton { background: #2e7d32; }")
        add_btn.clicked.connect(self._show_create_menu)
        toolbar.addWidget(add_btn)
        self._search = QLineEdit()
        self._search.setPlaceholderText("  All")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._on_search)
        self._search.addAction(qta.icon("fa5s.search", color="#d4d4d4"), QLineEdit.ActionPosition.LeadingPosition)
        toolbar.addWidget(self._search, 1)
        collapse_btn = QPushButton(qta.icon("fa5s.chevron-up", color="#d4d4d4"), "")
        collapse_btn.setFixedSize(*scale_xy(24, 24))
        collapse_btn.setToolTip("Collapse All")
        collapse_btn.clicked.connect(self._collapse_all)
        toolbar.addWidget(collapse_btn)
        layout.addLayout(toolbar)

        self._scene_header = QWidget()
        header_layout = QHBoxLayout(self._scene_header)
        header_layout.setContentsMargins(8, 0, 8, 2)
        header_layout.setSpacing(4)
        self._scene_icon = QLabel()
        icon_path = os.path.join(os.path.dirname(__file__), '..', '..', 'zarin_icon.svg')
        icon_pix = QPixmap(icon_path).scaled(16, 16, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._scene_icon.setPixmap(icon_pix)
        header_layout.addWidget(self._scene_icon)
        self._scene_label = QLabel()
        header_layout.addWidget(self._scene_label)
        header_layout.addStretch()
        self._update_scene_header()
        layout.addWidget(self._scene_header)

        self._tree = HierarchyTree(self)
        self._tree.setHeaderHidden(True)
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        self._tree.itemSelectionChanged.connect(self._on_selection_changed)
        self._tree.itemDoubleClicked.connect(self._on_item_double_click)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.entity_reparented.connect(self._on_reparent)
        self._tree.component_drop_requested.connect(self._on_component_drop)
        self._tree.delete_requested.connect(self._delete_entities_by_ids)
        self._tree.copy_requested.connect(self._on_copy)
        self._tree.paste_requested.connect(self._on_paste)
        layout.addWidget(self._tree)
        self.setWidget(w)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._periodic_refresh)
        self._refresh_timer.start(2000)
    def load_config(self, config) -> None:
        refresh_interval = config.get("hierarchy.refresh_interval", 2000)
        self._refresh_timer.setInterval(refresh_interval)
    def _update_scene_header(self):
        if self._scene:
            self._scene_label.setText(getattr(self._scene, 'name', 'Scene'))
        else:
            self._scene_label.setText("No Scene")
    def _on_scene_loaded(self, scene: Scene):
        self._scene = scene
        self._selected_entity = None
        self._last_render_version = -1
        self._update_scene_header()
        self._refresh()
    def _on_history_changed(self, cmd=None):
        QTimer.singleShot(0, self._refresh)

    def _refresh(self):
        if not self._scene:
            self._tree.clear()
            self._selected_entity = None
            return
        self._last_render_version = getattr(self._scene, '_render_version', -1)
        old_expanded = self._get_expanded_ids()
        old_selection = self._selected_entity.id if self._selected_entity else None
        self._tree.blockSignals(True)
        self._tree.clear()
        filter_text = self._search.text().strip().lower()
        root_entities = self._scene.get_root_entities()
        live_ids = set(self._scene._entities.keys())
        for entity in root_entities:
            if entity.id not in live_ids:
                continue
            self._add_entity_item(entity, self._tree.invisibleRootItem(), filter_text, live_ids)
        self._restore_expanded(old_expanded)
        if old_selection and old_selection in live_ids:
            self._restore_selection(old_selection)
        else:
            self._selected_entity = None
        self._tree.blockSignals(False)
    def _add_entity_item(self, entity: Entity, parent_item, filter_text: str, live_ids: set) -> bool:
        name = entity.name
        if entity.is_prefab_instance:
            from core.ecs.prefab import Prefab
            overrides = Prefab.compute_overrides(entity)
            override_mark = " *" if overrides else ""
            name = f"{name}{override_mark}"
        children = [c for c in entity.children if c.id in live_ids]
        has_visible_child = any(self._entity_matches_filter(c, filter_text, live_ids) for c in children)
        matches_filter = (not filter_text) or (filter_text in name.lower()) or has_visible_child
        if not matches_filter:
            return False
        item = QTreeWidgetItem(parent_item)
        item.setText(0, name)
        item.setData(0, Qt.ItemDataRole.UserRole, entity.id)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        icon_cls = None
        if len(entity._components) > 1:
            for c in entity.get_all_components():
                if getattr(type(c), '_show_gizmo_icon', True) and type(c).__name__ != "Transform":
                    icon_cls = type(c)
                    break
        if icon_cls:
            item.setIcon(0, QIcon(_get_component_icon_pixmap(icon_cls, 16)))
        if entity.is_prefab_instance:
            item.setForeground(0, QBrush(QColor("#88ccff")))
        if not entity.active:
            gray = self._tree.palette().color(self._tree.palette().ColorRole.PlaceholderText)
            item.setForeground(0, QBrush(gray))
        child_filter = "" if (has_visible_child and filter_text) else filter_text
        for child in children:
            self._add_entity_item(child, item, child_filter, live_ids)
        item.setExpanded(True)
        return True
    def _entity_matches_filter(self, entity: Entity, filter_text: str, live_ids: set = None) -> bool:
        if live_ids is not None and entity.id not in live_ids:
            return False
        if not filter_text:
            return True
        if filter_text in entity.name.lower():
            return True
        children = entity.children if live_ids is None else [c for c in entity.children if c.id in live_ids]
        return any(self._entity_matches_filter(c, filter_text, live_ids) for c in children)
    def _get_expanded_ids(self) -> set:
        expanded = set()
        self._collect_expanded(self._tree.invisibleRootItem(), expanded)
        return expanded
    def _collect_expanded(self, parent_item, expanded: set):
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if child.isExpanded():
                eid = child.data(0, Qt.ItemDataRole.UserRole)
                if eid:
                    expanded.add(eid)
            self._collect_expanded(child, expanded)
    def _restore_expanded(self, expanded: set):
        self._restore_expanded_items(self._tree.invisibleRootItem(), expanded)
    def _restore_expanded_items(self, parent_item, expanded: set):
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            eid = child.data(0, Qt.ItemDataRole.UserRole)
            if eid and eid in expanded:
                child.setExpanded(True)
            self._restore_expanded_items(child, expanded)
    def _restore_selection(self, eid: str):
        item = self._find_item(eid, self._tree.invisibleRootItem())
        if item:
            self._tree.setCurrentItem(item)
    def _find_item(self, eid: str, parent_item) -> Optional[QTreeWidgetItem]:
        for i in range(parent_item.childCount()):
            child = parent_item.child(i)
            if child.data(0, Qt.ItemDataRole.UserRole) == eid:
                return child
            found = self._find_item(eid, child)
            if found:
                return found
        return None
    def _on_selection_changed(self):
        items = self._tree.selectedItems()
        if not items or not self._scene:
            if self._selected_entity:
                self._selected_entity = None
                self.entity_selected.emit(None)
            return
        if len(items) == 1:
            eid = items[0].data(0, Qt.ItemDataRole.UserRole)
            if not eid:
                return
            entity = self._scene.get_entity(eid)
            if entity and entity != self._selected_entity:
                self._selected_entity = entity
                self.entity_selected.emit(entity)
        else:
            entities = []
            for item in items:
                eid = item.data(0, Qt.ItemDataRole.UserRole)
                if eid:
                    entity = self._scene.get_entity(eid)
                    if entity:
                        entities.append(entity)
            if entities:
                self._selected_entity = entities[0]
                self.entities_selected.emit(entities)
    def _on_item_double_click(self, item: QTreeWidgetItem, col: int):
        eid = item.data(0, Qt.ItemDataRole.UserRole)
        if self._scene and eid:
            self.entity_double_clicked.emit(eid)
    def _on_item_changed(self, item: QTreeWidgetItem, col: int):
        eid = item.data(0, Qt.ItemDataRole.UserRole)
        new_name = item.text(0)
        if not self._scene or not eid:
            return
        entity = self._scene.get_entity(eid)
        if entity and entity.name != new_name:
            entity.name = new_name
            self._scene.mark_dirty()
    def _on_search(self, text: str):
        self._refresh()

    def _collapse_all(self):
        def collapse_items(parent_item):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                child.setExpanded(False)
                collapse_items(child)
        collapse_items(self._tree.invisibleRootItem())
    def _on_reparent(self, dragged_eid: str, target_eid: Optional[str]):
        if not self._scene:
            return
        dragged = self._scene.get_entity(dragged_eid)
        if not dragged:
            return
        if target_eid is not None:
            target = self._scene.get_entity(target_eid)
            if not target:
                return
            if dragged == target:
                return
            if self._is_ancestor(dragged, target):
                return
            dragged.set_parent(target)
        else:
            dragged.set_parent(None)
        self._scene.mark_dirty()
        self._refresh()
        if self._selected_entity:
            self._restore_selection(self._selected_entity.id)
    def _on_component_drop(self, target_eid: str, data: dict, is_copy: bool):
        if not self._scene:
            return
        target_entity = self._scene.get_entity(target_eid)
        if not target_entity:
            return
        source_entity = self._scene.get_entity(data.get("entity_id", ""))
        if not source_entity:
            return
        if not is_copy and source_entity == target_entity:
            return
        comp_type_name = data.get("component_type", "")
        component_key = data.get("component_key", "")
        comp_data = data.get("component_data", {})
        if not comp_type_name or not comp_data:
            return
        from core.ecs.ecs import ComponentRegistry
        cls = ComponentRegistry.get(comp_type_name)
        if not cls:
            return
        can_multiple = getattr(cls, '_allow_multiple', False)
        from core.foundation.commands import MoveComponentCommand, CopyComponentCommand, CompoundCommand, get_history
        if is_copy:
            eid_set = {target_eid}
            for item in self._tree.selectedItems():
                eid = item.data(0, Qt.ItemDataRole.UserRole)
                if eid:
                    eid_set.add(eid)
            targets = []
            for eid in eid_set:
                e = self._scene.get_entity(eid)
                if e and e != source_entity:
                    if not can_multiple and e.has_component(cls):
                        continue
                    targets.append(e)
            if not targets:
                return
            cmds = [CopyComponentCommand(e, cls, comp_data, source_key=component_key) for e in targets]
            get_history().execute(CompoundCommand(cmds, f"Copy {comp_type_name} to {len(targets)} entities"))
        else:
            if not can_multiple and target_entity.has_component(cls):
                return
            get_history().execute(MoveComponentCommand(source_entity, target_entity, component_key, cls, comp_data))
        self._scene.mark_dirty()
        self.refresh()
        mw = self.parent()
        inspector = getattr(mw, '_inspector', None)
        if inspector:
            inspector._rebuild()

    def _is_ancestor(self, potential_ancestor: Entity, entity: Entity) -> bool:
        current = entity.parent
        while current:
            if current == potential_ancestor:
                return True
            current = current.parent
        return False
    def _periodic_refresh(self):
        if not self._scene:
            return
        rv = getattr(self._scene, '_render_version', -1)
        if rv != self._last_render_version:
            self._last_render_version = rv
            if self._tree.state() != QTreeWidget.State.EditingState and not self._tree._drag_started and self._tree._press_item is None:
                now = time.perf_counter()
                if now - getattr(self, '_last_refresh_time', 0) < 0.5:
                    return
                self._last_refresh_time = now
                self._refresh()
        if self._tree.currentItem() is None and self._selected_entity:
            self._restore_selection(self._selected_entity.id)
    def _show_context_menu(self, pos):
        item = self._tree.itemAt(pos)
        menu = QMenu(self)
        if item:
            eid = item.data(0, Qt.ItemDataRole.UserRole)
            entity = self._scene.get_entity(eid) if self._scene and eid else None
            if entity:
                rename_act = QAction("Rename\tF2", self)
                rename_act.triggered.connect(lambda: self._tree.editItem(item, 0))
                menu.addAction(rename_act)
                dup_act = QAction("Duplicate\tCtrl+D", self)
                dup_act.triggered.connect(lambda: self._duplicate_entity(entity))
                menu.addAction(dup_act)
                create_child_act = QAction("Create Child", self)
                create_child_act.triggered.connect(lambda: self._create_child(entity))
                menu.addAction(create_child_act)
                menu.addSeparator()
                if entity.is_prefab_instance:
                    from core.ecs.prefab import Prefab, PrefabLibrary
                    prefab_path = PrefabLibrary.path_for_guid(entity._prefab_guid)
                    apply_act = QAction("Apply", self)
                    apply_act.triggered.connect(lambda: self._apply_prefab(entity))
                    menu.addAction(apply_act)
                    revert_act = QAction("Revert", self)
                    revert_act.triggered.connect(lambda: self._revert_prefab(entity))
                    menu.addAction(revert_act)
                    if prefab_path:
                        select_act = QAction("Select Prefab Asset", self)
                        select_act.triggered.connect(
                            lambda: self._select_prefab_asset.emit(prefab_path))
                        menu.addAction(select_act)
                        open_edit_act = QAction("Open in Prefab Editor", self)
                        open_edit_act.triggered.connect(
                            lambda: self.open_prefab_editor.emit(prefab_path))
                        menu.addAction(open_edit_act)
                    unpack_act = QAction("Unpack", self)
                    unpack_act.triggered.connect(lambda: self._unpack_prefab(entity))
                    menu.addAction(unpack_act)
                    menu.addSeparator()
                set_active = QAction("Toggle Active", self)
                set_active.triggered.connect(lambda: self._toggle_active(entity))
                menu.addAction(set_active)
                menu.addSeparator()
                save_pref_act = QAction("Save as Prefab...", self)
                save_pref_act.triggered.connect(lambda: self._save_prefab(entity))
                menu.addAction(save_pref_act)
                menu.addSeparator()
                del_act = QAction("Delete", self)
                del_act.triggered.connect(lambda: self._delete_entity(entity))
                menu.addAction(del_act)
                menu.addSeparator()
        self._add_create_menu_entries(menu)
        menu.exec(self._tree.mapToGlobal(pos))
    def _add_create_menu_entries(self, menu: QMenu):
        create_empty = QAction("Create Empty", self)
        create_empty.triggered.connect(self._create_entity)
        menu.addAction(create_empty)

        obj_3d = menu.addMenu("3D Object")
        for name in ["Cube", "Sphere", "Plane"]:
            act = QAction(name, self)
            act.triggered.connect(lambda checked=False, n=name.lower(): self._create_primitive(n))
            obj_3d.addAction(act)
        probuilder = obj_3d.addMenu("ProBuilder Shape")
        from core.components.mesh_editor.primitives import get_primitive_names
        for name in get_primitive_names():
            act = QAction(name, self)
            act.triggered.connect(lambda checked=False, n=name: self._create_probuilder_primitive(n))
            probuilder.addAction(act)

        lights_menu = menu.addMenu("Light")
        for ltype, label in [("sun", "Sun"), ("directional", "Directional Light"), ("point", "Point Light"), ("spot", "Spot Light")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, lt=ltype: self._create_light(lt))
            lights_menu.addAction(act)

        effects = menu.addMenu("Effects")
        for label, comp_cls, setup_fn in [
            ("Sky", "Sky", None),
            ("Clouds", "Cloud", None),
            ("Particle System", "ParticleSystem", None),
        ]:
            act = QAction(label, self)
            act.triggered.connect(
                lambda checked=False, n=label, cc=comp_cls, sf=setup_fn:
                self._create_from_component(n, cc, sf)
            )
            effects.addAction(act)

        audio_menu = menu.addMenu("Audio")
        for label, comp_cls in [("Audio Source", "AudioSource"), ("Audio Listener", "AudioListener"), ("Reverb Zone", "ReverbZone")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: self._create_from_component(n, cc, None))
            audio_menu.addAction(act)

        physics_menu = menu.addMenu("Physics")
        for label, comp_cls in [("Rigidbody", "Rigidbody"), ("Box Collider", "BoxCollider"), ("Sphere Collider", "SphereCollider"), ("Capsule Collider", "CapsuleCollider"), ("Mesh Collider", "MeshCollider"), ("Character Controller", "CharacterController"), ("Joint", "Joint")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: self._create_from_component(n, cc, None))
            physics_menu.addAction(act)

        physics2d_menu = menu.addMenu("Physics 2D")
        for label, comp_cls in [("Rigidbody 2D", "Rigidbody2D"), ("Box Collider 2D", "BoxCollider2D"), ("Circle Collider 2D", "CircleCollider2D")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: self._create_from_component(n, cc, None))
            physics2d_menu.addAction(act)

        rendering_menu = menu.addMenu("Rendering")
        for label, comp_cls in [("Sprite Renderer", "SpriteRenderer"), ("SVG Renderer", "SvgRenderer")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: self._create_from_component(n, cc, None))
            rendering_menu.addAction(act)

        anim_menu = menu.addMenu("Animation")
        for label, comp_cls in [("Animation", "Animation"), ("Animator", "Animator")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: self._create_from_component(n, cc, None))
            anim_menu.addAction(act)

        constraints_menu = menu.addMenu("Constraints")
        for label, comp_cls in [
            ("Aim Constraint", "AimConstraint"),
            ("Follow Transform", "FollowTransformConstraint"),
            ("Look At", "LookAtConstraint"),
            ("Move Towards", "MoveTowardsConstraint"),
            ("Parent", "ParentConstraint"),
            ("Position", "PositionConstraint"),
            ("Rotate Towards", "RotateTowardsConstraint"),
            ("Rotation", "RotationConstraint"),
            ("Scale", "ScaleConstraint"),
            ("Scale To", "ScaleToConstraint"),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: self._create_from_component(n, cc, None))
            constraints_menu.addAction(act)

        net_menu = menu.addMenu("Networking")
        for label, comp_cls in [("Network Identity", "NetworkIdentity"), ("Remote Collaborator", "RemoteCollaborator")]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, n=label, cc=comp_cls: self._create_from_component(n, cc, None))
            net_menu.addAction(act)

        menu.addSeparator()
        cam_act = QAction("Camera", self)
        cam_act.triggered.connect(self._create_camera)
        menu.addAction(cam_act)
    def _show_create_menu(self):
        btn = self.sender()
        menu = QMenu(self)
        self._add_create_menu_entries(menu)
        menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
    def _collab_sync_create(self, entity):
        mgr = getattr(self._engine, "collab_manager", None)
        if mgr and mgr.connected:
            mgr.send_entity_create(entity.serialize())

    def _collab_sync_delete(self, entity_id: str):
        mgr = getattr(self._engine, "collab_manager", None)
        if mgr and mgr.connected:
            mgr.send_entity_delete(entity_id)

    def _create_entity(self):
        if not self._scene:
            return
        from core.foundation.commands import CreateEntityCommand, get_history
        cmd = CreateEntityCommand(self._scene, "GameObject")
        get_history().execute(cmd)
        e = self._scene.get_entity(cmd._entity_id)
        if e:
            from core.components import Transform
            e.add_component(Transform())
        self._refresh()
        if e:
            self._collab_sync_create(e)
            self._selected_entity = e
            self.entity_selected.emit(e)
            item = self._find_item(e.id, self._tree.invisibleRootItem())
            if item:
                self._tree.setCurrentItem(item)
                self._tree.editItem(item, 0)
    def _create_child(self, parent: Entity):
        if not self._scene:
            return
        from core.foundation.commands import CreateEntityCommand, get_history
        cmd = CreateEntityCommand(self._scene, "GameObject")
        get_history().execute(cmd)
        e = self._scene.get_entity(cmd._entity_id)
        if e:
            from core.components import Transform
            e.add_component(Transform())
            e.set_parent(parent)
        self._refresh()
        if e:
            self._collab_sync_create(e)
            self._selected_entity = e
            self.entity_selected.emit(e)
            item = self._find_item(e.id, self._tree.invisibleRootItem())
            if item:
                self._tree.setCurrentItem(item)
                self._tree.editItem(item, 0)
    def _create_primitive(self, mesh_name: str):
        if not self._scene:
            return
        from core.foundation.commands import CreateEntityCommand, get_history
        from core.components import Transform, MeshFilter, MeshRenderer
        cmd = CreateEntityCommand(self._scene, mesh_name.capitalize())
        get_history().execute(cmd)
        e = self._scene.get_entity(cmd._entity_id)
        if e:
            mf = MeshFilter()
            mf.mesh_name = mesh_name
            e.add_component(Transform())
            e.add_component(mf)
            e.add_component(MeshRenderer())
        self._refresh()
        if e:
            self._collab_sync_create(e)
            self._selected_entity = e
            self.entity_selected.emit(e)
            item = self._find_item(e.id, self._tree.invisibleRootItem())
            if item:
                self._tree.setCurrentItem(item)
    def _create_probuilder_primitive(self, name: str):
        if not self._scene:
            return
        from core.foundation.commands import CreateEntityCommand, get_history
        from core.components import Transform, MeshFilter, MeshRenderer
        from core.components.mesh_editor import ProBuilderMesh, create_primitive
        cmd = CreateEntityCommand(self._scene, name)
        get_history().execute(cmd)
        e = self._scene.get_entity(cmd._entity_id)
        if e:
            e.add_component(Transform())
            mf = MeshFilter()
            e.add_component(mf)
            e.add_component(MeshRenderer())
            pb = ProBuilderMesh()
            e.add_component(pb)
            positions, indices = create_primitive(name)
            pb.set_mesh_data(positions, indices)
            mf.mesh_name = f"ProBuilder_{e.id[:6]}"
        self._refresh()
        if e:
            self._collab_sync_create(e)
            self._selected_entity = e
            self.entity_selected.emit(e)
            item = self._find_item(e.id, self._tree.invisibleRootItem())
            if item:
                self._tree.setCurrentItem(item)

    def _create_light(self, ltype: str):
        if not self._scene:
            return
        from core.foundation.commands import CreateEntityCommand, get_history
        from core.components import Transform, Light, LightType
        from core.math.math3d import Vec3
        name_map = {"sun": "Sun", "directional": "Directional Light", "point": "Point Light", "spot": "Spot Light"}
        cmd = CreateEntityCommand(self._scene, name_map.get(ltype, "Light"))
        get_history().execute(cmd)
        e = self._scene.get_entity(cmd._entity_id)
        if e:
            t = Transform()
            if ltype == "sun":
                t.local_euler_angles = Vec3(-45, 45, 0)
            e.add_component(t)
            l = Light()
            type_map = {"sun": LightType.DIRECTIONAL, "directional": LightType.DIRECTIONAL, "point": LightType.POINT, "spot": LightType.SPOT}
            l.light_type = type_map[ltype]
            if ltype == "sun":
                l.procedural_sky_lighting = True
                l.cast_shadows = True
            e.add_component(l)
        self._refresh()
        if e:
            self._collab_sync_create(e)
            self._selected_entity = e
            self.entity_selected.emit(e)
            item = self._find_item(e.id, self._tree.invisibleRootItem())
            if item:
                self._tree.setCurrentItem(item)
    def _create_camera(self):
        if not self._scene:
            return
        from core.foundation.commands import CreateEntityCommand, get_history
        from core.components import Transform, Camera
        cmd = CreateEntityCommand(self._scene, "Camera")
        get_history().execute(cmd)
        e = self._scene.get_entity(cmd._entity_id)
        if e:
            e.add_component(Transform())
            e.add_component(Camera())
        self._refresh()
        if e:
            self._collab_sync_create(e)
            self._selected_entity = e
            self.entity_selected.emit(e)
            item = self._find_item(e.id, self._tree.invisibleRootItem())
            if item:
                self._tree.setCurrentItem(item)
    def _duplicate_entity(self, entity: Entity):
        if not self._scene:
            return
        import copy, uuid
        data = copy.deepcopy(entity.serialize())
        data["id"] = str(uuid.uuid4())
        data["name"] = entity.name
        data["parent"] = None
        for comp_data in data.get("components", []):
            pass
        from core.ecs.ecs import Entity as Ent
        from core.engine.engine import Engine
        new_e = Ent.deserialize(data, Engine.instance()._component_registry)
        self._scene.add_entity(new_e)
        if entity.parent:
            new_e.set_parent(entity.parent)
        self._collab_sync_create(new_e)
        self._refresh()
        self._selected_entity = new_e
        self.entity_selected.emit(new_e)
        item = self._find_item(new_e.id, self._tree.invisibleRootItem())
        if item:
            self._tree.setCurrentItem(item)
    def _create_from_component(self, name: str, comp_cls_name: str, extra_setup=None):
        if not self._scene:
            return
        from core.foundation.commands import CreateEntityCommand, get_history
        from core.ecs.ecs import ComponentRegistry
        from core.components import Transform
        cls = ComponentRegistry.get(comp_cls_name)
        if not cls:
            return
        cmd = CreateEntityCommand(self._scene, name)
        get_history().execute(cmd)
        e = self._scene.get_entity(cmd._entity_id)
        if e:
            e.add_component(Transform())
            e.add_component(cls())
            if extra_setup:
                extra_setup(e)
        self._refresh()
        if e:
            self._collab_sync_create(e)
            self._selected_entity = e
            self.entity_selected.emit(e)
            item = self._find_item(e.id, self._tree.invisibleRootItem())
            if item:
                self._tree.setCurrentItem(item)
    def _delete_entity(self, entity: Entity):
        if not self._scene:
            return
        from core.foundation.commands import DeleteEntityCommand, get_history
        was_selected = (self._selected_entity == entity)
        self._collab_sync_delete(entity.id)
        cmd = DeleteEntityCommand(self._scene, entity.id)
        get_history().execute(cmd)
        if was_selected:
            self._selected_entity = None
            self.entity_selected.emit(None)
        self._refresh()
    def _delete_entities_by_ids(self, eids: list):
        if not self._scene:
            return
        for eid in eids:
            entity = self._scene.get_entity(eid)
            if entity:
                self._delete_entity(entity)
    def _on_copy(self):
        if not self._scene:
            return
        items = self._tree.selectedItems()
        if not items:
            return
        import copy
        selected = []
        for item in items:
            eid = item.data(0, Qt.ItemDataRole.UserRole)
            if eid:
                e = self._scene.get_entity(eid)
                if e:
                    selected.append(e)
        if not selected:
            return
        seen = set()
        to_serialize = []
        for e in selected:
            if e.id in seen:
                continue
            stack = [(e, True)]
            while stack:
                current, is_top = stack.pop()
                if current.id in seen:
                    continue
                seen.add(current.id)
                to_serialize.append((current, is_top))
                for child in current.children:
                    stack.append((child, False))
        clipboard = []
        for e, is_top in to_serialize:
            data = copy.deepcopy(e.serialize())
            if is_top:
                data["parent"] = None
            t = e.transform
            if t:
                world_pos, world_rot, world_scale = t.world_matrix.decompose()
                for comp_data in data.get("components", []):
                    if comp_data.get("_key") == "Transform":
                        comp_data["local_position"] = world_pos.to_list()
                        comp_data["local_rotation"] = world_rot.to_list()
                        comp_data["local_scale"] = world_scale.to_list()
                        break
            clipboard.append(data)
        mw = self.parent()
        viewport = getattr(mw, '_viewport', None)
        if viewport:
            viewport._entity_clipboard = clipboard
    def _on_paste(self):
        mw = self.parent()
        viewport = getattr(mw, '_viewport', None)
        if viewport:
            viewport._paste_entities()
    def _delete_entity_by_id(self, eid: str):
        if not self._scene:
            return
        entity = self._scene.get_entity(eid)
        if entity:
            self._delete_entity(entity)
    def _toggle_active(self, entity: Entity):
        entity.active = not entity.active
        if self._scene:
            self._scene.mark_dirty()
        self._refresh()
        if self._selected_entity:
            self._restore_selection(self._selected_entity.id)

    def _toggle_active_by_ids(self, eids: list):
        if not self._scene:
            return
        changed = False
        for eid in eids:
            entity = self._scene.get_entity(eid)
            if entity:
                entity.active = not entity.active
                changed = True
        if changed:
            self._scene.mark_dirty()
            self._refresh()

    def _toggle_active_selected(self):
        items = self._tree.selectedItems()
        if not items or not self._scene:
            return
        eids = []
        for item in items:
            eid = item.data(0, Qt.ItemDataRole.UserRole)
            if eid:
                eids.append(eid)
        self._toggle_active_by_ids(eids)
    def _save_prefab(self, entity: Entity):
        from PyQt6.QtWidgets import QFileDialog
        from core.ecs.prefab import Prefab
        path, _ = QFileDialog.getSaveFileName(self, "Save Prefab", "prefabs/", "Prefabs (*.zpep)")
        if path:
            if not path.endswith(".zpep"):
                path += ".zpep"
            pref = Prefab(entity.name)
            pref.capture([entity])
            pref.save(path)
    def _apply_prefab(self, entity: Entity):
        from core.foundation.commands import ApplyPrefabOverridesCommand, get_history
        from core.ecs.prefab import Prefab
        roots = Prefab.get_prefab_roots([entity])
        get_history().execute(ApplyPrefabOverridesCommand(self._scene, roots))
        self._scene.mark_dirty()
        self._refresh()

    def _revert_prefab(self, entity: Entity):
        from core.foundation.commands import RevertPrefabInstanceCommand, get_history
        from core.ecs.prefab import Prefab
        roots = Prefab.get_prefab_roots([entity])
        cmd = RevertPrefabInstanceCommand(self._scene, roots)
        get_history().execute(cmd)
        self._scene.mark_dirty()
        self._refresh()

    def _unpack_prefab(self, entity: Entity):
        from core.foundation.commands import UnpackPrefabCommand, get_history
        from core.ecs.prefab import Prefab
        roots = Prefab.get_prefab_roots([entity])
        cmd = UnpackPrefabCommand(self._scene, roots)
        get_history().execute(cmd)
        self._scene.mark_dirty()
        self._refresh()

    def set_selected_entity(self, entity: Optional[Entity]):
        self._tree.blockSignals(True)
        self._selected_entity = entity
        if entity:
            self._restore_selection(entity.id)
        else:
            self._tree.clearSelection()
        self._tree.blockSignals(False)
    def set_selected_entities(self, entities: list):
        self._tree.blockSignals(True)
        self._tree.clearSelection()
        for entity in entities:
            if entity:
                item = self._find_item(entity.id, self._tree.invisibleRootItem())
                if item:
                    item.setSelected(True)
        self._tree.blockSignals(False)
        if entities:
            self._selected_entity = entities[0]
    def refresh(self):
        self._refresh()

    def reveal_entity(self, eid: str):
        entity = self._scene.get_entity(eid) if self._scene else None
        if not entity:
            return
        self._tree.blockSignals(True)
        self._selected_entity = entity
        item = self._find_item(eid, self._tree.invisibleRootItem())
        if item:
            self._tree.setCurrentItem(item)
        self._tree.blockSignals(False)
        if item:
            self._tree.scrollToItem(item)
            self._flash_item(item)
        self.entity_selected.emit(entity)

    def flash_entity(self, eid: str):
        item = self._find_item(eid, self._tree.invisibleRootItem())
        if item:
            self._tree.scrollToItem(item)
            self._flash_item(item)

    @staticmethod
    def _flash_item(item: QTreeWidgetItem):
        try:
            tree = item.treeWidget()
            if not tree:
                return
            _flash_overlay(tree.viewport(), tree.visualItemRect(item))
        except RuntimeError:
            pass
