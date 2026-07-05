# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional
from PyQt6.QtWidgets import QDoubleSpinBox, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QListWidget, \
    QListWidgetItem, QPushButton, QDialog, QApplication
from PyQt6.QtCore import Qt, QTimer, QMimeData
from PyQt6.QtGui import QPixmap, QFont, QColor, QPainter, QBrush, QPen, QFont as QF, QCursor, QIcon
from core.editor_scale import scale, scale_xy
from editor.inspector.constants import _accent
def _get_component_icon_pixmap(cls, size=16):
    from editor.inspector.helpers import get_component_icon_pixmap
    return get_component_icon_pixmap(cls, size)

class _FocusSpinBox(QDoubleSpinBox):
    def focusInEvent(self, event):
        super().focusInEvent(event)
        QTimer.singleShot(0, self.selectAll)

class _DragLabel(QLabel):
    def __init__(self, text, color, spinbox):
        super().__init__(text)
        self._spinbox = spinbox
        self._dragging = False
        self._start_x = 0
        self._start_val = 0
        self.setFixedWidth(scale(14))
        self.setCursor(Qt.CursorShape.SizeHorCursor)
        self.setStyleSheet(f"color: {color}; font-weight: bold;")
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._start_x = event.globalPosition().x()
            self._start_val = self._spinbox.value()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
    def mouseMoveEvent(self, event):
        if self._dragging:
            screen = self.screen()
            global_x = event.globalPosition().x()
            global_y = event.globalPosition().y()
            dx = global_x - self._start_x
            if screen:
                geo = screen.geometry()
                margin = 2
                if global_x >= geo.right() - margin:
                    landing_x = geo.left() + margin + 5
                    self._start_x -= global_x - landing_x
                    QCursor.setPos(int(landing_x), int(global_y))
                elif global_x <= geo.left() + margin:
                    landing_x = geo.right() - margin - 5
                    self._start_x -= global_x - landing_x
                    QCursor.setPos(int(landing_x), int(global_y))
            modifiers = QApplication.keyboardModifiers()
            ctrl_down = bool(modifiers & Qt.KeyboardModifier.ControlModifier)
            if modifiers & Qt.KeyboardModifier.ShiftModifier:
                factor = 0.001
            elif ctrl_down:
                factor = 0.1
            else:
                factor = 0.01
            new_val = self._start_val + dx * factor
            if not ctrl_down:
                try:
                    from core.engine import Engine
                    gizmo = Engine.instance().viewport.gizmo
                    if gizmo.snap_enabled and gizmo.snap_translate > 0:
                        snap = gizmo.snap_translate
                        new_val = round(new_val / snap) * snap
                except Exception:
                    pass
            self._spinbox.setValue(new_val)
            event.accept()
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()

class _DropLabelMixin:
    def _highlight(self, on=True):
        if on:
            if not hasattr(self, '_cached_style'):
                self._cached_style = self.styleSheet()
            self.setStyleSheet("border: 1px solid;")
        else:
            style = getattr(self, '_cached_style', "")
            self.setStyleSheet(style)

class _ResourceDropLabel(QLabel, _DropLabelMixin):
    def __init__(self, on_drop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_drop = on_drop
        self.setAcceptDrops(True)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            self._highlight(True)
            event.accept()
        else:
            super().dragEnterEvent(event)
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)
    def dragLeaveEvent(self, event):
        self._highlight(False)
        super().dragLeaveEvent(event)
    def dropEvent(self, event):
        self._highlight(False)
        path = None
        if event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()
        elif event.mimeData().hasText():
            path = event.mimeData().text().strip()
        if path and self._on_drop:
            self._on_drop(path)
        event.acceptProposedAction()

class _EntityDropLabel(QLabel, _DropLabelMixin):
    def __init__(self, on_drop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_drop = on_drop
        self.setAcceptDrops(True)
    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-zpe-entity"):
            event.setDropAction(Qt.DropAction.CopyAction)
            self._highlight(True)
            event.accept()
        else:
            super().dragEnterEvent(event)
    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-zpe-entity"):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            super().dragMoveEvent(event)
    def dragLeaveEvent(self, event):
        self._highlight(False)
        super().dragLeaveEvent(event)
    def dropEvent(self, event):
        self._highlight(False)
        if event.mimeData().hasFormat("application/x-zpe-entity"):
            data = bytes(event.mimeData().data("application/x-zpe-entity")).decode("utf-8")
            eid = data.split(",")[0]
            if eid and self._on_drop:
                self._on_drop(eid)
        event.acceptProposedAction()

class EntityPickerDialog(QDialog):
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Entity")
        self.setMinimumSize(300, 400)
        self.resize(320, 450)
        self._scene = scene
        self._selected_id: Optional[str] = None
        self._setup_ui()
        self._populate()
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search entities...")
        self._search.textChanged.connect(self._filter)
        layout.addWidget(self._search)
        self._list = QListWidget()
        self._list.setSpacing(1)
        self._list.itemDoubleClicked.connect(self._accept_selection)
        layout.addWidget(self._list, 1)
        btn_row = QHBoxLayout()
        self._select_btn = QPushButton("Select")
        self._select_btn.setEnabled(False)
        self._select_btn.clicked.connect(self._accept_selection)
        btn_row.addWidget(self._select_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
    def _populate(self, filter_text: str = ""):
        self._list.clear()
        filter_lower = filter_text.lower()
        if not self._scene:
            return
        for e in self._scene.get_all_entities():
            if filter_text and filter_lower not in e.name.lower():
                continue
            item = QListWidgetItem(f"  {e.name}")
            item.setData(Qt.ItemDataRole.UserRole, e.id)
            icon_cls = None
            for c in e.get_all_components():
                if getattr(type(c), '_show_gizmo_icon', True) and type(c).__name__ != "Transform":
                    icon_cls = type(c)
                    break
            if icon_cls:
                pix = _get_component_icon_pixmap(icon_cls, 16)
                item.setIcon(QIcon(pix))
            self._list.addItem(item)
    def _filter(self, text: str):
        self._populate(text)
    def _on_selection_changed(self):
        items = self._list.selectedItems()
        self._select_btn.setEnabled(len(items) > 0)
    def _accept_selection(self):
        items = self._list.selectedItems()
        if items:
            self._selected_id = items[0].data(Qt.ItemDataRole.UserRole)
            self.accept()
    def selected_id(self) -> Optional[str]:
        return self._selected_id
