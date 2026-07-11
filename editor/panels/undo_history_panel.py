# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from PyQt6.QtWidgets import QDockWidget, QListWidget, QListWidgetItem
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

class UndoHistoryPanel(QDockWidget):
    history_navigated = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__("Undo History", parent)
        self.setObjectName("UndoHistoryDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self._list = QListWidget()
        self._list.itemClicked.connect(self._on_item_clicked)
        self.setWidget(self._list)
        self._cache_key = ""
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._timed_refresh)
        self._refresh_timer.start(500)
    def _timed_refresh(self):
        from core.foundation.commands import get_history
        h = get_history()
        key = f"{h.undo_count}:{h.redo_count}"
        if key != self._cache_key:
            self._cache_key = key
            self.refresh()
    def refresh(self):
        from core.foundation.commands import get_history
        h = get_history()
        undo_descs = h.get_undo_descriptions()
        redo_descs = h.get_redo_descriptions()
        head = h.undo_count
        total = head + h.redo_count
        self._list.blockSignals(True)
        self._list.clear()
        sep_added = False
        for i in range(total - 1, -1, -1):
            is_redo = i >= head
            if not sep_added and not is_redo and total > head:
                sep = QListWidgetItem("--- Current State ---")
                sep.setFlags(sep.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                sep.setForeground(self._list.palette().mid())
                self._list.addItem(sep)
                sep_added = True
            desc = redo_descs[i - head] if is_redo else undo_descs[i]
            item = QListWidgetItem(desc)
            item.setData(Qt.ItemDataRole.UserRole, i + 1)
            if is_redo:
                item.setForeground(self._list.palette().window().color().darker(180))
                f = item.font()
                f.setItalic(True)
                item.setFont(f)
            self._list.addItem(item)
        if not sep_added and total > 0:
            sep = QListWidgetItem("--- Current State ---")
            sep.setFlags(sep.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            sep.setForeground(self._list.palette().mid())
            self._list.addItem(sep)
        self._list.blockSignals(False)
    def _on_item_clicked(self, item):
        target = item.data(Qt.ItemDataRole.UserRole)
        if target is None:
            return
        from core.foundation.commands import get_history
        get_history().seek(target)
        self.history_navigated.emit()
