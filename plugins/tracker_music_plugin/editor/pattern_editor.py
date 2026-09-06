# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QBrush
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

_HIGHLIGHT_BRUSH = QBrush(QColor(255, 190, 30, 100))
_BASE_BRUSH = QBrush(QColor(0, 0, 0, 0))


def cell_text(cell) -> str:
    note = getattr(cell, "period", "") or ""
    note_txt = note if note else "---"
    inst = int(getattr(cell, "instrument_idx", 0) or 0)
    inst_txt = f"{inst:02d}" if inst else "--"
    vol_cmd = getattr(cell, "vol_cmd", "") or ""
    vol_txt = "---"
    if vol_cmd:
        vv = getattr(cell, "vol_val", -1)
        vol_txt = f"{vol_cmd}{max(0, int(vv)):02d}" if isinstance(vv, int) and vv >= 0 else f"{vol_cmd}--"
    else:
        volume = getattr(cell, "volume", -1)
        if isinstance(volume, int) and volume >= 0:
            vol_txt = f"v{volume:02d}"
    eff = getattr(cell, "effect", "") or ""
    return f"{note_txt} {inst_txt} {vol_txt} {eff}"


class PatternEditorWidget(QTableWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._current_row: int = -1
        self._highlight_refs: list[tuple[int, QTableWidgetItem]] = []
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setDefaultSectionSize(20)
        self.horizontalHeader().setDefaultSectionSize(98)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setVisible(True)

    def clear_rows(self):
        self._current_row = -1
        self._highlight_refs.clear()
        self.clearSelection()

    def load_pattern(self, pattern) -> None:
        self.clear_rows()
        if pattern is None:
            self.clear()
            return
        rows = int(getattr(pattern, "n_rows", 0) or 0)
        channels = int(getattr(pattern, "n_channels", 0) or 0)
        self.setRowCount(rows)
        self.setColumnCount(channels)
        headers = [f"Ch {i}" for i in range(channels)]
        self.setHorizontalHeaderLabels(headers)
        data = getattr(pattern, "data", None) or []
        for ch in range(channels):
            if ch >= len(data):
                continue
            chan = data[ch]
            for r in range(rows):
                if r >= len(chan):
                    continue
                item = QTableWidgetItem(cell_text(chan[r]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.setItem(r, ch, item)

    def highlight_row(self, row: int) -> None:
        if row == self._current_row and self._highlight_refs:
            return
        for col_idx, item in self._highlight_refs:
            if item is not None:
                item.setBackground(_BASE_BRUSH)
        self._highlight_refs = []
        self._current_row = row
        if row < 0 or row >= self.rowCount():
            return
        refs = []
        for col in range(self.columnCount()):
            item = self.item(row, col)
            if item is not None:
                item.setBackground(_HIGHLIGHT_BRUSH)
                refs.append((col, item))
        self._highlight_refs = refs
        self.scrollToItem(self.item(row, 0) if self.item(row, 0) else None)