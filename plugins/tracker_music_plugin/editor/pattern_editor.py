# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import copy
import re

from PyQt6.QtCore import QItemSelection, QItemSelectionModel, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)

_HIGHLIGHT_BRUSH = QBrush(QColor(255, 190, 30, 100))
_BASE_BRUSH = QBrush(QColor(0, 0, 0, 0))

_NOTE_RE = re.compile(r"^([A-Ga-g])([-#])([0-9])$")
_EFFECT_RE = re.compile(r"^([0-9A-Fa-f]{3})$")
_VOLUME_RE = re.compile(r"^([a-z])([0-9]{1,2})$")

_NOTE_NAMES = ["C-", "C#", "D-", "D#", "E-", "F-", "F#", "G-", "G#", "A-", "A#", "B-"]

COLUMNS_PER_CHANNEL = 4
COL_NOTE = 0
COL_INS = 1
COL_VOL = 2
COL_EFF = 3

_COL_LABELS = ["Note", "Ins", "Vol", "Eff"]


def _note_text(note) -> str:
    p = getattr(note, "period", "") or ""
    return p if p else "---"


def _inst_text(note) -> str:
    i = int(getattr(note, "instrument_idx", 0) or 0)
    return f"{i:02d}" if i else "--"


def _vol_text(note) -> str:
    if hasattr(note, "vol_cmd"):
        cmd = str(getattr(note, "vol_cmd", "") or "")
        if cmd:
            v = int(getattr(note, "vol_val", -1) or -1)
            return f"{cmd}{v:02d}" if v >= 0 else f"{cmd}--"
        return "--"
    if hasattr(note, "volume"):
        v = int(getattr(note, "volume", -1) or -1)
        return f"v{v:02d}" if v >= 0 else "--"
    return "--"


def _eff_text(note) -> str:
    e = str(getattr(note, "effect", "") or "")
    return e if e else "---"


def _transpose_note(note_str: str, semis: int) -> str:
    m = _NOTE_RE.match(note_str)
    if not m:
        return note_str
    pitch = (m.group(1) + m.group(2)).upper()
    octave = int(m.group(3))
    idx = (octave - 1) * 12 + _NOTE_NAMES.index(pitch)
    idx += semis
    idx = max(0, min(95, idx))
    return f"{_NOTE_NAMES[idx % 12]}{idx // 12 + 1}"


def _parse_note(text: str):
    s = text.strip()
    if not s or s in ("---", "--"):
        return ""
    if s.lower() == "off":
        return "off"
    m = _NOTE_RE.match(s)
    if not m:
        return None
    return (m.group(1) + m.group(2)).upper() + m.group(3)


def _parse_inst(text: str):
    s = text.strip()
    if not s or s in ("---", "--"):
        return 0
    if not s.isdigit():
        return None
    v = int(s)
    if v > 255:
        return None
    return v


def _parse_effect(text: str):
    s = text.strip()
    if not s or s in ("---", "--"):
        return ""
    m = _EFFECT_RE.match(s)
    if not m:
        return None
    return m.group(1).upper()


def _parse_vol(text: str):
    s = text.strip()
    if not s or s in ("---", "--"):
        return ("", -1)
    m = _VOLUME_RE.match(s)
    if m:
        v = int(m.group(2))
        if v <= 64:
            return (m.group(1).lower(), v)
        return None
    if s.isdigit():
        v = int(s)
        if v <= 64:
            return ("v", v)
    return None


class PatternEditorWidget(QTableWidget):
    patternChanged = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._pattern = None
        self._loading = False
        self._current_row: int = -1
        self._highlight_refs: list[tuple[int, QTableWidgetItem]] = []
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._pending_char = ""
        self.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.verticalHeader().setDefaultSectionSize(20)
        self.horizontalHeader().setDefaultSectionSize(64)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().setVisible(True)
        self.itemChanged.connect(self._on_item_changed)

    def clear_rows(self):
        self._current_row = -1
        self._highlight_refs.clear()
        self.clearSelection()

    def load_pattern(self, pattern) -> None:
        self.setRowCount(0)
        self.setColumnCount(0)
        self._pattern = pattern
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._refresh_grid()

    def _refresh_grid(self) -> None:
        self._loading = True
        self._highlight_refs.clear()
        self._current_row = -1
        self.setRowCount(0)
        self.setColumnCount(0)
        if self._pattern is None:
            self._loading = False
            return
        rows = int(getattr(self._pattern, "n_rows", 0) or 0)
        channels = int(getattr(self._pattern, "n_channels", 0) or 0)
        self.setRowCount(rows)
        self.setColumnCount(channels * COLUMNS_PER_CHANNEL)
        headers = []
        for c in range(channels):
            for sub in range(COLUMNS_PER_CHANNEL):
                label = f"Ch{c} {_COL_LABELS[sub]}" if sub == COL_NOTE else _COL_LABELS[sub]
                headers.append(label)
        self.setHorizontalHeaderLabels(headers)
        data = getattr(self._pattern, "data", None) or []
        for ch in range(channels):
            if ch >= len(data):
                continue
            chan = data[ch]
            for r in range(rows):
                if r >= len(chan):
                    continue
                note = chan[r]
                texts = [_note_text(note), _inst_text(note), _vol_text(note), _eff_text(note)]
                for sub in range(COLUMNS_PER_CHANNEL):
                    item = QTableWidgetItem(texts[sub])
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.setItem(r, ch * COLUMNS_PER_CHANNEL + sub, item)
        self._current_row = -1
        self._loading = False

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._loading or self._pattern is None or item is None:
            return
        row = item.row()
        col = item.column()
        ch = col // COLUMNS_PER_CHANNEL
        field = col % COLUMNS_PER_CHANNEL
        data = getattr(self._pattern, "data", None) or []
        if ch >= len(data) or row >= len(data[ch]):
            return
        note = data[ch][row]
        parsed = self._parse_field(field, item.text())
        if parsed is None or self._parsed_is_current(note, field, parsed):
            self._set_item_text(item, self._field_text(note, field))
            return
        self._push_undo()
        self._apply_field(note, field, parsed)
        self._set_item_text(item, self._field_text(note, field))
        self.patternChanged.emit()

    def _parse_field(self, field: int, text: str):
        if field == COL_NOTE:
            return _parse_note(text)
        if field == COL_INS:
            return _parse_inst(text)
        if field == COL_VOL:
            return _parse_vol(text)
        if field == COL_EFF:
            return _parse_effect(text)
        return None

    def _parsed_is_current(self, note, field: int, parsed) -> bool:
        if field == COL_NOTE:
            return (getattr(note, "period", "") or "") == parsed
        if field == COL_INS:
            return int(getattr(note, "instrument_idx", 0) or 0) == int(parsed)
        if field == COL_VOL:
            cmd, val = parsed
            if hasattr(note, "vol_cmd"):
                return str(getattr(note, "vol_cmd", "") or "") == cmd and int(
                    getattr(note, "vol_val", -1) or -1) == val
            if hasattr(note, "volume"):
                return int(getattr(note, "volume", -1) or -1) == val
            return False
        if field == COL_EFF:
            return (getattr(note, "effect", "") or "") == parsed
        return False

    def _apply_field(self, note, field: int, parsed) -> None:
        if field == COL_NOTE:
            note.period = parsed
        elif field == COL_INS:
            note.instrument_idx = int(parsed)
        elif field == COL_VOL:
            cmd, val = parsed
            if hasattr(note, "vol_cmd"):
                note.vol_cmd = cmd
                note.vol_val = val
            elif hasattr(note, "volume"):
                note.volume = val
        elif field == COL_EFF:
            note.effect = parsed

    def _field_text(self, note, field: int) -> str:
        if field == COL_NOTE:
            return _note_text(note)
        if field == COL_INS:
            return _inst_text(note)
        if field == COL_VOL:
            return _vol_text(note)
        return _eff_text(note)

    def _set_item_text(self, item: QTableWidgetItem, text: str) -> None:
        self._loading = True
        try:
            item.setText(text)
        finally:
            self._loading = False

    def _empty_parsed(self, field: int):
        if field in (COL_NOTE, COL_EFF):
            return ""
        if field == COL_INS:
            return 0
        return ("", -1)

    def _current_cell(self):
        item = self.currentItem()
        if item is None:
            return None
        return item.row(), item.column()

    def _push_undo(self) -> None:
        if self._pattern is None:
            return
        self._undo_stack.append(copy.deepcopy(getattr(self._pattern, "data", None)))
        if len(self._undo_stack) > 200:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def undo(self) -> None:
        if not self._undo_stack or self._pattern is None:
            return
        self._redo_stack.append(copy.deepcopy(getattr(self._pattern, "data", None)))
        self._pattern.data = self._undo_stack.pop()
        self._refresh_after_structure()

    def redo(self) -> None:
        if not self._redo_stack or self._pattern is None:
            return
        self._undo_stack.append(copy.deepcopy(getattr(self._pattern, "data", None)))
        self._pattern.data = self._redo_stack.pop()
        self._refresh_after_structure()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def insert_row(self) -> None:
        if self._pattern is None:
            return
        row = max(0, self.currentRow())
        self._push_undo()
        data = getattr(self._pattern, "data", None) or []
        for ch in range(len(data)):
            chan = data[ch]
            if not chan:
                continue
            chan.insert(row, type(chan[row % len(chan)])())
            chan.pop()
        self._refresh_after_structure()

    def delete_row(self) -> None:
        if self._pattern is None:
            return
        row = self.currentRow()
        if row < 0:
            return
        data = getattr(self._pattern, "data", None) or []
        for ch in range(len(data)):
            chan = data[ch]
            if not chan:
                continue
            cls = type(chan[row])
            chan.pop(row)
            chan.append(cls())
        self._refresh_after_structure()

    def clear_row(self) -> None:
        if self._pattern is None:
            return
        row = self.currentRow()
        if row < 0:
            return
        self._push_undo()
        data = getattr(self._pattern, "data", None) or []
        for ch in range(len(data)):
            chan = data[ch]
            if row < len(chan):
                chan[row] = type(chan[row])()
        self._refresh_after_structure()

    def transpose_rows(self, semis: int) -> None:
        if self._pattern is None:
            return
        rows = sorted({i.row() for i in self.selectedItems()}) or [self.currentRow()]
        if not rows or rows[0] < 0:
            return
        self._push_undo()
        data = getattr(self._pattern, "data", None) or []
        for row in rows:
            for ch in range(len(data)):
                if row >= len(data[ch]):
                    continue
                note = data[ch][row]
                p = getattr(note, "period", "") or ""
                if p.lower() == "off":
                    continue
                transposed = _transpose_note(p, semis)
                if transposed != p:
                    note.period = transposed
        self._refresh_after_structure()

    def copy_row(self) -> None:
        if self._pattern is None:
            return
        row = self.currentRow()
        if row < 0:
            return
        data = getattr(self._pattern, "data", None) or []
        self._clipboard = [copy.deepcopy(data[ch][row]) for ch in range(len(data))]

    def cut_row(self) -> None:
        self.copy_row()
        self.clear_row()

    def paste_row(self) -> None:
        if self._pattern is None or not getattr(self, "_clipboard", None):
            return
        row = self.currentRow()
        if row < 0:
            return
        self._push_undo()
        data = getattr(self._pattern, "data", None) or []
        for ch in range(min(len(data), len(self._clipboard))):
            if row < len(data[ch]):
                data[ch][row] = copy.deepcopy(self._clipboard[ch])
        self._refresh_after_structure()

    def select_all_rows(self) -> None:
        if self._pattern is None:
            return
        row = self.currentRow()
        if row < 0:
            return
        if not self.columnCount():
            return
        sel = QItemSelection(self.model().index(row, 0), self.model().index(row, self.columnCount() - 1))
        self.selectionModel().select(sel, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _refresh_after_structure(self) -> None:
        current = self.currentItem()
        keep = (current.row(), current.column()) if current is not None else (0, 0)
        self._refresh_grid()
        if self.rowCount() and self.columnCount():
            r = max(0, min(keep[0], self.rowCount() - 1))
            c = max(0, min(keep[1], self.columnCount() - 1))
            self.setCurrentCell(r, c)
        self.patternChanged.emit()

    def highlight_row(self, row: int) -> None:
        if row == self._current_row and self._highlight_refs:
            return
        for _, item in self._highlight_refs:
            if item is not None:
                try:
                    item.setBackground(_BASE_BRUSH)
                except RuntimeError:
                    pass
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
        first = self.item(row, 0) if self.item(row, 0) else None
        self.scrollToItem(first)

    def _start_edit_with_char(self, item: QTableWidgetItem, text: str) -> None:
        self._pending_char = text
        self.editItem(item)
        QTimer.singleShot(0, self._prefill_editor)

    def _prefill_editor(self) -> None:
        ed = self.findChild(QLineEdit)
        if ed is not None:
            ed.setText(self._pending_char or "")
            ed.selectAll()
        self._pending_char = ""

    def keyPressEvent(self, event):
        if self.state() == QAbstractItemView.State.EditingState:
            super().keyPressEvent(event)
            return
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        item = self.currentItem()
        if ctrl and key == Qt.Key.Key_C:
            self.copy_row()
            return
        if ctrl and key == Qt.Key.Key_X:
            self.cut_row()
            return
        if ctrl and key == Qt.Key.Key_V:
            self.paste_row()
            return
        if ctrl and key == Qt.Key.Key_Z and not shift:
            self.undo()
            return
        if (ctrl and key == Qt.Key.Key_Y) or (ctrl and shift and key == Qt.Key.Key_Z):
            self.redo()
            return
        if key == Qt.Key.Key_Delete or key == Qt.Key.Key_Backspace:
            if item is not None:
                fields = self._field_item(item)
                if fields is not None:
                    note = fields[0]
                    field = fields[1]
                    if not self._parsed_is_current(note, field, self._empty_parsed(field)):
                        parsed = self._empty_parsed(field)
                        self._push_undo()
                        self._apply_field(note, field, parsed)
                        self._set_item_text(item, self._field_text(note, field))
                        self.patternChanged.emit()
            return
        if key == Qt.Key.Key_Insert:
            self.insert_row()
            return
        if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and item is not None:
            r = min(item.row() + 1, self.rowCount() - 1)
            self.setCurrentCell(max(0, r), item.column())
            return
        if item is not None and key in (Qt.Key.Key_Tab, Qt.Key.Key_Backtab):
            if shift:
                self.setCurrentCell(item.row(), max(0, item.column() - 1))
            else:
                self.setCurrentCell(item.row(), min(self.columnCount() - 1, item.column() + 1))
            return
        if self._edit_key(event) and item is not None:
            self._start_edit_with_char(item, event.text())
            return
        super().keyPressEvent(event)

    def _field_item(self, item: QTableWidgetItem):
        if self._pattern is None or item is None:
            return None
        ch = item.column() // COLUMNS_PER_CHANNEL
        row = item.row()
        data = getattr(self._pattern, "data", None) or []
        if ch >= len(data) or row >= len(data[ch]):
            return None
        return (data[ch][row], item.column() % COLUMNS_PER_CHANNEL)

    def _edit_key(self, event) -> bool:
        nav = (Qt.Key.Key_Up, Qt.Key.Key_Down, Qt.Key.Key_Left, Qt.Key.Key_Right,
               Qt.Key.Key_PageUp, Qt.Key.Key_PageDown, Qt.Key.Key_Home, Qt.Key.Key_End,
               Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Tab,
               Qt.Key.Key_Backtab, Qt.Key.Key_Delete, Qt.Key.Key_Backspace,
               Qt.Key.Key_Insert)
        if event.key() in nav:
            return False
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            return False
        return bool(event.text())