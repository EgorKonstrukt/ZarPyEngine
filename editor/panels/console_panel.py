# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QCheckBox, QLabel, QLineEdit,
                              QSplitter, QPlainTextEdit, QListWidget,
                              QListWidgetItem, QFrame, QStyle, QAbstractItemView)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor, QIcon, QKeyEvent, QKeySequence
from core.logger import Logger, LogLevel, LogEntry
import time as _time
import re
from core.editor_scale import scale

_NUM_RE = re.compile(r"\d+(\.\d+)?")


def _normalize(msg: str) -> str:
    return _NUM_RE.sub("{n}", msg)


def _fmt_time(ts: float) -> str:
    return _time.strftime("%H:%M:%S", _time.localtime(ts))


class ConsoleGroup:
    def __init__(self, entry: LogEntry):
        self.message: str = entry.message
        self._pattern: str = _normalize(entry.message)
        self.level: LogLevel = entry.level
        self.traceback: str = entry.traceback_str or ""
        self.count: int = 1
        self.entries: list[LogEntry] = [entry]
        self.list_item: QListWidgetItem | None = None

    def matches(self, entry: LogEntry) -> bool:
        return (self._pattern == _normalize(entry.message)
                and self.level == entry.level
                and (entry.traceback_str or "") == self.traceback)

    def display_text(self) -> str:
        ts = _fmt_time(self.entries[0].timestamp)
        level_names = {LogLevel.DEBUG: "DEBUG", LogLevel.INFO: "INFO",
                       LogLevel.WARNING: "WARNING", LogLevel.ERROR: "ERROR"}
        label = f"[{ts}] {level_names.get(self.level, 'INFO')}: {self.message}"
        if self.count > 1:
            label += f"  ({self.count})"
        return label


_LEVEL_COLORS = {
    LogLevel.DEBUG: (QColor("#c0c0c0"), QColor("#2d2d2d")),
    LogLevel.INFO: (QColor("#ffffff"), QColor("#1a2a4a")),
    LogLevel.WARNING: (QColor("#ffffff"), QColor("#3a2e00")),
    LogLevel.ERROR: (QColor("#ffffff"), QColor("#4a1a1a")),
}

_LEVEL_STD_ICONS = {
    LogLevel.DEBUG: QStyle.StandardPixmap.SP_ArrowRight,
    LogLevel.INFO: QStyle.StandardPixmap.SP_MessageBoxInformation,
    LogLevel.WARNING: QStyle.StandardPixmap.SP_MessageBoxWarning,
    LogLevel.ERROR: QStyle.StandardPixmap.SP_MessageBoxCritical,
}


class _LogList(QListWidget):
    def __init__(self, panel: ConsolePanel, parent=None):
        super().__init__(parent)
        self._panel = panel

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is None:
            self.clearSelection()
            self.setCurrentItem(None)
            self._panel._deselect()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.matches(QKeySequence.StandardKey.Copy):
            items = self.selectedItems()
            if items:
                from PyQt6.QtWidgets import QApplication
                lines = [item.text() for item in items]
                QApplication.clipboard().setText("\n".join(lines))
            return
        super().keyPressEvent(event)


class ConsolePanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Console", parent)
        self._show_debug: bool = True
        self._show_info: bool = True
        self._show_warning: bool = True
        self._show_error: bool = True
        self._filter_text: str = ""
        self._font_family: str = "Consolas"
        self._font_size: int = 9
        self._max_groups: int = 2000
        self._refresh_interval: int = 300
        self._groups: list[ConsoleGroup] = []
        self._pending: list[LogEntry] = []
        self._selected_group: ConsoleGroup | None = None
        self._level_icons: dict[LogLevel, QIcon] = {}
        self._setup_ui()
        Logger.add_listener(self._on_log_entry)
        self._rebuild()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._flush_pending)
        self._refresh_timer.start(self._refresh_interval)

    def load_config(self, config) -> None:
        self._font_family = config.get("console.font_family", self._font_family)
        self._font_size = config.get("console.font_size", self._font_size)
        self._max_groups = config.get("console.max_blocks", self._max_groups)
        self._refresh_interval = config.get("console.refresh_interval", self._refresh_interval)
        self._refresh_timer.setInterval(self._refresh_interval)
        font = QFont(self._font_family, self._font_size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._list.setFont(font)
        self._detail.setFont(font)

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(2)

        clear_btn = QPushButton("Clear")
        clear_btn.setFixedHeight(22)
        clear_btn.clicked.connect(self._clear)
        toolbar.addWidget(clear_btn)

        toolbar.addSpacing(4)

        collapse_btn = QPushButton("Collapse")
        collapse_btn.setFixedHeight(22)
        collapse_btn.clicked.connect(self._collapse_all)
        toolbar.addWidget(collapse_btn)

        toolbar.addSpacing(8)

        self._dbg_cb = QCheckBox("Debug")
        self._dbg_cb.setChecked(True)
        self._dbg_cb.toggled.connect(lambda v: setattr(self, "_show_debug", v) or self._rebuild())
        toolbar.addWidget(self._dbg_cb)

        self._info_cb = QCheckBox("Info")
        self._info_cb.setChecked(True)
        self._info_cb.toggled.connect(lambda v: setattr(self, "_show_info", v) or self._rebuild())
        toolbar.addWidget(self._info_cb)

        self._warn_cb = QCheckBox("Warning")
        self._warn_cb.setChecked(True)
        self._warn_cb.toggled.connect(lambda v: setattr(self, "_show_warning", v) or self._rebuild())
        toolbar.addWidget(self._warn_cb)

        self._err_cb = QCheckBox("Error")
        self._err_cb.setChecked(True)
        self._err_cb.toggled.connect(lambda v: setattr(self, "_show_error", v) or self._rebuild())
        toolbar.addWidget(self._err_cb)

        toolbar.addStretch()

        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("Filter...")
        self._filter_edit.setFixedHeight(22)
        self._filter_edit.textChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self._filter_edit)

        toolbar.addSpacing(8)

        self._dbg_count = QLabel("0")
        self._dbg_count.setStyleSheet("color: #999; padding: 0 4px;")
        toolbar.addWidget(self._dbg_count)

        self._info_count = QLabel("0")
        self._info_count.setStyleSheet("color: #aaaaff; padding: 0 4px;")
        toolbar.addWidget(self._info_count)

        self._warn_count = QLabel("0")
        self._warn_count.setStyleSheet("color: #ffcc00; padding: 0 4px;")
        toolbar.addWidget(self._warn_count)

        self._err_count = QLabel("0")
        self._err_count.setStyleSheet("color: #ff4444; padding: 0 4px;")
        toolbar.addWidget(self._err_count)

        layout.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setHandleWidth(3)

        self._list = _LogList(self)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        self._list.setWordWrap(True)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        font = QFont(self._font_family, self._font_size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        self._list.setFont(font)
        self._list.currentItemChanged.connect(self._on_item_changed)
        self._list.setStyleSheet("""
            QListWidget { background: transparent; border: none; padding: 2px; }
            QListWidget::item { padding: 1px 4px; }
        """)
        splitter.addWidget(self._list)

        self._detail = QPlainTextEdit()
        self._detail.setReadOnly(True)
        detail_font = QFont(self._font_family, self._font_size)
        detail_font.setStyleHint(QFont.StyleHint.Monospace)
        self._detail.setFont(detail_font)
        self._detail.setFrameShape(QFrame.Shape.NoFrame)
        splitter.addWidget(self._detail)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 150])

        layout.addWidget(splitter)
        self.setWidget(w)

        for level, sp in _LEVEL_STD_ICONS.items():
            self._level_icons[level] = self.style().standardIcon(sp)

    def _level_colors(self, level: LogLevel):
        return _LEVEL_COLORS.get(level, _LEVEL_COLORS[LogLevel.INFO])

    def _on_log_entry(self, entry: LogEntry):
        self._pending.append(entry)

    def _flush_pending(self):
        if not self._pending:
            return
        entries = self._pending[:]
        self._pending.clear()
        changed = False
        for entry in entries:
            if self._add_entry(entry):
                changed = True
        if changed:
            self._update_list()
            self._update_counts()

    def _add_entry(self, entry: LogEntry) -> bool:
        if not self._should_show(entry):
            return False
        for g in self._groups:
            if g.matches(entry):
                g.count += 1
                g.entries.append(entry)
                return True
        g = ConsoleGroup(entry)
        self._groups.append(g)
        if len(self._groups) > self._max_groups:
            removed = self._groups.pop(0)
            if removed.list_item:
                row = self._list.row(removed.list_item)
                taken = self._list.takeItem(row)
                del taken
                removed.list_item = None
        return True

    def _should_show(self, entry: LogEntry) -> bool:
        level_map = {
            LogLevel.DEBUG: self._show_debug,
            LogLevel.INFO: self._show_info,
            LogLevel.WARNING: self._show_warning,
            LogLevel.ERROR: self._show_error,
        }
        if not level_map.get(entry.level, True):
            return False
        if self._filter_text and self._filter_text.lower() not in entry.message.lower():
            return False
        return True

    def _update_list(self):
        sb = self._list.verticalScrollBar()
        at_bottom = sb.value() >= sb.maximum() - 5 if sb.maximum() > 0 else True
        saved_current = self._list.currentItem()
        saved_group = self._selected_group

        for g in self._groups:
            if not {LogLevel.DEBUG: self._show_debug, LogLevel.INFO: self._show_info,
                    LogLevel.WARNING: self._show_warning, LogLevel.ERROR: self._show_error}.get(g.level, True):
                if g.list_item:
                    row = self._list.row(g.list_item)
                    taken = self._list.takeItem(row)
                    del taken
                    g.list_item = None
                continue
            if self._filter_text and self._filter_text.lower() not in g.message.lower():
                if g.list_item:
                    row = self._list.row(g.list_item)
                    taken = self._list.takeItem(row)
                    del taken
                    g.list_item = None
                continue

            text = g.display_text()
            fg, bg = self._level_colors(g.level)
            if g.list_item:
                if g.list_item.text() != text:
                    g.list_item.setText(text)
                    g.list_item.setForeground(fg)
            else:
                item = QListWidgetItem(text)
                item.setForeground(fg)
                item.setBackground(bg)
                item.setIcon(self._level_icons.get(g.level, self._level_icons[LogLevel.INFO]))
                item.setData(Qt.ItemDataRole.UserRole, id(g))
                self._list.addItem(item)
                g.list_item = item

        if saved_group and saved_group.list_item:
            self._list.setCurrentItem(saved_group.list_item)
        elif saved_current and saved_current.isHidden():
            self._list.setCurrentRow(self._list.count() - 1 if self._list.count() > 0 else -1)
        else:
            self._list.setCurrentItem(saved_current)

        if at_bottom:
            self._list.scrollToBottom()

    def _on_item_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        if current is None:
            self._selected_group = None
            self._detail.clear()
            return
        for g in self._groups:
            if g.list_item is current:
                self._selected_group = g
                self._render_detail(g)
                return

    def _deselect(self):
        self._selected_group = None
        self._detail.clear()

    def _render_detail(self, group: ConsoleGroup):
        lines = []
        level_names = {LogLevel.DEBUG: "DEBUG", LogLevel.INFO: "INFO",
                       LogLevel.WARNING: "WARNING", LogLevel.ERROR: "ERROR"}
        lines.append(f"[{level_names.get(group.level, 'INFO')}] {group.message}")
        if group.count > 1:
            lines.append(f"Repeated {group.count} times")
        for entry in group.entries:
            ts = _fmt_time(entry.timestamp)
            lines.append(f"[{ts}] {entry.message}")
            if entry.traceback_str:
                lines.append(entry.traceback_str.strip())
        self._detail.setPlainText("\n".join(lines))
        self._detail.moveCursor(QTextCursor.MoveOperation.Start)

    def _update_counts(self):
        counts = {LogLevel.DEBUG: 0, LogLevel.INFO: 0, LogLevel.WARNING: 0, LogLevel.ERROR: 0}
        for g in self._groups:
            if g.level in counts:
                counts[g.level] += g.count
        self._dbg_count.setText(str(counts[LogLevel.DEBUG]))
        self._info_count.setText(str(counts[LogLevel.INFO]))
        self._warn_count.setText(str(counts[LogLevel.WARNING]))
        self._err_count.setText(str(counts[LogLevel.ERROR]))

    def _clear(self):
        Logger.clear()
        self._groups.clear()
        self._selected_group = None
        self._list.clear()
        self._detail.clear()
        self._update_counts()

    def _collapse_all(self):
        pass

    def _on_filter_changed(self, text: str):
        self._filter_text = text
        self._rebuild()

    def _rebuild(self):
        self._list.clear()
        for g in self._groups:
            g.list_item = None
        self._groups.clear()
        for entry in Logger.get_entries():
            for g in self._groups:
                if g.matches(entry):
                    g.count += 1
                    g.entries.append(entry)
                    break
            else:
                g = ConsoleGroup(entry)
                self._groups.append(g)
                if len(self._groups) > self._max_groups:
                    self._groups.pop(0)
        self._update_list()
        self._update_counts()
