# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout,
                             QToolBar, QLabel, QFileDialog,
                             QPlainTextEdit, QMessageBox, QTabWidget,
                             QSizePolicy, QComboBox, QMenuBar,
                             QStatusBar, QCompleter, QTextEdit, QMenu,
                             QInputDialog, QApplication, QDialog)
from PyQt6.QtCore import Qt, pyqtSignal, QSize, QStringListModel, QTimer
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QKeyEvent, QWheelEvent,
                         QSyntaxHighlighter, QTextCharFormat, QTextCursor,
                         QAction, QIcon, QKeySequence, QPainter, QPixmap,
                         QPen, QTextBlock, QMouseEvent)

if TYPE_CHECKING:
    from core.engine.engine import Engine

try:
    import qtawesome as qta
except ImportError:
    qta = None

from core.config.editor_scale import scale, scale_xy

try:
    from core.config.syntax_config import (KEYWORDS, BUILTINS, CONSTANTS, EXCEPTIONS,
                                           SYNTAX_COLORS, SYNTAX_STYLES)
except ImportError:
    try:
        from .syntax_config import (KEYWORDS, BUILTINS, CONSTANTS, EXCEPTIONS,
                                    SYNTAX_COLORS, SYNTAX_STYLES)
    except ImportError:
        from syntax_config import (KEYWORDS, BUILTINS, CONSTANTS, EXCEPTIONS,
                                   SYNTAX_COLORS, SYNTAX_STYLES)

from editor.panels.vcs_panel import _Git


_QTA_COLORS = {
    "new": "#d4d4d4",
    "open": "#d4d4d4",
    "save": "#d4d4d4",
    "save_as": "#d4d4d4",
    "cut": "#d4d4d4",
    "copy": "#d4d4d4",
    "paste": "#d4d4d4",
    "undo": "#d4d4d4",
    "redo": "#d4d4d4",
    "word_wrap": "#d4d4d4",
    "indent": "#d4d4d4",
    "zoom_in": "#d4d4d4",
    "zoom_out": "#d4d4d4",
    "run": "#9ccc65",
}


def _ot_transform_pos(pos: int, against_pos: int, against_removed: int, against_added_len: int) -> int:
    if pos > against_pos:
        if against_removed > 0 and pos <= against_pos + against_removed:
            return against_pos
        pos += against_added_len - against_removed
    return pos


def _ot_transform_op(op: tuple, against: tuple) -> tuple:
    pos, removed, added = op
    apos, aremoved, aadded = against
    new_pos = _ot_transform_pos(pos, apos, aremoved or 0, len(aadded or ""))
    return (new_pos, removed, added)


def _qta_icon(name: str) -> QIcon:
    if qta is None:
        return QIcon()
    names = {
        "new": "fa5s.file",
        "open": "fa5s.folder-open",
        "save": "fa5s.save",
        "save_as": "fa5s.save",
        "cut": "fa5s.cut",
        "copy": "fa5s.copy",
        "paste": "fa5s.paste",
        "undo": "fa5s.undo",
        "redo": "fa5s.redo",
        "word_wrap": "fa5s.align-left",
        "indent": "fa5s.indent",
        "zoom_in": "fa5s.search-plus",
        "zoom_out": "fa5s.search-minus",
        "run": "fa5s.play",
    }
    return qta.icon(names.get(name, "fa5s.file"), color=_QTA_COLORS.get(name, "#d4d4d4"))


class _PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._formats = {}
        for token, color in SYNTAX_COLORS.items():
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            style = SYNTAX_STYLES.get(token, {})
            if style.get("bold"):
                fmt.setFontWeight(QFont.Weight.Bold)
            if style.get("italic"):
                fmt.setFontItalic(True)
            self._formats[token] = fmt
        self._keywords = set(KEYWORDS)
        self._builtins = set(BUILTINS)
        self._constants = set(CONSTANTS)
        self._exceptions = set(EXCEPTIONS)

    def highlightBlock(self, text):
        self.setCurrentBlockState(0)
        start = 0
        prev = self.previousBlockState()
        if prev == 1 or prev == 2:
            delimiter = '"""' if prev == 1 else "'''"
            end = text.find(delimiter)
            if end == -1:
                self.setFormat(0, len(text), self._formats["string"])
                self.setCurrentBlockState(prev)
                return
            end += 3
            self.setFormat(0, end, self._formats["string"])
            start = end
        self._scan(text, start)

    def _scan(self, text, start):
        i = start
        n = len(text)
        expect_definition = False

        while i < n:
            c = text[i]

            if c.isspace():
                i += 1
                continue

            if c == "#":
                self.setFormat(i, n - i, self._formats["comment"])
                return

            if text.startswith('"""', i) or text.startswith("'''", i):
                delimiter = text[i:i + 3]
                state = 1 if delimiter == '"""' else 2
                end = text.find(delimiter, i + 3)
                if end == -1:
                    self.setFormat(i, n - i, self._formats["string"])
                    self.setCurrentBlockState(state)
                    return
                self.setFormat(i, end + 3 - i, self._formats["string"])
                i = end + 3
                expect_definition = False
                continue

            if c == '"' or c == "'":
                quote = c
                j = i + 1
                while j < n:
                    if text[j] == "\\":
                        j += 2
                        continue
                    if text[j] == quote:
                        j += 1
                        break
                    j += 1
                self.setFormat(i, j - i, self._formats["string"])
                i = j
                expect_definition = False
                continue

            if c == "@":
                j = i + 1
                while j < n and (text[j].isalnum() or text[j] in "._"):
                    j += 1
                if j > i + 1:
                    self.setFormat(i, j - i, self._formats["decorator"])
                    i = j
                    expect_definition = False
                    continue
                i += 1
                expect_definition = False
                continue

            if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
                j = i
                if c == "0" and i + 1 < n and text[i + 1] in "xXoObB":
                    j = i + 2
                    while j < n and (text[j].isalnum() or text[j] == "_"):
                        j += 1
                else:
                    while j < n and (text[j].isdigit() or text[j] in "._"):
                        j += 1
                    if j < n and text[j] in "eE":
                        j += 1
                        if j < n and text[j] in "+-":
                            j += 1
                        while j < n and text[j].isdigit():
                            j += 1
                self.setFormat(i, j - i, self._formats["number"])
                i = j
                expect_definition = False
                continue

            if c.isalpha() or c == "_":
                j = i
                while j < n and (text[j].isalnum() or text[j] == "_"):
                    j += 1
                word = text[i:j]
                k = j
                while k < n and text[k].isspace():
                    k += 1

                token = None
                if expect_definition:
                    token = "function"
                    expect_definition = False
                elif word in self._keywords:
                    token = "keyword"
                    if word in ("def", "class"):
                        expect_definition = True
                elif word in self._constants:
                    token = "constant"
                elif word in self._exceptions:
                    token = "exception"
                elif word in self._builtins:
                    token = "builtin"
                elif k < n and text[k] == "(":
                    token = "function"

                if token is not None:
                    self.setFormat(i, j - i, self._formats[token])
                i = j
                continue

            i += 1
            expect_definition = False


class _LineNumberArea(QWidget):
    def __init__(self, editor: "_CodeEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor.line_number_width(), 0)

    def paintEvent(self, event):
        self._editor.line_number_paint(event, self)

    def mouseMoveEvent(self, event: QMouseEvent):
        self._editor.line_number_mouse_move(event, self)

    def leaveEvent(self, event):
        self._editor.line_number_mouse_leave(event)


class _VcsBlameGutter(QWidget):
    def __init__(self, editor: "_CodeEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor.blame_gutter_width(), 0)

    def paintEvent(self, event):
        self._editor.blame_gutter_paint(event, self)


class _Minimap(QWidget):
    def __init__(self, editor: "_CodeEditor"):
        super().__init__(editor)
        self._editor = editor
        self.setFixedWidth(scale(110))

    def sizeHint(self):
        return QSize(scale(110), 0)

    def paintEvent(self, event):
        self._editor.minimap_paint(event, self)

    def mousePressEvent(self, event):
        self._editor.minimap_pressed(int(event.position().y()))
        event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._editor.minimap_dragged(int(event.position().y()))
            event.accept()


class _CodeEditor(QPlainTextEdit):
    MIN_FONT = 8
    MAX_FONT = 48
    DEFAULT_FONT = 12

    cursorMoved = pyqtSignal(int, int)
    _vcs_result_ready = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._font_size = self.DEFAULT_FONT
        self._wrap = False
        self._show_indent_guides = True
        self._show_blame = False

        self._vcs_result_ready.connect(self._on_vcs_result)

        self._vcs_git: _Git | None = None
        self._vcs_file_path: str = ""
        self._vcs_blame_data: dict[int, dict] = {}
        self._vcs_diff_data: dict[int, str] = {}
        self._vcs_status: str = ""
        self._vcs_branch: str = ""
        self._vcs_refreshing: bool = False
        self._vcs_pending: bool = False

        self._line_number = _LineNumberArea(self)
        self._blame_gutter = _VcsBlameGutter(self)
        self._minimap = _Minimap(self)
        self._minimap_cache = None
        self._minimap_cache_key = (-1, -1, -1, -1)

        self._remote_cursors: dict[str, dict] = {}
        self._ops_callback = None
        self._suppress_ops = False
        self._old_text = ""

        self.document().contentsChange.connect(self._on_contents_change)
        QTimer.singleShot(0, self._init_old_text)

        self._apply_font()
        self.setUndoRedoEnabled(True)
        self.setCursorWidth(scale(2))
        self.setMouseTracking(True)

        self._completer = QCompleter([], self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setModelSorting(QCompleter.ModelSorting.CaseInsensitivelySortedModel)
        self._completer.activated.connect(self._insert_completion)

        self.blockCountChanged.connect(self._update_line_number)
        self.updateRequest.connect(self._on_update_request)
        self.cursorPositionChanged.connect(self._emit_cursor)
        self.cursorPositionChanged.connect(self._update_current_line_highlight)
        self.textChanged.connect(self._invalidate_minimap)

        self._update_line_number()
        self._update_minimap()
        self._update_current_line_highlight()

    def set_indent_guides(self, on: bool):
        self._show_indent_guides = on
        self.viewport().update()

    def _indent_levels(self, block) -> int:
        text = block.text()
        column = 0
        for ch in text:
            if ch == " ":
                column += 1
            elif ch == "\t":
                column += 4
            else:
                break
        return column // 4

    def _indent_guide_records(self):
        viewport_h = self.viewport().height()
        first = self.firstVisibleBlock()
        if not first.isValid():
            return [], 0

        visible = []
        block = first
        while block.isValid():
            geom = self.blockBoundingGeometry(block).translated(self.contentOffset())
            top = int(geom.top())
            if top > viewport_h:
                break
            bottom = top + int(self.blockBoundingRect(block).height())
            visible.append([block, top, bottom])
            block = block.next()

        if not visible:
            return [], 0

        cursor_level = self._indent_levels(self.textCursor().block())

        limit = 200

        prev_level = None
        p = first.previous()
        steps = 0
        while p.isValid() and steps < limit:
            if p.text().strip():
                prev_level = self._indent_levels(p)
                break
            p = p.previous()
            steps += 1

        trailing_next = None
        t = visible[-1][0].next()
        steps = 0
        while t.isValid() and steps < limit:
            if t.text().strip():
                trailing_next = self._indent_levels(t)
                break
            t = t.next()
            steps += 1

        n = len(visible)
        raw = [self._indent_levels(v[0]) for v in visible]
        empty = [not v[0].text().strip() for v in visible]

        prev = [None] * n
        cur = prev_level
        for i in range(n):
            prev[i] = cur
            if not empty[i]:
                cur = raw[i]

        nxt = [None] * n
        cur = trailing_next
        for i in range(n - 1, -1, -1):
            nxt[i] = cur
            if not empty[i]:
                cur = raw[i]

        for i in range(n):
            if empty[i]:
                cand = [c for c in (prev[i], nxt[i]) if c is not None]
                visible[i].append(min(cand) if cand else 0)
            else:
                visible[i].append(raw[i])

        return visible, cursor_level

    def paintEvent(self, event):
        super().paintEvent(event)

        if self._show_indent_guides:
            self._draw_indent_guides(event)

        if self._vcs_diff_data:
            self._draw_diff_markers(event)

        if self._show_blame and self._vcs_blame_data:
            self._draw_blame_overlay(event)

        self._draw_remote_cursors(event)

    def _draw_indent_guides(self, event):
        tab_w = self.tabStopDistance()
        if tab_w <= 0:
            return

        fm = QFontMetrics(self.font())
        space_w = fm.horizontalAdvance(" ")
        if space_w <= 0:
            return

        records, cursor_level = self._indent_guide_records()
        if not records:
            return

        offset_x = self.contentOffset().x()
        width = self.viewport().width()
        rect_top = event.rect().top()
        rect_bottom = event.rect().bottom()
        guide_offset = 2 * space_w

        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        guide_color = QColor("#2f2f2f")
        active_color = QColor("#6b8cb3")

        for block, top, bottom, level in records:
            if level <= 0:
                continue

            seg_top = max(top, rect_top)
            seg_bottom = min(bottom, rect_bottom)
            if seg_top > seg_bottom:
                continue

            for L in range(1, level + 1):
                x = int(offset_x + L * tab_w - guide_offset)
                if x < 0 or x > width:
                    continue
                painter.setPen(active_color if L <= cursor_level else guide_color)
                painter.drawLine(x, seg_top, x, seg_bottom)

        painter.end()

    def _draw_diff_markers(self, event):
        viewport_h = self.viewport().height()
        first = self.firstVisibleBlock()
        if not first.isValid():
            return

        painter = QPainter(self.viewport())
        offset_x = self.contentOffset().x()
        marker_w = scale(3)

        block = first
        while block.isValid():
            line = block.blockNumber()
            status = self._vcs_diff_data.get(line, "")
            if status:
                geom = self.blockBoundingGeometry(block).translated(self.contentOffset())
                top = int(geom.top())
                bottom = top + int(self.blockBoundingRect(block).height())
                if top > viewport_h:
                    break
                if bottom >= event.rect().top():
                    if status == "added":
                        painter.fillRect(int(offset_x), top, marker_w, bottom - top, QColor("#3C9B3C"))
                    elif status == "modified":
                        painter.fillRect(int(offset_x), top, marker_w, bottom - top, QColor("#3C6E9B"))
                    elif status == "deleted":
                        painter.fillRect(int(offset_x), top + (bottom - top) // 2 - 1, marker_w, 2, QColor("#9B3C3C"))
            block = block.next()

        painter.end()

    def set_remote_cursors(self, cursors: dict[str, dict]):
        self._remote_cursors = dict(cursors)
        self._update_remote_extra_selections()
        self.viewport().update()

    def _update_remote_extra_selections(self):
        local = QTextEdit.ExtraSelection()
        local.format.setBackground(QColor("#282828"))
        local.format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
        local.cursor = self.textCursor()
        local.cursor.clearSelection()

        selections = [local]

        for peer_id, info in self._remote_cursors.items():
            if info["sel_anchor"] != info["sel_end"]:
                sel = QTextEdit.ExtraSelection()
                c = QColor(info["color"])
                c.setAlpha(60)
                sel.format.setBackground(c)
                tc = QTextCursor(self.document())
                tc.setPosition(info["sel_anchor"])
                tc.setPosition(info["sel_end"], QTextCursor.MoveMode.KeepAnchor)
                sel.cursor = tc
                selections.append(sel)

        self.setExtraSelections(selections)

    def _draw_remote_cursors(self, event):
        if not self._remote_cursors:
            return
        painter = QPainter(self.viewport())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        fm = painter.fontMetrics()

        for info in self._remote_cursors.values():
            color = QColor(info["color"])
            name = info["name"]
            pos = info["pos"]

            tc = QTextCursor(self.document())
            tc.setPosition(pos)
            rect = self.cursorRect(tc)
            x = rect.x()
            y_top = rect.y()

            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.drawLine(x, y_top, x, y_top + rect.height())

            label_w = fm.horizontalAdvance(name) + 6
            label_h = fm.height() + 2
            label_x = x
            label_y = y_top - label_h - 2

            bg = QColor(color)
            bg.setAlpha(200)
            painter.fillRect(label_x, label_y, label_w, label_h, bg)
            painter.setPen(Qt.GlobalColor.white)
            painter.drawText(label_x + 3, label_y + fm.ascent() + 1, name)

        painter.end()

    def _draw_blame_overlay(self, event):
        if not self._vcs_blame_data:
            return

        painter = QPainter(self.viewport())
        fm = painter.fontMetrics()
        text_color = QColor("#6b7888")

        first = self.firstVisibleBlock()
        if not first.isValid():
            painter.end()
            return

        viewport_w = self.viewport().width()
        viewport_h = self.viewport().height()

        block = first
        while block.isValid():
            line = block.blockNumber()
            info = self._vcs_blame_data.get(line)
            if info:
                geom = self.blockBoundingGeometry(block).translated(self.contentOffset())
                top = int(geom.top())
                if top > viewport_h:
                    break
                author = info.get("author", "?")
                when = info.get("time_str", "")
                label = f"{author} {when}"
                text_w = fm.horizontalAdvance(label)
                x = viewport_w - text_w - scale(8)
                if x > scale(40):
                    painter.setPen(text_color)
                    painter.drawText(x, top + fm.ascent(), label)
            block = block.next()

        painter.end()

    def _apply_font(self):
        font = QFont("Consolas", self._font_size)
        self.setFont(font)
        self.setTabStopDistance(QFontMetrics(font).horizontalAdvance(" ") * 4)
        self.setStyleSheet(
            "QPlainTextEdit { "
            f"background: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace; "
            f"font-size: {self._font_size}px; border: none; "
            "selection-background-color: #264f78; "
            "}"
        )
        self.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth if self._wrap
            else QPlainTextEdit.LineWrapMode.NoWrap
        )
        self._minimap_cache = None
        QTimer.singleShot(0, self._update_minimap)

    def _update_current_line_highlight(self):
        selections = []
        selection = QTextEdit.ExtraSelection()
        selection.format.setBackground(QColor("#282828"))
        selection.format.setProperty(QTextCharFormat.Property.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        selections.append(selection)
        self.setExtraSelections(selections)

    def _init_old_text(self):
        self._old_text = self.toPlainText()

    def _on_contents_change(self, position: int, chars_removed: int, chars_added: int):
        if self._suppress_ops or (chars_removed == 0 and chars_added == 0):
            return
        if self._ops_callback is None:
            return
        current = self.toPlainText()
        if chars_removed > 0 and position + chars_removed <= len(self._old_text):
            removed = self._old_text[position:position + chars_removed]
        else:
            removed = ""
        if chars_added > 0 and position + chars_added <= len(current):
            added = current[position:position + chars_added]
        else:
            added = ""
        self._old_text = current
        if removed or added:
            self._ops_callback(position, chars_removed, added)

    def set_ops_callback(self, cb):
        self._ops_callback = cb

    def line_number_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        fm = QFontMetrics(self.font())
        return 6 + int(fm.horizontalAdvance("9") * digits) + 6

    def blame_gutter_width(self) -> int:
        if not self._show_blame or not self._vcs_blame_data:
            return 0
        fm = QFontMetrics(self.font())
        max_w = 0
        for info in self._vcs_blame_data.values():
            author = info.get("author", "?")
            when = info.get("time_str", "")
            label = f"{author} {when}"
            w = fm.horizontalAdvance(label)
            if w > max_w:
                max_w = w
        return max_w + scale(16) if max_w > 0 else 0

    def _update_extra(self):
        lw = self.line_number_width()
        mw = self._minimap.width()
        if mw <= 0:
            mw = self._minimap.sizeHint().width()

        bw = self.blame_gutter_width()

        self._line_number.setFixedWidth(lw)
        self._blame_gutter.setFixedWidth(bw)
        self._minimap.setFixedWidth(mw)

        right_margin = mw + bw
        self.setViewportMargins(lw, 0, right_margin, 0)

        cr = self.contentsRect()
        self._line_number.setGeometry(cr.x(), cr.y(), lw, cr.height())
        blame_x = cr.x() + cr.width() - right_margin
        self._blame_gutter.setGeometry(blame_x, cr.y(), bw, cr.height())
        self._minimap.setGeometry(blame_x + bw, cr.y(), mw, cr.height())

        self._line_number.update()
        self._blame_gutter.update()
        self._minimap.update()

    def _update_line_number(self):
        self._update_extra()

    def _update_minimap(self):
        self._update_extra()

    def _invalidate_minimap(self):
        self._minimap_cache = None
        self._minimap.update()

    def _on_update_request(self, rect, dy):
        if dy:
            self._line_number.scroll(0, dy)
            self._blame_gutter.scroll(0, dy)
        else:
            self._line_number.update(0, rect.y(), self._line_number.width(), rect.height())
            self._blame_gutter.update(0, rect.y(), self._blame_gutter.width(), rect.height())
        self._minimap.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._minimap_cache = None
        self._update_line_number()
        self._update_minimap()

    def _minimap_line_nominal(self) -> int:
        return max(3, int(self._font_size * 0.45))

    def _rebuild_minimap(self):
        mw = max(1, self._minimap.width())
        mh = max(1, self._minimap.height())
        blocks = max(1, self.document().blockCount())
        key = (mw, mh, blocks, self._font_size)

        if self._minimap_cache is not None and self._minimap_cache_key == key:
            return self._minimap_cache

        pm = QPixmap(mw, mh)
        pm.fill(QColor("#161616"))

        painter = QPainter(pm)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        pad = scale(3)
        usable = max(0, mw - pad * 2)

        small_size = max(4, int(self._font_size * 0.42))
        mm_font = QFont("Consolas", small_size)
        painter.setFont(mm_font)

        fm = QFontMetrics(mm_font)
        ascent = fm.ascent()
        char_w = max(1, fm.horizontalAdvance("a"))

        nominal = self._minimap_line_nominal()
        text_color = QColor("#6b7888")
        bar_color = QColor("#4f5b66")

        if blocks * nominal <= mh:
            line_h = nominal
            draw_text = line_h >= ascent

            block = self.document().firstBlock()
            idx = 0
            while block.isValid():
                y = idx * line_h
                if y >= mh:
                    break

                raw = block.text()
                text = raw.strip()
                if text:
                    indent = len(raw) - len(raw.lstrip())
                    x = pad + min(usable // 2, int(indent * char_w * 0.5))
                    available = max(0, usable - (x - pad))

                    if draw_text:
                        available_chars = max(1, int(available / char_w))
                        painter.setPen(text_color)
                        painter.drawText(x, y + ascent, text[:available_chars])
                    else:
                        if available > 0:
                            length = min(available, max(2, int(len(text) * char_w * 0.6)))
                            if length > 0:
                                painter.setPen(bar_color)
                                painter.drawLine(x, y + line_h // 2, x + length, y + line_h // 2)

                block = block.next()
                idx += 1
        else:
            pixels_per_block = mh / blocks
            draw_text = pixels_per_block >= ascent

            if draw_text:
                block = self.document().firstBlock()
                idx = 0
                while block.isValid():
                    y = int(idx * pixels_per_block)
                    if y >= mh:
                        break

                    raw = block.text()
                    text = raw.strip()
                    if text:
                        indent = len(raw) - len(raw.lstrip())
                        x = pad + min(usable // 2, int(indent * char_w * 0.5))
                        available = max(0, usable - (x - pad))
                        available_chars = max(1, int(available / char_w))
                        painter.setPen(text_color)
                        painter.drawText(x, y + ascent, text[:available_chars])

                    block = block.next()
                    idx += 1
            else:
                for y in range(mh):
                    number = int(y * blocks / mh)
                    block = self.document().findBlockByNumber(number)
                    if not block.isValid():
                        continue

                    raw = block.text()
                    text = raw.strip()
                    if text:
                        indent = len(raw) - len(raw.lstrip())
                        x = pad + min(usable // 2, int(indent * char_w * 0.5))
                        available = max(0, usable - (x - pad))
                        if available > 0:
                            length = min(available, max(2, int(len(text) * char_w * 0.6)))
                            if length > 0:
                                painter.setPen(bar_color)
                                painter.drawLine(x, y, x + length, y)

        painter.end()

        self._minimap_cache = pm
        self._minimap_cache_key = key
        return pm

    def line_number_paint(self, event, area):
        painter = QPainter(area)
        painter.setFont(self.font())
        painter.fillRect(event.rect(), QColor("#181818"))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        cur_block = self.textCursor().blockNumber()
        fm = painter.fontMetrics()

        vcs_colors = {
            "added": QColor("#3C9B3C"),
            "modified": QColor("#3C6E9B"),
            "deleted": QColor("#9B3C3C"),
        }

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                line = block_number
                txt = str(line + 1)

                vcs_status = self._vcs_diff_data.get(line, "")

                if line == cur_block:
                    painter.fillRect(0, top, area.width(), int(self.blockBoundingRect(block).height()), QColor("#222222"))
                    painter.setPen(QColor("#c8c8c8"))
                elif vcs_status in vcs_colors:
                    painter.setPen(vcs_colors[vcs_status])
                else:
                    painter.setPen(QColor("#5a5a5a"))

                if vcs_status in vcs_colors:
                    marker_w = scale(3)
                    painter.fillRect(0, top, marker_w, int(self.blockBoundingRect(block).height()), vcs_colors[vcs_status])

                painter.drawText(0, top, area.width() - scale(4), fm.height(),
                                 Qt.AlignmentFlag.AlignRight, txt)

                if vcs_status:
                    painter.setPen(QColor("#5a5a5a"))
                    offset_x = area.width()
                    dot_size = scale(4)
                    painter.drawRect(offset_x - dot_size - scale(2), top + (fm.height() - dot_size) // 2, dot_size, dot_size)

            block = block.next()
            top = bottom
            if not block.isValid():
                break
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

        painter.end()

    def line_number_mouse_move(self, event: QMouseEvent, area):
        y = int(event.position().y())
        block = self._find_block_at_y(y)
        if block is not None:
            line = block.blockNumber()
            if line in self._vcs_blame_data:
                self._vcs_hover_line = line
                info = self._vcs_blame_data[line]
                author = info.get("author", "?")
                email = info.get("email", "")
                when = info.get("time_str", "")
                rev = info.get("short", "")
                msg = info.get("message", "")
                tip = f"{author} <{email}>  {when}\n{rev}  {msg}"
                self.setToolTip(tip)
            else:
                self._vcs_hover_line = -1
                self.setToolTip("")
        area.update()

    def line_number_mouse_leave(self, event):
        self._vcs_hover_line = -1
        self.setToolTip("")

    def _find_block_at_y(self, y: int) -> QTextBlock | None:
        block = self.firstVisibleBlock()
        while block.isValid():
            geom = self.blockBoundingGeometry(block).translated(self.contentOffset())
            top = int(geom.top())
            bottom = top + int(self.blockBoundingRect(block).height())
            if top <= y <= bottom:
                return block
            if top > y:
                return None
            block = block.next()
        return None

    def blame_gutter_paint(self, event, area):
        painter = QPainter(area)
        painter.fillRect(event.rect(), QColor("#1b1b1b"))
        painter.setFont(self.font())

        if not self._show_blame or not self._vcs_blame_data:
            painter.end()
            return

        text_color = QColor("#6b7888")
        fm = painter.fontMetrics()

        block = self.firstVisibleBlock()
        while block.isValid():
            line = block.blockNumber()
            info = self._vcs_blame_data.get(line)
            if info:
                geom = self.blockBoundingGeometry(block).translated(self.contentOffset())
                top = int(geom.top())
                author = info.get("author", "?")
                when = info.get("time_str", "")
                label = f"{author} {when}"
                painter.setPen(text_color)
                painter.drawText(scale(4), top + fm.ascent(), label)
            block = block.next()

        painter.end()

    def minimap_paint(self, event, area):
        painter = QPainter(area)
        painter.fillRect(event.rect(), QColor("#161616"))

        pm = self._rebuild_minimap()
        painter.drawPixmap(0, 0, pm)

        vsb = self.verticalScrollBar()
        if vsb is not None:
            total = vsb.maximum() + vsb.pageStep()
            if total > 0:
                y = int(vsb.value() * area.height() / total)
                h = int(vsb.pageStep() * area.height() / total)
            else:
                y = 0
                h = area.height()

            y = max(0, min(area.height() - 1, y))
            h = max(scale(8), min(area.height() - y, h))

            painter.setPen(QColor("#3a6ea5"))
            painter.setBrush(QColor(58, 110, 165, 60))
            painter.drawRect(0, y, area.width() - 1, h)

            painter.setPen(QColor("#222222"))
            painter.drawLine(0, 0, 0, area.height())

        painter.end()

    def minimap_pressed(self, y: int):
        self._minimap_scroll_to(y)

    def minimap_dragged(self, y: int):
        self._minimap_scroll_to(y)

    def _minimap_scroll_to(self, y: int):
        vsb = self.verticalScrollBar()
        if vsb is None:
            return

        total = vsb.maximum() + vsb.pageStep()
        if total <= 0:
            return

        mh = max(1, self._minimap.height())
        ratio = max(0.0, min(1.0, y / mh))
        target = int(ratio * total - vsb.pageStep() * 0.5)
        vsb.setValue(max(0, min(vsb.maximum(), target)))

    def _emit_cursor(self):
        QTimer.singleShot(0, self._report_cursor)

    def _report_cursor(self):
        line = self.textCursor().blockNumber() + 1
        col = self.textCursor().positionInBlock() + 1
        self.cursorMoved.emit(line, col)

    def set_font_size(self, size: int):
        self._font_size = max(self.MIN_FONT, min(self.MAX_FONT, size))
        self._apply_font()

    def zoom(self, delta: int):
        self.set_font_size(self._font_size + (1 if delta > 0 else -1))

    def set_wrap(self, enabled: bool):
        self._wrap = enabled
        self._apply_font()

    def refresh_completions(self):
        words = set(KEYWORDS) | set(BUILTINS) | set(CONSTANTS) | set(EXCEPTIONS)
        text = self.toPlainText()
        for match in re.finditer(r"[A-Za-z_]\w*", text):
            words.add(match.group(0))
        model = QStringListModel(sorted(words), self)
        self._completer.setModel(model)

    def _word_under_cursor(self) -> QTextCursor:
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        return cursor

    def _insert_completion(self, completion: str):
        cursor = self._word_under_cursor()
        extra = len(completion) - len(self._completer.completionPrefix())
        if extra > 0:
            cursor.insertText(completion[-extra:])
            self.setTextCursor(cursor)

    def _update_completer_popup(self):
        cursor = self._word_under_cursor()
        prefix = cursor.selectedText()
        if not prefix or prefix[0].isdigit():
            self._completer.popup().hide()
            return

        self._completer.setCompletionPrefix(prefix)
        if self._completer.completionCount() == 0:
            self._completer.popup().hide()
            return

        popup = self._completer.popup()
        if popup is None:
            return

        scroll = popup.verticalScrollBar()
        scroll_width = scroll.sizeHint().width() if scroll is not None else 0

        rect = self.cursorRect()
        rect.setWidth(popup.sizeHintForColumn(0) + scroll_width)
        self._completer.complete(rect)

    def keyPressEvent(self, event: QKeyEvent):
        popup = self._completer.popup()
        if popup.isVisible():
            if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return,
                               Qt.Key.Key_Tab, Qt.Key.Key_Escape):
                event.ignore()
                return

        if event.key() == Qt.Key.Key_Tab:
            if self.textCursor().hasSelection():
                self._indent_selected_lines()
            else:
                self.textCursor().insertText("    ")
            return

        if event.key() == Qt.Key.Key_Backtab:
            self._unindent_selected_lines()
            return

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._auto_indent(event)
            return

        if event.key() == Qt.Key.Key_Space and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.refresh_completions()
            self._update_completer_popup()
            return

        super().keyPressEvent(event)

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Space, Qt.Key.Key_Backspace):
            self.refresh_completions()

        if event.text() and event.text().isalnum() or event.key() == Qt.Key.Key_Period:
            self._update_completer_popup()

    def _auto_indent(self, event: QKeyEvent):
        cursor = self.textCursor()
        if cursor.hasSelection():
            cursor.insertText("\n")
            self.setTextCursor(cursor)
            return

        block = cursor.block()
        text = block.text()
        indent = len(text) - len(text.lstrip())
        indent_text = text[:indent]

        code = text.split("#", 1)[0].rstrip()
        if code.endswith(":"):
            indent_text += "    "

        cursor.insertText("\n" + indent_text)
        self.setTextCursor(cursor)

    def _indent_selected_lines(self):
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()

        cursor.setPosition(start)
        start_block = cursor.block().blockNumber()

        cursor.setPosition(end)
        end_block = cursor.block().blockNumber()
        if end_block > start_block and cursor.atBlockStart() and end != start:
            end_block -= 1

        if end_block < start_block:
            return

        cursor.beginEditBlock()
        for number in range(start_block, end_block + 1):
            block = self.document().findBlockByNumber(number)
            cursor.setPosition(block.position())
            cursor.insertText("    ")
        cursor.endEditBlock()

        cursor.setPosition(start)
        cursor.setPosition(end + 4 * (end_block - start_block + 1), QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)

    def _unindent_selected_lines(self):
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()

        cursor.setPosition(start)
        start_block = cursor.block().blockNumber()
        start_block_pos = self.document().findBlockByNumber(start_block).position()

        cursor.setPosition(end)
        end_block = cursor.block().blockNumber()
        if end_block > start_block and cursor.atBlockStart() and end != start:
            end_block -= 1

        if end_block < start_block:
            return

        cursor.beginEditBlock()
        removed_first = 0
        total_removed = 0

        for number in range(start_block, end_block + 1):
            block = self.document().findBlockByNumber(number)
            text = block.text()
            remove = 0

            if text.startswith("    "):
                remove = 4
            elif text.startswith("\t"):
                remove = 1
            else:
                while remove < 4 and remove < len(text) and text[remove] == " ":
                    remove += 1

            if remove:
                cursor.setPosition(block.position())
                cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, remove)
                cursor.removeSelectedText()

                if number == start_block:
                    removed_first = remove
                total_removed += remove

        cursor.endEditBlock()

        new_start = max(start - removed_first, start_block_pos)
        new_end = end - total_removed

        cursor.setPosition(new_start)
        cursor.setPosition(max(new_start, new_end), QTextCursor.MoveMode.KeepAnchor)
        self.setTextCursor(cursor)

    def focusInEvent(self, event):
        self._completer.setWidget(self)
        super().focusInEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)

    def vcs_set_git(self, git: _Git | None):
        self._vcs_git = git

    def vcs_set_file(self, path: str):
        self._vcs_file_path = path
        self.vcs_refresh()

    def vcs_refresh(self):
        self._vcs_blame_data = {}
        self._vcs_diff_data = {}
        self._vcs_status = ""
        self._vcs_branch = ""

        if not self._vcs_git or not self._vcs_git.available or not self._vcs_git.repo_root:
            self._update_extra()
            return

        if not self._vcs_file_path:
            self._update_extra()
            return

        try:
            rel = os.path.relpath(self._vcs_file_path, self._vcs_git.repo_root)
        except ValueError:
            self._update_extra()
            return
        rel = rel.replace("\\", "/")

        if self._vcs_refreshing:
            self._vcs_pending = True
            return
        self._vcs_refreshing = True
        threading.Thread(
            target=self._vcs_worker,
            args=(self._vcs_git, rel, self._vcs_file_path, self._show_blame),
            daemon=True,
        ).start()

    def _vcs_worker(self, git, rel: str, file_path: str, show_blame: bool):
        result: dict = {"branch": "", "status": "", "diff_data": {}, "blame_data": {}}
        try:
            result["branch"] = git.current_branch()

            rc, out, _ = git.status()
            if rc == 0:
                for entry in out.strip().split("\n"):
                    entry = entry.rstrip()
                    if not entry or len(entry) < 3:
                        continue
                    xy = entry[:2]
                    path = entry[3:]
                    if path == rel:
                        if xy[0] == "?" and xy[1] == "?":
                            result["status"] = "untracked"
                        elif xy[0] == "M" or xy[1] == "M" or xy[0] == "A":
                            result["status"] = "modified"
                        elif xy[0] == "D" or xy[1] == "D":
                            result["status"] = "deleted"
                        elif xy[0] == "U" or xy[1] == "U":
                            result["status"] = "conflict"
                        break

            diff_text = git.file_diff(rel)
            if diff_text:
                current = None
                for line in diff_text.split("\n"):
                    if line.startswith("@@ "):
                        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@", line)
                        if m:
                            current = int(m.group(1))
                    elif line.startswith("+"):
                        if current is not None:
                            result["diff_data"][current - 1] = "added"
                        current = (current or 0) + 1
                    elif line.startswith("-"):
                        pass
                    elif line.startswith(" "):
                        if current is not None:
                            current += 1

            if show_blame:
                if "GIT_AUTHOR_NAME" not in os.environ:
                    os.environ["GIT_AUTHOR_NAME"] = ""
                if "GIT_COMMITTER_NAME" not in os.environ:
                    os.environ["GIT_COMMITTER_NAME"] = ""
                blame_text = git.blame(file_path)
                if blame_text:
                    for bline in blame_text.strip().split("\n"):
                        if not bline.strip():
                            continue
                        parts = bline.split("\t")
                        if len(parts) >= 4:
                            try:
                                rev_info = parts[0].strip()
                                line_num_part = parts[2].strip().rstrip(")")
                                line_num = int(line_num_part.split()[0]) - 1
                                author = parts[1].strip()
                                time_str = parts[3].strip() if len(parts) > 3 else ""
                                result["blame_data"][line_num] = {
                                    "author": author,
                                    "short": rev_info[:7],
                                    "time_str": time_str,
                                    "email": "",
                                    "message": "",
                                }
                            except (ValueError, IndexError):
                                pass
        except Exception:
            pass
        self._vcs_result_ready.emit(result)

    def _on_vcs_result(self, result: dict):
        self._vcs_refreshing = False
        try:
            self._vcs_branch = result.get("branch", "")
            self._vcs_status = result.get("status", "")
            self._vcs_diff_data = result.get("diff_data", {})
            self._vcs_blame_data = result.get("blame_data", {})
            self._update_extra()
            self._line_number.update()
            self._blame_gutter.update()
            self.viewport().update()
        except RuntimeError:
            pass
        if self._vcs_pending:
            self._vcs_pending = False
            self.vcs_refresh()

    def vcs_set_blame_visible(self, visible: bool):
        self._show_blame = visible
        self.vcs_refresh()
        self._update_extra()
        self._blame_gutter.update()
        self.viewport().update()

    def vcs_is_blame_visible(self) -> bool:
        return self._show_blame

    def vcs_branch(self) -> str:
        return self._vcs_branch

    def vcs_file_status(self) -> str:
        return self._vcs_status


class _CloseableTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self._on_close_requested)
        self.setStyleSheet(
            "QTabWidget::pane { border: none; background: #1e1e1e; } "
            "QTabBar::tab { background: #252526; color: #d4d4d4; padding: 4px 10px; border: 1px solid #1e1e1e; border-bottom: none; } "
            "QTabBar::tab:selected { background: #1e1e1e; } "
            "QTabBar::tab:hover { background: #2a2d2e; }"
        )

    def _on_close_requested(self, index: int):
        widget = self.widget(index)
        if widget is not None:
            widget._close_self()

    def add_closeable_tab(self, widget, title: str) -> int:
        return self.addTab(widget, title)


class _ScriptTab(QWidget):
    closed = pyqtSignal(QWidget)

    def __init__(self, git: _Git | None = None, parent=None):
        super().__init__(parent)
        self._file_path: Optional[str] = None
        self._dirty = False
        self._vcs_git = git

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._editor = _CodeEditor()
        self._highlighter = _PythonHighlighter(self._editor.document())
        self._editor.textChanged.connect(self._on_text_changed)
        self._editor.refresh_completions()
        self._editor.vcs_set_git(git)

        layout.addWidget(self._editor)

    def set_font_size(self, size: int):
        self._editor.set_font_size(size)

    def set_wrap(self, enabled: bool):
        self._editor.set_wrap(enabled)

    def set_indent_guides(self, enabled: bool):
        self._editor.set_indent_guides(enabled)

    def _on_text_changed(self):
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _tab_title(self) -> str:
        base = os.path.basename(self._file_path) if self._file_path else "Untitled"
        suffix = "*" if self._dirty else ""
        return f"{base}{suffix}"

    def _update_title(self):
        tabs = self.parent()
        while tabs is not None and not isinstance(tabs, _CloseableTabWidget):
            tabs = tabs.parent()
        if tabs is not None:
            index = tabs.indexOf(self)
            if index >= 0:
                tabs.setTabText(index, self._tab_title())

    def _close_self(self):
        if self._dirty and not self._discard_prompt():
            return
        self.closed.emit(self)
        self.deleteLater()

    def _discard_prompt(self) -> bool:
        res = QMessageBox.question(
            self, "Unsaved Changes",
            f"Save changes to {self._tab_title()} before closing?",
            QMessageBox.StandardButton.Save |
            QMessageBox.StandardButton.Discard |
            QMessageBox.StandardButton.Cancel
        )
        if res == QMessageBox.StandardButton.Save:
            self.save()
            return not self._dirty
        return res == QMessageBox.StandardButton.Discard

    def set_content(self, text: str, from_remote: bool = False):
        if from_remote:
            parent_widget = self.parent()
            while parent_widget is not None and not isinstance(parent_widget, _ScriptEditorWidget):
                parent_widget = parent_widget.parent()
            if parent_widget and self._file_path:
                parent_widget.clear_pending_ops(self._file_path)
        self._editor.blockSignals(True)
        self._editor.setPlainText(text)
        self._editor.blockSignals(False)
        self._editor._old_text = text
        self._dirty = False
        self._editor.document().setModified(False)
        self._update_title()

    def open_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self._editor.blockSignals(True)
            self._editor.setPlainText(content)
            self._editor.blockSignals(False)
            self._editor._old_text = content
            self._file_path = path
            self._dirty = False
            self._editor.document().setModified(False)
            self._editor.vcs_set_file(path)
            self._update_title()
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open:\n{e}")

    def save(self):
        if self._file_path:
            try:
                with open(self._file_path, "w", encoding="utf-8") as f:
                    f.write(self._editor.toPlainText())
                self._dirty = False
                self._editor.document().setModified(False)
                self._editor.vcs_set_file(self._file_path)
                self._update_title()
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
        else:
            self.save_as()

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "",
            "Python Files (*.py)"
        )
        if path:
            if not path.endswith(".py"):
                path += ".py"
            self._file_path = path
            self.save()

    def new(self):
        self._editor.clear()
        self._editor._old_text = ""
        self._file_path = None
        self._dirty = False
        self._editor.document().setModified(False)
        self._editor._vcs_diff_data = {}
        self._editor._vcs_blame_data = {}
        self._editor._vcs_status = ""
        self._editor._vcs_branch = ""
        self._editor._update_extra()
        self._update_title()


class _ScriptEditorWidget(QWidget):
    tab_opened = pyqtSignal(str)
    tab_closed = pyqtSignal(str)
    tab_switched = pyqtSignal(str)
    collab_file_opened = pyqtSignal(str, str)
    collab_file_saved = pyqtSignal(str, str)
    collab_cursor_changed = pyqtSignal(str, int, int, int)
    collab_ops_ready = pyqtSignal(str, list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom = _CodeEditor.DEFAULT_FONT
        self._wrap = False

        self._remote_cursors: dict[str, dict[str, dict]] = {}
        self._ops_buffers: dict[str, list] = {}
        self._pending_local_ops: dict[str, list] = {}

        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.timeout.connect(self._do_auto_save)

        self._cursor_sync_timer = QTimer(self)
        self._cursor_sync_timer.setSingleShot(True)
        self._cursor_sync_timer.timeout.connect(self._send_cursor_sync)

        self._ops_flush_timer = QTimer(self)
        self._ops_flush_timer.setSingleShot(True)
        self._ops_flush_timer.timeout.connect(self._flush_and_send_ops)

        self._full_sync_timer = QTimer(self)
        self._full_sync_timer.timeout.connect(self._do_full_sync)
        self._full_sync_timer.start(10000)

        self._git = _Git()
        self._git_available = False
        self._vcs_refresh_timer = QTimer(self)
        self._vcs_refresh_timer.timeout.connect(self._vcs_timer_tick)
        self._vcs_refresh_timer.start(3000)

        self._setup_ui()
        self._try_detect_repo()

    def _try_detect_repo(self):
        eng = None
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
        except Exception:
            pass
        if eng:
            project_path = getattr(eng, "_project_path", "") or ""
            if project_path:
                self._git_available = self._git.detect(project_path)
                self._update_vcs_statusbar()
                return
        self._git_available = False
        self._update_vcs_statusbar()

    def _vcs_timer_tick(self):
        eng = None
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
        except Exception:
            pass
        if eng:
            project_path = getattr(eng, "_project_path", "") or ""
            if project_path:
                self._git_available = self._git.detect(project_path)
                tab = self._current_tab()
                if tab and tab._file_path and self._git_available:
                    tab._editor.vcs_refresh()
                self._update_vcs_statusbar()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._make_actions()

        menubar = QMenuBar()
        menubar.setNativeMenuBar(False)
        menubar.setStyleSheet(
            "QMenuBar { background: #1f1f1f; color: #d4d4d4; spacing: 2px; padding: 1px; } "
            "QMenuBar::item { padding: 3px 8px; border-radius: 3px; } "
            "QMenuBar::item:selected { background: #3e3e3e; } "
            "QMenu { background: #2d2d2d; color: #d4d4d4; border: 1px solid #444; } "
            "QMenu::item:selected { background: #264f78; }"
        )
        self._build_menubar(menubar)
        layout.addWidget(menubar)

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(*scale_xy(18, 18)))
        toolbar.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly if qta is None
            else Qt.ToolButtonStyle.ToolButtonIconOnly
        )
        toolbar.setStyleSheet(
            "QToolBar { background: #2d2d2d; border: none; spacing: 2px; padding: 2px; } "
            "QToolButton { background: transparent; border: none; padding: 3px; border-radius: 3px; } "
            "QToolButton:hover { background: #3e3e3e; } "
            "QToolButton:pressed { background: #505050; }"
        )

        toolbar.addAction(self._act_new)
        toolbar.addAction(self._act_open)
        toolbar.addAction(self._act_save)
        toolbar.addAction(self._act_save_as)
        toolbar.addSeparator()

        toolbar.addAction(self._act_undo)
        toolbar.addAction(self._act_redo)
        toolbar.addAction(self._act_cut)
        toolbar.addAction(self._act_copy)
        toolbar.addAction(self._act_paste)
        toolbar.addSeparator()

        toolbar.addAction(self._act_wrap)
        toolbar.addAction(self._act_indent)
        toolbar.addAction(self._act_zoom_out)

        self._zoom_combo = QComboBox()
        self._zoom_combo.setEditable(False)
        self._zoom_combo.addItems(["8", "9", "10", "11", "12", "14", "16", "18", "20", "24", "28"])
        self._zoom_combo.setCurrentText(str(self._zoom))
        self._zoom_combo.setFixedWidth(scale(56))
        self._zoom_combo.setToolTip("Font Size")
        self._zoom_combo.currentTextChanged.connect(
            lambda t: self._apply_zoom(0, int(t) if t.isdigit() else self._zoom)
        )
        toolbar.addWidget(self._zoom_combo)

        toolbar.addAction(self._act_zoom_in)
        toolbar.addSeparator()
        toolbar.addAction(self._act_run)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        self._vcs_branch_label = QLabel("")
        self._vcs_branch_label.setStyleSheet("color: #9a9a9a; padding: 0 4px; font-weight: bold;")
        toolbar.addWidget(self._vcs_branch_label)

        self._vcs_status_label = QLabel("")
        self._vcs_status_label.setStyleSheet("color: #9a9a9a; padding: 0 4px;")
        toolbar.addWidget(self._vcs_status_label)

        self._file_label = QLabel("  No file")
        self._file_label.setStyleSheet("color: #9a9a9a; padding: 0 6px;")
        toolbar.addWidget(self._file_label)

        layout.addWidget(toolbar)

        self._tabs = _CloseableTabWidget()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self._tabs)

        statusbar = QStatusBar()
        statusbar.setStyleSheet(
            "QStatusBar { background: #1f1f1f; color: #9a9a9a; padding: 1px 4px; } "
            "QStatusBar::item { border: none; }"
        )

        self._status_pos = QLabel("Ln 1, Col 1")
        self._status_info = QLabel("")

        statusbar.addPermanentWidget(self._status_info, 1)
        statusbar.addPermanentWidget(self._status_pos)

        layout.addWidget(statusbar)
        self._statusbar = statusbar

        self._new_tab()

    def _make_actions(self):
        self._act_new = QAction(_qta_icon("new"), "New", self)
        self._act_new.setToolTip("New Script")
        self._act_new.setShortcut(QKeySequence("Ctrl+N"))
        self._act_new.triggered.connect(self._new_tab)

        self._act_open = QAction(_qta_icon("open"), "Open", self)
        self._act_open.setToolTip("Open Script")
        self._act_open.setShortcut(QKeySequence("Ctrl+O"))
        self._act_open.triggered.connect(self._open_tab)

        self._act_save = QAction(_qta_icon("save"), "Save", self)
        self._act_save.setToolTip("Save")
        self._act_save.setShortcut(QKeySequence("Ctrl+S"))
        self._act_save.triggered.connect(self._save_current)

        self._act_save_as = QAction(_qta_icon("save_as"), "Save As", self)
        self._act_save_as.setToolTip("Save As")
        self._act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._act_save_as.triggered.connect(self._save_current_as)

        self._act_undo = QAction(_qta_icon("undo"), "Undo", self)
        self._act_undo.setToolTip("Undo (Ctrl+Z)")
        self._act_undo.triggered.connect(self._undo)

        self._act_redo = QAction(_qta_icon("redo"), "Redo", self)
        self._act_redo.setToolTip("Redo (Ctrl+Y)")
        self._act_redo.triggered.connect(self._redo)

        self._act_cut = QAction(_qta_icon("cut"), "Cut", self)
        self._act_cut.setToolTip("Cut")
        self._act_cut.triggered.connect(self._cut)

        self._act_copy = QAction(_qta_icon("copy"), "Copy", self)
        self._act_copy.setToolTip("Copy")
        self._act_copy.triggered.connect(self._copy)

        self._act_paste = QAction(_qta_icon("paste"), "Paste", self)
        self._act_paste.setToolTip("Paste")
        self._act_paste.triggered.connect(self._paste)

        self._act_wrap = QAction(_qta_icon("word_wrap"), "Word Wrap", self)
        self._act_wrap.setToolTip("Toggle Word Wrap")
        self._act_wrap.setCheckable(True)
        self._act_wrap.triggered.connect(self._toggle_wrap)

        self._act_indent = QAction(_qta_icon("indent"), "Indent Guides", self)
        self._act_indent.setToolTip("Toggle Indentation Guides")
        self._act_indent.setCheckable(True)
        self._act_indent.setChecked(True)
        self._act_indent.triggered.connect(self._toggle_indent)

        self._act_zoom_out = QAction(_qta_icon("zoom_out"), "Zoom Out", self)
        self._act_zoom_out.setToolTip("Zoom Out")
        self._act_zoom_out.triggered.connect(lambda: self._apply_zoom(-1))

        self._act_zoom_in = QAction(_qta_icon("zoom_in"), "Zoom In", self)
        self._act_zoom_in.setToolTip("Zoom In")
        self._act_zoom_in.triggered.connect(lambda: self._apply_zoom(1))

        self._act_run = QAction(_qta_icon("run"), "Run", self)
        self._act_run.setToolTip("Run Script")
        self._act_run.triggered.connect(self._run_current)

        self._act_blame = QAction("Annotate with Git Blame", self)
        self._act_blame.setToolTip("Toggle git blame annotations")
        self._act_blame.setCheckable(True)
        self._act_blame.setShortcut(QKeySequence("Ctrl+Shift+B"))
        self._act_blame.triggered.connect(self._toggle_blame)

        self._act_vcs_commit = QAction("Commit File...", self)
        self._act_vcs_commit.setToolTip("Commit current file")
        self._act_vcs_commit.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self._act_vcs_commit.triggered.connect(self._vcs_commit)

        self._act_vcs_diff = QAction("Diff with HEAD", self)
        self._act_vcs_diff.setToolTip("Show diff of current file against HEAD")
        self._act_vcs_diff.setShortcut(QKeySequence("Ctrl+Shift+D"))
        self._act_vcs_diff.triggered.connect(self._vcs_diff)

        self._act_vcs_history = QAction("Show History", self)
        self._act_vcs_history.setToolTip("Show git log for current file")
        self._act_vcs_history.setShortcut(QKeySequence("Ctrl+Shift+H"))
        self._act_vcs_history.triggered.connect(self._vcs_history)

        self._act_vcs_revert = QAction("Revert File...", self)
        self._act_vcs_revert.setToolTip("Discard changes and revert to HEAD")
        self._act_vcs_revert.triggered.connect(self._vcs_revert)

    def _build_menubar(self, menubar: QMenuBar):
        file_menu = menubar.addMenu("File")
        file_menu.addAction(self._act_new)
        file_menu.addAction(self._act_open)
        file_menu.addSeparator()
        file_menu.addAction(self._act_save)
        file_menu.addAction(self._act_save_as)

        edit_menu = menubar.addMenu("Edit")
        edit_menu.addAction(self._act_undo)
        edit_menu.addAction(self._act_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self._act_cut)
        edit_menu.addAction(self._act_copy)
        edit_menu.addAction(self._act_paste)

        view_menu = menubar.addMenu("View")

        self._act_wrap_m = QAction("Word Wrap", self)
        self._act_wrap_m.setCheckable(True)
        self._act_wrap_m.setChecked(self._wrap)
        self._act_wrap_m.triggered.connect(self._toggle_wrap_menu)
        view_menu.addAction(self._act_wrap_m)

        self._act_indent_m = QAction("Indentation Guides", self)
        self._act_indent_m.setCheckable(True)
        self._act_indent_m.setChecked(True)
        self._act_indent_m.triggered.connect(self._toggle_indent_menu)
        view_menu.addAction(self._act_indent_m)

        view_menu.addSeparator()
        view_menu.addAction(self._act_blame)

        act_zoom_in = QAction("Zoom In", self)
        act_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        act_zoom_in.triggered.connect(lambda: self._apply_zoom(1))
        view_menu.addAction(act_zoom_in)

        act_zoom_out = QAction("Zoom Out", self)
        act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        act_zoom_out.triggered.connect(lambda: self._apply_zoom(-1))
        view_menu.addAction(act_zoom_out)

        vcs_menu = menubar.addMenu("VCS")
        vcs_menu.addAction(self._act_vcs_commit)
        vcs_menu.addAction(self._act_vcs_diff)
        vcs_menu.addAction(self._act_vcs_history)
        vcs_menu.addSeparator()
        vcs_menu.addAction(self._act_vcs_revert)

        run_menu = menubar.addMenu("Run")
        self._act_run_m = QAction("Run Script", self)
        self._act_run_m.triggered.connect(self._run_current)
        run_menu.addAction(self._act_run_m)

    def _current_tab(self) -> Optional[_ScriptTab]:
        return self._tabs.currentWidget()

    def _active_editor(self) -> Optional[_CodeEditor]:
        tab = self._current_tab()
        return tab._editor if tab is not None else None

    def _bind_tab_signals(self, tab: _ScriptTab):
        tab._editor.cursorMoved.connect(self._on_cursor)
        tab._editor.modificationChanged.connect(self._on_modified)
        tab._editor.textChanged.connect(self._on_local_text_changed)
        tab._editor.cursorPositionChanged.connect(self._on_local_cursor_moved)
        tab._editor.set_ops_callback(lambda pos, removed, added: self._on_op_captured(tab, pos, removed, added))

    def _new_tab(self):
        tab = _ScriptTab(git=self._git)
        tab.closed.connect(self._on_tab_closed)
        tab.set_font_size(self._zoom)
        tab.set_wrap(self._wrap)
        self._bind_tab_signals(tab)

        self._tabs.add_closeable_tab(tab, "Untitled")
        self._tabs.setCurrentWidget(tab)

        self._update_label()
        self._update_vcs_statusbar()
        self._on_cursor(1, 1)
        self.tab_opened.emit("")

    def open_script(self, path: str):
        for i in range(self._tabs.count()):
            existing = self._tabs.widget(i)
            if existing._file_path == path:
                self._tabs.setCurrentWidget(existing)
                return

        tab = _ScriptTab(git=self._git)
        tab.closed.connect(self._on_tab_closed)
        tab.set_font_size(self._zoom)
        tab.set_wrap(self._wrap)
        self._bind_tab_signals(tab)

        tab.open_file(path)
        self._tabs.add_closeable_tab(tab, tab._tab_title())
        self._tabs.setCurrentWidget(tab)

        self._update_label()
        self._update_vcs_statusbar()
        self._on_cursor(1, 1)
        self.tab_opened.emit(path)
        self.collab_file_opened.emit(path, tab._editor.toPlainText())

    def _open_tab(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", "",
            "Python Files (*.py);;All Files (*)"
        )
        if path:
            self.open_script(path)

    def _save_current(self):
        tab = self._current_tab()
        if tab is not None:
            tab.save()
            self._update_label()
            self._update_vcs_statusbar()
            if tab._file_path:
                content = tab._editor.toPlainText()
                self.collab_file_saved.emit(tab._file_path, content)

    def _save_current_as(self):
        tab = self._current_tab()
        if tab is not None:
            tab.save_as()
            self._update_label()
            self._update_vcs_statusbar()

    def _undo(self):
        ed = self._active_editor()
        if ed is not None and ed.document().isUndoAvailable():
            ed.undo()

    def _redo(self):
        ed = self._active_editor()
        if ed is not None and ed.document().isRedoAvailable():
            ed.redo()

    def _cut(self):
        ed = self._active_editor()
        if ed is not None:
            ed.cut()

    def _copy(self):
        ed = self._active_editor()
        if ed is not None:
            ed.copy()

    def _paste(self):
        ed = self._active_editor()
        if ed is not None:
            ed.paste()

    def _toggle_wrap(self, checked: bool):
        self._wrap = checked
        self._act_wrap.setChecked(checked)
        if hasattr(self, "_act_wrap_m"):
            self._act_wrap_m.setChecked(checked)
        for i in range(self._tabs.count()):
            self._tabs.widget(i).set_wrap(self._wrap)

    def _toggle_wrap_menu(self, checked: bool):
        self._toggle_wrap(checked)

    def _toggle_indent(self, checked: bool):
        self._act_indent.setChecked(checked)
        if hasattr(self, "_act_indent_m"):
            self._act_indent_m.setChecked(checked)
        for i in range(self._tabs.count()):
            self._tabs.widget(i).set_indent_guides(checked)

    def _toggle_indent_menu(self, checked: bool):
        self._toggle_indent(checked)

    def _toggle_blame(self, checked: bool):
        for i in range(self._tabs.count()):
            tab = self._tabs.widget(i)
            if isinstance(tab, _ScriptTab):
                tab._editor.vcs_set_blame_visible(checked)

    def _apply_zoom(self, delta: int, size: int = 0):
        if delta != 0:
            self._zoom = max(_CodeEditor.MIN_FONT, min(_CodeEditor.MAX_FONT, self._zoom + delta))
        else:
            self._zoom = max(_CodeEditor.MIN_FONT, min(_CodeEditor.MAX_FONT, size))

        self._zoom_combo.setCurrentText(str(self._zoom))
        for i in range(self._tabs.count()):
            self._tabs.widget(i).set_font_size(self._zoom)

    def _run_current(self):
        tab = self._current_tab()
        if tab is None:
            return

        if tab._dirty or not tab._file_path:
            tab.save()

        if tab._file_path and not tab._dirty:
            try:
                cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                subprocess.Popen([sys.executable, tab._file_path], cwd=cwd)
            except Exception as e:
                QMessageBox.critical(self, "Run Error", f"Failed to run:\n{e}")

    def _vcs_commit(self):
        tab = self._current_tab()
        if not tab or not tab._file_path:
            return
        if not self._git_available or not self._git.repo_root:
            QMessageBox.information(self, "No Repository", "No git repository detected.")
            return

        try:
            rel = os.path.relpath(tab._file_path, self._git.repo_root)
        except ValueError:
            return
        rel = rel.replace("\\", "/")

        rc, _, _ = self._git.add([rel])
        if rc != 0:
            QMessageBox.warning(self, "Stage Failed", "Failed to stage file.")
            return

        msg, ok = QInputDialog.getText(self, "Commit", "Commit message:")
        if not ok or not msg.strip():
            self._git.unstage([rel])
            return

        rc, out, err = self._git.commit(msg.strip())
        if rc == 0:
            QMessageBox.information(self, "Committed", f"Committed:\n{msg.strip()}")
            tab._editor.vcs_refresh()
            self._update_vcs_statusbar()
        else:
            QMessageBox.critical(self, "Commit Failed", f"Error:\n{err or out}")

    def _vcs_diff(self):
        tab = self._current_tab()
        if not tab or not tab._file_path:
            return
        if not self._git_available or not self._git.repo_root:
            return

        try:
            rel = os.path.relpath(tab._file_path, self._git.repo_root)
        except ValueError:
            return
        rel = rel.replace("\\", "/")

        diff = self._git.file_diff(rel)
        if not diff:
            diff = self._git.file_diff(rel, staged=True)
        if not diff:
            diff = "No changes against HEAD"

        from editor.panels.vcs_panel import _DiffView
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Diff: {os.path.basename(tab._file_path)}")
        dlg.setMinimumSize(scale(700), scale(500))
        layout = QVBoxLayout(dlg)
        diff_view = _DiffView()
        diff_view.show_diff(diff)
        layout.addWidget(diff_view)
        dlg.exec()

    def _vcs_history(self):
        tab = self._current_tab()
        if not tab or not tab._file_path:
            return
        if not self._git_available or not self._git.repo_root:
            return

        try:
            rel = os.path.relpath(tab._file_path, self._git.repo_root)
        except ValueError:
            return
        rel = rel.replace("\\", "/")

        rc, out, _ = self._git.run_sync("log", "--oneline", "--decorate", "--", rel, timeout=15)
        if rc != 0 or not out.strip():
            QMessageBox.information(self, "History", "No history for this file.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"History: {os.path.basename(tab._file_path)}")
        dlg.setMinimumSize(scale(600), scale(400))
        layout = QVBoxLayout(dlg)
        te = QPlainTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Courier New", 10))
        te.setPlainText(out)
        layout.addWidget(te)
        dlg.exec()

    def _vcs_revert(self):
        tab = self._current_tab()
        if not tab or not tab._file_path:
            return
        if not self._git_available or not self._git.repo_root:
            return

        reply = QMessageBox.question(
            self, "Revert File",
            f"Discard all changes to {os.path.basename(tab._file_path)}?\n"
            f"This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            rel = os.path.relpath(tab._file_path, self._git.repo_root)
        except ValueError:
            return
        rel = rel.replace("\\", "/")

        rc, _, err = self._git.restore([rel])
        if rc == 0:
            tab.open_file(tab._file_path)
            self._update_vcs_statusbar()
        else:
            QMessageBox.critical(self, "Revert Failed", f"Error:\n{err}")

    def _on_tab_changed(self, index: int):
        self._update_label()
        self._update_vcs_statusbar()
        ed = self._active_editor()
        if ed is not None:
            line = ed.textCursor().blockNumber() + 1
            col = ed.textCursor().positionInBlock() + 1
            self._on_cursor(line, col)
        tab = self._current_tab()
        if tab is not None:
            self.tab_switched.emit(tab._file_path or "")
            fp = tab._file_path or ""
            if fp and fp in self._remote_cursors:
                tab._editor.set_remote_cursors(self._remote_cursors[fp])
            elif hasattr(tab, '_editor'):
                tab._editor.set_remote_cursors({})

    def _on_tab_closed(self, tab: QWidget):
        path = tab._file_path or ""
        self._remote_cursors.pop(path, None)
        index = self._tabs.indexOf(tab)
        if index >= 0:
            self._tabs.removeTab(index)
        if self._tabs.count() == 0:
            self._new_tab()
        self.tab_closed.emit(path)

    def _on_cursor(self, line: int, col: int):
        self._status_pos.setText(f"Ln {line}, Col {col}")

    def _on_modified(self, modified: bool):
        tab = self._current_tab()
        if tab is not None:
            self._status_info.setText("Modified" if modified else "")

    def _on_local_text_changed(self):
        self._auto_save_timer.start(2000)

    def _do_auto_save(self):
        tab = self._current_tab()
        if tab is None or not tab._file_path or not tab._dirty:
            return
        try:
            with open(tab._file_path, "w", encoding="utf-8") as f:
                f.write(tab._editor.toPlainText())
            tab._dirty = False
            tab._editor.document().setModified(False)
            tab._editor.vcs_set_file(tab._file_path)
            tab._update_title()
            if tab._file_path:
                self.collab_file_saved.emit(tab._file_path, tab._editor.toPlainText())
        except Exception as e:
            pass

    def _on_op_captured(self, tab, pos: int, removed: int, added: str):
        path = tab._file_path
        if not path:
            return
        self._pending_local_ops.setdefault(path, []).append((pos, removed, added))
        self._ops_buffers.setdefault(path, []).append((pos, removed, added))
        if not self._ops_flush_timer.isActive():
            self._ops_flush_timer.start(100)

    def _flush_and_send_ops(self):
        for path, buffer in list(self._ops_buffers.items()):
            if buffer:
                batch = list(buffer)
                buffer.clear()
                self.collab_ops_ready.emit(path, batch)

    def apply_remote_ops(self, path: str, ops: list):
        for i in range(self._tabs.count()):
            tab = self._tabs.widget(i)
            if isinstance(tab, _ScriptTab) and tab._file_path == path:
                editor = tab._editor
                pending = list(self._pending_local_ops.get(path, []))
                transformed = []
                for rop in ops:
                    for lop in pending:
                        rop = _ot_transform_op(rop, lop)
                    transformed.append(rop)
                editor._suppress_ops = True
                editor.document().blockSignals(True)
                for pos, removed, added in transformed:
                    if pos < 0:
                        continue
                    text_len = len(editor.toPlainText())
                    if removed and removed > 0 and pos + removed > text_len:
                        continue
                    tc = editor.textCursor()
                    tc.setPosition(pos)
                    if removed and removed > 0:
                        end = min(pos + removed, text_len)
                        tc.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
                    tc.insertText(added)
                editor.document().blockSignals(False)
                editor._old_text = editor.toPlainText()
                editor._suppress_ops = False
                new_pending = []
                for lop in pending:
                    for rop in transformed:
                        lop = _ot_transform_op(lop, rop)
                    new_pending.append(lop)
                self._pending_local_ops[path] = new_pending
                return

    def clear_pending_ops(self, path: str):
        self._pending_local_ops.pop(path, None)
        self._ops_buffers.pop(path, None)

    def _do_full_sync(self):
        tab = self._current_tab()
        if tab and tab._file_path:
            self.clear_pending_ops(tab._file_path)
            self.collab_file_saved.emit(tab._file_path, tab._editor.toPlainText())

    def _on_local_cursor_moved(self):
        self._cursor_sync_timer.start(50)

    def _send_cursor_sync(self):
        tab = self._current_tab()
        if tab is None or not tab._file_path:
            return
        cursor = tab._editor.textCursor()
        pos = cursor.position()
        self.collab_cursor_changed.emit(tab._file_path, pos, cursor.anchor(), pos)

    def update_remote_cursor(self, peer_id: str, path: str, pos: int,
                              sel_anchor: int, sel_end: int,
                              color: list[float], name: str):
        if not path:
            return
        cursors_for_path = self._remote_cursors.setdefault(path, {})
        cursors_for_path[peer_id] = {
            "pos": pos,
            "sel_anchor": sel_anchor,
            "sel_end": sel_end,
            "color": QColor.fromRgbF(*color[:3]),
            "name": name,
        }
        tab = self._current_tab()
        if tab and tab._file_path == path:
            tab._editor.set_remote_cursors(cursors_for_path)

    def _update_label(self):
        tab = self._current_tab()
        if tab is None:
            self._file_label.setText("  No file")
        elif tab._file_path:
            self._file_label.setText(f"  {os.path.basename(tab._file_path)}")
        else:
            self._file_label.setText("  Untitled Script")

    def _update_vcs_statusbar(self):
        branch = ""
        status = ""
        if self._git_available and self._git.repo_root:
            branch = self._git.current_branch()
            if not branch:
                branch = ""

            tab = self._current_tab()
            if isinstance(tab, _ScriptTab):
                ed = tab._editor
                file_status = ed.vcs_file_status()
                if file_status:
                    status_labels = {
                        "modified": "● modified",
                        "untracked": "● untracked",
                        "deleted": "● deleted",
                        "conflict": "● conflict",
                    }
                    status = status_labels.get(file_status, file_status)

        if branch:
            self._vcs_branch_label.setText(f"  [{branch}]  ")
            self._vcs_branch_label.show()
            self._vcs_status_label.setText(status)
            self._vcs_status_label.show()
        else:
            self._vcs_branch_label.hide()
            self._vcs_status_label.hide()


class ScriptEditorPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Script Editor", parent)
        self._engine = engine
        self.setObjectName("ScriptEditorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setMinimumWidth(200)

        self._script_widget = _ScriptEditorWidget()
        self.setWidget(self._script_widget)
