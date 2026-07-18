# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import keyword
from typing import TYPE_CHECKING, Optional
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout,
                              QToolBar, QToolButton, QLabel, QFileDialog,
                              QPlainTextEdit, QMessageBox, QTabWidget,
                              QSizePolicy, QComboBox, QFrame, QMenuBar,
                              QStatusBar, QCompleter, QMenu, QApplication)
from PyQt6.QtCore import Qt, QRegularExpression, pyqtSignal, QSize, QStringListModel, QTimer
from PyQt6.QtGui import (QColor, QFont, QFontMetrics, QKeyEvent, QWheelEvent,
                         QSyntaxHighlighter, QTextCharFormat, QTextCursor,
                          QAction, QIcon, QTextDocument, QKeySequence, QPainter, QPixmap)
if TYPE_CHECKING:
    from core.engine.engine import Engine

try:
    import qtawesome as qta
except ImportError:
    qta = None
from core.config.editor_scale import scale, scale_xy


_PY_KEYWORDS = list(keyword.kwlist)
_PY_BUILTINS = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable",
    "chr", "classmethod", "compile", "complex", "dict", "dir", "divmod", "enumerate",
    "eval", "filter", "float", "format", "frozenset", "getattr", "globals", "hasattr",
    "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass", "iter",
    "len", "list", "locals", "map", "max", "memoryview", "min", "next", "object",
    "oct", "open", "ord", "pow", "print", "property", "range", "repr", "reversed",
    "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type", "vars", "zip", "True", "False", "None", "self", "Exception",
    "ValueError", "TypeError", "KeyError", "IndexError", "RuntimeError", "AttributeError",
    "StopIteration", "FileNotFoundError",
]


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
    "zoom_in": "#d4d4d4",
    "zoom_out": "#d4d4d4",
    "run": "#9ccc65",
}


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
        "zoom_in": "fa5s.search-plus",
        "zoom_out": "fa5s.search-minus",
        "run": "fa5s.play",
    }
    return qta.icon(names.get(name, "fa5s.file"), color=_QTA_COLORS.get(name, "#d4d4d4"))


class _PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rules = []

        kw_fmt = QTextCharFormat()
        kw_fmt.setForeground(QColor("#569cd6"))
        kw_fmt.setFontWeight(QFont.Weight.Bold)
        keywords = (r'\b(def|class|return|if|elif|else|for|while|import|from|'
                    r'try|except|finally|with|as|yield|lambda|pass|break|'
                    r'continue|and|or|not|in|is|True|False|None|self|print|'
                    r'raise|del|global|nonlocal|assert|async|await)\b')
        self._rules.append((QRegularExpression(keywords), kw_fmt))

        builtins_fmt = QTextCharFormat()
        builtins_fmt.setForeground(QColor("#4ec9b0"))
        builtins = r'\b(int|float|str|bool|list|dict|tuple|set|type|len|range|enumerate|zip|map|filter|super|property|staticmethod|classmethod|isinstance|issubclass|hasattr|getattr|setattr|open|abs|min|max|sum|sorted|reversed|any|all|iter|next|object|Exception|ValueError|TypeError|KeyError|IndexError)\b'
        self._rules.append((QRegularExpression(builtins), builtins_fmt))

        string_fmt = QTextCharFormat()
        string_fmt.setForeground(QColor("#ce9178"))
        self._rules.append((QRegularExpression(r'""".*?"""|\'\'\'.*?\'\'\'|".*?"|\'.*?\''), string_fmt))

        number_fmt = QTextCharFormat()
        number_fmt.setForeground(QColor("#b5cea8"))
        self._rules.append((QRegularExpression(r'\b\d+\.?\d*\b'), number_fmt))

        comment_fmt = QTextCharFormat()
        comment_fmt.setForeground(QColor("#6a9955"))
        comment_fmt.setFontItalic(True)
        self._rules.append((QRegularExpression(r'#[^\n]*'), comment_fmt))

        decorator_fmt = QTextCharFormat()
        decorator_fmt.setForeground(QColor("#dcdcaa"))
        self._rules.append((QRegularExpression(r'@\w+'), decorator_fmt))

        func_fmt = QTextCharFormat()
        func_fmt.setForeground(QColor("#dcdcaa"))
        self._rules.append((QRegularExpression(r'\b\w+(?=\()'), func_fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self._rules:
            match_it = pattern.globalMatch(text)
            while match_it.hasNext():
                match = match_it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class _LineNumberArea(QWidget):
    def __init__(self, editor: "_CodeEditor"):
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self):
        return QSize(self._editor.line_number_width(), 0)

    def paintEvent(self, event):
        self._editor.line_number_paint(event, self)


class _Minimap(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, editor: "_CodeEditor"):
        super().__init__(editor)
        self._editor = editor
        self.setMinimumWidth(scale(90))
        self.setMaximumWidth(scale(140))

    def sizeHint(self):
        return QSize(scale(110), 0)

    def paintEvent(self, event):
        self._editor.minimap_paint(event, self)

    def mousePressEvent(self, event):
        ratio = event.position().y() / self.height()
        self.clicked.emit(int(ratio * self._editor.document().blockCount()))


class _CodeEditor(QPlainTextEdit):
    MIN_FONT = 8
    MAX_FONT = 48
    DEFAULT_FONT = 12

    cursorMoved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._font_size = self.DEFAULT_FONT
        self._wrap = False
        self._apply_font()
        self.setTabStopDistance(QFontMetrics(QFont("Consolas", 10)).horizontalAdvance(' ') * 4)
        self.setUndoRedoEnabled(True)

        self._completer = QCompleter([], self)
        self._completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self._completer.setWidget(self)
        self._completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        self._completer.setModelSorting(QCompleter.ModelSorting.CaseInsensitivelySortedModel)
        self._completer.activated.connect(self._insert_completion)

        self._line_number = _LineNumberArea(self)
        self._minimap = _Minimap(self)
        self._minimap.clicked.connect(self._goto_line)

        self._minimap_cache = None
        self._minimap_cache_key = (-1, -1, -1)

        self.blockCountChanged.connect(self._update_line_number)
        self.updateRequest.connect(self._on_update_request)
        self.cursorPositionChanged.connect(self._emit_cursor)
        self.textChanged.connect(self._invalidate_minimap)

        self._update_line_number()
        self._update_minimap()

    def _apply_font(self):
        self.setStyleSheet(f"""
            QPlainTextEdit {{
                background: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, monospace;
                font-size: {self._font_size}px;
                border: none;
                selection-background-color: #264f78;
            }}
        """)
        self.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth if self._wrap
            else QPlainTextEdit.LineWrapMode.NoWrap)
        QTimer.singleShot(0, self._update_minimap)

    def line_number_width(self) -> int:
        digits = max(2, len(str(max(1, self.blockCount()))))
        fm = QFontMetrics(QFont("Consolas", self._font_size))
        return 6 + int(fm.horizontalAdvance('9') * digits) + 6

    def _update_extra(self):
        lw = self.line_number_width()
        mw = self._minimap.width()
        self._line_number.setFixedWidth(lw)
        self.setViewportMargins(lw, 0, mw, 0)
        cr = self.contentsRect()
        self._line_number.setGeometry(cr.x(), cr.y(), lw, cr.height())
        self._minimap.setGeometry(cr.x() + cr.width() - mw, cr.y(), mw, cr.height())
        self._line_number.update()
        self._minimap.update()

    def _update_line_number(self):
        self._update_extra()

    def _update_minimap(self):
        self._update_extra()

    def _invalidate_minimap(self):
        self._minimap_cache = None

    def _on_update_request(self, rect, dy):
        if dy:
            self._line_number.scroll(0, dy)
        else:
            self._line_number.update(0, rect.y(), self._line_number.width(), rect.height())
        self._minimap.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._minimap_cache = None
        self._update_line_number()
        self._update_minimap()

    def _line_number_geometry(self):
        cr = self.contentsRect()
        return cr.x(), cr.y(), self._line_number.width(), cr.height()

    def _minimap_geometry(self):
        cr = self.contentsRect()
        return cr.x() + cr.width() - self._minimap.width(), cr.y(), self._minimap.width(), cr.height()

    def _rebuild_minimap(self):
        mw = self._minimap.width()
        mh = max(1, self._minimap.height())
        blocks = self.document().blockCount()
        line_h = max(1, int(self._font_size * 0.5))
        scale_y = min(1.0, mh / max(1, blocks * line_h))
        key = (mw, mh, blocks)
        if self._minimap_cache is not None and self._minimap_cache_key == key:
            return self._minimap_cache, scale_y, line_h
        pm = QPixmap(mw, mh)
        pm.fill(QColor("#161616"))
        painter = QPainter(pm)
        mm_font = QFont("Consolas", max(4, int(self._font_size * 0.42)))
        painter.setFont(mm_font)
        fm = painter.fontMetrics()
        ascent = fm.ascent()
        pad = 3
        block = self.document().firstBlock()
        cur_block = self.textCursor().blockNumber()
        idx = 0
        while block.isValid():
            y = int(idx * line_h * scale_y)
            if y > mh:
                break
            text = block.text().strip()
            if not text:
                block = block.next()
                idx += 1
                continue
            painter.setPen(QColor("#9ccc65") if block.blockNumber() == cur_block else QColor("#6a7888"))
            painter.drawText(pad, y + ascent, text[:40])
            block = block.next()
            idx += 1
        painter.end()
        self._minimap_cache = pm
        self._minimap_cache_key = key
        return pm, scale_y, line_h

    def line_number_paint(self, event, area):
        painter = QPainter(area)
        painter.fillRect(event.rect(), QColor("#181818"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())
        cur_block = self.textCursor().blockNumber()
        fm = painter.fontMetrics()
        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                txt = str(block_number + 1)
                if block_number == cur_block:
                    painter.fillRect(0, top, area.width(), int(self.blockBoundingRect(block).height()), QColor("#222"))
                    painter.setPen(QColor("#c8c8c8"))
                else:
                    painter.setPen(QColor("#5a5a5a"))
                painter.drawText(0, top, area.width() - scale(4), fm.height(),
                                 Qt.AlignmentFlag.AlignRight, txt)
            block = block.next()
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1
        painter.end()

    def minimap_paint(self, event, area):
        painter = QPainter(area)
        painter.fillRect(event.rect(), QColor("#161616"))

        pm, scale_y, line_h = self._rebuild_minimap()
        painter.drawPixmap(0, 0, pm)

        cur_y = int(self.textCursor().blockNumber() * line_h * scale_y)
        painter.fillRect(0, cur_y - scale(1), area.width(), max(2, int(line_h * scale_y)), QColor(156, 204, 101, 60))

        vsb = self.verticalScrollBar()
        if vsb is not None and self.document().blockCount() > 0:
            doc_h = max(1, self.document().blockCount() * line_h)
            y_top = int(self.contentOffset().y() / doc_h * area.height())
            view_h = int(self.viewport().height() / doc_h * area.height())
            y_top = max(0, min(area.height() - 1, y_top))
            view_h = max(scale(6), min(area.height() - y_top, view_h))
            painter.setPen(QColor("#3a6ea5"))
            painter.setBrush(QColor(58, 110, 165, 70))
            painter.drawRect(0, y_top, area.width() - 1, view_h)

        painter.end()

    def _goto_line(self, line: int):
        block = self.document().findBlockByNumber(max(0, line - 1))
        if block.isValid():
            cursor = self.textCursor()
            cursor.setPosition(block.position())
            self.setTextCursor(cursor)
            self.centerCursor()

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
        words = set(_PY_KEYWORDS) | set(_PY_BUILTINS)
        text = self.toPlainText()
        for m in __import__("re").finditer(r"[A-Za-z_]\w*", text):
            words.add(m.group(0))
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
        rect = self.cursorRect()
        rect.setWidth(self._completer.popup().sizeHintForColumn(0)
                      + self._completer.popup().verticalScrollBar().sizeHint().width())
        self._completer.complete(rect)

    def keyPressEvent(self, event: QKeyEvent):
        if self._completer.popup().isVisible():
            if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return,
                               Qt.Key.Key_Tab, Qt.Key.Key_Escape):
                event.ignore()
                return
        if event.key() == Qt.Key.Key_Tab:
            cursor = self.textCursor()
            cursor.insertText("    ")
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

    def focusInEvent(self, event):
        self._completer.setWidget(self)
        super().focusInEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom(event.angleDelta().y())
            event.accept()
            return
        super().wheelEvent(event)


class _CloseableTabWidget(QTabWidget):
    tabCloseRequested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabCloseRequested.connect(self._on_close_requested)

    def _on_close_requested(self, index: int):
        widget = self.widget(index)
        if widget is not None:
            widget._close_self()

    def add_closeable_tab(self, widget, title: str) -> int:
        return self.addTab(widget, title)


class _ScriptTab(QWidget):
    closed = pyqtSignal(QWidget)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_path: Optional[str] = None
        self._dirty = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._editor = _CodeEditor()
        self._highlighter = _PythonHighlighter(self._editor.document())
        self._editor.textChanged.connect(self._on_text_changed)
        self._editor.refresh_completions()
        layout.addWidget(self._editor)

    def set_font_size(self, size: int):
        self._editor.set_font_size(size)

    def set_wrap(self, enabled: bool):
        self._editor.set_wrap(enabled)

    def _on_text_changed(self):
        if not self._dirty:
            self._dirty = True
            self._update_title()

    def _tab_title(self) -> str:
        base = os.path.basename(self._file_path) if self._file_path else "Untitled"
        return f"{base}*" if self._dirty else base

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
            QMessageBox.StandardButton.Cancel)
        if res == QMessageBox.StandardButton.Save:
            self.save()
            return not self._dirty
        return res == QMessageBox.StandardButton.Discard

    def open_file(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._editor.setPlainText(content)
            self._file_path = path
            self._dirty = False
            self._update_title()
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Failed to open:\n{e}")

    def save(self):
        if self._file_path:
            try:
                with open(self._file_path, 'w', encoding='utf-8') as f:
                    f.write(self._editor.toPlainText())
                self._dirty = False
                self._update_title()
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Failed to save:\n{e}")
        else:
            self.save_as()

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Script", "",
            "Python Files (*.py)")
        if path:
            if not path.endswith('.py'):
                path += '.py'
            self._file_path = path
            self.save()

    def new(self):
        self._editor.clear()
        self._file_path = None
        self._dirty = False
        self._update_title()


class _ScriptEditorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom = _CodeEditor.DEFAULT_FONT
        self._wrap = False
        self._setup_ui()

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
            "QMenu::item:selected { background: #264f78; }")
        self._build_menubar(menubar)
        layout.addWidget(menubar)

        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(*scale_xy(18, 18)))
        toolbar.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly if qta is None
            else Qt.ToolButtonStyle.ToolButtonIconOnly)
        toolbar.setStyleSheet(
            "QToolBar { background: #2d2d2d; border: none; spacing: 2px; padding: 2px; } "
            "QToolButton { background: transparent; border: none; padding: 3px; border-radius: 3px; } "
            "QToolButton:hover { background: #3e3e3e; } "
            "QToolButton:pressed { background: #505050; }")

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
        toolbar.addAction(self._act_zoom_out)

        self._zoom_combo = QComboBox()
        self._zoom_combo.setEditable(False)
        self._zoom_combo.addItems(["8", "9", "10", "11", "12", "14", "16", "18", "20", "24", "28"])
        self._zoom_combo.setCurrentText(str(self._zoom))
        self._zoom_combo.setFixedWidth(scale(56))
        self._zoom_combo.setToolTip("Font Size")
        self._zoom_combo.currentTextChanged.connect(
            lambda t: self._apply_zoom(0, int(t) if t.isdigit() else self._zoom))
        toolbar.addWidget(self._zoom_combo)

        toolbar.addAction(self._act_zoom_in)
        toolbar.addSeparator()
        toolbar.addAction(self._act_run)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

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
            "QStatusBar::item { border: none; }")
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

        self._act_zoom_out = QAction(_qta_icon("zoom_out"), "Zoom Out", self)
        self._act_zoom_out.setToolTip("Zoom Out")
        self._act_zoom_out.triggered.connect(lambda: self._apply_zoom(-1))

        self._act_zoom_in = QAction(_qta_icon("zoom_in"), "Zoom In", self)
        self._act_zoom_in.setToolTip("Zoom In")
        self._act_zoom_in.triggered.connect(lambda: self._apply_zoom(1))

        self._act_run = QAction(_qta_icon("run"), "Run", self)
        self._act_run.setToolTip("Run Script")
        self._act_run.triggered.connect(self._run_current)

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
        act_zoom_in = QAction("Zoom In", self)
        act_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        act_zoom_in.triggered.connect(lambda: self._apply_zoom(1))
        view_menu.addAction(act_zoom_in)
        act_zoom_out = QAction("Zoom Out", self)
        act_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        act_zoom_out.triggered.connect(lambda: self._apply_zoom(-1))
        view_menu.addAction(act_zoom_out)

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

    def _new_tab(self):
        tab = _ScriptTab()
        tab.closed.connect(self._on_tab_closed)
        tab.set_font_size(self._zoom)
        tab.set_wrap(self._wrap)
        self._bind_tab_signals(tab)
        self._tabs.add_closeable_tab(tab, "Untitled")
        self._tabs.setCurrentWidget(tab)
        self._update_label()
        self._on_cursor(1, 1)

    def _open_tab(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", "",
            "Python Files (*.py);;All Files (*)")
        if path:
            for i in range(self._tabs.count()):
                existing = self._tabs.widget(i)
                if existing._file_path == path:
                    self._tabs.setCurrentWidget(existing)
                    return
            tab = _ScriptTab()
            tab.closed.connect(self._on_tab_closed)
            tab.set_font_size(self._zoom)
            tab.set_wrap(self._wrap)
            self._bind_tab_signals(tab)
            tab.open_file(path)
            self._tabs.add_closeable_tab(tab, tab._tab_title())
            self._tabs.setCurrentWidget(tab)
            self._update_label()
            self._on_cursor(1, 1)

    def _save_current(self):
        tab = self._current_tab()
        if tab is not None:
            tab.save()
            self._update_label()

    def _save_current_as(self):
        tab = self._current_tab()
        if tab is not None:
            tab.save_as()
            self._update_label()

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
        self._act_wrap_m.setChecked(checked)
        self._act_wrap.setChecked(checked)
        for i in range(self._tabs.count()):
            self._tabs.widget(i).set_wrap(self._wrap)

    def _toggle_wrap_menu(self, checked: bool):
        self._toggle_wrap(checked)

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
        if tab._file_path:
            import subprocess
            try:
                subprocess.Popen([".venv/Scripts/python", tab._file_path],
                                 cwd=os.path.dirname(os.path.abspath(__file__)) + "/../..")
            except Exception as e:
                QMessageBox.critical(self, "Run Error", f"Failed to run:\n{e}")

    def _on_tab_changed(self, index: int):
        self._update_label()
        ed = self._active_editor()
        if ed is not None:
            line = ed.textCursor().blockNumber() + 1
            col = ed.textCursor().positionInBlock() + 1
            self._on_cursor(line, col)

    def _on_tab_closed(self, tab: QWidget):
        index = self._tabs.indexOf(tab)
        if index >= 0:
            self._tabs.removeTab(index)
        if self._tabs.count() == 0:
            self._new_tab()

    def _on_cursor(self, line: int, col: int):
        self._status_pos.setText(f"Ln {line}, Col {col}")

    def _on_modified(self, modified: bool):
        tab = self._current_tab()
        if tab is not None:
            self._status_info.setText("Modified" if modified else "")

    def _update_label(self):
        tab = self._current_tab()
        if tab is None:
            self._file_label.setText("  No file")
        elif tab._file_path:
            self._file_label.setText(f"  {os.path.basename(tab._file_path)}")
        else:
            self._file_label.setText("  Untitled Script")


class ScriptEditorPanel(QDockWidget):
    def __init__(self, engine: Engine, parent=None):
        super().__init__("Script Editor", parent)
        self._engine = engine
        self.setObjectName("ScriptEditorDock")
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.setMinimumWidth(200)
        self._script_widget = _ScriptEditorWidget()
        self.setWidget(self._script_widget)
