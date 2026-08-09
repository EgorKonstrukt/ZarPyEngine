# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import sys
import io
import traceback
import subprocess
import math
import json
import time as _time_module
from typing import Any
from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QTextEdit, QTabWidget, QPlainTextEdit,
                              QCompleter, QApplication)
from PyQt6.QtCore import Qt, QStringListModel, pyqtSignal, QTimer, QThread
import qtawesome as qta
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont, QKeyEvent, QKeySequence

from core.engine.engine import Engine
from core.foundation.logger import Logger
from core.config.editor_scale import scale


class _ReplInput(QPlainTextEdit):
    """Multiline input widget for REPL.
    - Enter = execute (if brace-balanced) or newline
    - Shift+Enter = force newline
    - Ctrl+Enter = force execute
    - Up/Down = history navigation (at first/last line)
    """
    execute_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._history: list[str] = []
        self._history_index: int = -1
        self._saved_text: str = ""
        self.setTabChangesFocus(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setMaximumBlockCount(100)
        self.setPlaceholderText(">>>")

    def keyPressEvent(self, event: QKeyEvent):
        mods = event.modifiers()
        key = event.key()

        if key == Qt.Key.Key_Return or key == Qt.Key.Key_Enter:
            if mods & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            if mods & Qt.KeyboardModifier.ControlModifier:
                self._do_execute()
                return
            text = self.toPlainText()
            if _is_balanced(text) or text.strip().startswith(("class ", "def ", "@", "if ", "elif ", "else:", "for ", "while ", "try:", "with ")):
                self._do_execute()
            else:
                super().keyPressEvent(event)
            return

        if key == Qt.Key.Key_Up:
            cursor = self.textCursor()
            if cursor.blockNumber() == 0 and not cursor.movePosition(QTextCursor.MoveOperation.Up):
                self._history_up()
                return
        elif key == Qt.Key.Key_Down:
            cursor = self.textCursor()
            if cursor.blockNumber() == self.blockCount() - 1 and not cursor.movePosition(QTextCursor.MoveOperation.Down):
                self._history_down()
                return
        elif event.matches(QKeySequence.StandardKey.Copy):
            if not self.textCursor().hasSelection():
                cursor = self.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.StartOfLine)
                cursor.movePosition(QTextCursor.MoveOperation.EndOfLine, QTextCursor.MoveMode.KeepAnchor)
                QApplication.clipboard().setText(cursor.selectedText())
                return

        super().keyPressEvent(event)

    def _do_execute(self):
        code = self.toPlainText()
        if code.strip():
            self._history.append(code)
            self._history_index = len(self._history)
            self.execute_requested.emit(code)
        self.clear()

    def _history_up(self):
        if not self._history:
            return
        if self._history_index == len(self._history):
            self._saved_text = self.toPlainText()
        if self._history_index > 0:
            self._history_index -= 1
            self.setPlainText(self._history[self._history_index])
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.setTextCursor(cursor)

    def _history_down(self):
        if not self._history:
            return
        if self._history_index < len(self._history) - 1:
            self._history_index += 1
            self.setPlainText(self._history[self._history_index])
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.setTextCursor(cursor)
        elif self._history_index == len(self._history) - 1:
            self._history_index = len(self._history)
            self.setPlainText(self._saved_text)
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.setTextCursor(cursor)


def _is_balanced(text: str) -> bool:
    """Check if braces/parens/brackets are balanced and the code is 'complete'."""
    depth = 0
    in_str = False
    str_char = None
    for c in text:
        if in_str:
            if c == str_char:
                in_str = False
            continue
        if c in ("'", '"'):
            in_str = True
            str_char = c
            continue
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
            if depth < 0:
                return True
    return depth == 0


_CODE_COLORS = {
    "keyword": QColor("#569cd6"),
    "string": QColor("#ce9178"),
    "number": QColor("#b5cea8"),
    "builtin": QColor("#dcdcaa"),
    "comment": QColor("#6a9955"),
    "error": QColor("#f44747"),
    "result": QColor("#9cdcfe"),
    "prompt": QColor("#6a9955"),
    "stdout": QColor("#ffffff"),
}


class _CaptureWriter(io.StringIO):
    """Captures writes for later retrieval."""


class TerminalTab(QWidget):
    _font_family: str = "Courier New"
    _font_size: int = 10

    def __init__(self, namespace: dict[str, Any], font_family: str = "",
                 font_size: int = 0, parent=None):
        super().__init__(parent)
        self._namespace = namespace
        if font_family:
            self._font_family = font_family
        if font_size:
            self._font_size = font_size
        self._setup_ui()

    def _make_font(self) -> QFont:
        font = QFont(self._font_family, self._font_size)
        font.setStyleHint(QFont.StyleHint.Monospace)
        return font

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(self._make_font())
        self._output.setMinimumHeight(60)
        layout.addWidget(self._output)

        self._input = _ReplInput()
        self._input.setFont(self._make_font())
        self._input.setMaximumBlockCount(50)
        self._input.setMinimumHeight(28)
        self._input.setMaximumHeight(120)
        self._input.execute_requested.connect(self._execute)
        layout.addWidget(self._input)

    def _execute(self, code: str):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        proxy = _CaptureWriter()
        sys.stdout = proxy
        sys.stderr = proxy

        error_text = None

        try:
            compiled = compile(code, "<repl>", "exec")
            ns = self._namespace
            exec(compiled, ns)
        except Exception:
            error_text = traceback.format_exc().rstrip()

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        self._write_output(f">>> {code}", _CODE_COLORS["prompt"])

        if error_text:
            self._write_output(error_text, _CODE_COLORS["error"])
            self._namespace.pop("_", None)
            return

        out = proxy.getvalue()
        if out:
            self._write_output(out.rstrip(), _CODE_COLORS["stdout"])

        self._namespace.pop("_", None)

        try:
            expr_val = eval(code, self._namespace)
            if expr_val is not None:
                self._namespace["_"] = expr_val
                self._write_output(repr(expr_val), _CODE_COLORS["result"])
        except Exception:
            pass

    def _write_output(self, text: str, color: QColor):
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor.insertText(text + "\n", fmt)
        scroll = self._output.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def set_namespace(self, ns: dict):
        self._namespace = ns


class PsReader(QThread):
    line_received = pyqtSignal(str)

    def __init__(self, proc: subprocess.Popen):
        super().__init__()
        self._proc = proc

    def run(self):
        for line in iter(self._proc.stdout.readline, ""):
            line = line.rstrip("\r\n")
            if line:
                self.line_received.emit(line)


class PowershellInput(QPlainTextEdit):
    send_cmd = pyqtSignal(str)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            mods = event.modifiers()
            if mods & Qt.KeyboardModifier.ShiftModifier:
                super().keyPressEvent(event)
                return
            text = self.toPlainText().strip()
            if text:
                self.send_cmd.emit(text)
                self.clear()
        else:
            super().keyPressEvent(event)


class PowershellTab(QWidget):
    _font_family: str = "Segoe UI"
    _font_size: int = 10

    def __init__(self, assets_path: str, font_family: str = "",
                 font_size: int = 0, parent=None):
        super().__init__(parent)
        self._proc: subprocess.Popen | None = None
        self._reader: PsReader | None = None
        if font_family:
            self._font_family = font_family
        if font_size:
            self._font_size = font_size
        self._setup_ui()
        self._start_powershell(assets_path)

    def _make_font(self) -> QFont:
        return QFont(self._font_family, self._font_size)

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setFont(self._make_font())
        self._output.setMinimumHeight(60)
        self._output.setStyleSheet(
            "QTextEdit { background: #012456; color: #ffffff;"
            " border: 1px solid #003366; }")
        layout.addWidget(self._output)

        input_layout = QHBoxLayout()
        input_layout.setSpacing(4)

        self._input = PowershellInput()
        self._input.setFont(self._make_font())
        self._input.setMaximumBlockCount(10)
        self._input.setMinimumHeight(28)
        self._input.setMaximumHeight(80)
        self._input.setStyleSheet(
            "QPlainTextEdit { background: #001a4a; color: #ffffff;"
            " border: 1px solid #003366; padding: 4px 8px; }")
        self._input.setPlaceholderText("Enter PowerShell command...")
        self._input.send_cmd.connect(self._send_command)
        input_layout.addWidget(self._input)

        layout.addLayout(input_layout)

    def _start_powershell(self, assets_path: str):
        try:
            setup = (
                f'[Console]::OutputEncoding = [Text.Encoding]::UTF8; '
                f'$OutputEncoding = [Text.Encoding]::UTF8; '
                f'cd "{assets_path}"'
            )
            self._proc = subprocess.Popen(
                ["powershell.exe", "-NoExit", "-Command", setup],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=assets_path,
                bufsize=0,
                text=True,
                encoding="utf-8",
            )
            self._reader = PsReader(self._proc)
            self._reader.line_received.connect(self._on_line)
            self._reader.start()
        except Exception as e:
            self._write_output(f"Failed to start PowerShell: {e}", QColor("#f44747"))
            self._input.setEnabled(False)

    def _on_line(self, text: str):
        QTimer.singleShot(0, lambda: self._write_output(text, QColor("#ffffff")))

    def _write_output(self, text: str, color: QColor):
        cursor = self._output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(color)
        cursor.insertText(text + "\n", fmt)
        scroll = self._output.verticalScrollBar()
        scroll.setValue(scroll.maximum())

    def _send_command(self, cmd: str):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(cmd + "\n")
                self._proc.stdin.flush()
            except Exception as e:
                self._write_output(f"Error: {e}", QColor("#f44747"))

    def terminate(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass


class TerminalPanel(QDockWidget):
    _SCENE_REFRESH_MS = 500

    def __init__(self, parent=None):
        super().__init__("Terminal", parent)
        self._tab_count: int = 0
        self._font_family: str = "Segoe UI"
        self._font_size: int = 10
        self._namespace: dict[str, Any] = {}
        self._pws_tabs: list[PowershellTab] = []
        self._setup_ui()
        self._init_namespace()
        eng = Engine.instance()
        if eng:
            eng.on("scene_loaded", self._on_scene_loaded)

    def _init_namespace(self):
        ns = self._namespace
        ns.clear()
        eng = Engine.instance()
        if eng:
            ns["engine"] = eng
            ns["scene"] = eng.scene
        ns["Logger"] = Logger
        ns["os"] = os
        ns["sys"] = sys
        ns["json"] = json
        ns["math"] = _time_module
        from core.maths.math3d import Vec2, Vec3, Vec4, Quat, Mat4
        ns["Vec2"] = Vec2
        ns["Vec3"] = Vec3
        ns["Vec4"] = Vec4
        ns["Quat"] = Quat
        ns["Mat4"] = Mat4
        try:
            import numpy as np
            ns["np"] = np
        except ImportError:
            pass
        try:
            from core.ecs.ecs import Entity, Component
            ns["Entity"] = Entity
            ns["Component"] = Component
        except ImportError:
            pass

    def _on_scene_loaded(self, scene):
        self._namespace["scene"] = scene

    def _setup_ui(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(4)

        add_btn = QPushButton(qta.icon("fa5s.plus", color="#9ccc65"), "")
        add_btn.setToolTip("New Python Tab")
        add_btn.clicked.connect(self._add_tab)
        toolbar.addWidget(add_btn)

        ps_btn = QPushButton(qta.icon("fa5s.terminal", color="#d4d4d4"), "")
        ps_btn.setToolTip("New PowerShell Tab (embedded)")
        ps_btn.clicked.connect(self._open_powershell)
        toolbar.addWidget(ps_btn)

        toolbar.addStretch()

        clear_btn = QPushButton(qta.icon("fa5s.eraser", color="#d4d4d4"), "")
        clear_btn.setToolTip("Clear Current Tab")
        clear_btn.clicked.connect(self._clear_current)
        toolbar.addWidget(clear_btn)

        reset_btn = QPushButton(qta.icon("fa5s.sync-alt", color="#d4d4d4"), "")
        reset_btn.setToolTip("Reset Python Namespace")
        reset_btn.clicked.connect(self._reset_namespace)
        toolbar.addWidget(reset_btn)

        layout.addLayout(toolbar)

        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)
        layout.addWidget(self._tabs)

        self.setWidget(w)
        self._add_tab()

    def _get_assets_path(self) -> str:
        assets_path = os.path.abspath("assets")
        if not os.path.isdir(assets_path):
            assets_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "..", "assets"))
        if not os.path.isdir(assets_path):
            assets_path = os.getcwd()
        return assets_path

    def _add_tab(self):
        self._tab_count += 1
        tab = TerminalTab(self._namespace, self._font_family,
                          self._font_size, self)
        idx = self._tabs.addTab(tab, f"Python {self._tab_count}")
        self._tabs.setCurrentIndex(idx)

    def _open_powershell(self):
        self._tab_count += 1
        tab = PowershellTab(self._get_assets_path(),
                            self._font_family, self._font_size, self)
        self._pws_tabs.append(tab)
        idx = self._tabs.addTab(tab, f"PS {self._tab_count}")
        self._tabs.setCurrentIndex(idx)

    def _close_tab(self, index: int):
        if self._tabs.count() <= 1:
            return
        w = self._tabs.widget(index)
        if isinstance(w, PowershellTab):
            w.terminate()
            if w in self._pws_tabs:
                self._pws_tabs.remove(w)
        self._tabs.removeTab(index)
        w.deleteLater()

    def _clear_current(self):
        tab = self._tabs.currentWidget()
        if tab and hasattr(tab, "_output"):
            tab._output.clear()

    def _reset_namespace(self):
        self._init_namespace()

    def load_config(self, config) -> None:
        self._font_family = config.get("terminal.font_family", self._font_family)
        self._font_size = config.get("terminal.font_size", self._font_size)

    def save_config(self, config) -> None:
        config.set("terminal.font_family", self._font_family)
        config.set("terminal.font_size", self._font_size)

    def execute_python(self, code: str):
        tab = self._tabs.currentWidget()
        if isinstance(tab, TerminalTab):
            tab._input.setPlainText(code)
            tab._execute(code)
