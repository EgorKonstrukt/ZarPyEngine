# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QTextEdit
from PyQt6.QtGui import QFont, QColor
from editor.inspector.constants import _FUSION_BG, _FUSION_BORDER, _FUSION_TEXT_DIM

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

class SourceViewerDialog(QDialog):
    def __init__(self, file_path: str, line_number: int = 1, title: str = "Source", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Source - {title}")
        self.setMinimumSize(500, 400)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        abs_path = file_path
        if not os.path.isabs(file_path):
            abs_path = os.path.join(_PROJECT_ROOT, file_path)
        path_label = QLabel(f"  {abs_path} (line {line_number})")
        path_label.setStyleSheet(f"color: {_FUSION_TEXT_DIM}; font-size: 10px; padding: 2px 0;")
        layout.addWidget(path_label)
        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self._text_edit.setStyleSheet(f"""
            QTextEdit {{
                background-color: {_FUSION_BG};
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid {_FUSION_BORDER};
            }}
        """)
        try:
            if os.path.exists(abs_path):
                with open(abs_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                self._text_edit.setPlainText("".join(lines))
                self._highlight_line(line_number)
        except Exception:
            self._text_edit.setPlainText(f"# Could not read file: {abs_path}")
        layout.addWidget(self._text_edit)
    def _highlight_line(self, line_num: int):
        try:
            document = self._text_edit.document()
            block = document.findBlockByLineNumber(line_num - 1)
            if not block.isValid():
                return
            cursor = self._text_edit.textCursor()
            cursor.setPosition(block.position())
            cursor.select(cursor.BlockUnderCursor)
            fmt = cursor.charFormat()
            bg_color = QColor(40, 60, 25, 180)
            fmt.setBackground(bg_color)
            fmt.setFontWeight(QFont.Weight.Bold)
            cursor.setCharFormat(fmt)
            self._text_edit.setTextCursor(cursor)
            self._text_edit.ensureCursorVisible()
        except Exception:
            pass
