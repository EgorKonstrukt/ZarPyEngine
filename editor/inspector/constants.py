# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette

_FUSION_BG = "#1e1e1e"
_FUSION_BG_CARD = "#252525"
_FUSION_BG_HEADER = "#2b2b2b"
_FUSION_BG_HOVER = "#333333"
_FUSION_BG_INPUT = "#2a2a2a"
_FUSION_BORDER = "#3c3c3c"
_FUSION_BORDER_LIGHT = "#4a4a4a"
_FUSION_TEXT = "#cccccc"
_FUSION_TEXT_DIM = "#888888"
_FUSION_TEXT_BRIGHT = "#eeeeee"
_FUSION_TEXT_DISABLED = "#666666"
_FUSION_ACCENT_GREEN = "#4ec9b0"
_FUSION_ACCENT_RED = "#f44747"
_FUSION_ACCENT_ORANGE = "#ce9178"
_FUSION_ACCENT_YELLOW = "#dcdcaa"
_FUSION_CARD_RADIUS = "4px"
_FUSION_INPUT_RADIUS = "3px"

_XYZ_COLORS = {"X": "#f44747", "Y": "#4ec9b0", "Z": "#5a9cf5"}

_COMPONENT_MIME = "application/x-zpe-component"

_FUSION_ACCENT_CACHE = None

def _accent() -> str:
    global _FUSION_ACCENT_CACHE
    if _FUSION_ACCENT_CACHE is None:
        app = QApplication.instance()
        if app:
            c = app.palette().color(QPalette.ColorRole.Highlight)
            _FUSION_ACCENT_CACHE = c.name()
        else:
            _FUSION_ACCENT_CACHE = "#5a9cf5"
    return _FUSION_ACCENT_CACHE

def _border_focus() -> str:
    return _accent()

def _checkbox_style() -> str:
    a = _accent()
    return (
        f"QCheckBox {{ color: {_FUSION_TEXT}; spacing: 4px; background: transparent; }}"
        f"QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {_FUSION_BORDER_LIGHT}; border-radius: 2px; background: {_FUSION_BG_INPUT}; }}"
        f"QCheckBox::indicator:checked {{ background: {a}; border-color: {a}; }}"
        f"QCheckBox::indicator:hover {{ border-color: {a}; }}"
    )
