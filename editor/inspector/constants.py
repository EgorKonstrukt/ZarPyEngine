# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette

_FUSION_ACCENT_GREEN = "#4ec9b0"
_FUSION_ACCENT_RED = "#f44747"
_FUSION_ACCENT_ORANGE = "#ce9178"
_FUSION_ACCENT_YELLOW = "#dcdcaa"
_FUSION_CARD_RADIUS = "4px"
_FUSION_INPUT_RADIUS = "3px"

_XYZ_COLORS = {"X": "#f44747", "Y": "#4ec9b0", "Z": "#5a9cf5"}

_COMPONENT_MIME = "application/x-zpe-component"

_FUSION_ACCENT_CACHE = None


def _palette_color(role, fallback: str = "#000000") -> str:
    app = QApplication.instance()
    if app:
        return app.palette().color(role).name()
    return fallback


def _accent() -> str:
    global _FUSION_ACCENT_CACHE
    if _FUSION_ACCENT_CACHE is None:
        _FUSION_ACCENT_CACHE = _palette_color(QPalette.ColorRole.Highlight, "#5a9cf5")
    return _FUSION_ACCENT_CACHE


def _text() -> str:
    return _palette_color(QPalette.ColorRole.Text, "#cccccc")


def _window_text() -> str:
    return _palette_color(QPalette.ColorRole.WindowText, "#cccccc")


def _base() -> str:
    return _palette_color(QPalette.ColorRole.Base, "#1e1e1e")


def _alternate() -> str:
    return _palette_color(QPalette.ColorRole.AlternateBase, "#252526")


def _mid() -> str:
    return _palette_color(QPalette.ColorRole.Mid, "#555555")


def _button() -> str:
    return _palette_color(QPalette.ColorRole.Button, "#2d2d2d")


def _highlight() -> str:
    return _palette_color(QPalette.ColorRole.Highlight, "#5a9cf5")


def _highlighted_text() -> str:
    return _palette_color(QPalette.ColorRole.HighlightedText, "#ffffff")


def _placeholder() -> str:
    return _palette_color(QPalette.ColorRole.PlaceholderText, "#888888")


def _border_focus() -> str:
    return _accent()
