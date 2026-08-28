# MIT License
#
# Copyright (c) 2026 Zarrakun
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
from typing import List
from PyQt6.QtGui import QColor

_PALETTE: List[str] = [
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
    "#1abc9c",
    "#e67e22",
    "#34495e",
    "#e91e63",
    "#00bcd4",
    "#8bc34a",
    "#ff5722",
]

_line_idx = 0
_scatter_idx = 0


def next_line_color() -> str:
    global _line_idx
    c = _PALETTE[_line_idx % len(_PALETTE)]
    _line_idx += 1
    return c


def next_scatter_color() -> str:
    global _scatter_idx
    offset = len(_PALETTE) // 3
    c = _PALETTE[(_scatter_idx + offset) % len(_PALETTE)]
    _scatter_idx += 1
    return c


def reset_colors():
    global _line_idx, _scatter_idx
    _line_idx = 0
    _scatter_idx = 0


def set_palette(colors: List[str]):
    global _PALETTE
    _PALETTE = list(colors)
    reset_colors()


def contrast_color(base: QColor) -> QColor:
    lum = 0.299 * base.redF() + 0.587 * base.greenF() + 0.114 * base.blueF()
    return QColor("#000000") if lum > 0.5 else QColor("#ffffff")