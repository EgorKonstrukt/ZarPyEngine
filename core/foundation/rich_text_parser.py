# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import re
from typing import Optional


MINECRAFT_COLORS: dict[str, tuple[int, int, int]] = {
    '0': (0, 0, 0),
    '1': (0, 0, 170),
    '2': (0, 170, 0),
    '3': (0, 170, 170),
    '4': (170, 0, 0),
    '5': (170, 0, 170),
    '6': (255, 170, 0),
    '7': (170, 170, 170),
    '8': (85, 85, 85),
    '9': (85, 85, 255),
    'a': (85, 255, 85),
    'b': (85, 255, 255),
    'c': (255, 85, 85),
    'd': (255, 85, 255),
    'e': (255, 255, 85),
    'f': (255, 255, 255),
}

MINECRAFT_EFFECTS: dict[str, str] = {
    'l': 'bold',
    'm': 'strikethrough',
    'n': 'underline',
    'o': 'italic',
}

ANSI_COLORS: dict[int, tuple[int, int, int]] = {
    30: (0, 0, 0),
    31: (170, 0, 0),
    32: (0, 170, 0),
    33: (170, 85, 0),
    34: (0, 0, 170),
    35: (170, 0, 170),
    36: (0, 170, 170),
    37: (170, 170, 170),
    90: (85, 85, 85),
    91: (255, 85, 85),
    92: (85, 255, 85),
    93: (255, 255, 85),
    94: (85, 85, 255),
    95: (255, 85, 255),
    96: (85, 255, 255),
    97: (255, 255, 255),
}

_ANSI_PATTERN = re.compile(r'\x1b\[([\d;]*)m')


class RichTextSegment:
    def __init__(self, text: str, color: list[float], bold: bool = False, italic: bool = False, underline: bool = False, strikethrough: bool = False):
        self.text = text
        self.color = color
        self.bold = bold
        self.italic = italic
        self.underline = underline
        self.strikethrough = strikethrough


def _mc_color_to_float(c: tuple[int, int, int], base_alpha: float) -> list[float]:
    return [c[0] / 255.0, c[1] / 255.0, c[2] / 255.0, base_alpha]


def _apply_minecraft(text: str, default_color: list[float], default_bold: bool, default_italic: bool, default_underline: bool, default_strikethrough: bool) -> list[RichTextSegment]:
    segments: list[RichTextSegment] = []
    cur_color = list(default_color)
    cur_bold = default_bold
    cur_italic = default_italic
    cur_underline = default_underline
    cur_strikethrough = default_strikethrough
    buf: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '\xa7' or ch == 'В§':
            i += 1
            if i >= len(text):
                break
            code = text[i].lower()
            i += 1
            if buf:
                segments.append(RichTextSegment(''.join(buf), cur_color, cur_bold, cur_italic, cur_underline, cur_strikethrough))
                buf = []
            if code in MINECRAFT_COLORS:
                cur_color = _mc_color_to_float(MINECRAFT_COLORS[code], default_color[3])
            elif code == 'r':
                cur_color = list(default_color)
                cur_bold = default_bold
                cur_italic = default_italic
                cur_underline = default_underline
                cur_strikethrough = default_strikethrough
            elif code in MINECRAFT_EFFECTS:
                effect = MINECRAFT_EFFECTS[code]
                if effect == 'bold':
                    cur_bold = True
                elif effect == 'italic':
                    cur_italic = True
                elif effect == 'underline':
                    cur_underline = True
                elif effect == 'strikethrough':
                    cur_strikethrough = True
        else:
            buf.append(ch)
            i += 1
    if buf:
        segments.append(RichTextSegment(''.join(buf), cur_color, cur_bold, cur_italic, cur_underline, cur_strikethrough))
    return segments


def _apply_ansi(text: str, default_color: list[float], default_bold: bool, default_italic: bool, default_underline: bool, default_strikethrough: bool) -> list[RichTextSegment]:
    segments: list[RichTextSegment] = []
    cur_color = list(default_color)
    cur_bold = default_bold
    cur_italic = default_italic
    cur_underline = default_underline
    cur_strikethrough = default_strikethrough
    buf: list[str] = []
    pos = 0
    for m in _ANSI_PATTERN.finditer(text):
        start = m.start()
        if start > pos:
            buf.append(text[pos:start])
        params_str = m.group(1)
        if buf:
            segments.append(RichTextSegment(''.join(buf), cur_color, cur_bold, cur_italic, cur_underline, cur_strikethrough))
            buf = []
        if params_str:
            params = [int(x) for x in params_str.split(';') if x]
        else:
            params = [0]
        j = 0
        while j < len(params):
            p = params[j]
            if p == 0:
                cur_color = list(default_color)
                cur_bold = default_bold
                cur_italic = default_italic
                cur_underline = default_underline
                cur_strikethrough = default_strikethrough
            elif p == 1:
                cur_bold = True
            elif p == 3:
                cur_italic = True
            elif p == 4:
                cur_underline = True
            elif p == 9:
                cur_strikethrough = True
            elif p == 22:
                cur_bold = False
            elif p == 23:
                cur_italic = False
            elif p == 24:
                cur_underline = False
            elif p == 29:
                cur_strikethrough = False
            elif 30 <= p <= 37 or 90 <= p <= 97:
                c = ANSI_COLORS.get(p)
                if c:
                    cur_color = _mc_color_to_float(c, default_color[3])
            elif p == 38 and j + 1 < len(params):
                if params[j + 1] == 2 and j + 4 < len(params):
                    cur_color = [params[j + 2] / 255.0, params[j + 3] / 255.0, params[j + 4] / 255.0, default_color[3]]
                    j += 4
            j += 1
        pos = m.end()
    if pos < len(text):
        buf.append(text[pos:])
    if buf:
        segments.append(RichTextSegment(''.join(buf), cur_color, cur_bold, cur_italic, cur_underline, cur_strikethrough))
    return segments


def parse_rich_text(text: str, default_color: Optional[list[float]] = None, default_bold: bool = False, default_italic: bool = False, default_underline: bool = False, default_strikethrough: bool = False) -> list[RichTextSegment]:
    if default_color is None:
        default_color = [1, 1, 1, 1]
    if '\x1b' in text:
        return _apply_ansi(text, default_color, default_bold, default_italic, default_underline, default_strikethrough)
    return _apply_minecraft(text, default_color, default_bold, default_italic, default_underline, default_strikethrough)
