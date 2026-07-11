# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import platform
from typing import Optional


_SYSTEM_FONT_CANDIDATES: list[str] = []

if platform.system() == "Windows":
    _windir = os.environ.get("WINDIR", "C:\\Windows")
    _SYSTEM_FONT_CANDIDATES = [
        os.path.join(_windir, "Fonts", "seguiui.ttf"),
        os.path.join(_windir, "Fonts", "arial.ttf"),
        os.path.join(_windir, "Fonts", "tahoma.ttf"),
        os.path.join(_windir, "Fonts", "calibri.ttf"),
        os.path.join(_windir, "Fonts", "consola.ttf"),
    ]
elif platform.system() == "Darwin":
    _SYSTEM_FONT_CANDIDATES = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
else:
    _SYSTEM_FONT_CANDIDATES = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]

_UNICODE_RANGES = [
    (32, 127),
    (160, 256),
    (1024, 1280),
    (0x400, 0x4FF + 1),
]


def get_default_font_path() -> str:
    for p in _SYSTEM_FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return ""


def _collect_chars() -> list[str]:
    seen: set[int] = set()
    chars: list[str] = []
    for start, end in _UNICODE_RANGES:
        for cp in range(start, end):
            if cp not in seen:
                seen.add(cp)
                try:
                    chars.append(chr(cp))
                except ValueError:
                    pass
    return chars


class FontAtlas:
    font_path: str
    base_size: int
    texture: Optional[np.ndarray]
    glyphs: dict[str, dict]
    texture_width: int
    texture_height: int
    line_height: float
    ascender: float
    descender: float
    max_cp: int = 1281

    def __init__(self, font_path: str, base_size: int = 128):
        self.font_path = font_path
        self.base_size = base_size
        self.texture = None
        self.glyphs = {}
        self.texture_width = 0
        self.texture_height = 0
        self.line_height = 1.0
        self.ascender = 0.0
        self.descender = 0.0
        self._gp_advance = np.zeros(self.max_cp, dtype=np.float32)
        self._gp_bearing_x = np.zeros(self.max_cp, dtype=np.float32)
        self._gp_bearing_y = np.zeros(self.max_cp, dtype=np.float32)
        self._gp_glyph_w = np.zeros(self.max_cp, dtype=np.float32)
        self._gp_glyph_h = np.zeros(self.max_cp, dtype=np.float32)
        self._gp_uv = np.zeros((self.max_cp, 4), dtype=np.float32)
        self._gp_uv_bold = np.zeros((self.max_cp, 4), dtype=np.float32)
        self._gp_valid = np.zeros(self.max_cp, dtype=bool)
        self._build()

    def _load_font(self) -> ImageFont.FreeTypeFont:
        path = self.font_path
        if not path or not os.path.exists(path):
            default = get_default_font_path()
            if default:
                path = default
            else:
                return ImageFont.load_default()
        return ImageFont.truetype(path, self.base_size)

    def _build(self):
        font = self._load_font()
        ascent, descent = font.getmetrics()
        self.ascender = float(ascent)
        self.descender = float(descent)
        self.line_height = float(ascent + descent)

        chars = _collect_chars()
        rendered: list[tuple[str, Image.Image, int, int, int, int]] = []
        total_width = 0

        space_bbox = font.getbbox(" ")
        space_advance = font.getlength(" ") if space_bbox else float(self.base_size) * 0.25

        for ch in chars:
            if ch == " ":
                sw = max(int(space_advance) + 2, 4)
                img = Image.new("L", (sw, 2), 0)
                rendered.append((ch, img, sw, 2, 0, 0))
                total_width += sw + 4
                continue
            bbox = font.getbbox(ch)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            gw = x1 - x0
            gh = y1 - y0
            if gw <= 0 or gh <= 0:
                continue
            img = Image.new("L", (gw + 2, gh + 2), 0)
            draw = ImageDraw.Draw(img)
            draw.text((-x0 + 1, -y0 + 1), ch, font=font, fill=255)
            advance = font.getlength(ch)
            rendered.append((ch, img, gw + 2, gh + 2, x0, y0))
            total_width += gw + 4

        pad = 2
        if not rendered:
            self.texture = np.zeros((4, 4, 4), dtype=np.uint8)
            self.texture_width = 4
            self.texture_height = 4
            return

        atlas_w = 1
        while atlas_w < total_width:
            atlas_w <<= 1
        if atlas_w > 8192:
            atlas_w = 8192

        rows: list[list] = []
        current_row: list = []
        row_y = pad
        row_h = 0
        cursor_x = pad

        for ch, img, iw, ih, bbox_x0, bbox_y0 in rendered:
            if cursor_x + iw + pad > atlas_w:
                rows.append((current_row, row_y, row_h))
                row_y += row_h + pad
                current_row = []
                cursor_x = pad
                row_h = 0
            current_row.append((ch, img, iw, ih, cursor_x, row_y, bbox_x0, bbox_y0))
            cursor_x += iw + pad
            if ih > row_h:
                row_h = ih
        if current_row:
            rows.append((current_row, row_y, row_h))

        regular_h = row_y + row_h + pad

        bold_rendered: list[tuple[str, Image.Image, int, int, int, int]] = []
        bold_total_w = 0
        for ch in chars:
            if ch == " ":
                sw = max(int(space_advance) + 2, 4)
                img = Image.new("L", (sw, 2), 0)
                bold_rendered.append((ch, img, sw, 2, 0, 0))
                bold_total_w += sw + 4
                continue
            bbox = font.getbbox(ch)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            gw = x1 - x0
            gh = y1 - y0
            if gw <= 0 or gh <= 0:
                continue
            bp = 2
            img_w = gw + bp * 2
            img_h = gh + bp * 2
            arr = np.zeros((img_h, img_w), dtype=np.uint8)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    layer = Image.new("L", (img_w, img_h), 0)
                    d2 = ImageDraw.Draw(layer)
                    d2.text((bp - x0 + dx, bp - y0 + dy), ch, font=font, fill=255)
                    arr = np.maximum(arr, np.array(layer))
            img = Image.fromarray(arr)
            bold_rendered.append((ch, img, img_w, img_h, x0, y0))
            bold_total_w += img_w + 4

        bold_atlas_w = 1
        while bold_atlas_w < bold_total_w:
            bold_atlas_w <<= 1
        if bold_atlas_w > 8192:
            bold_atlas_w = 8192

        bold_rows: list[list] = []
        current_row = []
        row_y = pad
        row_h = 0
        cursor_x = pad
        for ch, img, iw, ih, bbox_x0, bbox_y0 in bold_rendered:
            if cursor_x + iw + pad > bold_atlas_w:
                bold_rows.append((current_row, row_y, row_h))
                row_y += row_h + pad
                current_row = []
                cursor_x = pad
                row_h = 0
            current_row.append((ch, img, iw, ih, cursor_x, row_y, bbox_x0, bbox_y0))
            cursor_x += iw + pad
            if ih > row_h:
                row_h = ih
        if current_row:
            bold_rows.append((current_row, row_y, row_h))

        bold_total_h = row_y + row_h + pad
        final_w = atlas_w + bold_atlas_w
        final_h = max(regular_h, bold_total_h)
        atlas_h = 1
        while atlas_h < final_h:
            atlas_h <<= 1
        if atlas_h > 8192:
            atlas_h = 8192

        atlas = Image.new("L", (atlas_w + bold_atlas_w, atlas_h), 0)
        self.glyphs = {}
        for row_data, base_y, _ in rows:
            for ch, img, iw, ih, cx, cy, bbox_x0, bbox_y0 in row_data:
                atlas.paste(img, (cx, cy))
                advance_val = font.getlength(ch) if ch != " " else space_advance
                self.glyphs[ch] = {
                    "x": cx,
                    "y": cy,
                    "w": iw,
                    "h": ih,
                    "atlas_w": atlas_w + bold_atlas_w,
                    "atlas_h": atlas_h,
                    "bearing_x": float(bbox_x0),
                    "bearing_y": float(bbox_y0),
                    "advance": float(advance_val),
                    "glyph_w": iw - 2 if ch != " " else 0,
                    "glyph_h": ih - 2 if ch != " " else 0,
                }

        for row_data, base_y, _ in bold_rows:
            for ch, img, iw, ih, cx, cy, bbox_x0, bbox_y0 in row_data:
                atlas.paste(img, (atlas_w + cx, cy))
                if ch in self.glyphs:
                    g = self.glyphs[ch]
                    g["bold_x"] = atlas_w + cx
                    g["bold_y"] = cy
                    g["bold_w"] = iw
                    g["bold_h"] = ih

        self.texture_width = atlas_w + bold_atlas_w
        self.texture_height = atlas_h
        arr = np.array(atlas, dtype=np.uint8)
        self.texture = np.repeat(arr[:, :, np.newaxis], 4, axis=2)
        self.texture[:, :, 3] = arr
        for c in range(3):
            self.texture[:, :, c] = 255
        self._populate_glyph_arrays()

    def _populate_glyph_arrays(self):
        for ch, data in self.glyphs.items():
            cp = ord(ch)
            if cp >= self.max_cp:
                continue
            self._gp_advance[cp] = data["advance"]
            self._gp_bearing_x[cp] = data["bearing_x"]
            self._gp_bearing_y[cp] = data["bearing_y"]
            self._gp_glyph_w[cp] = data["glyph_w"]
            self._gp_glyph_h[cp] = data["glyph_h"]
            u0 = data["x"] / data["atlas_w"]
            v0 = data["y"] / data["atlas_h"]
            u1 = (data["x"] + data["w"]) / data["atlas_w"]
            v1 = (data["y"] + data["h"]) / data["atlas_h"]
            self._gp_uv[cp] = [u0, v0, u1, v1]
            bx = data.get("bold_x")
            if bx is not None:
                bu0 = bx / data["atlas_w"]
                bv0 = data["bold_y"] / data["atlas_h"]
                bu1 = (bx + data["bold_w"]) / data["atlas_w"]
                bv1 = (data["bold_y"] + data["bold_h"]) / data["atlas_h"]
                self._gp_uv_bold[cp] = [bu0, bv0, bu1, bv1]
            else:
                self._gp_uv_bold[cp] = [u0, v0, u1, v1]
            self._gp_valid[cp] = True

    def get_glyph(self, char: str) -> Optional[dict]:
        return self.glyphs.get(char)

    def get_uv(self, char: str) -> tuple[float, float, float, float]:
        g = self.glyphs.get(char)
        if g is None:
            return 0.0, 0.0, 0.0, 0.0
        u0 = g["x"] / g["atlas_w"]
        v0 = g["y"] / g["atlas_h"]
        u1 = (g["x"] + g["w"]) / g["atlas_w"]
        v1 = (g["y"] + g["h"]) / g["atlas_h"]
        return u0, v0, u1, v1

    def measure_line(self, text: str) -> tuple[float, float]:
        total_w = 0.0
        max_h = 0.0
        for ch in text:
            g = self.glyphs.get(ch)
            if g is None:
                continue
            total_w += g["advance"]
            gh = g["glyph_h"]
            if gh > max_h:
                max_h = gh
        return total_w * self._inv_lh(), max_h * self._inv_lh()

    def measure_multiline(self, text: str, line_spacing: float = 1.2) -> tuple[float, float]:
        lines = text.split("\n")
        max_w = 0.0
        total_h = 0.0
        for i, line in enumerate(lines):
            w, h = self.measure_line(line)
            if w > max_w:
                max_w = w
            total_h += self.line_height * self._inv_lh() * line_spacing
        if lines:
            total_h -= self.line_height * self._inv_lh() * (line_spacing - 1.0)
        return max_w, total_h

    def get_text_bounds(self, text: str, font_size: int, line_spacing: float = 1.2) -> tuple[float, float]:
        w, h = self.measure_multiline(text, line_spacing)
        scale = float(font_size) * self._inv_lh() * 0.01
        return w * scale, h * scale

    def _inv_lh(self) -> float:
        return 1.0 / self.line_height if self.line_height > 0 else 1.0

    def measure_text(self, text: str) -> tuple[float, float]:
        return self.measure_line(text)
