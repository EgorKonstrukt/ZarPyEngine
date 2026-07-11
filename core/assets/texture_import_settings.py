# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
from enum import Enum

import moderngl
from dataclasses import dataclass
from typing import Any, Optional


class FilterMode(Enum):
    POINT = "point"
    BILINEAR = "bilinear"
    TRILINEAR = "trilinear"


class WrapMode(Enum):
    REPEAT = "repeat"
    CLAMP = "clamp"
    MIRRORED_REPEAT = "mirrored_repeat"


class CompressionQuality(Enum):
    NONE = "none"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


DEFAULT_SETTINGS = {
    "filter_mode": "trilinear",
    "anisotropic": 1,
    "max_size": 2048,
    "wrap_mode": "clamp",
    "compression": "none",
    "srgb": True,
}


@dataclass
class TextureImportSettings:
    filter_mode: str = "trilinear"
    anisotropic: int = 1
    max_size: int = 2048
    wrap_mode: str = "clamp"
    compression: str = "none"
    srgb: bool = True

    @classmethod
    def for_file(cls, path: str) -> TextureImportSettings:
        import_path = path + ".import"
        settings = cls()
        if os.path.exists(import_path):
            try:
                with open(import_path) as f:
                    data = json.load(f)
                settings.filter_mode = data.get("filter_mode", settings.filter_mode)
                settings.anisotropic = data.get("anisotropic", settings.anisotropic)
                settings.max_size = data.get("max_size", settings.max_size)
                settings.wrap_mode = data.get("wrap_mode", settings.wrap_mode)
                settings.compression = data.get("compression", settings.compression)
                settings.srgb = data.get("srgb", settings.srgb)
            except Exception:
                pass
        return settings

    @staticmethod
    def import_mtime(path: str) -> float:
        import_path = path + ".import"
        try:
            return os.path.getmtime(import_path)
        except OSError:
            return 0.0

    def apply_to_texture(self, tex: moderngl.Texture) -> None:
        if self.filter_mode == "point":
            tex.build_mipmaps()
            tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        elif self.filter_mode == "bilinear":
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        # Anisotropic
        if self.anisotropic > 1:
            try:
                tex.anisotropy = float(self.anisotropic)
            except Exception:
                pass
        # Wrap
        repeat = self.wrap_mode != "clamp"
        tex.repeat_x = repeat
        tex.repeat_y = repeat

    def to_dict(self) -> dict:
        return {
            "filter_mode": self.filter_mode,
            "anisotropic": self.anisotropic,
            "max_size": self.max_size,
            "wrap_mode": self.wrap_mode,
            "compression": self.compression,
            "srgb": self.srgb,
        }
