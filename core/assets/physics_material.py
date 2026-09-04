from __future__ import annotations
import json
import os
from enum import Enum
from typing import Optional


PHYSIC_MATERIAL_EXTENSION = ".zphysmat"


class PhysicCombineMode(Enum):
    AVERAGE = "average"
    MINIMUM = "minimum"
    MAXIMUM = "maximum"
    MULTIPLY = "multiply"

    @classmethod
    def coerce(cls, value) -> "PhysicCombineMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value or "average").strip().lower())
        except Exception:
            return cls.AVERAGE

    @property
    def label(self) -> str:
        return {
            PhysicCombineMode.AVERAGE: "Average",
            PhysicCombineMode.MINIMUM: "Minimum",
            PhysicCombineMode.MAXIMUM: "Maximum",
            PhysicCombineMode.MULTIPLY: "Multiply",
        }[self]


def combine_friction(mode, a: float, b: float) -> float:
    m = PhysicCombineMode.coerce(mode)
    fa = max(0.0, float(a))
    fb = max(0.0, float(b))
    if m == PhysicCombineMode.MINIMUM:
        return min(fa, fb)
    if m == PhysicCombineMode.MAXIMUM:
        return max(fa, fb)
    if m == PhysicCombineMode.MULTIPLY:
        return fa * fb
    return (fa + fb) * 0.5


def combine_bounciness(mode, a: float, b: float) -> float:
    m = PhysicCombineMode.coerce(mode)
    ba = max(0.0, float(a))
    bb = max(0.0, float(b))
    if m == PhysicCombineMode.MINIMUM:
        return min(ba, bb)
    if m == PhysicCombineMode.MAXIMUM:
        return max(ba, bb)
    if m == PhysicCombineMode.MULTIPLY:
        return ba * bb
    return (ba + bb) * 0.5


class PhysicsMaterial:
    def __init__(self, name: str = "New PhysicsMaterial"):
        self.name: str = name
        self.dynamic_friction: float = 0.6
        self.static_friction: float = 0.6
        self.bounciness: float = 0.0
        self.friction_combine: PhysicCombineMode = PhysicCombineMode.AVERAGE
        self.bounce_combine: PhysicCombineMode = PhysicCombineMode.AVERAGE
        self._path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dynamic_friction": float(self.dynamic_friction),
            "static_friction": float(self.static_friction),
            "bounciness": float(self.bounciness),
            "friction_combine": PhysicCombineMode.coerce(self.friction_combine).value,
            "bounce_combine": PhysicCombineMode.coerce(self.bounce_combine).value,
        }

    @classmethod
    def from_dict(cls, data: dict, path: str = "") -> "PhysicsMaterial":
        data = data if isinstance(data, dict) else {}
        try:
            name = str(data.get("name", "")) or "PhysicsMaterial"
        except Exception:
            name = "PhysicsMaterial"
        mat = cls(name)
        try:
            mat.dynamic_friction = max(0.0, float(data.get("dynamic_friction", 0.6)))
        except Exception:
            mat.dynamic_friction = 0.6
        try:
            mat.static_friction = max(0.0, float(data.get("static_friction", 0.6)))
        except Exception:
            mat.static_friction = 0.6
        try:
            mat.bounciness = max(0.0, float(data.get("bounciness", 0.0)))
        except Exception:
            mat.bounciness = 0.0
        mat.friction_combine = PhysicCombineMode.coerce(data.get("friction_combine", "average"))
        mat.bounce_combine = PhysicCombineMode.coerce(data.get("bounce_combine", "average"))
        if path:
            mat._path = os.path.normpath(path)
        return mat

    def save(self, path: str, project_root: str = "") -> str:
        root = project_root or os.getcwd()
        abs_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))
        abs_path = os.path.normpath(abs_path)
        if os.path.splitext(abs_path)[1].lower() != PHYSIC_MATERIAL_EXTENSION:
            abs_path += PHYSIC_MATERIAL_EXTENSION
        parent = os.path.dirname(abs_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        self._path = abs_path
        _MATERIAL_CACHE[abs_path] = (_cache_mtime(abs_path), self)
        if os.path.normpath(path) != abs_path:
            _MATERIAL_CACHE[os.path.normpath(path)] = (_cache_mtime(abs_path), self)
        return abs_path

    @classmethod
    def load(cls, path: str, project_root: str = "") -> Optional["PhysicsMaterial"]:
        if not path:
            return None
        root = project_root or os.getcwd()
        abs_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))
        abs_path = os.path.normpath(abs_path)
        if not abs_path.lower().endswith(PHYSIC_MATERIAL_EXTENSION):
            return None
        if not os.path.exists(abs_path):
            return None
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        return cls.from_dict(data, abs_path)

    @classmethod
    def load_cached(cls, path: str, project_root: str = "") -> Optional["PhysicsMaterial"]:
        if not path:
            return None
        root = project_root or os.getcwd()
        abs_path = path if os.path.isabs(path) else os.path.normpath(os.path.join(root, path))
        abs_path = os.path.normpath(abs_path)
        if not abs_path.lower().endswith(PHYSIC_MATERIAL_EXTENSION):
            return None
        try:
            mtime = _cache_mtime(abs_path)
        except Exception:
            mtime = -1.0
        cached = _MATERIAL_CACHE.get(abs_path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        mat = cls.load(abs_path)
        if mat is None:
            _MATERIAL_CACHE.pop(abs_path, None)
            return None
        _MATERIAL_CACHE[abs_path] = (mtime, mat)
        return mat

    @classmethod
    def invalidate(cls, path: str = ""):
        if not path:
            _MATERIAL_CACHE.clear()
            return
        try:
            key = os.path.normpath(path)
        except Exception:
            return
        _MATERIAL_CACHE.pop(key, None)


def _cache_mtime(abs_path: str) -> float:
    try:
        return float(os.path.getmtime(abs_path))
    except Exception:
        return -1.0


_MATERIAL_CACHE: dict[str, tuple[float, PhysicsMaterial]] = {}
