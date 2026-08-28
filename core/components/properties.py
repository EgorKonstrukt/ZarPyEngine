# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Iterator, Optional

_VEC_LEAVES = {"vec2": ("x", "y"), "vec3": ("x", "y", "z"), "vec4": ("x", "y", "z", "w")}
_SUB_INDEX = {"x": 0, "y": 1, "z": 2, "w": 3}
_SKIP_ATTRS = frozenset({
    "entity", "transform", "scene", "parent", "children", "engine",
    "gizmo", "gizmo_icon", "gizmo_lines", "gizmo_primitives", "gizmo_instance_data",
    "gizmo_cache_sig", "gizmo_meshes", "gizmo_collect",
})


def _vec_leaf_for(value: Any) -> Optional[tuple[str, ...]]:
    from core.maths.math3d import Vec2, Vec3, Vec4, Quat
    if isinstance(value, Quat):
        return ("x", "y", "z", "w")
    if isinstance(value, Vec4):
        return ("x", "y", "z", "w")
    if isinstance(value, Vec3):
        return ("x", "y", "z")
    if isinstance(value, Vec2):
        return ("x", "y")
    return None


def _read_field(comp, name: str) -> Any:
    read = getattr(comp, "get_field_value", None)
    if callable(read):
        try:
            return read(name)
        except Exception:
            pass
    return getattr(comp, name, None)


def _leaf_value(value: Any, leaf: str) -> Any:
    if value is None:
        return None
    try:
        return getattr(value, leaf)
    except (AttributeError, RuntimeError):
        pass
    if isinstance(value, (list, tuple)):
        idx = _SUB_INDEX.get(leaf)
        if idx is not None and len(value) > idx:
            return value[idx]
    return None


def _generic_prop_entries(comp, cname: str):
    entries: list[tuple[str, str, Optional[tuple[str, ...]]]] = []
    for name in dir(comp):
        if name.startswith("_") or name in _SKIP_ATTRS:
            continue
        try:
            val = getattr(comp, name)
        except Exception:
            continue
        if callable(val):
            continue
        if isinstance(val, (str, bytes, list, tuple, dict, set)):
            continue
        leaf = _vec_leaf_for(val)
        if leaf:
            entries.append((name, f"{cname}/{name}", leaf))
            continue
        if isinstance(val, bool):
            entries.append((name, f"{cname}/{name}", None))
            continue
        if isinstance(val, (int, float)):
            entries.append((name, f"{cname}/{name}", None))
            continue
        if isinstance(val, Enum):
            if isinstance(val.value, (int, float, bool)):
                entries.append((name, f"{cname}/{name}", None))
            continue
        if hasattr(val, "__float__"):
            entries.append((name, f"{cname}/{name}", None))
    return entries


def iter_entity_prop_groups(entity) -> list[tuple[str, object, list[tuple[str, str, Optional[tuple[str, ...]]]]]]:
    groups: list[tuple[str, object, list[tuple[str, str, Optional[tuple[str, ...]]]]]] = []
    for comp in entity.get_all_components():
        cname = type(comp).__name__
        fields = getattr(type(comp), "_inspector_fields", None)
        entries: list[tuple[str, str, Optional[tuple[str, ...]]]] = []
        if fields is not None:
            try:
                flist = list(fields())
            except Exception:
                flist = []
        else:
            flist = []
        for f in flist:
            leaf = f.field_type.value if hasattr(f.field_type, "value") else str(f.field_type)
            leaf = _VEC_LEAVES.get(leaf)
            base = f"{cname}/{f.name}"
            if leaf:
                entries.append((f.label or f.name, base, tuple(leaf)))
            else:
                entries.append((f.label or f.name, base, None))
        if not entries:
            entries = _generic_prop_entries(comp, cname)
        if entries:
            groups.append((cname, comp, entries))
    return groups


def iter_entity_props(entity) -> list[tuple[str, str]]:
    props: list[tuple[str, str]] = []
    for cname, comp, entries in iter_entity_prop_groups(entity):
        for label, base, leaf in entries:
            if leaf:
                for item in leaf:
                    props.append((f"{label}.{item}", f"{base}.{item}"))
            else:
                props.append((label, base))
    return props


def read_prop(entity, path: str) -> Any:
    if not path or entity is None:
        return None
    parts = path.split("/", 1)
    if len(parts) != 2:
        return None
    cname, rest = parts
    comp = entity.get_component_by_name(cname)
    if comp is None:
        return None
    if "." in rest:
        name, leaf = rest.split(".", 1)
    else:
        name, leaf = rest, None
    val = _read_field(comp, name)
    if leaf:
        val = _leaf_value(val, leaf)
    return val


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    val = getattr(value, "value", None)
    if val is not None:
        if isinstance(val, bool):
            return 1.0 if val else 0.0
        if isinstance(val, (int, float)):
            return float(val)
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f


def _vec_ctor_and_len(value: Any) -> tuple[Optional[type], int]:
    from core.maths.math3d import Vec2, Vec3, Vec4, Quat
    if isinstance(value, Quat):
        return Quat, 4
    if isinstance(value, Vec4):
        return Vec4, 4
    if isinstance(value, Vec3):
        return Vec3, 3
    if isinstance(value, Vec2):
        return Vec2, 2
    return None, 0


def _replace_leaf(value: Any, leaf: str, scalar: Any) -> Any:
    idx = _SUB_INDEX.get(leaf)
    if idx is None or value is None:
        return None
    ctor, size = _vec_ctor_and_len(value)
    if ctor is not None:
        comps = []
        for i in range(size):
            comps.append(float(getattr(value, "xyzw"[i])))
        comps[idx] = float(scalar)
        return ctor(*comps)
    if isinstance(value, (list, tuple)):
        out = list(value)
        if len(out) > idx:
            out[idx] = scalar
        return out
    return None


def write_prop(entity, path: str, value: Any) -> bool:
    if not path or entity is None:
        return False
    parts = path.split("/", 1)
    if len(parts) != 2:
        return False
    cname, rest = parts
    comp = entity.get_component_by_name(cname)
    if comp is None:
        return False
    if "." in rest:
        name, leaf = rest.split(".", 1)
    else:
        name, leaf = rest, None
    if leaf:
        nval = _replace_leaf(_read_field(comp, name), leaf, value)
        if nval is None:
            return False
        value = nval
    setter = getattr(comp, "set_field_value", None)
    if callable(setter):
        try:
            setter(name, value)
            return True
        except Exception:
            pass
    try:
        setattr(comp, name, value)
        return True
    except Exception:
        return False


def make_prop_reader(scene_getter: Callable[[], Any], entity_id: str, path: str) -> Callable[[], Optional[float]]:
    def _reader() -> Optional[float]:
        scene = scene_getter()
        if scene is None:
            return None
        entity = scene.get_entity(entity_id)
        if entity is None:
            return None
        return to_float(read_prop(entity, path))
    return _reader


def iter_entity_numeric_props(entity) -> Iterator[tuple[str, str]]:
    for label, path in iter_entity_props(entity):
        if to_float(read_prop(entity, path)) is not None:
            yield label, path