# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
import glob


def get_scene(engine):
    if engine is None:
        return None
    return engine.scene


def get_entity_by_id_or_name(scene, entity_id="", entity_name=""):
    if scene is None:
        return None
    if entity_id:
        return scene.get_entity(entity_id)
    if entity_name:
        return scene.get_entity_by_name(entity_name)
    return None


def serialize_component(comp):
    try:
        return comp.serialize()
    except Exception:
        base = {"type": type(comp).__name__, "enabled": getattr(comp, "enabled", True)}
        for attr in dir(comp):
            if attr.startswith("_"):
                continue
            val = getattr(comp, attr, None)
            if callable(val):
                continue
            try:
                json.dumps({attr: val})
                base[attr] = val
            except (TypeError, OverflowError):
                base[attr] = str(val)
        return base


def find_project_files(root, pattern):
    files = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    rel = [os.path.relpath(f, root).replace("\\", "/") for f in files]
    return sorted(rel)


def resolve_path(engine, path):
    if os.path.isabs(path):
        return path
    return os.path.join(engine.project_root, path)


def component_inspector_fields(component_type):
    from core.ecs.ecs import ComponentRegistry
    cls = ComponentRegistry.get(component_type)
    if cls is None:
        return {}
    method = getattr(cls, "_inspector_fields", None)
    if not callable(method):
        return {}
    try:
        return {f.name: f for f in method()}
    except Exception:
        return {}


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "on", "y"):
            return True
        if v in ("false", "0", "no", "off", "n"):
            return False
    return bool(value)


def _to_int(value):
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_enum(enum_cls, value):
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        for member in enum_cls:
            if member.name.lower() == v or str(member.value).lower() == v:
                return member
    try:
        return enum_cls(value)
    except Exception:
        return None


def _num_list(value, n):
    if isinstance(value, (list, tuple)):
        seq = list(value)
    elif isinstance(value, str):
        seq = value.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")
    else:
        seq = [value]
    nums = []
    for item in seq[:n]:
        try:
            nums.append(float(item))
        except (TypeError, ValueError):
            nums.append(0.0)
    while len(nums) < n:
        nums.append(0.0)
    return nums


def _num_sequence(value):
    if isinstance(value, (list, tuple)):
        seq = list(value)
    elif isinstance(value, str):
        seq = value.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")
    else:
        seq = [value]
    result = []
    for item in seq:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            result.append(0.0)
    return result


def _vec_dims(vec_cls):
    return {"Vec2": 2, "Vec3": 3, "Vec4": 4}.get(getattr(vec_cls, "__name__", ""), 3)


def coerce_value(comp, name, value, field=None):
    from enum import Enum
    from core.maths.math3d import Vec2, Vec3, Vec4
    if field is not None:
        ft = getattr(field.field_type, "name", "")
        if ft == "BOOL":
            return _to_bool(value)
        if ft in ("FLOAT", "SLIDER"):
            return _to_float(value)
        if ft in ("INT", "INT_SLIDER", "LAYER"):
            return _to_int(value)
        if ft == "ENUM" and getattr(field, "enum_class", None) is not None:
            result = _to_enum(field.enum_class, value)
            if result is not None:
                return result
        if ft == "COLOR":
            current = getattr(comp, name, None)
            seq = _num_sequence(value)
            if isinstance(current, (Vec2, Vec3, Vec4)):
                return type(current)(*seq[:_vec_dims(type(current))])
            return seq
        if ft == "VEC2":
            return Vec2(*_num_list(value, 2))
        if ft == "VEC3":
            return Vec3(*_num_list(value, 3))
        if ft == "VEC4":
            return Vec4(*_num_list(value, 4))
        if ft in ("STRING", "TEXTAREA", "RESOURCE_PATH", "GAMEOBJECT", "ASSET"):
            return str(value)
    current = getattr(comp, name, None)
    if current is None:
        return value
    if isinstance(current, bool):
        return _to_bool(value)
    if isinstance(current, int):
        return _to_int(value)
    if isinstance(current, float):
        return _to_float(value)
    if isinstance(current, str):
        return str(value)
    if isinstance(current, Enum):
        result = _to_enum(type(current), value)
        return result if result is not None else current
    if isinstance(current, (Vec2, Vec3, Vec4)):
        return type(current)(*_num_list(value, _vec_dims(type(current))))
    if isinstance(current, (list, tuple)):
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            parts = value.replace("[", "").replace("]", "").replace("(", "").replace(")", "").split(",")
            return [p.strip() for p in parts]
        return [value]
    return value


def value_to_json(val):
    from enum import Enum
    from core.maths.math3d import Vec2, Vec3, Vec4
    if isinstance(val, Vec2):
        return [float(val.x), float(val.y)]
    if isinstance(val, (Vec3, Vec4)):
        parts = [val.x, val.y, val.z]
        if hasattr(val, "w"):
            parts.append(val.w)
        return [float(p) for p in parts]
    if isinstance(val, Enum):
        return {"name": val.name, "value": val.value}
    if isinstance(val, (list, tuple)):
        return list(val)
    return val


def value_type_name(val):
    from enum import Enum
    from core.maths.math3d import Vec2, Vec3, Vec4
    if val is None:
        return "any"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        return "string"
    if isinstance(val, Enum):
        return "enum"
    if isinstance(val, (Vec2, Vec3, Vec4)):
        return type(val).__name__
    if isinstance(val, (list, tuple)):
        return "array"
    return type(val).__name__
