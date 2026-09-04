# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
import numpy as np
from typing import Optional, TYPE_CHECKING
from core.maths.math3d import Vec3
from core.assets.physics_material import PhysicsMaterial, PhysicCombineMode


def default_physics_material_dict() -> dict:
    return {
        "path": "",
        "dynamic_friction": 0.6,
        "static_friction": 0.6,
        "bounciness": 0.0,
        "friction_combine": PhysicCombineMode.AVERAGE.value,
        "bounce_combine": PhysicCombineMode.AVERAGE.value,
    }


def resolve_physics_material(comp) -> dict:
    try:
        fallback_friction = float(getattr(comp, "material_friction", 0.6))
    except Exception:
        fallback_friction = 0.6
    try:
        fallback_bounce = float(getattr(comp, "material_bounciness", 0.0))
    except Exception:
        fallback_bounce = 0.0
    if fallback_friction != fallback_friction or fallback_friction < 0.0:
        fallback_friction = 0.6
    if fallback_bounce != fallback_bounce or fallback_bounce < 0.0:
        fallback_bounce = 0.0
    try:
        path = getattr(comp, "physic_material", "") or ""
    except Exception:
        path = ""
    if isinstance(path, str) and path:
        try:
            mat = PhysicsMaterial.load_cached(path)
        except Exception:
            mat = None
        if mat is not None:
            return {
                "path": path,
                "dynamic_friction": float(mat.dynamic_friction),
                "static_friction": float(mat.static_friction),
                "bounciness": float(mat.bounciness),
                "friction_combine": PhysicCombineMode.coerce(mat.friction_combine).value,
                "bounce_combine": PhysicCombineMode.coerce(mat.bounce_combine).value,
            }
    return {
        "path": "",
        "dynamic_friction": fallback_friction,
        "static_friction": fallback_friction,
        "bounciness": fallback_bounce,
        "friction_combine": PhysicCombineMode.AVERAGE.value,
        "bounce_combine": PhysicCombineMode.AVERAGE.value,
    }

if TYPE_CHECKING:
    from core.ecs.ecs import Entity

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

SHAPE_TYPE_MAP = {
    "BoxCollider": "box",
    "SphereCollider": "sphere",
    "CapsuleCollider": "capsule",
    "MeshCollider": "mesh",
    "TerrainCollider": "heightfield",
    "BoxCollider2D": "box",
    "CircleCollider2D": "sphere",
}

PRIMITIVE_SHAPE_TYPES = ("box", "sphere", "capsule", "cylinder")

_HALF_SQRT2 = 0.7071067811865476
_CAPSULE_PART_QUAT = {
    0: (0.0, 0.0, _HALF_SQRT2, _HALF_SQRT2),
    1: (0.0, 0.0, 0.0, 1.0),
    2: (_HALF_SQRT2, 0.0, 0.0, _HALF_SQRT2),
}
_PYBULLET_CAPSULE_EULER = {
    0: (0.0, 1.5707963267948966, 0.0),
    1: (-1.5707963267948966, 0.0, 0.0),
    2: (0.0, 0.0, 0.0),
}


def capsule_part_quat(direction: int) -> tuple[float, float, float, float]:
    try:
        return _CAPSULE_PART_QUAT[int(direction) % 3]
    except Exception:
        return (0.0, 0.0, 0.0, 1.0)


def capsule_section_height(radius: float, height: float) -> tuple[float, float]:
    r = max(float(radius), 1e-4)
    try:
        h = max(float(height) - 2.0 * r, 1e-4)
    except Exception:
        h = 1e-4
    return r, h


def capsule_volume(radius: float, height: float) -> float:
    import math
    r, h = capsule_section_height(radius, height)
    return math.pi * r * r * h + 4.0 / 3.0 * math.pi * r ** 3


def part_volume(shape_type: str, params: dict) -> float:
    import math
    try:
        if shape_type == "box":
            s = params.get("size", [1, 1, 1])
            return max(float(s[0]) * float(s[1]) * float(s[2]), 1e-9)
        if shape_type == "sphere":
            r = max(float(params.get("radius", 0.5)), 1e-4)
            return 4.0 / 3.0 * math.pi * r ** 3
        if shape_type == "capsule":
            return max(capsule_volume(float(params.get("radius", 0.5)), float(params.get("height", 2.0))), 1e-9)
        if shape_type == "cylinder":
            r = max(float(params.get("radius", 0.5)), 1e-4)
            h = max(float(params.get("height", 1.0)), 1e-4)
            return math.pi * r * r * h
    except Exception:
        pass
    return 1.0


def part_center(shape_type: str, params: dict) -> list[float]:
    c = params.get("center", [0.0, 0.0, 0.0])
    try:
        return [float(c[0]), float(c[1]), float(c[2])]
    except Exception:
        return [0.0, 0.0, 0.0]


SHAPE_INFO_CACHE_KEYS = {
    "BoxCollider": ("type", "size", "center", "friction", "restitution", "is_trigger"),
    "SphereCollider": ("type", "radius", "center", "friction", "restitution", "is_trigger"),
    "CapsuleCollider": ("type", "radius", "height", "center", "direction", "friction", "restitution", "is_trigger"),
    "MeshCollider": ("type", "file", "collision_mode", "max_vertices", "scale", "center", "friction", "restitution", "is_trigger"),
    "TerrainCollider": ("type", "size", "resolution", "height_scale", "center", "friction", "restitution", "is_trigger"),
    "BoxCollider2D": ("type", "size", "center", "friction", "restitution", "is_trigger"),
    "CircleCollider2D": ("type", "radius", "center", "friction", "restitution", "is_trigger"),
}


def read_import_scale(mesh_path: str) -> float | None:
    """Read import scale from .mesh_path.import JSON file."""
    if not mesh_path:
        return None
    from core.assets.asset_importer import _resolve_mesh_import_path
    import_path = _resolve_mesh_import_path(mesh_path)
    if os.path.exists(import_path):
        try:
            with open(import_path) as f:
                return json.load(f).get("scale", 1.0)
        except Exception:
            pass
    return None


def _shape_info_for(cname: str, comp, transform=None) -> Optional[dict]:
    params: dict = {}
    friction = 0.6
    restitution = 0.0
    is_trigger = False
    mat = resolve_physics_material(comp)

    if cname == "BoxCollider":
        params["size"] = [comp.scaled_size.x, comp.scaled_size.y, comp.scaled_size.z]
        params["center"] = [comp.scaled_center.x, comp.scaled_center.y, comp.scaled_center.z]
        friction = mat["dynamic_friction"]
        restitution = mat["bounciness"]
        is_trigger = comp.is_trigger
    elif cname == "SphereCollider":
        params["radius"] = comp.scaled_radius
        params["center"] = [comp.scaled_center.x, comp.scaled_center.y, comp.scaled_center.z]
        friction = mat["dynamic_friction"]
        restitution = mat["bounciness"]
        is_trigger = comp.is_trigger
    elif cname == "CapsuleCollider":
        params["radius"] = comp.scaled_radius
        params["height"] = comp.scaled_height
        params["center"] = [comp.scaled_center.x, comp.scaled_center.y, comp.scaled_center.z]
        params["direction"] = comp.direction
        friction = mat["dynamic_friction"]
        restitution = mat["bounciness"]
        is_trigger = comp.is_trigger
    elif cname == "BoxCollider2D":
        sz = comp.scaled_size
        params["size"] = [sz.x, sz.y, 1.0]
        off = comp.scaled_offset
        params["center"] = [off.x, off.y, 0.0]
        friction = mat["dynamic_friction"]
        restitution = mat["bounciness"]
        is_trigger = comp.is_trigger
    elif cname == "CircleCollider2D":
        params["radius"] = comp.scaled_radius
        off = comp.scaled_offset
        params["center"] = [off.x, off.y, 0.0]
        friction = mat["dynamic_friction"]
        restitution = mat["bounciness"]
        is_trigger = comp.is_trigger
    elif cname == "MeshCollider":
        params["file"] = comp.mesh_path
        params["collision_mode"] = comp.collision_mode.value
        params["max_vertices"] = comp.max_vertices
        params["center"] = [comp.scaled_center.x, comp.scaled_center.y, comp.scaled_center.z]
        friction = mat["dynamic_friction"]
        restitution = mat["bounciness"]
        is_trigger = comp.is_trigger
    elif cname == "TerrainCollider":
        params["size"] = [comp.size.x, comp.size.y, comp.size.z]
        params["resolution"] = comp.resolution
        params["height_scale"] = comp.height_scale
        params["center"] = [comp.center.x, comp.center.y, comp.center.z]
        hd = getattr(comp, "height_data", None)
        if hd is not None and getattr(hd, "ndim", 0) == 2:
            params["height_data"] = hd.astype(np.float32)
        elif hd is not None:
            params["height_data"] = np.asarray(hd, dtype=np.float32)
        friction = mat["dynamic_friction"]
        restitution = mat["bounciness"]
        is_trigger = comp.is_trigger
    else:
        return None

    scale = read_import_scale(params.get("file", ""))
    s = transform.local_scale if transform else Vec3.one()
    if scale is not None:
        params["scale"] = [scale * s.x, scale * s.y, scale * s.z]
    elif cname == "MeshCollider":
        params["scale"] = [s.x, s.y, s.z]

    return {
        "cname": cname,
        "type": SHAPE_TYPE_MAP[cname],
        "params": params,
        "friction": friction,
        "restitution": restitution,
        "material": mat,
        "is_trigger": is_trigger,
        "layer": getattr(comp, 'layer', 0),
        "mask": getattr(comp, 'mask', 0xFFFF),
    }


def find_shapes_info(entity: Entity, transform=None) -> list[dict]:
    out: list[dict] = []
    for comp in entity.get_all_components():
        cname = type(comp).__name__
        if cname not in SHAPE_TYPE_MAP:
            continue
        info = _shape_info_for(cname, comp, transform)
        if info is not None:
            out.append(info)
    if out:
        return out

    from core.components import MeshFilter
    mf = entity.get_component(MeshFilter)
    if mf and mf.mesh_path:
        s = transform.local_scale if transform else Vec3.one()
        scale = read_import_scale(mf.mesh_path)
        if scale is not None:
            params = {"file": mf.mesh_path, "collision_mode": "auto", "max_vertices": 2000, "scale": [scale * s.x, scale * s.y, scale * s.z]}
        else:
            params = {"file": mf.mesh_path, "collision_mode": "auto", "max_vertices": 2000, "scale": [s.x, s.y, s.z]}
        params["center"] = [0.0, 0.0, 0.0]
        return [{
            "cname": "MeshCollider",
            "type": "mesh",
            "params": params,
            "friction": 0.6,
            "restitution": 0.0,
            "material": default_physics_material_dict(),
            "is_trigger": False,
            "layer": 0,
            "mask": 0xFFFF,
        }]
    return []


def find_shape_info(entity: Entity, transform=None) -> Optional[dict]:
    shapes = find_shapes_info(entity, transform)
    return shapes[0] if shapes else None


def make_shape_key(entity: Entity, shape_info: dict) -> tuple:
    cname = None
    for comp in entity.get_all_components():
        if type(comp).__name__ in SHAPE_TYPE_MAP:
            cname = type(comp).__name__
            break
    if cname is None:
        return ()
    keys = SHAPE_INFO_CACHE_KEYS.get(cname, ())
    parts = [shape_info["type"]]
    for k in keys:
        if k == "type":
            continue
        parts.append(_key_part(shape_info["params"].get(k, shape_info.get(k))))
    return tuple(parts)


def _key_part(val):
    if isinstance(val, list):
        return tuple(val)
    try:
        import numpy as _np
        if isinstance(val, _np.ndarray):
            try:
                return ("ndarray", tuple(val.shape), int(val.nbytes), float(_np.sum(val)))
            except Exception:
                return ("ndarray", tuple(val.shape), int(val.nbytes), 0.0)
    except Exception:
        pass
    return val


def make_shapes_key(shapes: list[dict]) -> tuple:
    parts: list[tuple] = []
    for info in shapes:
        cname = info.get("cname")
        if cname is None:
            for comp_cname, stype in SHAPE_TYPE_MAP.items():
                if stype == info.get("type"):
                    cname = comp_cname
                    break
        keys = SHAPE_INFO_CACHE_KEYS.get(cname or "", ())
        item: list = [info.get("type")]
        for k in keys:
            if k == "type":
                continue
            item.append(_key_part(info["params"].get(k, info.get(k))))
        parts.append(tuple(item))
    return tuple(parts)


def downgrade_mesh_to_box(shape: dict) -> Optional[dict]:
    try:
        from core.components.physics.mesh_collider import _load_mesh_data
    except Exception:
        return None
    try:
        params = shape["params"]
        md = _load_mesh_data(params.get("file", ""))
        if md is None or md.get("num_verts", 0) == 0:
            return None
        sc = params.get("scale", [1.0, 1.0, 1.0])
        try:
            sx, sy, sz = float(sc[0]), float(sc[1]), float(sc[2])
        except Exception:
            sx = sy = sz = 1.0
        try:
            k = float(md.get("import_scale", 1.0) or 1.0)
        except Exception:
            k = 1.0
        if abs(k) < 1e-9:
            k = 1.0
        s = np.array([sx / k, sy / k, sz / k], dtype=np.float64)
        s = np.where(np.abs(s) < 1e-9, 1.0, s)
        mins = md["mins"].astype(np.float64) * s
        maxs = md["maxs"].astype(np.float64) * s
        half = np.maximum((maxs - mins) * 0.5, 1e-4)
        mc = params.get("center", [0.0, 0.0, 0.0])
        try:
            moff = np.array([float(mc[0]), float(mc[1]), float(mc[2])], dtype=np.float64)
        except Exception:
            moff = np.zeros(3, dtype=np.float64)
        center = (mins + maxs) * 0.5 + moff
        try:
            material = dict(shape.get("material", None) or default_physics_material_dict())
        except Exception:
            material = default_physics_material_dict()
        return {
            "cname": "BoxCollider",
            "type": "box",
            "params": {
                "size": [float(half[0] * 2.0), float(half[1] * 2.0), float(half[2] * 2.0)],
                "center": [float(center[0]), float(center[1]), float(center[2])],
            },
            "friction": shape.get("friction", 0.6),
            "restitution": shape.get("restitution", 0.0),
            "material": material,
            "is_trigger": shape.get("is_trigger", False),
            "layer": shape.get("layer", 0),
            "mask": shape.get("mask", 0xFFFF),
        }
    except Exception:
        return None


def partition_compound_shapes(shapes: list[dict], dynamic: bool) -> tuple[list[dict], list[dict]]:
    prims = [s for s in shapes if s.get("type") in PRIMITIVE_SHAPE_TYPES]
    rest = [s for s in shapes if s.get("type") not in PRIMITIVE_SHAPE_TYPES]
    separate: list[dict] = []
    if dynamic:
        for s in rest:
            if s.get("type") == "mesh":
                box = downgrade_mesh_to_box(s)
                if box is not None:
                    try:
                        from core.foundation.logger import Logger
                        Logger.warning(f"MeshCollider '{s['params'].get('file', '')}' shares a dynamic body, using box approximation")
                    except Exception:
                        pass
                    prims.append(box)
                else:
                    separate.append(s)
            else:
                separate.append(s)
    else:
        separate.extend(rest)
    return prims, separate
