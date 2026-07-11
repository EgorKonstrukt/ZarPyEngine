# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
from typing import Optional, TYPE_CHECKING
from core.math.math3d import Vec3

if TYPE_CHECKING:
    from core.ecs.ecs import Entity

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

SHAPE_TYPE_MAP = {
    "BoxCollider": "box",
    "SphereCollider": "sphere",
    "CapsuleCollider": "capsule",
    "MeshCollider": "mesh",
    "BoxCollider2D": "box",
    "CircleCollider2D": "cylinder",
}

SHAPE_INFO_CACHE_KEYS = {
    "BoxCollider": ("type", "size", "center", "friction", "restitution", "is_trigger"),
    "SphereCollider": ("type", "radius", "center", "friction", "restitution", "is_trigger"),
    "CapsuleCollider": ("type", "radius", "height", "center", "direction", "friction", "restitution", "is_trigger"),
    "MeshCollider": ("type", "file", "collision_mode", "max_vertices", "scale", "friction", "restitution", "is_trigger"),
    "BoxCollider2D": ("type", "size", "center", "friction", "restitution", "is_trigger"),
    "CircleCollider2D": ("type", "radius", "height", "center", "friction", "restitution", "is_trigger"),
}


def read_import_scale(mesh_path: str) -> float | None:
    """Read import scale from .mesh_path.import JSON file."""
    if not mesh_path:
        return None
    import_path = mesh_path + ".import"
    if not os.path.isabs(import_path):
        import_path = os.path.normpath(os.path.join(_PROJECT_ROOT, import_path))
    if os.path.exists(import_path):
        try:
            with open(import_path) as f:
                return json.load(f).get("scale", 1.0)
        except Exception:
            pass
    return None


def find_shape_info(entity: Entity, transform=None) -> Optional[dict]:
    """Extract shape parameters from collider components on entity."""
    for comp in entity.get_all_components():
        cname = type(comp).__name__
        if cname not in SHAPE_TYPE_MAP:
            continue
        params = {}
        friction = 0.6
        restitution = 0.0
        is_trigger = False

        if cname == "BoxCollider":
            params["size"] = [comp.scaled_size.x, comp.scaled_size.y, comp.scaled_size.z]
            params["center"] = [comp.scaled_center.x, comp.scaled_center.y, comp.scaled_center.z]
            friction = comp.material_friction
            restitution = comp.material_bounciness
            is_trigger = comp.is_trigger
        elif cname == "SphereCollider":
            params["radius"] = comp.scaled_radius
            params["center"] = [comp.scaled_center.x, comp.scaled_center.y, comp.scaled_center.z]
            friction = comp.material_friction
            restitution = comp.material_bounciness
            is_trigger = comp.is_trigger
        elif cname == "CapsuleCollider":
            params["radius"] = comp.scaled_radius
            params["height"] = comp.scaled_height
            params["center"] = [comp.scaled_center.x, comp.scaled_center.y, comp.scaled_center.z]
            params["direction"] = comp.direction
            is_trigger = comp.is_trigger
        elif cname == "BoxCollider2D":
            sz = comp.scaled_size
            params["size"] = [sz.x, sz.y, 1.0]
            off = comp.scaled_offset
            params["center"] = [off.x, off.y, 0.0]
            friction = comp.material_friction
            restitution = comp.material_bounciness
            is_trigger = comp.is_trigger
        elif cname == "CircleCollider2D":
            params["radius"] = comp.scaled_radius
            params["height"] = 1.0
            off = comp.scaled_offset
            params["center"] = [off.x, off.y, 0.0]
            friction = comp.material_friction
            restitution = comp.material_bounciness
            is_trigger = comp.is_trigger
        elif cname == "MeshCollider":
            params["file"] = comp.mesh_path
            params["collision_mode"] = comp.collision_mode.value
            params["max_vertices"] = comp.max_vertices
            friction = comp.material_friction
            restitution = comp.material_bounciness
            is_trigger = comp.is_trigger

        scale = read_import_scale(params.get("file", ""))
        s = transform.local_scale if transform else Vec3.one()
        if scale is not None:
            params["scale"] = [scale * s.x, scale * s.y, scale * s.z]
        elif cname == "MeshCollider":
            params["scale"] = [s.x, s.y, s.z]

        return {
            "type": SHAPE_TYPE_MAP[cname],
            "params": params,
            "friction": friction,
            "restitution": restitution,
            "is_trigger": is_trigger,
            "layer": getattr(comp, 'layer', 0),
            "mask": getattr(comp, 'mask', 0xFFFF),
        }

    from core.components import MeshFilter
    mf = entity.get_component(MeshFilter)
    if mf and mf.mesh_path:
        s = transform.local_scale if transform else Vec3.one()
        scale = read_import_scale(mf.mesh_path)
        if scale is not None:
            params = {"file": mf.mesh_path, "collision_mode": "auto", "max_vertices": 2000, "scale": [scale * s.x, scale * s.y, scale * s.z]}
        else:
            params = {"file": mf.mesh_path, "collision_mode": "auto", "max_vertices": 2000, "scale": [s.x, s.y, s.z]}
        return {
            "type": "mesh",
            "params": params,
            "friction": 0.6,
            "restitution": 0.0,
            "is_trigger": False,
            "layer": 0,
            "mask": 0xFFFF,
        }
    return None


def make_shape_key(entity: Entity, shape_info: dict) -> tuple:
    """Create a cache key for a shape based on its parameters."""
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
        val = shape_info["params"].get(k, shape_info.get(k))
        if isinstance(val, list):
            parts.append(tuple(val))
        else:
            parts.append(val)
    return tuple(parts)
