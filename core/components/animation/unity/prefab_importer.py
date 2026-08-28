# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""Import Unity prefab/scene rig hierarchies into the engine.

Unity serializes prefabs and scenes as YAML streams of ``GameObject``
(``!u!1``), ``Transform`` (``!u!4``) and component documents. This module
reconstructs the GameObject/Transform hierarchy — names, local position /
rotation / scale and father-child links — exactly as Unity presents it, so
animation clips authored against bone paths such as ``Armature/Hips/Tail``
bind to the same named entities in the engine.

Usage::

    from core.components.animation.unity.prefab_importer import (
        import_prefab, import_prefab_entities, find_asset_by_guid,
    )

    model = import_prefab("path/to/Vulper.prefab")
    roots, model = import_prefab_entities(scene, "path/to/Vulper.prefab")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from core.components.animation.unity.yaml_util import parse_unity_documents

_CLASS_GAME_OBJECT = 1
_CLASS_TRANSFORM = 4
_CLASS_ANIMATOR = 95


def _ref_id(v: Any) -> Optional[int]:
    """Read a Unity reference dict's fileID (0 counts as "no reference")."""
    if not isinstance(v, dict):
        return None
    fid = v.get("fileID")
    if not isinstance(fid, int) or fid == 0:
        return None
    return fid


@dataclass
class PrefabNode:
    """One imported GameObject with its Transform state."""

    file_id: int
    name: str = ""
    parent_file_id: Optional[int] = None
    child_file_ids: list = field(default_factory=list)
    component_file_ids: list = field(default_factory=list)
    transform_file_id: Optional[int] = None
    local_position: Optional[dict] = None
    local_rotation: Optional[dict] = None
    local_scale: Optional[dict] = None
    is_active: bool = True

    def path_name(self, model: "PrefabModel") -> str:
        """Unity-style slash path from the rig root to this node."""
        parts = []
        cur: Optional[int] = self.file_id
        visited = set()
        while cur is not None and cur not in visited:
            node = model.nodes.get(cur)
            if node is None:
                break
            visited.add(cur)
            parts.append(node.name)
            cur = node.parent_file_id
        return "/".join(reversed(parts))


@dataclass
class PrefabModel:
    """Parsed prefab hierarchy, keyed by Unity file ids."""

    nodes: dict = field(default_factory=dict)  # node.file_id -> PrefabNode
    by_transform: dict = field(default_factory=dict)  # transform fid -> node fid
    roots: list = field(default_factory=list)
    animators: list = field(default_factory=list)


def _children_of_transform(data: dict) -> list:
    ch = data.get("m_Children")
    if not isinstance(ch, list):
        return []
    out = []
    for item in ch:
        fid = _ref_id(item)
        if fid is not None:
            out.append(fid)
    return out


def _components_of_game_object(data: dict) -> list:
    comps = data.get("m_Component")
    if not isinstance(comps, list):
        return []
    out = []
    for item in comps:
        ref = item.get("component") if isinstance(item, dict) else item
        fid = _ref_id(ref)
        if fid is not None:
            out.append(fid)
    return out


def import_prefab(source: str) -> PrefabModel:
    """Parse a .prefab / .unity file (or raw YAML text) into a PrefabModel."""
    if os.path.isfile(source):
        with open(source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = source

    docs = parse_unity_documents(text)
    game_objects: dict[int, dict] = {}
    transforms: dict[int, dict] = {}
    animators: list[dict] = []
    for doc in docs:
        if doc["file_id"] is None or doc["class_id"] is None:
            continue
        data = doc["data"]
        if not isinstance(data, dict) or doc["file_id"] == 0:
            continue
        cid = doc["class_id"]
        if cid == _CLASS_GAME_OBJECT:
            game_objects[doc["file_id"]] = data
        elif cid == _CLASS_TRANSFORM:
            transforms[doc["file_id"]] = data
        elif cid == _CLASS_ANIMATOR:
            animators.append(data)

    model = PrefabModel(animators=animators)
    for fid, data in game_objects.items():
        model.nodes[fid] = PrefabNode(
            file_id=fid,
            name=str(data.get("m_Name", "")),
            component_file_ids=_components_of_game_object(data),
            is_active=bool(data.get("m_IsActive", 1)),
        )

    for tfid, data in transforms.items():
        gid = _ref_id(data.get("m_GameObject"))
        if gid is not None and gid not in model.nodes:
            model.nodes[gid] = PrefabNode(file_id=gid)
        if gid is not None:
            model.by_transform[tfid] = gid

    for tfid, data in transforms.items():
        gid = model.by_transform.get(tfid)
        if gid is None:
            continue
        node = model.nodes[gid]
        node.transform_file_id = tfid
        node.local_position = data.get("m_LocalPosition")
        node.local_rotation = data.get("m_LocalRotation")
        node.local_scale = data.get("m_LocalScale")
        parent_tr = _ref_id(data.get("m_Father"))
        if parent_tr is not None:
            node.parent_file_id = model.by_transform.get(parent_tr)
        child_ids = []
        for cid in _children_of_transform(data):
            gid = model.by_transform.get(cid)
            if gid is not None:
                child_ids.append(gid)
        node.child_file_ids = child_ids

    model.roots = [gid for gid, n in model.nodes.items() if n.parent_file_id is None]
    return model


def _vec3_from_flow(v: Optional[dict]):
    if not isinstance(v, dict):
        return None
    try:
        return (float(v["x"]), float(v["y"]), float(v["z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _quat_from_flow(v: Optional[dict]):
    if not isinstance(v, dict):
        return None
    try:
        return (float(v["x"]), float(v["y"]), float(v["z"]), float(v["w"]))
    except (KeyError, TypeError, ValueError):
        return None


def _build_entity(scene, model: PrefabModel, node: PrefabNode, by_fid: dict) -> Any:
    from core.components import Transform

    ent = scene.create_entity(node.name)
    by_fid[node.file_id] = ent
    tr = Transform()
    pos = _vec3_from_flow(node.local_position)
    rot = _quat_from_flow(node.local_rotation)
    scl = _vec3_from_flow(node.local_scale)
    if pos is not None:
        tr.local_position = pos
    if rot is not None:
        from core.maths.math3d import Quat

        tr.local_rotation = Quat(rot[0], rot[1], rot[2], rot[3])
    if scl is not None:
        tr.local_scale = scl
    ent.add_component(tr)
    for cid in node.child_file_ids:
        child = model.nodes.get(cid)
        if child is None:
            continue
        child_ent = _build_entity(scene, model, child, by_fid)
        child_ent.set_parent(ent, preserve_world=False)
    return ent


def import_prefab_entities(scene, source: str, name: Optional[str] = None,
                           world_pos=None) -> tuple:
    """Build engine entities mirroring a Unity prefab's GameObject hierarchy.

    Returns ``(roots, model)`` where ``roots`` is the list of created root
    entities and ``model`` is the parsed :class:`PrefabModel`.
    """
    model = import_prefab(source)
    by_fid: dict = {}
    roots = []
    for gid in model.roots:
        node = model.nodes.get(gid)
        if node is None:
            continue
        roots.append(_build_entity(scene, model, node, by_fid))
    if world_pos is not None:
        for r in roots:
            if r.transform is not None:
                r.transform.local_position = world_pos
    return roots, model


def find_asset_by_guid(subtree: str, guid: str) -> Optional[str]:
    """Locate an asset file whose sibling ``.meta`` carries ``guid``.

    Mirrors Unity's guid-to-asset resolution for object references.
    """
    if not os.path.isdir(subtree):
        return None
    for dirpath, _dirnames, filenames in os.walk(subtree):
        for fn in filenames:
            if not fn.endswith(".meta"):
                continue
            meta_path = os.path.join(dirpath, fn)
            try:
                with open(meta_path, encoding="utf-8", errors="replace") as fh:
                    head = fh.read(1024)
            except OSError:
                continue
            if guid in head:
                return meta_path[:-5] if meta_path.endswith(".meta") else meta_path
    return None