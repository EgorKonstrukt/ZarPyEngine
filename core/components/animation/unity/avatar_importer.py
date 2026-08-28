# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

"""Import Unity Avatar (``!u!9000001``) assets.

Unity rig avatars live in ``.avatar`` files and describe the skeleton that an
Animator/AnimationClip binds to: the bone hierarchy (slash paths like
``Armature/Hips/Tail``), each bone's rest pose (position/rotation/scale) and
the humanoid bone mapping. This module parses the avatar document into the
same engine-side :class:`PrefabModel` the prefab importer produces, so one
code path builds the rig entities.

The skeleton container is located by recursion, so any layout works:

* ``m_Skeleton``: a block sequence of ``SkeletonBone`` entries with
  ``boneName``, ``humanName``, ``position``, ``rotation``, ``scale``.
* ``m_Human``: a block sequence of human bone entries: ``humanBoneBoneName``
  (pre-2019) or ``boneName`` (2019+), plus ``humanName``.

Example::

    from core.components.animation.unity.avatar_importer import (
        import_avatar, import_avatar_entities,
    )

    model = import_avatar("path/to/Vulper.avatar")
    root, model = import_avatar_entities(scene, "path/to/Vulper.avatar")

The returned :class:`PrefabModel` uses the avatar's own skeleton indices
(0..N-1) as fake file ids; slash paths become parent links.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

from core.components.animation.unity.prefab_importer import (
    PrefabModel,
    PrefabNode,
)
from core.components.animation.unity.yaml_util import parse_unity_documents

_CLASS_AVATAR = 9000001


@dataclass
class AvatarBone:
    """One ``SkeletonBone`` from an avatar's human description."""

    bone_name: str  # slash path, e.g. "Armature/Hips/Tail/Tail.001"
    human_name: str
    position: Optional[dict] = None
    rotation: Optional[dict] = None
    scale: Optional[dict] = None

    @property
    def path_parts(self) -> list:
        return [p for p in self.bone_name.split("/") if p]


@dataclass
class AvatarModel:
    name: str = ""
    root_bone: str = ""
    bones: list = field(default_factory=list)
    human: list = field(default_factory=list)  # (bone_name, human_name) pairs
    has_human: bool = False

    def to_prefab_model(self) -> PrefabModel:
        """Convert to a :class:`PrefabModel` so entity building is shared."""
        model = PrefabModel()
        for idx, bone in enumerate(self.bones):
            parts = bone.path_parts
            node = PrefabNode(file_id=idx, name=parts[-1] if parts else bone.bone_name)
            node.local_position = bone.position
            node.local_rotation = bone.rotation
            node.local_scale = bone.scale
            if len(parts) > 1:
                parent_path = "/".join(parts[:-1])
                for j in range(len(self.bones)):
                    if self.bones[j].bone_name == parent_path:
                        node.parent_file_id = j
                        break
            model.nodes[idx] = node
        for idx in list(model.nodes):
            parent = model.nodes[idx].parent_file_id
            if parent is not None and parent in model.nodes:
                model.nodes[parent].child_file_ids.append(idx)
        model.roots = [
            gid for gid, n in model.nodes.items() if n.parent_file_id is None and n.name
        ]
        return model


def _parse_skel(doc: dict) -> list:
    skeleton = doc.get("m_Skeleton") or doc.get("skeleton")
    out: list[AvatarBone] = []
    if not isinstance(skeleton, list):
        return out
    for item in skeleton:
        if not isinstance(item, dict):
            continue
        bname = item.get("boneName") or item.get("m_BoneName") or ""
        out.append(AvatarBone(
            bone_name=str(bname),
            human_name=str(item.get("humanName") or item.get("m_HumanName") or ""),
            position=item.get("position") or item.get("m_Position"),
            rotation=item.get("rotation") or item.get("m_Rotation"),
            scale=item.get("scale") or item.get("m_Scale"),
        ))
    return out


def _parse_human(doc: dict) -> tuple:
    human = doc.get("m_Human") or doc.get("human")
    out: list = []
    if not isinstance(human, list):
        return out, False
    for item in human:
        if not isinstance(item, dict):
            continue
        hname = item.get("humanName") or item.get("m_HumanName") or ""
        bname = (item.get("boneName")
                 or item.get("humanBoneBoneName")
                 or item.get("m_HumanBoneBoneName")
                 or "")
        out.append((str(bname), str(hname)))
    return out, len(out) > 0


def _find_desc(data: Any) -> Optional[dict]:
    """Locate the dict that holds the skeleton, wherever Unity nests it."""
    stack = [data]
    seen = set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, dict):
            skel = obj.get("m_Skeleton") or obj.get("skeleton")
            if isinstance(skel, list) and skel:
                return obj
            for v in obj.values():
                stack.append(v)
        elif isinstance(obj, list):
            for v in obj:
                stack.append(v)
    return None


def import_avatar(source: str) -> AvatarModel:
    if os.path.isfile(source):
        with open(source, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    else:
        text = source

    av = AvatarModel()
    for doc in parse_unity_documents(text):
        if doc["class_id"] != _CLASS_AVATAR:
            continue
        data = doc["data"]
        if not isinstance(data, dict):
            continue
        if not av.name:
            av.name = str(data.get("m_Name", ""))
        desc = _find_desc(data)
        if desc is None:
            continue
        av.bones = _parse_skel(desc)
        av.human, av.has_human = _parse_human(desc)
        if not av.name:
            av.name = str(desc.get("m_AvatarRootBoneName")
                          or desc.get("avatarRootBoneName") or "")

    if not av.root_bone and av.bones:
        av.root_bone = av.bones[0].path_parts[0] if av.bones[0].path_parts else ""
    return av


def import_avatar_entities(scene, source: str) -> tuple:
    """Build engine entities from an avatar's skeleton, returning the root."""
    from core.components.animation.unity.prefab_importer import _build_entity

    avatar = import_avatar(source)
    model = avatar.to_prefab_model()
    if not model.roots:
        return None, avatar
    by_fid = {}
    roots = [_build_entity(scene, model, model.nodes[r], by_fid) for r in model.roots]
    return (roots[0] if len(roots) == 1 else roots), avatar