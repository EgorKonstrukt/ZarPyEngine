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
