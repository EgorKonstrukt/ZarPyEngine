# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os


def register(registry, engine):

    @registry.tool(
        "project_get_info",
        "Get project root path and name",
        {"type": "object", "properties": {}},
    )
    def project_get_info():
        root = engine.project_root
        return {
            "project_root": root,
            "project_name": os.path.basename(root) if root else "",
        }

    @registry.tool(
        "project_get_settings",
        "Get all project settings",
        {"type": "object", "properties": {}},
    )
    def project_get_settings():
        from core.config import get_project_config
        cfg = get_project_config(".", lazy=True)
        if cfg is None:
            return {"error": "No project config"}
        return {"settings": cfg._data}

    @registry.tool(
        "project_set_settings",
        "Set a project setting (dot notation for nested keys)",
        {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Setting key, e.g. 'physics.gravity_y'"},
                "value": {"description": "Value to set"},
            },
            "required": ["key", "value"],
        },
    )
    def project_set_settings(key="", value=None):
        if value is None:
            value = ""
        from core.config import get_project_config
        cfg = get_project_config(".", lazy=True)
        if cfg is None:
            return {"error": "No project config"}
        cfg.set(key, value)
        cfg.save()
        return {"message": f"Set {key} = {value}"}

    @registry.tool(
        "project_get_physics_settings",
        "Get physics settings from project config",
        {"type": "object", "properties": {}},
    )
    def project_get_physics_settings():
        from core.config import get_project_config
        cfg = get_project_config(".", lazy=True)
        if cfg is None:
            return {"error": "No project config"}
        physics = cfg.get("physics", {})
        return {"physics_settings": physics}

    @registry.tool(
        "project_list_scenes",
        "List all .scene files in the project",
        {"type": "object", "properties": {}},
    )
    def project_list_scenes():
        root = engine.project_root
        if not root:
            return {"error": "No project root"}
        from .util import find_project_files
        scenes = find_project_files(root, "*.scene")
        return {"scenes": scenes, "count": len(scenes)}
