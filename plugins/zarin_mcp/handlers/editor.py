# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os


def register(registry, engine):

    @registry.tool(
        "editor_get_selection",
        "Get currently selected entity IDs and names",
        {"type": "object", "properties": {}},
    )
    def editor_get_selection():
        viewport = getattr(engine, "viewport", None)
        if viewport is None:
            return {"selection": [], "message": "No viewport (headless mode)"}
        selected = getattr(viewport, "_selected_entities", [])
        return {
            "selection": [
                {"id": e.id, "name": e.name} for e in selected
            ],
            "count": len(selected),
        }

    @registry.tool(
        "editor_set_selection",
        "Set the current entity selection by IDs",
        {
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity UUIDs to select",
                }
            },
            "required": ["entity_ids"],
        },
    )
    def editor_set_selection(entity_ids=None):
        if entity_ids is None:
            entity_ids = []
        viewport = getattr(engine, "viewport", None)
        if viewport is None:
            return {"error": "No viewport (headless mode)"}
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        entities = []
        for eid in entity_ids:
            e = scene.get_entity(eid)
            if e:
                entities.append(e)
        if hasattr(viewport, "set_selected_entities"):
            viewport.set_selected_entities(entities)
        elif hasattr(viewport, "set_selected_entity") and len(entities) == 1:
            viewport.set_selected_entity(entities[0])
        return {"message": f"Selected {len(entities)} entities"}

    @registry.tool(
        "editor_clear_selection",
        "Clear the current entity selection",
        {"type": "object", "properties": {}},
    )
    def editor_clear_selection():
        viewport = getattr(engine, "viewport", None)
        if viewport is None:
            return {"error": "No viewport"}
        if hasattr(viewport, "set_selected_entities"):
            viewport.set_selected_entities([])
        elif hasattr(viewport, "set_selected_entity"):
            viewport.set_selected_entity(None)
        return {"message": "Selection cleared"}

    @registry.tool(
        "editor_undo",
        "Undo the last editor action",
        {"type": "object", "properties": {}},
    )
    def editor_undo():
        from core.foundation.commands import get_history
        history = get_history()
        if history is None:
            return {"error": "No command history"}
        try:
            if history.can_undo:
                desc = history.undo_description()
                history.undo()
                return {"message": f"Undone: {desc}"}
        except TypeError:
            if history.can_undo():
                desc = history.undo_description()
                history.undo()
                return {"message": f"Undone: {desc}"}
        return {"message": "Nothing to undo"}

    @registry.tool(
        "editor_redo",
        "Redo the last undone editor action",
        {"type": "object", "properties": {}},
    )
    def editor_redo():
        from core.foundation.commands import get_history
        history = get_history()
        if history is None:
            return {"error": "No command history"}
        try:
            if history.can_redo:
                desc = history.redo_description()
                history.redo()
                return {"message": f"Redone: {desc}"}
        except TypeError:
            if history.can_redo():
                desc = history.redo_description()
                history.redo()
                return {"message": f"Redone: {desc}"}
        return {"message": "Nothing to redo"}

    @registry.tool(
        "editor_get_undo_history",
        "Get the undo/redo command history list",
        {"type": "object", "properties": {}},
    )
    def editor_get_undo_history():
        from core.foundation.commands import get_history
        history = get_history()
        if history is None:
            return {"error": "No command history"}
        try:
            can_undo = history.can_undo
        except TypeError:
            can_undo = history.can_undo()
        try:
            can_redo = history.can_redo
        except TypeError:
            can_redo = history.can_redo()
        return {
            "undo_stack": [c.description for c in history._undo_stack],
            "redo_stack": [c.description for c in history._redo_stack],
            "can_undo": can_undo,
            "can_redo": can_redo,
            "saved_index": history._saved_index,
            "current_index": history._current_index,
        }

    @registry.tool(
        "prefab_list",
        "List all prefab files (.zpep) in the project",
        {"type": "object", "properties": {}},
    )
    def prefab_list():
        root = engine.project_root
        if not root:
            return {"error": "No project root"}
        from .util import find_project_files
        prefabs = find_project_files(root, "*.zpep")
        return {"prefabs": prefabs, "count": len(prefabs)}

    @registry.tool(
        "prefab_instantiate",
        "Instantiate a prefab into the scene",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to .zpep prefab file (relative to project)"},
                "name": {"type": "string", "description": "Optional name for the instance", "default": ""},
            },
            "required": ["path"],
        },
    )
    def prefab_instantiate(path="", name=""):
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        full = os.path.join(engine.project_root, path) if not os.path.isabs(path) else path
        if not os.path.isfile(full):
            return {"error": f"Prefab not found: {path}"}
        from core.ecs.prefab import Prefab
        try:
            prefab = Prefab.load(full)
            entity = prefab.instantiate(scene)
            if name:
                entity.name = name
            return {"id": entity.id, "name": entity.name, "message": f"Instantiated prefab '{path}'"}
        except Exception as ex:
            return {"error": f"Failed to instantiate prefab: {ex}"}

    @registry.tool(
        "prefab_create",
        "Create a prefab from an existing entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID to save as prefab"},
                "save_path": {"type": "string", "description": "Output path (relative to project, e.g. prefabs/MyPrefab.zpep)"},
            },
            "required": ["entity_id", "save_path"],
        },
    )
    def prefab_create(entity_id="", save_path=""):
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        e = scene.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        full = os.path.join(engine.project_root, save_path) if not os.path.isabs(save_path) else save_path
        from core.ecs.prefab import Prefab
        try:
            os.makedirs(os.path.dirname(full), exist_ok=True)
            prefab = Prefab(e.name)
            prefab.capture([e])
            prefab.save(full)
            return {"message": f"Saved prefab to {save_path}"}
        except Exception as ex:
            return {"error": f"Failed to create prefab: {ex}"}
