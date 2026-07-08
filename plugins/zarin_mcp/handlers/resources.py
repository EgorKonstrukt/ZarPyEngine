# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from .util import serialize_component, find_project_files


def register(registry, engine):

    @registry.resource(
        "scene://entities",
        name="Scene Entities",
        description="Full list of all entities with component data",
    )
    def res_scene_entities():
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        results = []
        for e in scene.get_all_entities():
            results.append({
                "id": e.id,
                "name": e.name,
                "active": e.active,
                "component_count": len(e.get_all_components()),
                "component_types": [type(c).__name__ for c in e.get_all_components()],
                "child_count": len(e.children),
                "parent_id": e.parent.id if e.parent else None,
                "layer": e.layer,
                "tags": list(e.tags),
                "is_prefab_instance": e.is_prefab_instance,
            })
        return {"entities": results, "count": len(results), "scene_name": scene.name}

    @registry.resource(
        "scene://hierarchy",
        name="Scene Hierarchy",
        description="Entity hierarchy tree",
    )
    def res_scene_hierarchy():
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        def build_tree(e):
            comps = [type(c).__name__ for c in e.get_all_components()]
            return {
                "id": e.id, "name": e.name, "active": e.active,
                "components": comps,
                "children": [build_tree(c) for c in e.children],
            }
        roots = [build_tree(e) for e in scene.get_root_entities()]
        return {"hierarchy": roots, "scene_name": scene.name}

    @registry.resource(
        "project://info",
        name="Project Info",
        description="Project root, settings, scenes, assets overview",
    )
    def res_project_info():
        root = engine.project_root
        scene = engine.scene
        return {
            "project_root": root,
            "project_name": os.path.basename(root) if root else "",
            "loaded_scene": scene.name if scene else None,
            "entity_count": len(scene.get_all_entities()) if scene else 0,
        }

    @registry.resource(
        "engine://status",
        name="Engine Status",
        description="Engine runtime status snapshot",
    )
    def res_engine_status():
        scene = engine.scene
        return {
            "play_mode": engine.play_mode,
            "fps": engine.fps,
            "frame_count": engine.frame_count,
            "time_scale": engine.time_scale,
            "scene_name": scene.name if scene else None,
            "entity_count": len(scene.get_all_entities()) if scene else 0,
        }

    @registry.resource(
        "editor://selection",
        name="Editor Selection",
        description="Currently selected entities",
    )
    def res_editor_selection():
        viewport = getattr(engine, "viewport", None)
        if viewport is None:
            return {"selection": []}
        selected = getattr(viewport, "_selected_entities", [])
        return {
            "selection": [
                {"id": e.id, "name": e.name} for e in selected
            ],
            "count": len(selected),
        }

    @registry.resource(
        "console://log",
        name="Console Log",
        description="Recent console log entries",
    )
    def res_console_log():
        from core.logger import Logger
        entries = Logger.get_entries()[-100:]
        return {
            "entries": [
                {
                    "level": e.level.name,
                    "message": e.message,
                    "timestamp": e.timestamp,
                }
                for e in entries
            ],
            "count": len(entries),
            "total": len(Logger.get_entries()),
        }

    @registry.resource(
        "project://scenes",
        name="Project Scenes",
        description="List of all .scene files in the project",
    )
    def res_project_scenes():
        root = engine.project_root
        if not root:
            return {"scenes": []}
        files = find_project_files(root, "*.scene")
        return {"scenes": files, "count": len(files)}

    @registry.resource(
        "prefab://list",
        name="Prefab List",
        description="All prefab files in the project",
    )
    def res_prefab_list():
        root = engine.project_root
        if not root:
            return {"prefabs": []}
        files = find_project_files(root, "*.zpep")
        return {"prefabs": files, "count": len(files)}

    @registry.resource(
        "asset://overview",
        name="Asset Overview",
        description="Asset counts by category",
    )
    def res_asset_overview():
        root = engine.project_root
        if not root:
            return {"error": "No project root"}
        assets_dir = os.path.join(root, "assets")
        if not os.path.isdir(assets_dir):
            return {"assets": {}}
        return {
            "assets": {
                "meshes": len(find_project_files(assets_dir, "*.{obj,fbx,gltf,glb}")),
                "textures": len(find_project_files(assets_dir, "*.{png,jpg,jpeg,tga,dds}")),
                "audio": len(find_project_files(assets_dir, "*.{wav,ogg,mp3}")),
                "scripts": len(find_project_files(assets_dir, "*.{py,lua}")),
            }
        }

    @registry.resource_template(
        "scene://entity/{entity_id}",
        name="Single Entity",
        description="Full entity data by UUID",
    )
    def res_entity_template(entity_id=""):
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        e = scene.get_entity(entity_id)
        if e is None:
            return {"error": f"Entity not found: {entity_id}"}
        return {
            "id": e.id,
            "name": e.name,
            "active": e.active,
            "parent_id": e.parent.id if e.parent else None,
            "children": [{"id": c.id, "name": c.name} for c in e.children],
            "tags": list(e.tags),
            "layer": e.layer,
            "is_prefab_instance": e.is_prefab_instance,
            "components": [serialize_component(c) for c in e.get_all_components()],
        }

    @registry.resource_template(
        "scene://component/{component_type}",
        name="Component Type Index",
        description="All entities with a given component type",
    )
    def res_component_index(component_type=""):
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        from core.ecs import ComponentRegistry
        cls = ComponentRegistry.get(component_type)
        if cls is None:
            return {"error": f"Unknown component: {component_type}"}
        entities = scene.get_entities_with_component(cls)
        return {
            "component_type": component_type,
            "entities": [
                {"id": e.id, "name": e.name} for e in entities
            ],
            "count": len(entities),
        }

    @registry.prompt(
        "analyze_scene",
        "Analyze the current scene structure and provide a summary",
        [
            {
                "name": "detail",
                "description": "Detail level: basic, full",
                "required": False,
            }
        ],
    )
    def prompt_analyze_scene(detail="basic"):
        scene = engine.scene
        if scene is None:
            return {
                "messages": [
                    {
                        "role": "user",
                        "content": {
                            "type": "text",
                            "text": "No scene is currently loaded in the engine.",
                        },
                    }
                ]
            }
        total = len(scene.get_all_entities())
        root_count = len(scene.get_root_entities())
        comp_types = {}
        for e in scene.get_all_entities():
            for c in e.get_all_components():
                name = type(c).__name__
                comp_types[name] = comp_types.get(name, 0) + 1
        text = f"Scene: {scene.name}\nEntities: {total}\nRoot entities: {root_count}\nComponents: {comp_types}"
        if detail == "full":
            text += f"\nPath: {scene.path}\nDirty: {scene.dirty}\n"
            text += f"Play mode: {engine.play_mode}\nFPS: {engine.fps:.1f}"
        return {
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": text},
                }
            ]
        }

    @registry.prompt(
        "project_overview",
        "Overview of the entire project",
        [],
    )
    def prompt_project_overview():
        root = engine.project_root
        scene = engine.scene
        text = f"Project: {os.path.basename(root) if root else 'N/A'}\n"
        text += f"Root: {root}\n"
        text += f"Loaded scene: {scene.name if scene else 'None'}\n"
        text += f"Entities: {len(scene.get_all_entities()) if scene else 0}\n"
        text += f"Play mode: {engine.play_mode}\n"
        text += f"FPS: {engine.fps:.1f}\n"
        text += f"Frame: {engine.frame_count}\n"
        text += f"Time scale: {engine.time_scale}\n"
        return {
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": text},
                }
            ]
        }
