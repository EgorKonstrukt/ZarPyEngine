# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from .util import find_project_files, resolve_path, serialize_component


def register(registry, engine):
    _root = lambda: engine.project_root

    @registry.tool(
        "asset_list",
        "List all asset files in the project by type",
        {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "File glob pattern, e.g. '*.png', '*.obj', '*.wav'",
                    "default": "*",
                },
                "directory": {
                    "type": "string",
                    "description": "Subdirectory within project (default: assets/)",
                    "default": "assets",
                },
            },
        },
    )
    def asset_list(pattern="*", directory="assets"):
        root = _root()
        if not root:
            return {"error": "No project root"}
        search_dir = os.path.join(root, directory) if directory else root
        if not os.path.isdir(search_dir):
            return {"error": f"Directory not found: {directory}"}
        files = find_project_files(search_dir, pattern)
        return {"assets": files, "count": len(files), "directory": directory}

    @registry.tool(
        "asset_list_meshes",
        "List all 3D mesh files in the project",
        {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Subdirectory to search (default: assets/)",
                    "default": "assets",
                }
            },
        },
    )
    def asset_list_meshes(directory="assets"):
        root = _root()
        if not root:
            return {"error": "No project root"}
        search_dir = os.path.join(root, directory) if directory else root
        if not os.path.isdir(search_dir):
            return {"error": f"Directory not found: {directory}"}
        files = find_project_files(search_dir, "*.{obj,fbx,gltf,glb,dae,blend,3ds,stl}")
        return {"meshes": files, "count": len(files)}

    @registry.tool(
        "asset_list_textures",
        "List all texture/image files in the project",
        {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Subdirectory to search (default: assets/)",
                    "default": "assets",
                }
            },
        },
    )
    def asset_list_textures(directory="assets"):
        root = _root()
        if not root:
            return {"error": "No project root"}
        search_dir = os.path.join(root, directory) if directory else root
        if not os.path.isdir(search_dir):
            return {"error": f"Directory not found: {directory}"}
        files = find_project_files(search_dir, "*.{png,jpg,jpeg,tga,bmp,dds,exr,hdr,psd,svg}")
        return {"textures": files, "count": len(files)}

    @registry.tool(
        "asset_list_audio",
        "List all audio files in the project",
        {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Subdirectory to search (default: assets/)",
                    "default": "assets",
                }
            },
        },
    )
    def asset_list_audio(directory="assets"):
        root = _root()
        if not root:
            return {"error": "No project root"}
        search_dir = os.path.join(root, directory) if directory else root
        if not os.path.isdir(search_dir):
            return {"error": f"Directory not found: {directory}"}
        files = find_project_files(search_dir, "*.{wav,ogg,mp3,flac,aiff,aac,m4a}")
        return {"audio_clips": files, "count": len(files)}

    @registry.tool(
        "asset_import_mesh",
        "Import a 3D mesh file using the asset importer",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to mesh file (relative to project)"},
            },
            "required": ["path"],
        },
    )
    def asset_import_mesh(path=""):
        full = resolve_path(engine, path)
        if not os.path.isfile(full):
            return {"error": f"File not found: {path}"}
        from core.assets.asset_importer import load_mesh, load_obj
        ext = os.path.splitext(full)[1].lower()
        try:
            if ext == ".obj":
                data = load_obj(full)
            else:
                data = load_mesh(full)
            return {
                "message": f"Imported mesh from {path}",
                "vertex_count": len(data.vertices) if hasattr(data, "vertices") else 0,
                "index_count": len(data.indices) if hasattr(data, "indices") else 0,
                "has_normals": hasattr(data, "normals") and len(data.normals) > 0,
                "has_uvs": hasattr(data, "uvs") and len(data.uvs) > 0,
            }
        except Exception as ex:
            return {"error": f"Failed to import mesh: {ex}"}

    @registry.tool(
        "audio_list_clips",
        "List all loaded audio clips",
        {"type": "object", "properties": {}},
    )
    def audio_list_clips():
        try:
            from core.audio.audio_system import AudioSystem, AudioSourceManager
            system = AudioSystem.instance()
            if system is None:
                return {"audio_clips": [], "message": "Audio system not active"}
            mgr = AudioSourceManager.instance()
            if mgr is None:
                return {"audio_clips": []}
            active = [{"source_id": sid} for sid in mgr._active_sources] if hasattr(mgr, "_active_sources") else []
            return {"audio_clips": active, "active_count": len(active)}
        except Exception as ex:
            return {"error": f"Audio system error: {ex}"}

    @registry.tool(
        "audio_play_clip",
        "Play an audio clip by file path",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to audio file (relative to project)"},
                "volume": {"type": "number", "description": "Volume 0-1", "default": 1.0},
                "loop": {"type": "boolean", "description": "Loop playback", "default": False},
            },
            "required": ["path"],
        },
    )
    def audio_play_clip(path="", volume=1.0, loop=False):
        full = resolve_path(engine, path)
        if not os.path.isfile(full):
            return {"error": f"File not found: {path}"}
        try:
            from core.audio.audio_system import AudioSystem, AudioSourceManager
            system = AudioSystem.instance()
            if system is None:
                return {"error": "Audio system not active"}
            clip = system.load_clip(full)
            if clip is None:
                return {"error": "Failed to load audio clip"}
            mgr = AudioSourceManager.instance()
            source_id = mgr.play(clip, volume=volume, loop=loop)
            return {"message": f"Playing {path}", "source_id": source_id}
        except Exception as ex:
            return {"error": f"Audio play error: {ex}"}

    @registry.tool(
        "audio_stop_all",
        "Stop all active audio playback",
        {"type": "object", "properties": {}},
    )
    def audio_stop_all():
        try:
            from core.audio.audio_system import AudioSourceManager
            mgr = AudioSourceManager.instance()
            if mgr:
                mgr.stop_all()
                return {"message": "All audio stopped"}
            return {"error": "Audio system not active"}
        except Exception as ex:
            return {"error": f"Audio error: {ex}"}

    @registry.tool(
        "audio_set_volume",
        "Set audio volume levels",
        {
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "enum": ["master", "sfx", "music"],
                    "description": "Audio channel",
                    "default": "master",
                },
                "volume": {
                    "type": "number",
                    "description": "Volume 0.0 - 1.0",
                },
            },
            "required": ["channel", "volume"],
        },
    )
    def audio_set_volume(channel="master", volume=1.0):
        try:
            from core.audio.audio_system import AudioSystem
            system = AudioSystem.instance()
            if system is None:
                return {"error": "Audio system not active"}
            volume = max(0.0, min(1.0, volume))
            if channel == "master":
                system.master_volume = volume
            elif channel == "sfx":
                from core.config.config import get_project_config
                cfg = get_project_config(".", lazy=True)
                if cfg:
                    cfg.set("audio.sfx_volume", volume)
                    cfg.save()
            elif channel == "music":
                from core.config.config import get_project_config
                cfg = get_project_config(".", lazy=True)
                if cfg:
                    cfg.set("audio.music_volume", volume)
                    cfg.save()
            return {"message": f"Set {channel} volume to {volume}"}
        except Exception as ex:
            return {"error": f"Audio error: {ex}"}

    @registry.tool(
        "material_list",
        "List all materials in the scene (entities with Material component)",
        {"type": "object", "properties": {}},
    )
    def material_list():
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        from core.ecs.ecs import ComponentRegistry
        mat_cls = ComponentRegistry.get("Material")
        if mat_cls is None:
            return {"materials": [], "message": "No Material component registered"}
        entities = scene.get_entities_with_component(mat_cls)
        results = []
        for e in entities:
            mat = e.get_component(mat_cls)
            results.append({
                "entity_id": e.id,
                "entity_name": e.name,
                "material": serialize_component(mat) if mat else None,
            })
        return {"materials": results, "count": len(results)}

    @registry.tool(
        "material_get",
        "Get material properties from an entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID with Material component"},
            },
            "required": ["entity_id"],
        },
    )
    def material_get(entity_id=""):
        scene = engine.scene
        if scene is None:
            return {"error": "No scene loaded"}
        from core.ecs.ecs import ComponentRegistry
        e = scene.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        mat_cls = ComponentRegistry.get("Material")
        if mat_cls is None:
            return {"error": "No Material component registered"}
        mat = e.get_component(mat_cls)
        if mat is None:
            return {"error": "Entity has no Material"}
        return {"material": serialize_component(mat)}


