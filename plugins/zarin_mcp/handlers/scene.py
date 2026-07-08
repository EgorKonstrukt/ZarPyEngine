# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
from .util import get_scene, get_entity_by_id_or_name, serialize_component


def register(registry, engine):
    _scene = lambda: engine.scene

    @registry.tool(
        "scene_list_entities",
        "List all entities with basic info (id, name, active, components, children)",
        {
            "type": "object",
            "properties": {
                "include_components": {
                    "type": "boolean",
                    "description": "Include full component data",
                    "default": False,
                }
            },
        },
    )
    def scene_list_entities(include_components=False):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        results = []
        for e in s.get_all_entities():
            entry = {
                "id": e.id,
                "name": e.name,
                "active": e.active,
                "component_count": len(e.get_all_components()),
                "component_types": [type(c).__name__ for c in e.get_all_components()],
                "child_count": len(e.children),
                "parent_id": e.parent.id if e.parent else None,
                "layer": e.layer,
                "tags": list(e.tags),
            }
            if include_components:
                entry["components"] = [serialize_component(c) for c in e.get_all_components()]
            results.append(entry)
        return {"entities": results, "count": len(results)}

    @registry.tool(
        "scene_get_entity",
        "Get full details of an entity by id or name",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "entity_name": {"type": "string", "description": "Entity name fallback"},
            },
        },
    )
    def scene_get_entity(entity_id="", entity_name=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id, entity_name)
        if e is None:
            return {"error": "Entity not found"}
        return {
            "id": e.id,
            "name": e.name,
            "active": e.active,
            "parent_id": e.parent.id if e.parent else None,
            "children": [{"id": c.id, "name": c.name} for c in e.children],
            "components": [serialize_component(c) for c in e.get_all_components()],
            "tags": list(e.tags),
            "layer": e.layer,
            "is_prefab_instance": e.is_prefab_instance,
            "prefab_guid": e.prefab_guid,
        }

    @registry.tool(
        "scene_create_entity",
        "Create a new empty entity with Transform",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Entity name", "default": "GameObject"},
                "parent_id": {"type": "string", "description": "Optional parent UUID"},
            },
        },
    )
    def scene_create_entity(name="GameObject", parent_id=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        from core.ecs import Entity
        from core.components.transform.transform import Transform
        e = s.create_entity(name)
        e.add_component(Transform())
        if parent_id:
            parent = s.get_entity(parent_id)
            if parent:
                e.set_parent(parent)
        return {"id": e.id, "name": e.name, "message": f"Created entity '{e.name}'"}

    @registry.tool(
        "scene_duplicate_entity",
        "Duplicate an entity and its children",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID to duplicate"},
                "new_name": {"type": "string", "description": "Optional new name", "default": ""},
            },
            "required": ["entity_id"],
        },
    )
    def scene_duplicate_entity(entity_id="", new_name=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        src = s.get_entity(entity_id)
        if src is None:
            return {"error": "Entity not found"}
        import json, copy
        from core.ecs import Entity, ComponentRegistry
        data = json.loads(json.dumps(src.serialize(), default=str))
        data["id"] = str(__import__("uuid").uuid4())
        data["parent"] = src.parent.id if src.parent else None
        if new_name:
            data["name"] = new_name
        dup = Entity.deserialize(data, ComponentRegistry)
        s.add_entity(dup)
        if src.parent:
            dup.set_parent(src.parent)
        return {"id": dup.id, "name": dup.name, "message": f"Duplicated '{src.name}' as '{dup.name}'"}

    @registry.tool(
        "scene_delete_entity",
        "Delete an entity by id",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID to delete"},
            },
            "required": ["entity_id"],
        },
    )
    def scene_delete_entity(entity_id=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = s.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        name, eid = e.name, e.id
        s.remove_entity(entity_id)
        return {"message": f"Deleted entity '{name}' ({eid})"}

    @registry.tool(
        "scene_batch_delete",
        "Delete multiple entities at once",
        {
            "type": "object",
            "properties": {
                "entity_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of entity UUIDs to delete",
                }
            },
            "required": ["entity_ids"],
        },
    )
    def scene_batch_delete(entity_ids=None):
        if entity_ids is None:
            entity_ids = []
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        deleted = []
        for eid in entity_ids:
            e = s.get_entity(eid)
            if e:
                deleted.append(e.name)
                s.remove_entity(eid)
        return {"message": f"Deleted {len(deleted)} entities", "names": deleted}

    @registry.tool(
        "scene_rename_entity",
        "Rename an entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "new_name": {"type": "string", "description": "New name"},
            },
            "required": ["entity_id", "new_name"],
        },
    )
    def scene_rename_entity(entity_id="", new_name=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = s.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        old = e.name
        e.name = new_name
        return {"message": f"Renamed '{old}' to '{new_name}'"}

    @registry.tool(
        "scene_set_entity_active",
        "Set an entity's active state",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "active": {"type": "boolean", "description": "Active state"},
            },
            "required": ["entity_id", "active"],
        },
    )
    def scene_set_entity_active(entity_id="", active=True):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = s.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        e.active = active
        return {"message": f"Set '{e.name}' active={active}"}

    @registry.tool(
        "scene_reparent_entity",
        "Change an entity's parent",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID to reparent"},
                "new_parent_id": {
                    "type": "string",
                    "description": "New parent UUID (empty string to detach)",
                },
            },
            "required": ["entity_id", "new_parent_id"],
        },
    )
    def scene_reparent_entity(entity_id="", new_parent_id=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = s.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        if new_parent_id:
            parent = s.get_entity(new_parent_id)
            if parent is None:
                return {"error": "Parent entity not found"}
            e.set_parent(parent)
        else:
            e.set_parent(None)
        return {"message": f"Reparented '{e.name}'"}

    @registry.tool(
        "scene_get_hierarchy",
        "Get the full entity hierarchy tree",
        {"type": "object", "properties": {}},
    )
    def scene_get_hierarchy():
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        def build_tree(e):
            comps = [type(c).__name__ for c in e.get_all_components()]
            return {
                "id": e.id, "name": e.name, "active": e.active,
                "components": comps, "layer": e.layer,
                "children": [build_tree(c) for c in e.children],
            }
        roots = [build_tree(e) for e in s.get_root_entities()]
        return {"hierarchy": roots, "scene_name": s.name}

    @registry.tool(
        "scene_get_entities_by_component",
        "Find all entities with a specific component type",
        {
            "type": "object",
            "properties": {
                "component_type": {
                    "type": "string",
                    "description": "Component class name (e.g. MeshRenderer, Light)",
                }
            },
            "required": ["component_type"],
        },
    )
    def scene_get_entities_by_component(component_type=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        from core.ecs import ComponentRegistry
        cls = ComponentRegistry.get(component_type)
        if cls is None:
            return {"error": f"Unknown component: {component_type}"}
        entities = s.get_entities_with_component(cls)
        return {
            "entities": [
                {"id": e.id, "name": e.name} for e in entities
            ],
            "count": len(entities),
        }

    @registry.tool(
        "scene_set_entity_tag",
        "Set tags on an entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of tags",
                },
            },
            "required": ["entity_id", "tags"],
        },
    )
    def scene_set_entity_tag(entity_id="", tags=None):
        if tags is None:
            tags = []
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = s.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        e.tags.clear()
        e.tags.update(tags)
        return {"message": f"Set tags on '{e.name}': {tags}"}

    @registry.tool(
        "scene_get_entities_by_tag",
        "Find all entities with a specific tag",
        {
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Tag to search for"},
            },
            "required": ["tag"],
        },
    )
    def scene_get_entities_by_tag(tag=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        results = []
        for e in s.get_all_entities():
            if tag in e.tags:
                results.append({"id": e.id, "name": e.name})
        return {"entities": results, "count": len(results)}

    @registry.tool(
        "scene_set_entity_layer",
        "Set layer on an entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "layer": {"type": "integer", "description": "Layer index (0-31)"},
            },
            "required": ["entity_id", "layer"],
        },
    )
    def scene_set_entity_layer(entity_id="", layer=0):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = s.get_entity(entity_id)
        if e is None:
            return {"error": "Entity not found"}
        e.layer = layer
        return {"message": f"Set layer={layer} on '{e.name}'"}

    @registry.tool(
        "scene_create_primitive",
        "Create a 3D primitive (cube, sphere, plane)",
        {
            "type": "object",
            "properties": {
                "mesh": {
                    "type": "string",
                    "enum": ["cube", "sphere", "plane"],
                    "description": "Primitive type",
                },
                "name": {"type": "string", "description": "Optional name", "default": ""},
            },
            "required": ["mesh"],
        },
    )
    def scene_create_primitive(mesh="", name=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        from core.ecs import Entity
        from core.components.transform.transform import Transform
        from core.components.rendering.mesh_filter import MeshFilter
        from core.components.rendering.mesh_renderer import MeshRenderer
        e = s.create_entity(name or mesh.capitalize())
        e.add_component(Transform())
        mf = MeshFilter()
        mf.mesh_name = mesh
        e.add_component(mf)
        e.add_component(MeshRenderer())
        return {"id": e.id, "name": e.name, "message": f"Created {mesh} '{e.name}'"}

    @registry.tool(
        "scene_create_light",
        "Create a light (directional, point, spot)",
        {
            "type": "object",
            "properties": {
                "light_type": {
                    "type": "string",
                    "enum": ["directional", "point", "spot"],
                    "description": "Light type",
                },
                "name": {"type": "string", "description": "Optional name", "default": ""},
            },
            "required": ["light_type"],
        },
    )
    def scene_create_light(light_type="", name=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        from core.ecs import Entity
        from core.components.transform.transform import Transform
        from core.components.lighting.light import Light, LightType
        name_map = {"directional": "Directional Light", "point": "Point Light", "spot": "Spot Light"}
        type_map = {"directional": LightType.DIRECTIONAL, "point": LightType.POINT, "spot": LightType.SPOT}
        if light_type not in type_map:
            return {"error": f"Unknown light type: {light_type}"}
        e = s.create_entity(name or name_map[light_type])
        e.add_component(Transform())
        lc = Light()
        lc.light_type = type_map[light_type]
        e.add_component(lc)
        return {"id": e.id, "name": e.name, "message": f"Created {light_type} light"}

    @registry.tool(
        "scene_create_camera",
        "Create a Camera entity",
        {"type": "object", "properties": {}},
    )
    def scene_create_camera():
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        from core.ecs import Entity
        from core.components.transform.transform import Transform
        from core.components.rendering.camera import Camera
        e = s.create_entity("Camera")
        e.add_component(Transform())
        e.add_component(Camera())
        return {"id": e.id, "name": e.name, "message": "Created Camera"}

    @registry.tool(
        "scene_save",
        "Save current scene to its file path",
        {"type": "object", "properties": {}},
    )
    def scene_save():
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        if not s.path:
            return {"error": "Scene has no path. Save via editor first."}
        engine.save_scene()
        return {"message": f"Scene saved to {s.path}"}

    @registry.tool(
        "scene_load",
        "Load a scene file. Lists available scenes if path is empty.",
        {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to scene file (relative to project)"},
            },
        },
    )
    def scene_load(path=""):
        if not path:
            from .project import register as _proj_reg
            scenes = _get_project_scenes(engine)
            return {"message": "Provide a path. Available scenes:", "scenes": scenes}
        full = os.path.join(engine.project_root, path) if not os.path.isabs(path) else path
        scene = engine.load_scene(full)
        if scene is None:
            return {"error": f"Failed to load scene: {path}"}
        return {"message": f"Loaded scene '{scene.name}'"}

    @registry.tool(
        "scene_new",
        "Create a new empty scene",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Scene name", "default": "NewScene"},
            },
        },
    )
    def scene_new(name="NewScene"):
        engine.new_scene(name)
        return {"message": f"Created new scene '{name}'"}


def _get_project_scenes(engine):
    root = engine.project_root
    if not root:
        return []
    import glob
    scenes = glob.glob(os.path.join(root, "**", "*.scene"), recursive=True)
    return sorted(os.path.relpath(s, root).replace("\\", "/") for s in scenes)
