# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from .util import get_scene, get_entity_by_id_or_name, serialize_component


def register(registry, engine):
    _scene = lambda: engine.scene

    @registry.tool(
        "component_list_types",
        "List all registered component types with categories",
        {"type": "object", "properties": {}},
    )
    def component_list_types():
        from core.ecs import ComponentRegistry
        all_comps = ComponentRegistry.all()
        cats = ComponentRegistry.all_categories()
        result = {}
        for name, cls in all_comps.items():
            module = getattr(cls, "__module__", "")
            cat = cats.get(name, ["Other"])
            result[name] = {
                "category": cat[0] if cat else "Other",
                "module": module,
                "allow_multiple": getattr(cls, "_allow_multiple", False),
            }
        return {"component_types": result, "count": len(result)}

    @registry.tool(
        "component_add",
        "Add a component to an entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "component_type": {"type": "string", "description": "Component class name"},
            },
            "required": ["entity_id", "component_type"],
        },
    )
    def component_add(entity_id="", component_type=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        from core.ecs import ComponentRegistry
        cls = ComponentRegistry.get(component_type)
        if cls is None:
            return {"error": f"Unknown component: {component_type}"}
        try:
            inst = cls()
            e.add_component(inst)
            return {"message": f"Added {component_type} to '{e.name}'", "component": serialize_component(inst)}
        except Exception as ex:
            return {"error": f"Failed to add component: {ex}"}

    @registry.tool(
        "component_remove",
        "Remove a component from an entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "component_type": {"type": "string", "description": "Component class name to remove"},
            },
            "required": ["entity_id", "component_type"],
        },
    )
    def component_remove(entity_id="", component_type=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        from core.ecs import ComponentRegistry
        cls = ComponentRegistry.get(component_type)
        if cls is None:
            return {"error": f"Unknown component: {component_type}"}
        if not e.has_component(cls):
            return {"error": f"Entity has no {component_type}"}
        if component_type == "Transform":
            return {"error": "Cannot remove Transform"}
        e.remove_component(cls)
        return {"message": f"Removed {component_type} from '{e.name}'"}

    @registry.tool(
        "component_get",
        "Get all properties of a component on an entity",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "component_type": {"type": "string", "description": "Component class name"},
            },
            "required": ["entity_id", "component_type"],
        },
    )
    def component_get(entity_id="", component_type=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        from core.ecs import ComponentRegistry
        cls = ComponentRegistry.get(component_type)
        if cls is None:
            return {"error": f"Unknown component: {component_type}"}
        comp = e.get_component(cls)
        if comp is None:
            return {"error": f"Entity has no {component_type}"}
        return {"component": serialize_component(comp)}

    @registry.tool(
        "component_set_property",
        "Set a property on a component. For transform use transform_* tools.",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "component_type": {"type": "string", "description": "Component class name"},
                "property": {"type": "string", "description": "Property name"},
                "value": {"description": "Value (number, string, bool, or array for vectors)"},
            },
            "required": ["entity_id", "component_type", "property", "value"],
        },
    )
    def component_set_property(entity_id="", component_type="", property="", value=None):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        from core.ecs import ComponentRegistry
        cls = ComponentRegistry.get(component_type)
        if cls is None:
            return {"error": f"Unknown component: {component_type}"}
        comp = e.get_component(cls)
        if comp is None:
            return {"error": f"Entity has no {component_type}"}
        try:
            setattr(comp, property, value)
            return {"message": f"Set {component_type}.{property} = {value}"}
        except Exception as ex:
            return {"error": f"Failed to set property: {ex}"}

    @registry.tool(
        "component_move",
        "Re-order a component on an entity (move up/down)",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "component_key": {"type": "string", "description": "Component key (e.g. 'MeshFilter')"},
                "direction": {
                    "type": "integer",
                    "description": "1 = move down, -1 = move up",
                },
            },
            "required": ["entity_id", "component_key", "direction"],
        },
    )
    def component_move(entity_id="", component_key="", direction=0):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        try:
            e.move_component(component_key, direction)
            return {"message": f"Moved component '{component_key}' on '{e.name}'"}
        except Exception as ex:
            return {"error": f"Failed to move component: {ex}"}

    @registry.tool(
        "transform_get",
        "Get entity transform (position, rotation, scale, direction vectors)",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
            },
            "required": ["entity_id"],
        },
    )
    def transform_get(entity_id=""):
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        t = e.transform
        if t is None:
            return {"error": "Entity has no Transform"}
        return {
            "position": [t.position.x, t.position.y, t.position.z],
            "local_position": [t.local_position.x, t.local_position.y, t.local_position.z],
            "local_euler": [t.local_euler_angles.x, t.local_euler_angles.y, t.local_euler_angles.z],
            "local_scale": [t.local_scale.x, t.local_scale.y, t.local_scale.z],
            "forward": [t.forward.x, t.forward.y, t.forward.z],
            "up": [t.up.x, t.up.y, t.up.z],
            "right": [t.right.x, t.right.y, t.right.z],
        }

    @registry.tool(
        "transform_set_position",
        "Set entity world position [x, y, z]",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] world position",
                },
            },
            "required": ["entity_id", "position"],
        },
    )
    def transform_set_position(entity_id="", position=None):
        if position is None:
            position = [0, 0, 0]
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        t = e.transform
        if t is None:
            return {"error": "Entity has no Transform"}
        from core.math3d import Vec3
        t.position = Vec3(*position)
        return {"message": f"Set position to {position}"}

    @registry.tool(
        "transform_set_local_position",
        "Set entity local position [x, y, z]",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "position": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] local position",
                },
            },
            "required": ["entity_id", "position"],
        },
    )
    def transform_set_local_position(entity_id="", position=None):
        if position is None:
            position = [0, 0, 0]
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        t = e.transform
        if t is None:
            return {"error": "Entity has no Transform"}
        from core.math3d import Vec3
        t.local_position = Vec3(*position)
        return {"message": f"Set local position to {position}"}

    @registry.tool(
        "transform_set_rotation",
        "Set entity local euler rotation [pitch, yaw, roll] in degrees",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "euler": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[pitch, yaw, roll] in degrees",
                },
            },
            "required": ["entity_id", "euler"],
        },
    )
    def transform_set_rotation(entity_id="", euler=None):
        if euler is None:
            euler = [0, 0, 0]
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        t = e.transform
        if t is None:
            return {"error": "Entity has no Transform"}
        from core.math3d import Vec3
        t.local_euler_angles = Vec3(*euler)
        return {"message": f"Set rotation to {euler}"}

    @registry.tool(
        "transform_set_scale",
        "Set entity local scale [sx, sy, sz]",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[sx, sy, sz] scale factors",
                },
            },
            "required": ["entity_id", "scale"],
        },
    )
    def transform_set_scale(entity_id="", scale=None):
        if scale is None:
            scale = [1, 1, 1]
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        t = e.transform
        if t is None:
            return {"error": "Entity has no Transform"}
        from core.math3d import Vec3
        t.local_scale = Vec3(*scale)
        return {"message": f"Set scale to {scale}"}

    @registry.tool(
        "transform_look_at",
        "Make entity look at a target position",
        {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "Entity UUID"},
                "target": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] target world position",
                },
                "up": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "[x, y, z] up vector, default [0,1,0]",
                },
            },
            "required": ["entity_id", "target"],
        },
    )
    def transform_look_at(entity_id="", target=None, up=None):
        if target is None:
            target = [0, 0, 0]
        if up is None:
            up = [0, 1, 0]
        s = _scene()
        if s is None:
            return {"error": "No scene loaded"}
        e = get_entity_by_id_or_name(s, entity_id)
        if e is None:
            return {"error": "Entity not found"}
        t = e.transform
        if t is None:
            return {"error": "Entity has no Transform"}
        from core.math3d import Vec3, Quat
        direction = Vec3(*target) - t.position
        if direction.length() < 1e-10:
            return {"error": "Target is at the same position as entity"}
        t.rotation = Quat.look_rotation(direction, Vec3(*up))
        return {"message": f"Look at {target}"}
