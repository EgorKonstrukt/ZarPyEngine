# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
from typing import Optional, Any, get_type_hints
from core.ecs.ecs import Component, ComponentRegistry
from core.foundation.logger import Logger
from core.components.inspector_meta import FieldType, InspectorField
from core.maths.math3d import Vec2, Vec3, Vec4
from core.foundation.curve import Curve
from core.input.input_system import Input, KeyCode
import importlib.util
import os

RESOURCE_TYPE_FILTERS = {
    "mesh": "Models (*.obj *.fbx *.stl *.gltf *.glb *.usdz *.dae *.3ds *.blend)",
    "material": "Materials (*.zpem *.mat)",
    "texture": "Images (*.png *.jpg *.jpeg *.bmp *.tga *.tif *.tiff *.webp *.hdr *.exr *.dds *.svg)",
    "audio": "Audio (*.wav *.mp3 *.ogg *.flac *.aiff *.m4a)",
    "script": "Python Scripts (*.py)",
    "prefab": "Prefabs (*.zpep)",
    "scene": "Scenes (*.zpes)",
    "animclip": "Animation Clips (*.animclip)",
    "animcontroller": "Animator Controllers (*.animcontroller)",
}

PY_TYPE_TO_FIELD = {
    float: FieldType.FLOAT,
    int: FieldType.INT,
    bool: FieldType.BOOL,
    str: FieldType.STRING,
    Vec2: FieldType.VEC2,
    Vec3: FieldType.VEC3,
    Vec4: FieldType.VEC4,
}

@ComponentRegistry.register
class ScriptComponent(Component):
    _icon = "Script.png"
    _allow_multiple = True
    _gizmo_pass = "script"
    _show_gizmo_icon: bool = False

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("script_path", "Script", FieldType.RESOURCE_PATH, file_filter="Python Scripts (*.py)"),
        ]

    def __init__(self):
        super().__init__()
        self.script_path: str = ""
        self._py_instance: Optional[Any] = None
        self._py_module: Optional[Any] = None
        self._py_class: Optional[type] = None
        self._field_values: dict[str, Any] = {}
        self._cached_fields: list[InspectorField] = []
        self._cached_hints: dict[str, Any] | None = None
        self._py_has_update: bool = False
        self._py_has_fixed_update: bool = False
        self._py_has_awake: bool = False
        self._py_has_start: bool = False
        self._py_has_destroy: bool = False

    def get_script_public_fields(self) -> list[InspectorField]:
        if not self.script_path:
            self._cached_fields = []
            return []
        if self._cached_fields:
            return self._cached_fields
        if self._py_class is None:
            self._load_script_class()
        if self._py_class is None:
            return []
        self._cached_fields = self._build_fields_from_class(self._py_class)
        return self._cached_fields

    def _load_script_class(self):
        if not self.script_path:
            return
        script_path = self.script_path
        if not os.path.isabs(script_path) and not os.path.exists(script_path):
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng is not None:
                candidate = os.path.normpath(os.path.join(eng.project_root, script_path))
                if os.path.exists(candidate):
                    script_path = candidate
        try:
            spec = importlib.util.spec_from_file_location("_user_script_inspect", script_path)
            if spec is None:
                Logger.warning(f"Script inspect spec is None for '{self.script_path}'")
                return
            mod = importlib.util.module_from_spec(spec)
            mod.Input = Input
            mod.KeyCode = KeyCode
            spec.loader.exec_module(mod)
            self._py_module = mod
            for attr in dir(mod):
                obj = getattr(mod, attr)
                if isinstance(obj, type) and (hasattr(obj, "on_update") or hasattr(obj, "_inspector_buttons")):
                    if getattr(obj, "__module__", None) == mod.__name__:
                        self._py_class = obj
                        return
        except Exception as e:
            Logger.error(f"Script inspect error '{self.script_path}': {e}")

    def _build_fields_from_class(self, cls) -> list[InspectorField]:
        fields = []
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = getattr(cls, '__annotations__', {})
        for name, ann_type in hints.items():
            if name.startswith('_'):
                continue
            default = getattr(cls, name, None)
            if name not in self._field_values:
                self._field_values[name] = default
            ft = self._py_type_to_field_type(ann_type)
            if ft == FieldType.ENUM:
                fields.append(InspectorField(name, name.replace('_', ' ').title(), ft, enum_class=ann_type))
            else:
                fields.append(InspectorField(name, name.replace('_', ' ').title(), ft))
        for attr_name in dir(cls):
            if attr_name.startswith('_') or attr_name in hints:
                continue
            val = getattr(cls, attr_name, None)
            if isinstance(val, (int, float, bool, str)):
                if attr_name not in self._field_values:
                    self._field_values[attr_name] = val
                py_type = type(val)
                ft = PY_TYPE_TO_FIELD.get(py_type, FieldType.STRING)
                fields.append(InspectorField(attr_name, attr_name.replace('_', ' ').title(), ft))
        buttons = getattr(cls, '_inspector_buttons', None)
        if isinstance(buttons, list):
            for entry in buttons:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    method_name, label = entry[0], entry[1]
                    fields.append(InspectorField(method_name, label, FieldType.BUTTON))
        return fields

    def _py_type_to_field_type(self, t) -> FieldType:
        origin = getattr(t, '__origin__', None)
        if origin is not None:
            t = origin
        if isinstance(t, type) and issubclass(t, Enum):
            return FieldType.ENUM
        if t is float:
            return FieldType.FLOAT
        if t is int:
            return FieldType.INT
        if t is bool:
            return FieldType.BOOL
        if t is str:
            return FieldType.STRING
        if t is Vec2:
            return FieldType.VEC2
        if t is Vec3:
            return FieldType.VEC3
        if t is Vec4:
            return FieldType.VEC4
        if t is Curve:
            return FieldType.CURVE
        if isinstance(t, str):
            t_clean = t.strip("'\"")
            if t_clean == 'Entity':
                return FieldType.GAMEOBJECT
        elif t.__name__ == 'Entity':
            return FieldType.GAMEOBJECT
        return FieldType.FLOAT

    def get_field_value(self, name: str) -> Any:
        return self._field_values.get(name, None)

    def set_field_value(self, name: str, value: Any):
        self._field_values[name] = value

    def _resolve_entity(self, value):
        if isinstance(value, str) and value and self._entity and self._entity._scene:
            return self._entity._scene.get_entity(value)
        return value

    def _apply_fields_to_instance(self):
        if not self._py_instance:
            return
        hints = self._cached_hints
        if hints is None:
            hints = {}
            if self._py_class:
                try:
                    hints = get_type_hints(self._py_class)
                except Exception:
                    hints = getattr(self._py_class, '__annotations__', {})
            self._cached_hints = hints
        for name, value in self._field_values.items():
            try:
                hint = hints.get(name)
                if isinstance(value, list) and hint is Vec2:
                    setattr(self._py_instance, name, Vec2(value[0], value[1]))
                elif isinstance(value, list) and hint is Vec3:
                    setattr(self._py_instance, name, Vec3(value[0], value[1], value[2]))
                elif isinstance(value, list) and hint is Vec4:
                    setattr(self._py_instance, name, Vec4(value[0], value[1], value[2], value[3]))
                elif isinstance(hint, str) and hint.strip("'\"") == 'Entity':
                    setattr(self._py_instance, name, self._resolve_entity(value))
                elif hint is not None and hasattr(hint, '__name__') and hint.__name__ == 'Entity':
                    setattr(self._py_instance, name, self._resolve_entity(value))
                elif isinstance(hint, type) and issubclass(hint, Enum):
                    if not isinstance(value, hint):
                        value = hint(value)
                    setattr(self._py_instance, name, value)
                else:
                    setattr(self._py_instance, name, value)
            except Exception:
                pass
        self._py_instance._entity = self._entity

    def _load_script(self):
        if not self.script_path:
            return
        try:
            self.get_script_public_fields()
            self._load_script_class()
            if self._py_class:
                self._py_instance = self._py_class()
                try:
                    self._cached_hints = get_type_hints(self._py_class)
                except Exception:
                    self._cached_hints = getattr(self._py_class, '__annotations__', {})
                inst = self._py_instance
                self._py_has_update = hasattr(inst, "on_update")
                self._py_has_fixed_update = hasattr(inst, "on_fixed_update")
                self._py_has_awake = hasattr(inst, "on_awake")
                self._py_has_start = hasattr(inst, "on_start")
                self._py_has_destroy = hasattr(inst, "on_destroy")
                self._apply_fields_to_instance()
        except Exception as e:
            Logger.error(f"Script load error '{self.script_path}': {e}")

    def on_start(self):
        if not self._py_instance and self.script_path:
            self._load_script()
        self._apply_fields_to_instance()
        if self._py_instance and self._py_has_awake:
            try:
                self._py_instance.on_awake()
            except Exception as e:
                Logger.error(f"Script awake error: {e}")
        if self._py_instance and self._py_has_start:
            try:
                self._py_instance.on_start()
            except Exception as e:
                Logger.error(f"Script start error: {e}")

    def on_update(self, dt: float):
        if self._py_instance and self._py_has_update:
            self._apply_fields_to_instance()
            try:
                self._py_instance.on_update(dt)
            except Exception as e:
                Logger.error(f"Script update error: {e}")

    def on_fixed_update(self, dt: float):
        if self._py_instance and self._py_has_fixed_update:
            self._apply_fields_to_instance()
            try:
                self._py_instance.on_fixed_update(dt)
            except Exception as e:
                Logger.error(f"Script fixed_update error: {e}")

    def on_destroy(self):
        if self._py_instance and self._py_has_destroy:
            try:
                self._py_instance.on_destroy()
            except Exception as e:
                Logger.error(f"Script destroy error: {e}")

    def gizmo_lines(self):
        if not self._py_instance and self.script_path:
            self._load_script()
        if self._py_instance and hasattr(self._py_instance, "gizmo_lines"):
            try:
                return self._py_instance.gizmo_lines()
            except Exception as e:
                Logger.error(f"Script gizmo_lines error: {e}")
        return []

    def gizmo_meshes(self):
        if not self._py_instance and self.script_path:
            self._load_script()
        if self._py_instance and hasattr(self._py_instance, "gizmo_meshes"):
            try:
                return self._py_instance.gizmo_meshes()
            except Exception as e:
                Logger.error(f"Script gizmo_meshes error: {e}")
        return []

    @classmethod
    def gizmo_collect_meshes(cls, scene):
        meshes = []
        for entity in scene.get_entities_with_component(cls):
            if not entity.active:
                continue
            for c in entity.get_components(cls):
                try:
                    msh = c.gizmo_meshes()
                    if msh:
                        meshes.extend(msh)
                except Exception:
                    pass
        return meshes

    def __getattr__(self, name):
        if name.startswith('_script_'):
            field_name = name[8:]
            if field_name in self._field_values:
                return self._field_values[field_name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name.startswith('_script_'):
            field_name = name[8:]
            self._field_values[field_name] = value
        else:
            super().__setattr__(name, value)

    def serialize(self) -> dict:
        d = super().serialize()
        d["script_path"] = self.script_path
        fields = {}
        for name, value in self._field_values.items():
            if isinstance(value, Vec2):
                fields[name] = [value.x, value.y]
            elif isinstance(value, Vec3):
                fields[name] = [value.x, value.y, value.z]
            elif isinstance(value, Vec4):
                fields[name] = [value.x, value.y, value.z, value.w]
            elif isinstance(value, Enum):
                fields[name] = value.value
            elif isinstance(value, Curve):
                fields[name] = value.to_dict()
            else:
                fields[name] = value
        d["script_fields"] = fields
        return d

    @classmethod
    def deserialize(cls, data: dict) -> ScriptComponent:
        sc = cls()
        sc.enabled = data.get("enabled", True)
        sc.script_path = data.get("script_path", "")
        raw_fields = data.get("script_fields", {})
        field_values = {}
        for name, value in raw_fields.items():
            if isinstance(value, dict) and "keys" in value:
                try:
                    field_values[name] = Curve.from_dict(value)
                    continue
                except Exception:
                    pass
            field_values[name] = value
        sc._field_values = field_values
        return sc
