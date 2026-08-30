# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Type, TypeVar, Optional
import numpy as np
from core.spatial import Octree, AABB
try:
    from core._constraint_update import batch_update_constraints as _batch_constraints
except ImportError:
    _batch_constraints = None
try:
    from core._ik import batch_update_ik as _batch_ik
except ImportError:
    _batch_ik = None

try:
    from core._scene_query import fast_get_entities_with_component as _fast_get
    _HAS_FAST_QUERY = True
except ImportError:
    _fast_get = None
    _HAS_FAST_QUERY = False

T = TypeVar("T", bound="Component")

_UNSET = object()

_GIZMO_PASSES: dict[str, list[type[Component]]] = {}
_GIZMO_PASS_ORDER: list[str] = ["collider", "particle", "force_field", "camera", "audio", "light", "script", "nav", "armature"]

_TRANSFORM_NAME = "Transform"

def _get_engine():
    try:
        from core.engine.engine import Engine
        return Engine.instance()
    except Exception:
        return None


class Component:
    _entity: Optional[Entity] = None
    _key: str = ""
    _enabled: bool = True
    _allow_multiple: bool = False
    _updates: bool = False
    _fixed_updates: bool = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, v: bool):
        if self._enabled == v:
            return
        self._enabled = bool(v)
        ent = self._entity
        if ent is not None and ent._scene is not None:
            sc = ent._scene
            sc._invalidate_update_cache()
            if self._enabled:
                if self._updates:
                    sc._active_update_components.add(self)
                if self._fixed_updates:
                    sc._active_fixed_components.add(self)
            else:
                sc._active_update_components.discard(self)
                sc._active_fixed_components.discard(self)
    _gizmo_icon_color: tuple[int, int, int] = (140, 60, 200)
    _gizmo_icon_label: str = "?"
    _gizmo_icon_path: Optional[str] = None
    _show_gizmo_icon: bool = True
    _transform: Any = _UNSET
    _gizmo_pass: str = ""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._updates = cls.on_update is not Component.on_update
        cls._fixed_updates = cls.on_fixed_update is not Component.on_fixed_update
        pname = cls.__dict__.get('_gizmo_pass', '')
        if pname:
            if pname not in _GIZMO_PASSES:
                _GIZMO_PASSES[pname] = []
            _GIZMO_PASSES[pname].append(cls)

    def on_awake(self): pass
    def on_start(self): pass
    def on_update(self, dt: float): pass
    def on_fixed_update(self, dt: float): pass
    def on_destroy(self): pass
    def on_enable(self): pass
    def on_disable(self): pass

    @property
    def entity(self) -> Optional[Entity]:
        return self._entity

    @property
    def transform(self):
        cached = self._transform
        if cached is not _UNSET:
            return cached
        ent = self._entity
        if ent is None:
            self._transform = None
            return None
        tr = ent._transform
        if tr is not None:
            self._transform = tr
            return tr
        t_type = ent._transform_type
        if t_type is None:
            t_type = ent._get_transform_type()
            if t_type is None:
                self._transform = None
                return None
        t_list = ent._type_map.get(t_type)
        result = t_list[0] if t_list else None
        self._transform = result
        return result

    @property
    def gizmo_icon(self) -> Optional[tuple[int, int, int, str]]:
        if not self._show_gizmo_icon:
            return None
        return (self._gizmo_icon_color[0], self._gizmo_icon_color[1], self._gizmo_icon_color[2], self._gizmo_icon_label)

    def gizmo_lines(self) -> list[tuple[Any, Any, list[float]]]:
        return []

    def gizmo_primitives(self):
        return None

    def gizmo_instance_data(self):
        return None

    def gizmo_cache_sig(self):
        attrs = getattr(type(self), "_gizmo_cache_attrs", None)
        if not attrs:
            return None
        tr = self.transform
        if tr is None:
            return None
        try:
            wm = tr.world_matrix._d.tobytes()
        except Exception:
            return None
        from core.maths.math3d import Vec2, Vec3, Vec4
        from enum import Enum
        parts = [wm]
        for a in attrs:
            v = getattr(self, a, None)
            if v is None:
                parts.append(None)
            elif isinstance(v, (list, tuple)):
                parts.append(tuple(v))
            elif isinstance(v, (Vec2, Vec3, Vec4)):
                parts.append((v.x, v.y, v.z, getattr(v, "w", 0.0)))
            elif isinstance(v, Enum):
                parts.append(v.value)
            else:
                parts.append(v)
        return tuple(parts)

    def gizmo_meshes(self) -> list[tuple[list, list, list]]:
        return []

    def gizmo(self):
        prims = self.gizmo_primitives()
        if prims is not None:
            s, e, c = prims
            if s.shape[0] > 0:
                return [GizmoPrimitive(s, e, c)]
        lines = self.gizmo_lines()
        if lines:
            return [GizmoPrimitive.from_lines(lines)]
        return []

    @classmethod
    def gizmo_collect(cls, pipe, scene):
        for entity in scene.get_entities_with_component(cls):
            if not entity.active:
                continue
            for comp in entity.get_components(cls):
                try:
                    inst = comp.gizmo_instance_data()
                    if inst is not None:
                        pipe.add_instance(inst.shape_type, inst.transform_flat, inst.color)
                        continue
                    for prim in comp.gizmo():
                        if prim.starts.shape[0] > 0:
                            pipe.add(prim)
                except Exception:
                    pass

    @classmethod
    def gizmo_collect_meshes(cls, scene):
        return []

    def serialize(self) -> dict:
        return {"type": type(self).__name__, "enabled": self.enabled}

    @classmethod
    def deserialize(cls, data: dict) -> Component:
        inst = cls()
        inst._enabled = bool(data.get("enabled", True))
        inst._entity = None
        return inst


@dataclass
class GizmoStyle:
    glow: bool = False
    dashed: bool = False
    pulsating: bool = False
    color_cycling: bool = False
    xray: bool = False
    line_width: float = 1.0
    dash_length: float = 0.3
    gap_length: float = 0.15
    glow_layers: int = 3
    glow_intensity: float = 0.4
    pulse_speed: float = 2.0
    pulse_min_alpha: float = 0.2
    cycle_speed: float = 1.0
    cull_distance: float = 0.0

    DEFAULT: Optional[GizmoStyle] = None


GizmoStyle.DEFAULT = GizmoStyle()


@dataclass
class GizmoPrimitive:
    starts: np.ndarray
    ends: np.ndarray
    colors: np.ndarray
    style: GizmoStyle = field(default_factory=lambda: GizmoStyle.DEFAULT)

    @classmethod
    def from_lines(cls, lines: list, style: GizmoStyle = None) -> GizmoPrimitive:
        n = len(lines)
        starts = np.zeros((n, 3), dtype=np.float32)
        ends = np.zeros((n, 3), dtype=np.float32)
        colors = np.zeros((n, 4), dtype=np.float32)
        for i, (s, e, c) in enumerate(lines):
            starts[i, 0] = s.x; starts[i, 1] = s.y; starts[i, 2] = s.z
            ends[i, 0] = e.x; ends[i, 1] = e.y; ends[i, 2] = e.z
            colors[i, 0] = c[0]; colors[i, 1] = c[1]; colors[i, 2] = c[2]
            colors[i, 3] = c[3] if len(c) > 3 else 1.0
        return cls(starts, ends, colors, style or GizmoStyle.DEFAULT)


@dataclass
class InstancePrimitive:
    shape_type: str
    transform_flat: np.ndarray
    color: list


class Entity:
    __slots__ = (
        '_id', '_name', '_type_map', '_type_name_map', '_components',
        '_update_list', '_fixed_update_list',
        '_active', '_parent', '_children', '_tags', '_layer',
        '_scene', '_prefab_guid', '_prefab_source_id',
        '_transform_type', '_transform', '_embed_resources',
    )

    def __init__(self, name: str = "Entity", eid: Optional[str] = None,
                 prefab_guid: Optional[str] = None):
        self._id: str = eid or str(uuid.uuid4())
        self._name: str = name
        self._type_map: dict[type, list[Component]] = {}
        self._type_name_map: dict[str, type] = {}
        self._components: dict[str, Component] = {}
        self._update_list: list[Component] = []
        self._fixed_update_list: list[Component] = []
        self._active: bool = True
        self._parent: Optional[Entity] = None
        self._children: list[Entity] = []
        self._tags: set[str] = set()
        self._layer: int = 0
        self._scene: Optional[Scene] = None
        self._prefab_guid: Optional[str] = prefab_guid
        self._prefab_source_id: Optional[str] = None
        self._transform_type: Optional[type] = None
        self._transform: Optional[Component] = None
        self._embed_resources: bool = False

    def _get_transform_type(self):
        tt = self._transform_type
        if tt is not None:
            return tt
        tm = self._type_map
        for t in tm:
            if t.__name__ == _TRANSFORM_NAME:
                self._transform_type = t
                return t
        from core.components.transform import Transform
        self._transform_type = Transform
        return Transform

    @property
    def transform(self):
        tr = self._transform
        if tr is not None:
            return tr
        tt = self._transform_type
        if tt is not None:
            clist = self._type_map.get(tt)
            if clist:
                self._transform = clist[0]
                return clist[0]
        tm = self._type_map
        for t in tm:
            if t.__name__ == _TRANSFORM_NAME:
                self._transform_type = t
                clist = tm.get(t)
                if clist:
                    self._transform = clist[0]
                    return clist[0]
                return None
        return None

    @property
    def id(self) -> str: return self._id

    @property
    def name(self) -> str: return self._name

    @name.setter
    def name(self, v: str): self._name = v

    @property
    def active(self) -> bool: return self._active

    @active.setter
    def active(self, v: bool):
        if self._active == v:
            return
        self._active = v
        sc = self._scene
        if sc:
            sc._render_version += 1
            if v:
                sc._active_update_components.update(c for c in self._update_list if c.enabled)
                sc._active_fixed_components.update(c for c in self._fixed_update_list if c.enabled)
                sc._spatial_dirty = True
                sc._spatial_dirty_entities.add(self._id)
            else:
                sc._active_update_components.difference_update(self._update_list)
                sc._active_fixed_components.difference_update(self._fixed_update_list)
                sc._spatial.remove(self._id)
                sc._spatial_dirty_entities.discard(self._id)
                sc._spatial_known_entities.discard(self._id)
        cb = (lambda c: c.on_enable()) if v else (lambda c: c.on_disable())
        comps = self._components
        for c in comps.values():
            if c.enabled:
                cb(c)

    @property
    def parent(self) -> Optional[Entity]: return self._parent

    @property
    def children(self) -> list[Entity]: return self._children

    @property
    def tags(self) -> set[str]: return self._tags

    @property
    def layer(self) -> int: return self._layer

    @layer.setter
    def layer(self, v: int): self._layer = v

    @property
    def embed_resources(self) -> bool: return self._embed_resources

    @embed_resources.setter
    def embed_resources(self, v: bool):
        if self._embed_resources == v:
            return
        self._embed_resources = bool(v)
        sc = self._scene
        if sc:
            sc._render_version += 1

    @property
    def prefab_guid(self) -> Optional[str]: return self._prefab_guid

    @prefab_guid.setter
    def prefab_guid(self, v: Optional[str]): self._prefab_guid = v

    @property
    def prefab_source_id(self) -> Optional[str]: return self._prefab_source_id

    @prefab_source_id.setter
    def prefab_source_id(self, v: Optional[str]): self._prefab_source_id = v

    @property
    def is_prefab_instance(self) -> bool:
        return self._prefab_guid is not None

    def set_parent(self, parent: Optional[Entity], preserve_world: bool = True):
        t = self.transform
        if t and preserve_world:
            world = t.world_matrix
        else:
            world = None
        old = self._parent
        if old is not None:
            ch = old._children
            try:
                ch.remove(self)
            except ValueError:
                pass
        self._parent = parent
        if parent is not None:
            parent._children.append(self)
        if world is not None:
            t.world_matrix = world
        if t is not None:
            t._mark_dirty()
        sc = self._scene
        if sc:
            sc._roots_cache_valid = False

    def _invalidate_transform_cache(self):
        comps = self._components
        for c in comps.values():
            c._transform = _UNSET

    def _make_component_key(self, comp: Component) -> str:
        t = type(comp)
        base = t.__name__
        if t._allow_multiple:
            return base + "." + str(uuid.uuid4())[:8]
        return base

    def add_component(self, comp: Component, key: Optional[str] = None) -> Component:
        if key is None:
            t = type(comp)
            base = t.__name__
            if t._allow_multiple:
                key = base + "." + str(uuid.uuid4())[:8]
            else:
                key = base
        comp._entity = self
        comp._key = key
        comp_type = type(comp)
        comps = self._components
        comps[key] = comp
        type_map = self._type_map
        lst = type_map.get(comp_type)
        if lst is None:
            type_map[comp_type] = [comp]
            self._type_name_map[comp_type.__name__] = comp_type
        else:
            lst.append(comp)
        sc = self._scene
        is_active = self._active
        if comp._updates:
            self._update_list.append(comp)
            if sc and is_active and comp.enabled:
                sc._active_update_components.add(comp)
        if comp._fixed_updates:
            self._fixed_update_list.append(comp)
            if sc and is_active and comp.enabled:
                sc._active_fixed_components.add(comp)
        if sc is not None:
            comp_name = comp_type.__name__
            idx = sc._component_indices
            s = idx.get(comp_name)
            if s is None:
                idx[comp_name] = {self._id}
            else:
                s.add(self._id)
            sc._render_version += 1
            sc._invalidate_update_cache()
        if comp_type.__name__ == _TRANSFORM_NAME:
            self._transform_type = comp_type
            self._transform = comp
            for c in comps.values():
                c._transform = _UNSET
            if sc is not None and getattr(comp, "_dirty", False):
                sc._dirty_roots.add(comp)
        comp.on_awake()
        return comp

    def remove_component(self, cls: Type[T]):
        clist = self._type_map.get(cls)
        if not clist:
            return
        comp = clist.pop(0)
        key = comp._key
        comp.on_destroy()
        self._components.pop(key, None)
        if not clist:
            del self._type_map[cls]
            self._type_name_map.pop(cls.__name__, None)
            if cls.__name__ == "Transform":
                self._transform_type = None
                self._transform = None
                self._invalidate_transform_cache()
        sc = self._scene
        if comp._updates:
            try: self._update_list.remove(comp)
            except ValueError: pass
            if sc:
                sc._active_update_components.discard(comp)
        if comp._fixed_updates:
            try: self._fixed_update_list.remove(comp)
            except ValueError: pass
            if sc:
                sc._active_fixed_components.discard(comp)
        if sc:
            base = cls.__name__
            if base == "Transform":
                self._transform_type = None
                self._transform = None
            idx = sc._component_indices.get(base)
            if idx:
                idx.discard(self._id)
            sc._render_version += 1
            sc._invalidate_update_cache()

    def remove_all_components(self, cls: Type[T]):
        clist = self._type_map.pop(cls, None)
        if not clist:
            return
        base = cls.__name__
        sc = self._scene
        upd = self._update_list
        fupd = self._fixed_update_list
        for comp in clist:
            comp.on_destroy()
            self._components.pop(comp._key, None)
            if comp._updates:
                try: upd.remove(comp)
                except ValueError: pass
                if sc:
                    sc._active_update_components.discard(comp)
            if comp._fixed_updates:
                try: fupd.remove(comp)
                except ValueError: pass
                if sc:
                    sc._active_fixed_components.discard(comp)
        if not clist:
            del self._type_map[cls]
            self._type_name_map.pop(cls.__name__, None)
        if base == "Transform":
            self._transform_type = None
            self._transform = None
            self._invalidate_transform_cache()
        if sc:
            idx = sc._component_indices.get(base)
            if idx:
                idx.discard(self._id)
            sc._invalidate_update_cache()

    def remove_component_by_key(self, key: str):
        comp = self._components.pop(key, None)
        if comp is None:
            return
        comp.on_destroy()
        comp_type = type(comp)
        clist = self._type_map.get(comp_type)
        if clist:
            try: clist.remove(comp)
            except ValueError: pass
            if not clist:
                del self._type_map[comp_type]
                self._type_name_map.pop(comp_type.__name__, None)
                if comp_type.__name__ == "Transform":
                    self._transform_type = None
                    self._transform = None
                    self._invalidate_transform_cache()
        sc = self._scene
        if comp._updates:
            try: self._update_list.remove(comp)
            except ValueError: pass
            if sc:
                sc._active_update_components.discard(comp)
        if comp._fixed_updates:
            try: self._fixed_update_list.remove(comp)
            except ValueError: pass
            if sc:
                sc._active_fixed_components.discard(comp)
        base = key.split(".")[0]
        if sc:
            if base == "Transform":
                self._transform_type = None
                self._transform = None
            idx = sc._component_indices.get(base)
            if idx:
                idx.discard(self._id)
            sc._invalidate_update_cache()

    def get_component(self, cls: Type[T]) -> Optional[T]:
        clist = self._type_map.get(cls)
        return clist[0] if clist else None

    def get_components(self, cls: Type[T]) -> list[T]:
        return list(self._type_map.get(cls, []))

    def get_component_by_name(self, name: str) -> Optional[Component]:
        t = self._type_name_map.get(name)
        if t is not None:
            clist = self._type_map.get(t)
            if clist:
                return clist[0]
        c = self._components.get(name)
        if c is not None:
            return c
        prefix = name + "."
        for k, c in self._components.items():
            if k.startswith(prefix):
                return c
        return None

    def has_component(self, cls: Type[T]) -> bool:
        return cls in self._type_map

    def get_all_components(self) -> list[Component]:
        result = []
        for clist in self._type_map.values():
            result.extend(clist)
        return result

    def move_component(self, key: str, direction: int):
        keys = list(self._components.keys())
        if key not in keys:
            return
        idx = keys.index(key)
        new_idx = idx + direction
        if new_idx < 0 or new_idx >= len(keys):
            return
        keys[idx], keys[new_idx] = keys[new_idx], keys[idx]
        self._components = {k: self._components[k] for k in keys}

    def get_component_in_children(self, cls: Type[T]) -> Optional[T]:
        for child in self._children:
            c = child.get_component(cls)
            if c:
                return c
            c = child.get_component_in_children(cls)
            if c:
                return c
        return None

    def serialize(self) -> dict:
        d = {
            "id": self._id, "name": self._name, "active": self._active,
            "tags": list(self._tags), "layer": self._layer,
            "parent": self._parent.id if self._parent else None,
            "components": [{"_key": k, **c.serialize()} for k, c in self._components.items()]
        }
        if self._prefab_guid:
            d["prefab_guid"] = self._prefab_guid
        if self._prefab_source_id:
            d["prefab_source_id"] = self._prefab_source_id
        if self._embed_resources:
            d["embed_resources"] = True
        return d

    @classmethod
    def deserialize(cls, data: dict, registry: ComponentRegistry) -> Entity:
        prefab_guid = data.get("prefab_guid")
        e = cls(data["name"], data["id"], prefab_guid=prefab_guid)
        e._active = data.get("active", True)
        e._tags = set(data.get("tags", []))
        e._layer = data.get("layer", 0)
        e._prefab_source_id = data.get("prefab_source_id")
        e._embed_resources = bool(data.get("embed_resources", False))
        for cd in data.get("components", []):
            ctype = cd.get("type")
            comp_cls = registry.get(ctype)
            if comp_cls:
                comp = comp_cls.deserialize(cd)
                key = cd.get("_key", None)
                e.add_component(comp, key=key)
        return e


class ComponentRegistry:
    _registry: dict[str, Type[Component]] = {}
    _categories: dict[str, list[str]] = {}
    _category_name_map: dict[str, str] = {
        "transform": "Transform",
        "rendering": "Rendering",
        "physics": "Physics",
        "physics2d": "Physics 2D",
        "lighting": "Lighting",
        "audio": "Audio",
        "constraints": "Constraints",
        "network": "Network",
        "scripting": "Scripting",
    }

    @classmethod
    def register(cls, comp_cls: Type[Component]):
        name = comp_cls.__name__
        cls._registry[name] = comp_cls
        category = cls._infer_category(comp_cls)
        if category:
            cls._categories[name] = [category]
        return comp_cls

    @classmethod
    def _infer_category(cls, comp_cls: Type[Component]) -> Optional[str]:
        module = getattr(comp_cls, "__module__", "")
        parts = module.split(".")
        for i, part in enumerate(parts):
            if part == "components" and i + 1 < len(parts):
                sub = parts[i + 1]
                return cls._category_name_map.get(sub, sub.capitalize())
        return None

    @classmethod
    def get(cls, name: str) -> Optional[Type[Component]]:
        return cls._registry.get(name)

    @classmethod
    def all(cls) -> dict[str, Type[Component]]:
        return dict(cls._registry)

    @classmethod
    def get_categories(cls, comp_name: str) -> list[str]:
        return list(cls._categories.get(comp_name, []))

    @classmethod
    def all_categories(cls) -> dict[str, list[str]]:
        return dict(cls._categories)


class Scene:
    def __init__(self, name: str = "Scene"):
        self._name: str = name
        self._entities: dict[str, Entity] = {}
        self._entities_cache: list[Entity] = []
        self._entities_cache_valid: bool = False
        self._systems: list[Any] = []
        self._path: Optional[str] = None
        self._dirty: bool = False
        self._component_indices: dict[str, set[str]] = {}
        self._render_version: int = 0
        self._engine_ref = None
        self._scene_prof: Any = None
        self._embed_all: bool = False
        self._compress_resources: bool = False
        self._compress_level = None
        self._embedded_resources: dict = {}
        self._active_update_components: set[Component] = set()
        self._active_fixed_components: set[Component] = set()
        self._update_list_cache: list[Component] = []
        self._fixed_list_cache: list[Component] = []
        self._update_cache_valid: bool = False
        self._fixed_cache_valid: bool = False
        self._update_constraints_cache: list[Component] = []
        self._update_iks_cache: list[Component] = []
        self._update_others_cache: list[Component] = []
        self._update_partition_valid: bool = False
        self._dirty_roots: set = set()
        self._depth_cache: dict[str, int] = {}
        self._component_entity_frame_cache: dict = {}
        self._spatial: Octree = Octree(world_size=1000.0)
        self._spatial_dirty: bool = True
        self._spatial_dirty_entities: set[str] = set()
        self._spatial_known_entities: set[str] = set()
        self._roots_cache: list[Entity] = []
        self._roots_cache_valid: bool = False

    def _batch_sync_entities(self, entities: dict[str, Entity]):
        idx = self._component_indices
        auc = self._active_update_components
        afc = self._active_fixed_components
        for eid, e in entities.items():
            is_active = e._active
            for comp_type, clist in e._type_map.items():
                comp_name = comp_type.__name__
                if comp_name not in idx:
                    idx[comp_name] = set()
                idx[comp_name].add(eid)
                if is_active:
                    for comp in clist:
                        if comp.enabled:
                            if comp._updates:
                                auc.add(comp)
                            if comp._fixed_updates:
                                afc.add(comp)
        self._entities_cache_valid = False
        self._roots_cache_valid = False
        self._spatial_dirty = True
        self._spatial_dirty_entities.update(entities.keys())
        self._invalidate_update_cache()
        self._dirty = True
        self._render_version += 1

    def _invalidate_update_cache(self):
        self._update_cache_valid = False
        self._fixed_cache_valid = False
        self._update_partition_valid = False

    def _get_update_list(self) -> list[Component]:
        if not self._update_cache_valid:
            self._update_list_cache = [c for c in self._active_update_components if c.enabled]
            self._update_cache_valid = True
        return self._update_list_cache

    def _get_fixed_list(self) -> list[Component]:
        if not self._fixed_cache_valid:
            self._fixed_list_cache = [c for c in self._active_fixed_components if c.enabled]
            self._fixed_cache_valid = True
        return self._fixed_list_cache

    def _get_partitioned_update(self):
        if self._update_partition_valid:
            return self._update_constraints_cache, self._update_iks_cache, self._update_others_cache
        lst = self._get_update_list()
        CT = self._CONSTRAINT_TYPES
        IK = self._IK_TYPES
        cons = []
        iks = []
        others = []
        ac = cons.append
        ai = iks.append
        ao = others.append
        for c in lst:
            tn = type(c).__name__
            if tn in CT:
                ac(c)
            elif tn in IK:
                ai(c)
            else:
                ao(c)
        self._update_constraints_cache = cons
        self._update_iks_cache = iks
        self._update_others_cache = others
        self._update_partition_valid = True
        return cons, iks, others

    @property
    def _engine(self):
        if self._engine_ref is None:
            from core.engine.engine import Engine
            self._engine_ref = Engine.instance()
        return self._engine_ref

    def _ensure_entities_cache(self):
        if not self._entities_cache_valid:
            self._entities_cache = list(self._entities.values())
            self._entities_cache_valid = True
        return self._entities_cache

    @property
    def name(self) -> str: return self._name

    @name.setter
    def name(self, v: str):
        self._name = v
        self._dirty = True

    @property
    def path(self) -> Optional[str]: return self._path

    @path.setter
    def path(self, v: str): self._path = v

    @property
    def dirty(self) -> bool: return self._dirty

    def mark_dirty(self): self._dirty = True
    def mark_clean(self): self._dirty = False

    def _get_entity_depth(self, e: Entity) -> int:
        eid = e._id
        cached = self._depth_cache.get(eid)
        if cached is not None:
            return cached
        depth = 0
        p = e._parent
        while p is not None:
            depth += 1
            p = p._parent
        self._depth_cache[eid] = depth
        return depth

    def flush_transforms(self):
        dr = self._dirty_roots
        if not dr:
            return 0
        needs_bfs = False
        for root in dr:
            ent = root._entity
            if ent is None:
                continue
            if ent._parent is not None or ent._children:
                needs_bfs = True
                break
        if not needs_bfs:
            collected = [r for r in dr if r._dirty and r._entity is not None]
            if not collected:
                dr.clear()
                return 0
            try:
                from core._ecs_batch import batch_update_flat
                batch_update_flat(collected)
            except ImportError:
                from core.components.transform import Transform
                Transform.batch_update_world_matrices(collected)
            dr.clear()
            self._depth_cache.clear()
            return len(collected)
        collected = []
        visited = set()
        q = deque()
        add_q = q.append
        popleft = q.popleft
        for root in list(dr):
            if root._dirty and root._entity is not None and id(root) not in visited:
                visited.add(id(root))
                add_q(root)
        append_c = collected.append
        while q:
            t = popleft()
            append_c(t)
            ent = t._entity
            if ent is None:
                continue
            for child in ent._children:
                ct = child._transform_type
                if ct is not None:
                    lst = child._type_map.get(ct)
                    c = lst[0] if lst else None
                else:
                    c = child.transform
                if c is not None and c._dirty and id(c) not in visited:
                    visited.add(id(c))
                    add_q(c)
        if not collected:
            dr.clear()
            return 0
        dc = self._depth_cache
        def _depth_key(t):
            e = t._entity
            eid = e._id
            d = dc.get(eid)
            if d is not None:
                return d
            depth = 0
            p = e._parent
            while p is not None:
                depth += 1
                p = p._parent
            dc[eid] = depth
            return depth
        collected.sort(key=_depth_key)
        from core.components.transform import Transform
        Transform.batch_update_world_matrices(collected)
        dr.clear()
        dc.clear()
        return len(collected)

    def create_entity(self, name: str = "Entity",
                      prefab_guid: Optional[str] = None) -> Entity:
        e = Entity(name, prefab_guid=prefab_guid)
        e._scene = self
        self._entities[e.id] = e
        self._dirty = True
        self._render_version += 1
        self._entities_cache_valid = False
        self._roots_cache_valid = False
        self._spatial_dirty = True
        self._spatial_dirty_entities.add(e.id)
        return e

    def add_entity(self, e: Entity):
        e._scene = self
        self._entities[e.id] = e
        eid = e.id
        idx = self._component_indices
        is_active = e._active
        for comp_type, clist in e._type_map.items():
            comp_name = comp_type.__name__
            if comp_name not in idx:
                idx[comp_name] = set()
            idx[comp_name].add(eid)
            if is_active:
                for comp in clist:
                    if comp.enabled:
                        if comp._updates:
                            self._active_update_components.add(comp)
                        if comp._fixed_updates:
                            self._active_fixed_components.add(comp)
        t = e.transform
        if t and t._dirty:
            self._dirty_roots.add(t)
        self._invalidate_update_cache()
        self._dirty = True
        self._render_version += 1
        self._entities_cache_valid = False
        self._roots_cache_valid = False
        self._spatial_dirty = True
        self._spatial_dirty_entities.add(eid)
        self._spatial_known_entities.discard(eid)

    def remove_entity(self, eid: str):
        e = self._entities.pop(eid, None)
        if not e:
            return
        self._spatial.remove(eid)
        self._spatial_dirty_entities.discard(eid)
        self._spatial_known_entities.discard(eid)
        for child in list(e._children):
            self.remove_entity(child._id)
        auc = self._active_update_components
        afc = self._active_fixed_components
        idx = self._component_indices
        for c in e._components.values():
            c.on_destroy()
        for comp_type, clist in e._type_map.items():
            comp_name = comp_type.__name__
            s = idx.get(comp_name)
            if s:
                s.discard(eid)
            for comp in clist:
                auc.discard(comp)
                afc.discard(comp)
        self._invalidate_update_cache()
        self._dirty = True
        self._render_version += 1
        self._entities_cache_valid = False
        self._roots_cache_valid = False
        self._spatial_dirty = True

    def duplicate_entity(self, entity: Entity, new_name: str = "") -> Entity:
        """Duplicate an entity together with its whole descendant subtree.

        Skinned-mesh objects keep working after duplication: every entity id is
        remapped (all new uuids) and each copied Armature component is rebound
        to the copied bone entities instead of pointing at the source skeleton.
        The copied root keeps the same local transform and same parent as the
        source.
        """
        if entity._scene is not self:
            entity = self.get_entity(entity.id) or entity
        import copy as _copy

        def walk(e: Entity) -> list:
            data = _copy.deepcopy(e.serialize())
            data["id"] = str(uuid.uuid4())
            data["parent"] = None
            nodes = [(e, data)]
            for ch in list(e._children):
                nodes.extend(walk(ch))
            return nodes

        nodes = walk(entity)
        id_map = {src.id: data["id"] for src, data in nodes}
        new_entities: list[Entity] = []
        for src, data in nodes:
            data["parent"] = None
            new_e = Entity.deserialize(data, ComponentRegistry)
            self.add_entity(new_e)
            new_entities.append(new_e)
        roots = [new_entities[0]]
        new_by_id = {src.id: e for (src, _), e in zip(nodes, new_entities)}
        for (src, _data), new_e in zip(nodes, new_entities):
            if src.parent is None:
                continue
            if src.parent.id in new_by_id:
                new_e.set_parent(new_by_id[src.parent.id], preserve_world=False)
            else:
                new_e.set_parent(src.parent, preserve_world=False)
        self._rebind_armatures(new_entities, id_map)
        if new_name:
            roots[0].name = new_name
        return roots[0]

    def _rebind_armatures(self, entities: list, id_map: dict) -> None:
        """Point every copied Armature's bone_entity_ids at the freshly created bone entities."""
        for e in entities:
            arm = e.get_component_by_name("Armature")
            if arm is not None and getattr(arm, "bone_entity_ids", None):
                arm.bone_entity_ids = [id_map.get(bid, bid) for bid in arm.bone_entity_ids]

    def paste_entities(self, clipboard_data: list, registry) -> list:
        """Deserialize serialized entities (a copied subtree) into this scene.

        Remaps every entity id to a fresh uuid, restores inner parent links,
        and rebinds Armature components to the pasted bone entities so skinned
        meshes keep working instead of being tied to the source skeleton.
        Returns the list of spawned entity objects.
        """
        import copy as _copy
        id_map: dict = {}
        spawned: list = []
        for data in clipboard_data:
            d = _copy.deepcopy(data)
            old_id = d["id"]
            new_id = str(uuid.uuid4())
            d["id"] = new_id
            id_map[old_id] = new_id
            e = Entity.deserialize(d, registry)
            self.add_entity(e)
            spawned.append(e)
        all_by_id = {e.id: e for e in spawned}
        for data in clipboard_data:
            parent_id = data.get("parent")
            if not parent_id or parent_id not in id_map:
                continue
            child = all_by_id.get(id_map[data["id"]])
            new_parent = all_by_id.get(id_map[parent_id])
            if child and new_parent:
                child.set_parent(new_parent, preserve_world=False)
        self._rebind_armatures(spawned, id_map)
        return spawned

    def get_entity(self, eid: str) -> Optional[Entity]:
        return self._entities.get(eid)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        for e in self._entities.values():
            if e._name == name:
                return e
        return None

    def get_all_entities(self) -> list[Entity]:
        return self._ensure_entities_cache()

    def get_root_entities(self) -> list[Entity]:
        if not self._roots_cache_valid:
            self._roots_cache = [e for e in self._entities.values() if e._parent is None]
            self._roots_cache_valid = True
        return self._roots_cache

    def get_entities_with_component(self, cls: Type[T]) -> list[Entity]:
        if _HAS_FAST_QUERY:
            return _fast_get(self._component_indices, self._entities, cls.__name__, self._render_version, self._component_entity_frame_cache)
        key = cls.__name__
        s = self._component_indices.get(key)
        if not s:
            return []
        rv = self._render_version
        cache_tag = (key, rv)
        cc = self._component_entity_frame_cache
        cached = cc.get(cache_tag)
        if cached is not None:
            return cached
        ents = self._entities
        result = [ents[eid] for eid in s if eid in ents]
        cc[cache_tag] = result
        if len(cc) > 256:
            cc.clear()
            cc[cache_tag] = result
        return result

    def _insert_spatial_single(self, e):
        from core.maths.math3d import Vec3
        import numpy as np
        from core.components.rendering.renderers.mesh_filter import MeshFilter
        from core.components.rendering.renderers.mesh_renderer import MeshRenderer
        tr = e.transform
        if not tr:
            return
        mf = e.get_component(MeshFilter)
        mr = e.get_component(MeshRenderer)
        if mf and mr and mr.enabled:
            try:
                from core.engine.engine import Engine
                eng = Engine.instance()
                if eng:
                    r = getattr(eng, '_renderer', None)
                    if r is None:
                        vp = getattr(eng, 'viewport', None)
                        if vp:
                            r = getattr(vp, '_renderer', None)
                    if r:
                        name = mf.mesh_name or "cube"
                        mesh = r._meshes.get(name)
                        if mesh is None and mf.mesh_path:
                            mesh = r._meshes.get(mf.mesh_path)
                        if mesh is not None and len(mesh.vertices) > 0:
                            ax, ay, az = mesh.aabb_min
                            bx, by, bz = mesh.aabb_max
                            corners = np.array([
                                [ax, ay, az, 1], [bx, ay, az, 1],
                                [bx, by, az, 1], [ax, by, az, 1],
                                [ax, ay, bz, 1], [bx, ay, bz, 1],
                                [bx, by, bz, 1], [ax, by, bz, 1],
                            ], dtype=np.float32)
                            pts = corners @ tr.world_matrix._d
                            bmin = pts[:, :3].min(axis=0)
                            bmax = pts[:, :3].max(axis=0)
                            aabb = AABB(Vec3(float(bmin[0]), float(bmin[1]), float(bmin[2])),
                                        Vec3(float(bmax[0]), float(bmax[1]), float(bmax[2])))
                            self._spatial.insert(e.id, aabb)
                            return
            except Exception:
                pass
        pos = tr.position
        self._spatial.insert(e.id, AABB.from_center_size(pos, Vec3(5.0, 5.0, 5.0)))

    def rebuild_spatial(self):
        if not self._spatial_dirty:
            return
        dirty = self._spatial_dirty_entities
        if dirty and len(dirty) < len(self._entities) * 0.6:
            for eid in list(dirty):
                self._spatial.remove(eid)
                e = self._entities.get(eid)
                if e is None or not e.active:
                    continue
                self._insert_spatial_single(e)
            dirty.clear()
            if not dirty:
                self._spatial_dirty = False
            return
        from core.maths.math3d import Vec3
        import numpy as np
        self._spatial.clear()
        self._spatial_known_entities.clear()
        for e in self._ensure_entities_cache():
            if not e.active:
                continue
            self._insert_spatial_single(e)
            self._spatial_known_entities.add(e.id)
        dirty.clear()
        self._spatial_dirty = False

    def spatial_query(self, aabb: AABB) -> list[str]:
        if self._spatial_dirty:
            self.rebuild_spatial()
        return self._spatial.query(aabb)

    def spatial_raycast(self, origin: 'Vec3', direction: 'Vec3', max_dist: float = 100.0) -> list[tuple[str, float]]:
        if self._spatial_dirty:
            self.rebuild_spatial()
        return self._spatial.raycast(origin, direction, max_dist)

    def mark_spatial_dirty(self):
        self._spatial_dirty = True

    def _rebuild_component_index(self, comp_cls_name: str):
        indices: set[str] = set()
        for eid, e in self._entities.items():
            for t in e._type_map:
                if t.__name__ == comp_cls_name:
                    indices.add(eid)
                    break
        self._component_indices[comp_cls_name] = indices

    def _get_profiler(self):
        p = self._scene_prof
        if p is not None:
            return p if p is not False else None
        eng = self._engine
        if eng and hasattr(eng, '_profiler'):
            self._scene_prof = eng._profiler
            return self._scene_prof
        self._scene_prof = False
        return None

    _CONSTRAINT_TYPES = frozenset({
        "PositionConstraint", "RotationConstraint", "ScaleConstraint",
        "ParentConstraint", "MoveTowardsConstraint", "RotateTowardsConstraint",
        "ScaleToConstraint", "AimConstraint", "LookAtConstraint",
        "FollowTransformConstraint",
    })

    _IK_TYPES = frozenset({
        "TwoBoneIK", "FABRIKChain",
    })

    def update(self, dt: float):
        prof = self._get_profiler()
        if prof is None:
            return
        prof.start("scene_update")
        log_error = None
        constraints, iks, others = self._get_partitioned_update()
        if not constraints and not iks and not others:
            prof.stop("scene_update")
            return
        if constraints:
            if _batch_constraints is not None:
                try:
                    _batch_constraints(constraints, dt)
                except Exception as ex:
                    if log_error is None:
                        from core.foundation.logger import Logger
                        log_error = Logger.error
                    log_error(f"Constraint batch update error: {ex}")
            else:
                for c in constraints:
                    try:
                        c.on_update(dt)
                    except Exception as ex:
                        if log_error is None:
                            from core.foundation.logger import Logger
                            log_error = Logger.error
                        ent = c._entity
                        log_error(f"Update error in {ent._name if ent else '?'}/{type(c).__name__}: {ex}")
        for c in others:
            try:
                c.on_update(dt)
            except Exception as ex:
                if log_error is None:
                    from core.foundation.logger import Logger
                    log_error = Logger.error
                ent = c._entity
                log_error(f"Update error in {ent._name if ent else '?'}/{type(c).__name__}: {ex}")
        if iks:
            if _batch_ik is not None:
                try:
                    _batch_ik(iks, dt)
                except Exception as ex:
                    if log_error is None:
                        from core.foundation.logger import Logger
                        log_error = Logger.error
                    log_error(f"IK batch update error: {ex}")
                    for c in iks:
                        try:
                            c.on_update(dt)
                        except Exception as ex2:
                            if log_error is None:
                                from core.foundation.logger import Logger
                                log_error = Logger.error
                            ent = c._entity
                            log_error(f"Update error in {ent._name if ent else '?'}/{type(c).__name__}: {ex2}")
            else:
                for c in iks:
                    try:
                        c.on_update(dt)
                    except Exception as ex:
                        if log_error is None:
                            from core.foundation.logger import Logger
                            log_error = Logger.error
                        ent = c._entity
                        log_error(f"Update error in {ent._name if ent else '?'}/{type(c).__name__}: {ex}")
        prof.stop("scene_update")

    def fixed_update(self, dt: float):
        prof = self._get_profiler()
        if prof is None:
            return
        prof.start("scene_fixed_update")
        log_error = None
        for c in self._get_fixed_list():
            try:
                c.on_fixed_update(dt)
            except Exception as ex:
                if log_error is None:
                    from core.foundation.logger import Logger
                    log_error = Logger.error
                ent = c._entity
                log_error(f"FixedUpdate error in {ent._name if ent else '?'}/{type(c).__name__}: {ex}")
        prof.stop("scene_fixed_update")

    def start(self):
        prof = self._get_profiler()
        if prof is None:
            return
        prof.start("scene_start")
        for e in list(self._entities.values()):
            for c in e.get_all_components():
                try:
                    c.on_start()
                except Exception as ex:
                    from core.foundation.logger import Logger
                    Logger.error(f"Start error: {ex}")
        prof.stop("scene_start")

    def serialize(self) -> dict:
        data = {"name": self._name, "entities": {eid: e.serialize() for eid, e in self._entities.items()}}
        if self._embed_all:
            data["embed_all"] = True
        if self._compress_resources:
            data["compress_resources"] = True
        if self._compress_level is not None:
            data["compress_level"] = self._compress_level
        prof = self._get_profiler()
        if prof is not None:
            prof.set_value("scene_serialize", 0)
        return data

    @property
    def compress_resources(self) -> bool:
        return self._compress_resources

    @compress_resources.setter
    def compress_resources(self, v: bool):
        if self._compress_resources == bool(v):
            return
        self._compress_resources = bool(v)
        self._render_version += 1

    @property
    def compress_level(self):
        return self._compress_level

    @compress_level.setter
    def compress_level(self, v):
        self._compress_level = int(v) if isinstance(v, bool) else v

    @property
    def embed_all(self) -> bool:
        return self._embed_all

    @embed_all.setter
    def embed_all(self, v: bool):
        if self._embed_all == bool(v):
            return
        self._embed_all = bool(v)
        self._render_version += 1

    @property
    def embedded_resources(self) -> dict:
        return self._embedded_resources

    @embedded_resources.setter
    def embedded_resources(self, v: dict):
        self._embedded_resources = v or {}

    @classmethod
    def deserialize(cls, data: dict, registry: ComponentRegistry) -> Scene:
        s = cls(data["name"])
        s._embed_all = bool(data.get("embed_all", False))
        s._compress_resources = bool(data.get("compress_resources", False))
        s._compress_level = data.get("compress_level")
        raw = data.get("entities", {})
        entities: dict[str, Entity] = {}
        parent_map: dict[str, Optional[str]] = {}
        for eid, ed in raw.items():
            e = Entity.deserialize(ed, registry)
            entities[eid] = e
            parent_map[eid] = ed.get("parent")
        for eid, e in entities.items():
            pid = parent_map.get(eid)
            if pid and pid in entities:
                e.set_parent(entities[pid], preserve_world=False)
            s._entities[e.id] = e
            e._scene = s
        s._batch_sync_entities(entities)
        return s
