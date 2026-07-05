# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
import uuid
import copy
from enum import Enum
from typing import Optional, Any
from core.ecs import Scene, Entity, ComponentRegistry
from core.logger import Logger


class PrefabOverrideType(Enum):
    PROPERTY = "property"
    ADDED_COMPONENT = "added_component"
    REMOVED_COMPONENT = "removed_component"
    ADDED_CHILD = "added_child"
    REMOVED_CHILD = "removed_child"


_ROOT_TRANFORM_EXCLUDED = ("local_position", "local_rotation")


class Prefab:
    TRANSFORM_PASSTHROUGH = {"local_position", "local_rotation"}

    def __init__(self, name: str = "Prefab", guid: Optional[str] = None,
                 base_guid: Optional[str] = None):
        self.name: str = name
        self.guid: str = guid or str(uuid.uuid4())
        self.roots_data: list[dict] = []
        self.base_guid: Optional[str] = base_guid

    @property
    def is_variant(self) -> bool:
        return self.base_guid is not None

    def capture(self, entities: list[Entity]):
        self.roots_data = [self._capture_entity_data(e) for e in entities]

    def _capture_entity_data(self, entity: Entity) -> dict:
        data = entity.serialize()
        data.pop("parent", None)
        data.pop("prefab_guid", None)
        data.pop("prefab_source_id", None)
        children = []
        for child in entity.children:
            children.append(self._capture_entity_data(child))
        if children:
            data["children"] = children
        if "id" in data:
            data["source_id"] = data.pop("id")
        return data

    def instantiate(self, scene: Scene, registry: ComponentRegistry,
                    parent: Optional[Entity] = None) -> list[Entity]:
        if not self.roots_data:
            Logger.warning("Prefab has no root data.")
            return []
        spawned: list[Entity] = []
        for rd in self.roots_data:
            data = copy.deepcopy(rd)
            self._remap_ids(data)
            source_id = rd.get("source_id", data.get("id"))
            e = Entity.deserialize(data, registry)
            e._prefab_guid = self.guid
            e._prefab_source_id = source_id
            scene.add_entity(e)
            if parent:
                e.set_parent(parent)
            spawned.append(e)
            self._restore_children(e, data, rd, scene, registry)
        return spawned

    def _restore_children(self, parent_entity: Entity, data: dict, source_data: dict,
                          scene: Scene, registry: ComponentRegistry):
        for cd, sd in zip(data.get("children", []), source_data.get("children", [])):
            self._remap_ids(cd)
            child_source_id = sd.get("source_id", cd.get("id"))
            child = Entity.deserialize(cd, registry)
            child._prefab_guid = self.guid
            child._prefab_source_id = child_source_id
            scene.add_entity(child)
            child.set_parent(parent_entity)
            self._restore_children(child, cd, sd, scene, registry)

    def _remap_ids(self, data: dict):
        data["id"] = str(uuid.uuid4())
        data["parent"] = None
        for child in data.get("children", []):
            self._remap_ids(child)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        out = {
            "guid": self.guid,
            "name": self.name,
            "roots": self.roots_data
        }
        if self.base_guid:
            out["base_guid"] = self.base_guid
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        Logger.info(f"Prefab saved: {path}")

    @classmethod
    def load(cls, path: str) -> Optional[Prefab]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            p = cls(data.get("name", "Prefab"), data.get("guid"),
                    data.get("base_guid"))
            roots = data.get("roots")
            if roots is not None:
                p.roots_data = roots
            elif "entity" in data:
                p.roots_data = [data["entity"]]
            elif "entities" in data:
                p.roots_data = list(data["entities"].values())
            return p
        except Exception as e:
            Logger.error(f"Failed to load prefab '{path}': {e}", exc=e)
            return None

    @staticmethod
    def compute_overrides(entity: Entity) -> list[dict]:
        if not entity._prefab_guid:
            return []
        prefab_lib = None
        try:
            from core.prefab import PrefabLibrary
            prefab_lib = PrefabLibrary
        except ImportError:
            return []
        path = prefab_lib.path_for_guid(entity._prefab_guid)
        if not path:
            return []
        prefab = prefab_lib.load(path)
        if not prefab:
            return []
        source_id = entity._prefab_source_id
        prefab_data = Prefab._find_source_data(prefab.roots_data, source_id)
        if prefab_data is None:
            source_id_fallback = entity._prefab_source_id
            if source_id_fallback:
                prefab_data = Prefab._find_source_data(prefab.roots_data, source_id_fallback)
            if prefab_data is None:
                for rd in prefab.roots_data:
                    prefab_data = rd
                    break
            if prefab_data is None:
                return []
        current = entity.serialize()
        overrides = Prefab._diff_entity(current, prefab_data, entity,
                                        is_root=entity._parent is None or
                                        not entity._parent.is_prefab_instance)
        return overrides

    @staticmethod
    def compute_all_overrides(entities: list[Entity]) -> list[dict]:
        all_overrides = []
        for e in entities:
            all_overrides.extend(Prefab.compute_overrides(e))
            all_overrides.extend(Prefab.compute_all_overrides(e.children))
        return all_overrides

    @staticmethod
    def _find_source_data(roots_data: list[dict], source_id: str) -> Optional[dict]:
        def walk(items):
            for item in items:
                sid = item.get("source_id") or item.get("id")
                if sid == source_id:
                    return item
                found = walk(item.get("children", []))
                if found:
                    return found
            return None
        return walk(roots_data)

    @staticmethod
    def _find_entity_source_data(prefab: Prefab, entity: Entity) -> Optional[dict]:
        for rd in prefab.roots_data:
            found = Prefab._find_source_data([rd], entity._prefab_source_id)
            if found:
                return found
        return None

    @staticmethod
    def _diff_entity(current: dict, prefab_data: dict, entity: Entity,
                     is_root: bool = False) -> list[dict]:
        overrides = []

        if current.get("name") != prefab_data.get("name"):
            overrides.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "source_id": entity._prefab_source_id,
                "type": PrefabOverrideType.PROPERTY.value,
                "property": "name",
                "old_value": prefab_data.get("name"),
                "new_value": current.get("name")
            })

        if set(current.get("tags", [])) != set(prefab_data.get("tags", [])):
            overrides.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "source_id": entity._prefab_source_id,
                "type": PrefabOverrideType.PROPERTY.value,
                "property": "tags",
                "old_value": prefab_data.get("tags", []),
                "new_value": current.get("tags", [])
            })

        if current.get("layer") != prefab_data.get("layer"):
            overrides.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "source_id": entity._prefab_source_id,
                "type": PrefabOverrideType.PROPERTY.value,
                "property": "layer",
                "old_value": prefab_data.get("layer"),
                "new_value": current.get("layer")
            })

        cur_comps = {c.get("type"): c for c in current.get("components", [])}
        snap_comps = {c.get("type"): c for c in prefab_data.get("components", [])}

        for comp_type, snap_c in snap_comps.items():
            cur_c = cur_comps.get(comp_type)
            if cur_c is None:
                overrides.append({
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "source_id": entity._prefab_source_id,
                    "type": PrefabOverrideType.REMOVED_COMPONENT.value,
                    "component_type": comp_type,
                    "old_value": snap_c,
                    "new_value": None
                })
            else:
                comp_overrides = Prefab._diff_component(cur_c, snap_c, entity, comp_type, is_root)
                overrides.extend(comp_overrides)

        for comp_type, cur_c in cur_comps.items():
            if comp_type not in snap_comps:
                overrides.append({
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "source_id": entity._prefab_source_id,
                    "type": PrefabOverrideType.ADDED_COMPONENT.value,
                    "component_type": comp_type,
                    "old_value": None,
                    "new_value": cur_c
                })

        cur_children = current.get("children", [])
        snap_children = prefab_data.get("children", [])
        snap_child_ids = {c.get("source_id", c.get("id")) for c in snap_children}
        cur_child_ids = {c.get("id") for c in cur_children}

        if snap_child_ids or cur_child_ids:
            for sc in snap_children:
                sc_id = sc.get("source_id", sc.get("id"))
                found = False
                for child_ent in entity.children:
                    if child_ent._prefab_source_id == sc_id:
                        found = True
                        break
                if not found:
                    overrides.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "source_id": entity._prefab_source_id,
                        "type": PrefabOverrideType.REMOVED_CHILD.value,
                        "child_source_id": sc_id,
                        "child_name": sc.get("name", "?"),
                        "old_value": sc,
                        "new_value": None
                    })

            for child_ent in entity.children:
                if not child_ent._prefab_source_id or child_ent._prefab_source_id not in snap_child_ids:
                    overrides.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "source_id": entity._prefab_source_id,
                        "type": PrefabOverrideType.ADDED_CHILD.value,
                        "child_id": child_ent.id,
                        "child_name": child_ent.name,
                        "old_value": None,
                        "new_value": child_ent.serialize()
                    })

        return overrides

    @staticmethod
    def _diff_component(comp_cur: dict, comp_snap: dict, entity: Entity,
                        comp_type: str, is_root: bool) -> list[dict]:
        overrides = []
        all_keys = set(comp_cur.keys()) | set(comp_snap.keys())
        skip_keys = {"type", "enabled", "_key"}

        for key in all_keys:
            if key in skip_keys:
                continue
            cur_val = comp_cur.get(key)
            snap_val = comp_snap.get(key)
            if cur_val != snap_val:
                if is_root and comp_type == "Transform" and key in _ROOT_TRANFORM_EXCLUDED:
                    continue
                overrides.append({
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "source_id": entity._prefab_source_id,
                    "type": PrefabOverrideType.PROPERTY.value,
                    "component_type": comp_type,
                    "property": f"{comp_type}.{key}",
                    "old_value": snap_val,
                    "new_value": cur_val
                })

        if comp_cur.get("enabled") != comp_snap.get("enabled"):
            if not any(o["property"] == f"{comp_type}.enabled" for o in overrides):
                overrides.append({
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "source_id": entity._prefab_source_id,
                    "type": PrefabOverrideType.PROPERTY.value,
                    "component_type": comp_type,
                    "property": f"{comp_type}.enabled",
                    "old_value": comp_snap.get("enabled"),
                    "new_value": comp_cur.get("enabled")
                })

        return overrides

    @staticmethod
    def has_overrides(entities: list[Entity]) -> bool:
        return len(Prefab.compute_all_overrides(entities)) > 0

    @staticmethod
    def apply_overrides(entity_root: Entity):
        prefab_path = None
        try:
            from core.prefab import PrefabLibrary
            prefab_path = PrefabLibrary.path_for_guid(entity_root._prefab_guid)
        except ImportError:
            pass
        if not prefab_path:
            Logger.warning("Cannot apply overrides: prefab asset not found.")
            return
        overrides = Prefab.compute_all_overrides([entity_root])
        prefab = Prefab.load(prefab_path)
        if not prefab:
            return
        roots_data = Prefab._apply_overrides_to_data(prefab.roots_data, overrides, entity_root, prefab)
        prefab.roots_data = roots_data
        prefab.save(prefab_path)
        try:
            from core.prefab import PrefabLibrary
            PrefabLibrary.invalidate(prefab_path)
        except ImportError:
            pass
        Logger.info(f"Applied overrides to prefab: {prefab_path}")

    @staticmethod
    def _apply_overrides_to_data(roots_data: list[dict], overrides: list[dict],
                                  entity_root: Entity, prefab: Prefab) -> list[dict]:
        result = copy.deepcopy(roots_data)
        name_changes: dict[str, str] = {}

        for ov in overrides:
            otype = ov["type"]
            source_id = ov.get("source_id")

            if otype == PrefabOverrideType.PROPERTY.value:
                prop = ov["property"]
                if "." in prop:
                    comp_type, prop_key = prop.split(".", 1)
                else:
                    comp_type = None
                    prop_key = prop

                def apply_prop(items):
                    for item in items:
                        sid = item.get("source_id", item.get("id"))
                        if sid == source_id:
                            if comp_type:
                                for c in item.get("components", []):
                                    if c.get("type") == comp_type:
                                        c[prop_key] = ov["new_value"]
                            else:
                                if prop_key == "name":
                                    item[prop_key] = ov["new_value"]
                                elif prop_key in ("tags", "layer"):
                                    item[prop_key] = ov["new_value"]
                        apply_prop(item.get("children", []))
                apply_prop(result)

            elif otype == PrefabOverrideType.ADDED_COMPONENT.value:
                def add_comp(items):
                    for item in items:
                        sid = item.get("source_id", item.get("id"))
                        if sid == source_id:
                            existing = [c for c in item.get("components", [])
                                       if c.get("type") == ov["component_type"]]
                            if not existing:
                                item.setdefault("components", []).append(ov["new_value"])
                        add_comp(item.get("children", []))
                add_comp(result)

            elif otype == PrefabOverrideType.REMOVED_COMPONENT.value:
                def rem_comp(items):
                    for item in items:
                        sid = item.get("source_id", item.get("id"))
                        if sid == source_id:
                            item["components"] = [c for c in item.get("components", [])
                                                  if c.get("type") != ov["component_type"]]
                        rem_comp(item.get("children", []))
                rem_comp(result)

        return result

    @staticmethod
    def revert_instance(root_entity: Entity, registry: ComponentRegistry):
        prefab_path = None
        try:
            from core.prefab import PrefabLibrary
            prefab_path = PrefabLibrary.path_for_guid(root_entity._prefab_guid)
        except ImportError:
            pass
        if not prefab_path:
            Logger.warning("Cannot revert: prefab asset not found.")
            return
        prefab = Prefab.load(prefab_path)
        if not prefab:
            return
        def revert_subtree(entity):
            source_id = entity._prefab_source_id
            source_data = Prefab._find_source_data(prefab.roots_data, source_id) if source_id else None
            if source_data:
                entity._name = source_data.get("name", entity._name)
                comp_types_to_remove = list(entity._components.keys())
                for ct_key in comp_types_to_remove:
                    comp = entity._components[ct_key]
                    comp.on_destroy()
                entity._components.clear()
                entity._type_map.clear()
                entity._type_name_map.clear()
                entity._update_list.clear()
                entity._fixed_update_list.clear()
                for cd in source_data.get("components", []):
                    ctype = cd.get("type")
                    comp_cls = registry.get(ctype)
                    if comp_cls:
                        comp = comp_cls.deserialize(cd)
                        entity.add_component(comp)
            child_overrides_added = set()
            child_overrides_removed = set()
            source_children = source_data.get("children", []) if source_data else []
            source_child_ids = {sc.get("source_id", sc.get("id")) for sc in source_children}
            for child in list(entity.children):
                if child._prefab_source_id and child._prefab_source_id in source_child_ids:
                    revert_subtree(child)
                else:
                    entity._scene.remove_entity(child.id)
            for sc in source_children:
                sc_id = sc.get("source_id", sc.get("id"))
                found = any(c._prefab_source_id == sc_id for c in entity.children)
                if not found:
                    cd = copy.deepcopy(sc)
                    cd["id"] = str(uuid.uuid4())
                    cd["parent"] = None
                    child = Entity.deserialize(cd, registry)
                    child._prefab_guid = prefab.guid
                    child._prefab_source_id = sc_id
                    entity._scene.add_entity(child)
                    child.set_parent(entity)
                    Prefab._restore_source_children(child, sc.get("children", []), registry, prefab)
        revert_subtree(root_entity)

    @staticmethod
    def _restore_source_children(parent_entity, source_children, registry, prefab):
        for sc in source_children:
            sc_id = sc.get("source_id", sc.get("id"))
            cd = copy.deepcopy(sc)
            cd["id"] = str(uuid.uuid4())
            cd["parent"] = None
            child = Entity.deserialize(cd, registry)
            child._prefab_guid = prefab.guid
            child._prefab_source_id = sc_id
            parent_entity._scene.add_entity(child)
            child.set_parent(parent_entity)
            Prefab._restore_source_children(child, sc.get("children", []), registry, prefab)

    @staticmethod
    def get_prefab_roots(instances: list[Entity]) -> list[Entity]:
        roots = []
        for e in instances:
            if not e.is_prefab_instance:
                continue
            p = e._parent
            while p and p.is_prefab_instance and p._prefab_guid == e._prefab_guid:
                p = p._parent
            if p is None or not p.is_prefab_instance or p._prefab_guid != e._prefab_guid:
                roots.append(e)
        return roots

    @staticmethod
    def unpack(entity_root: Entity):
        def walk(e):
            e._prefab_guid = None
            e._prefab_source_id = None
            for c in e.children:
                walk(c)
        walk(entity_root)

    @staticmethod
    def create_variant(base_prefab: Prefab, name: str, guid: Optional[str] = None) -> Prefab:
        variant = Prefab(name, guid, base_guid=base_prefab.guid)
        variant.roots_data = copy.deepcopy(base_prefab.roots_data)
        return variant

    @staticmethod
    def get_source(prefab_or_entity) -> Optional[str]:
        if isinstance(prefab_or_entity, Prefab):
            return prefab_or_entity.base_guid
        entity = prefab_or_entity
        return entity._prefab_guid

    @staticmethod
    def get_original_source(prefab_or_entity, library=None) -> Optional[str]:
        if isinstance(prefab_or_entity, Prefab):
            p = prefab_or_entity
            while p.base_guid:
                base = None
                for path, cached in (library._prefabs.items() if library else {}).items():
                    if cached.guid == p.base_guid:
                        base = cached
                        break
                if base is None:
                    break
                p = base
            return p.guid
        return prefab_or_entity._prefab_guid


class PrefabLibrary:
    _prefabs: dict[str, Prefab] = {}
    _guids: dict[str, str] = {}

    @classmethod
    def register(cls, path: str, prefab: Prefab):
        cls._prefabs[path] = prefab
        cls._guids[prefab.guid] = path

    @classmethod
    def load(cls, path: str) -> Optional[Prefab]:
        if not os.path.exists(path):
            return None
        if path in cls._prefabs:
            return cls._prefabs[path]
        p = Prefab.load(path)
        if p:
            cls._prefabs[path] = p
            cls._guids[p.guid] = path
        return p

    @classmethod
    def get_all(cls) -> dict[str, Prefab]:
        return dict(cls._prefabs)

    @classmethod
    def path_for_guid(cls, guid: str) -> Optional[str]:
        return cls._guids.get(guid)

    @classmethod
    def invalidate(cls, path: str):
        p = cls._prefabs.pop(path, None)
        if p:
            cls._guids.pop(p.guid, None)

    @classmethod
    def clear(cls):
        cls._prefabs.clear()
        cls._guids.clear()
