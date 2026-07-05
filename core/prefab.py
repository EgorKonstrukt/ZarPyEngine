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
    MODIFIED_PROPERTY = "modified_property"
    ADDED_COMPONENT = "added_component"
    REMOVED_COMPONENT = "removed_component"
    ADDED_CHILD = "added_child"
    REMOVED_CHILD = "removed_child"


ROOT_TRANSFORM_EXCLUDED = frozenset({"local_position", "local_rotation"})
_ROOT_TRANSFORM_SKIP_COMPARE = frozenset({"local_position", "local_rotation", "position", "rotation"})


class Prefab:
    TRANSFORM_PASSTHROUGH = frozenset({"local_position", "local_rotation"})

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
        self.roots_data = []
        for e in entities:
            data = self._capture_entity_data(e)
            data.pop("prefab_guid", None)
            data.pop("prefab_source_id", None)
            self.roots_data.append(data)

    def _capture_entity_data(self, entity: Entity) -> dict:
        data = entity.serialize()
        old_id = data.get("id")
        data["source_id"] = old_id
        data.pop("parent", None)
        data["children"] = []
        for child in entity.children:
            child_data = self._capture_entity_data(child)
            data["children"].append(child_data)
        return data

    def instantiate(self, scene: Scene, registry: ComponentRegistry,
                    parent: Optional[Entity] = None) -> list[Entity]:
        if not self.roots_data:
            Logger.warning("Prefab has no root data.")
            return []
        spawned: list[Entity] = []
        for rd in self.roots_data:
            data = copy.deepcopy(rd)
            id_map: dict[str, str] = {}
            self._remap_ids(data, id_map)
            source_id = rd.get("source_id", rd.get("id"))
            children_data = data.pop("children", [])
            e = Entity.deserialize(data, registry)
            e._prefab_guid = self.guid
            e._prefab_source_id = source_id
            scene.add_entity(e)
            if parent:
                e.set_parent(parent)
            spawned.append(e)
            self._instantiate_children(e, children_data, id_map, scene, registry)
        return spawned

    def _instantiate_children(self, parent_entity: Entity, children_data: list[dict],
                               id_map: dict[str, str], scene: Scene,
                               registry: ComponentRegistry):
        for cd in children_data:
            child_source_id = cd.get("source_id", cd.get("id"))
            sub_children = cd.pop("children", [])
            child_id = cd.get("id")
            if child_id in id_map:
                cd["id"] = id_map[child_id]
            e = Entity.deserialize(cd, registry)
            nested_prefab_guid = cd.get("prefab_guid")
            if nested_prefab_guid:
                e._prefab_guid = nested_prefab_guid
            else:
                e._prefab_guid = self.guid
            e._prefab_source_id = child_source_id
            scene.add_entity(e)
            e.set_parent(parent_entity)
            self._instantiate_children(e, sub_children, id_map, scene, registry)

    def _remap_ids(self, data: dict, id_map: dict[str, str]):
        old_id = data.get("id")
        new_id = str(uuid.uuid4())
        if old_id:
            id_map[old_id] = new_id
        data["id"] = new_id
        for child in data.get("children", []):
            self._remap_ids(child, id_map)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        roots = self._serialize_roots(self.roots_data)
        for root in roots:
            root.pop("prefab_guid", None)
            root.pop("prefab_source_id", None)
        out = {
            "guid": self.guid,
            "name": self.name,
            "roots": roots
        }
        if self.base_guid:
            out["base_guid"] = self.base_guid
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        Logger.info(f"Prefab saved: {path}")

    def _serialize_roots(self, roots: list[dict]) -> list[dict]:
        result = []
        for root in roots:
            data = copy.deepcopy(root)
            data.pop("parent", None)
            if "id" in data and "source_id" not in data:
                data["source_id"] = data.pop("id")
            for comp in data.get("components", []):
                comp.pop("_key", None)
            if "children" in data:
                data["children"] = self._serialize_roots(data["children"])
            result.append(data)
        return result

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
        prefab = Prefab._resolve_prefab_for_entity(entity)
        if prefab is None:
            return []
        source_id = entity._prefab_source_id
        prefab_data = Prefab._find_source_data(prefab.roots_data, source_id)
        if prefab_data is None:
            for rd in prefab.roots_data:
                prefab_data = Prefab._find_source_data([rd], source_id)
                if prefab_data:
                    break
        if prefab_data is None:
            return []
        current = entity.serialize()
        is_root_of_instance = Prefab._is_root_of_instance(entity)
        overrides = Prefab._diff_entity(current, prefab_data, entity, is_root_of_instance)
        return overrides

    @staticmethod
    def _resolve_prefab_for_entity(entity: Entity) -> Optional[Prefab]:
        from core.prefab import PrefabLibrary
        guid = entity._prefab_guid
        if not guid:
            return None
        path = PrefabLibrary.path_for_guid(guid)
        if not path:
            return None
        prefab = PrefabLibrary.load(path)
        if prefab is None:
            return None
        if prefab.is_variant:
            return Prefab._resolve_variant_data(prefab, entity)
        return prefab

    @staticmethod
    def _resolve_variant_data(variant: Prefab, entity: Entity) -> Optional[Prefab]:
        from core.prefab import PrefabLibrary
        chain = []
        current = variant
        while current:
            chain.append(current)
            if not current.base_guid:
                break
            base_path = PrefabLibrary.path_for_guid(current.base_guid)
            if not base_path:
                break
            base = PrefabLibrary.load(base_path)
            if base is None:
                break
            current = base
        if not chain:
            return None
        resolved = Prefab(variant.name, variant.guid, variant.base_guid)
        resolved.roots_data = copy.deepcopy(chain[-1].roots_data)
        for layer in reversed(chain[:-1]):
            resolved.roots_data = Prefab._merge_overrides(resolved.roots_data, layer.roots_data)
        return resolved

    @staticmethod
    def _merge_overrides(base_roots: list[dict], override_roots: list[dict]) -> list[dict]:
        merged = copy.deepcopy(base_roots)
        for ov_root in override_roots:
            ov_source_id = ov_root.get("source_id", ov_root.get("id"))
            found = False
            for base in merged:
                base_sid = base.get("source_id", base.get("id"))
                if base_sid == ov_source_id:
                    Prefab._merge_entity_data(base, ov_root)
                    found = True
                    break
            if not found:
                merged.append(copy.deepcopy(ov_root))
        return merged

    @staticmethod
    def _merge_entity_data(base: dict, override: dict):
        if override.get("name") != base.get("name"):
            base["name"] = override["name"]
        if override.get("active") != base.get("active"):
            base["active"] = override.get("active", True)
        ov_comps = {c.get("type"): c for c in override.get("components", [])}
        base_comps = {c.get("type"): c for c in base.get("components", [])}
        for ctype, ov_c in ov_comps.items():
            if ctype in base_comps:
                base_c = base_comps[ctype]
                for k, v in ov_c.items():
                    if k not in ("type", "enabled", "_key"):
                        base_c[k] = v
            else:
                base.setdefault("components", []).append(copy.deepcopy(ov_c))
        for ctype in list(base_comps.keys()):
            if ctype not in ov_comps:
                base["components"] = [c for c in base.get("components", [])
                                      if c.get("type") != ctype]
        ov_child_map = {}
        for c in override.get("children", []):
            sid = c.get("source_id", c.get("id"))
            ov_child_map[sid] = c
        base_child_map = {}
        for c in base.get("children", []):
            sid = c.get("source_id", c.get("id"))
            base_child_map[sid] = c
        for sid, ov_c in ov_child_map.items():
            if sid in base_child_map:
                Prefab._merge_entity_data(base_child_map[sid], ov_c)
            else:
                base.setdefault("children", []).append(copy.deepcopy(ov_c))
        for sid in list(base_child_map.keys()):
            if sid not in ov_child_map:
                base["children"] = [c for c in base.get("children", [])
                                    if c.get("source_id", c.get("id")) != sid]

    @staticmethod
    def compute_all_overrides(entities: list[Entity]) -> list[dict]:
        all_overrides = []
        for e in entities:
            all_overrides.extend(Prefab.compute_overrides(e))
            all_overrides.extend(Prefab.compute_all_overrides(e.children))
        return all_overrides

    @staticmethod
    def _is_root_of_instance(entity: Entity) -> bool:
        if not entity._prefab_guid:
            return False
        p = entity._parent
        while p:
            if p._prefab_guid == entity._prefab_guid:
                return False
            p = p._parent
        return True

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
    def _diff_entity(current: dict, prefab_data: dict, entity: Entity,
                     is_root: bool = False) -> list[dict]:
        overrides = []

        if current.get("name") != prefab_data.get("name"):
            overrides.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "source_id": entity._prefab_source_id,
                "type": PrefabOverrideType.MODIFIED_PROPERTY.value,
                "property": "name",
                "old_value": prefab_data.get("name"),
                "new_value": current.get("name")
            })

        if current.get("active") != prefab_data.get("active"):
            overrides.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "source_id": entity._prefab_source_id,
                "type": PrefabOverrideType.MODIFIED_PROPERTY.value,
                "property": "active",
                "old_value": prefab_data.get("active", True),
                "new_value": current.get("active", True)
            })

        if set(current.get("tags", [])) != set(prefab_data.get("tags", [])):
            overrides.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "source_id": entity._prefab_source_id,
                "type": PrefabOverrideType.MODIFIED_PROPERTY.value,
                "property": "tags",
                "old_value": prefab_data.get("tags", []),
                "new_value": current.get("tags", [])
            })

        if current.get("layer") != prefab_data.get("layer"):
            overrides.append({
                "entity_id": entity.id,
                "entity_name": entity.name,
                "source_id": entity._prefab_source_id,
                "type": PrefabOverrideType.MODIFIED_PROPERTY.value,
                "property": "layer",
                "old_value": prefab_data.get("layer"),
                "new_value": current.get("layer")
            })

        cur_comps = {}
        for c in current.get("components", []):
            ct = c.get("type")
            key = c.get("_key", ct)
            cur_comps[key] = c
        snap_comps = {}
        for c in prefab_data.get("components", []):
            ct = c.get("type")
            key = c.get("_key", ct)
            snap_comps[key] = c

        snap_by_type = {}
        for key, c in snap_comps.items():
            ct = c.get("type")
            snap_by_type.setdefault(ct, []).append(c)

        for comp_type, snap_list in snap_by_type.items():
            matching_cur = [c for k, c in cur_comps.items() if c.get("type") == comp_type]
            for idx, snap_c in enumerate(snap_list):
                if idx < len(matching_cur):
                    cur_c = matching_cur[idx]
                    comp_overrides = Prefab._diff_component(cur_c, snap_c, entity, comp_type, is_root)
                    overrides.extend(comp_overrides)
                else:
                    overrides.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "source_id": entity._prefab_source_id,
                        "type": PrefabOverrideType.REMOVED_COMPONENT.value,
                        "component_type": comp_type,
                        "component_key": "",
                        "old_value": snap_c,
                        "new_value": None
                    })

        for cur_key, cur_c in cur_comps.items():
            comp_type = cur_c.get("type")
            snap_count = len(snap_by_type.get(comp_type, []))
            cur_count = len([c for k, c in cur_comps.items() if c.get("type") == comp_type])
            if cur_count > snap_count:
                is_excess = False
                type_list = [c for k, c in sorted(cur_comps.items()) if c.get("type") == comp_type]
                if type_list.index(cur_c) >= snap_count:
                    is_excess = True
                if is_excess:
                    overrides.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "source_id": entity._prefab_source_id,
                        "type": PrefabOverrideType.ADDED_COMPONENT.value,
                        "component_type": comp_type,
                        "component_key": cur_key,
                        "old_value": None,
                        "new_value": cur_c
                    })

        for snap_key, snap_c in snap_comps.items():
            if snap_key not in cur_comps:
                comp_type = snap_c.get("type")
                already_removed = any(
                    o["type"] == PrefabOverrideType.REMOVED_COMPONENT.value
                    and o["component_type"] == comp_type
                    for o in overrides
                )
                if not already_removed:
                    overrides.append({
                        "entity_id": entity.id,
                        "entity_name": entity.name,
                        "source_id": entity._prefab_source_id,
                        "type": PrefabOverrideType.REMOVED_COMPONENT.value,
                        "component_type": comp_type,
                        "component_key": snap_key,
                        "old_value": snap_c,
                        "new_value": None
                    })

        cur_children = current.get("children", [])
        snap_children = prefab_data.get("children", [])
        snap_child_ids = {}
        for sc in snap_children:
            sc_id = sc.get("source_id", sc.get("id"))
            snap_child_ids[sc_id] = sc

        cur_child_ids = {}
        for child_ent in entity.children:
            sid = child_ent._prefab_source_id
            if sid:
                cur_child_ids[sid] = child_ent

        if snap_child_ids or cur_child_ids:
            for sc_id, sc in snap_child_ids.items():
                if sc_id not in cur_child_ids:
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
                    if not child_ent._prefab_guid:
                        serialized = child_ent.serialize()
                        serialized.pop("parent", None)
                        overrides.append({
                            "entity_id": entity.id,
                            "entity_name": entity.name,
                            "source_id": entity._prefab_source_id,
                            "type": PrefabOverrideType.ADDED_CHILD.value,
                            "child_id": child_ent.id,
                            "child_name": child_ent.name,
                            "old_value": None,
                            "new_value": serialized
                        })

        return overrides

    @staticmethod
    def _diff_component(comp_cur: dict, comp_snap: dict, entity: Entity,
                        comp_type: str, is_root: bool) -> list[dict]:
        overrides = []
        all_keys = set(comp_cur.keys()) | set(comp_snap.keys())
        skip_keys = {"type", "_key"}

        for key in all_keys:
            if key in skip_keys:
                continue
            cur_val = comp_cur.get(key)
            snap_val = comp_snap.get(key)
            if cur_val != snap_val:
                if is_root and comp_type == "Transform" and key in _ROOT_TRANSFORM_SKIP_COMPARE:
                    continue
                overrides.append({
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "source_id": entity._prefab_source_id,
                    "type": PrefabOverrideType.MODIFIED_PROPERTY.value,
                    "component_type": comp_type,
                    "property": f"{comp_type}.{key}",
                    "old_value": snap_val,
                    "new_value": cur_val
                })

        if comp_cur.get("enabled") != comp_snap.get("enabled"):
            if not any(o.get("property") == f"{comp_type}.enabled" for o in overrides):
                overrides.append({
                    "entity_id": entity.id,
                    "entity_name": entity.name,
                    "source_id": entity._prefab_source_id,
                    "type": PrefabOverrideType.MODIFIED_PROPERTY.value,
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
        from core.prefab import PrefabLibrary
        prefab_path = PrefabLibrary.path_for_guid(entity_root._prefab_guid)
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
        PrefabLibrary.invalidate(prefab_path)
        Logger.info(f"Applied overrides to prefab: {prefab_path}")

    @staticmethod
    def _apply_overrides_to_data(roots_data: list[dict], overrides: list[dict],
                                  entity_root: Entity, prefab: Prefab) -> list[dict]:
        result = copy.deepcopy(roots_data)

        for ov in overrides:
            otype = ov["type"]
            source_id = ov.get("source_id")

            if otype == PrefabOverrideType.MODIFIED_PROPERTY.value:
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
                                elif prop_key in ("active", "tags", "layer"):
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
                                new_comp = copy.deepcopy(ov["new_value"])
                                new_comp.pop("_key", None)
                                item.setdefault("components", []).append(new_comp)
                        add_comp(item.get("children", []))
                add_comp(result)

            elif otype == PrefabOverrideType.REMOVED_COMPONENT.value:
                def rem_comp(items):
                    for item in items:
                        sid = item.get("source_id", item.get("id"))
                        if sid == source_id:
                            item["components"] = [c for c in item.get("components", [])
                                                  if not (c.get("type") == ov["component_type"]
                                                          and c.get("_key", c.get("type")) == ov.get("component_key", ""))]
                        rem_comp(item.get("children", []))
                rem_comp(result)

            elif otype == PrefabOverrideType.ADDED_CHILD.value:
                def add_child(items):
                    for item in items:
                        sid = item.get("source_id", item.get("id"))
                        if sid == source_id:
                            child_data = copy.deepcopy(ov["new_value"])
                            child_data.pop("id", None)
                            child_data.pop("parent", None)
                            child_data["source_id"] = str(uuid.uuid4())
                            if "children" not in child_data:
                                child_data["children"] = []
                            item.setdefault("children", []).append(child_data)
                        add_child(item.get("children", []))
                add_child(result)

            elif otype == PrefabOverrideType.REMOVED_CHILD.value:
                def remove_child(items):
                    for item in items:
                        sid = item.get("source_id", item.get("id"))
                        if sid == source_id:
                            child_sid = ov.get("child_source_id")
                            item["children"] = [c for c in item.get("children", [])
                                                if c.get("source_id", c.get("id")) != child_sid]
                        remove_child(item.get("children", []))
                remove_child(result)

        return result

    @staticmethod
    def revert_instance(root_entity: Entity, registry: ComponentRegistry):
        from core.prefab import PrefabLibrary
        prefab_path = PrefabLibrary.path_for_guid(root_entity._prefab_guid)
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
                entity._active = source_data.get("active", True)
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
                    cd_copy = copy.deepcopy(cd)
                    cd_copy.pop("_key", None)
                    ctype = cd_copy.get("type")
                    comp_cls = registry.get(ctype)
                    if comp_cls:
                        comp = comp_cls.deserialize(cd_copy)
                        entity.add_component(comp)

            source_children = source_data.get("children", []) if source_data else []
            source_child_ids = {}
            for sc in source_children:
                sc_id = sc.get("source_id", sc.get("id"))
                source_child_ids[sc_id] = sc

            for child in list(entity.children):
                if child._prefab_source_id and child._prefab_source_id in source_child_ids:
                    revert_subtree(child)
                else:
                    entity._scene.remove_entity(child.id)

            for sc_id, sc in source_child_ids.items():
                found = any(c._prefab_source_id == sc_id for c in entity.children)
                if not found:
                    cd = copy.deepcopy(sc)
                    cd["id"] = str(uuid.uuid4())
                    cd.pop("parent", None)
                    cd.pop("_key", None)
                    child = Entity.deserialize(cd, registry)
                    if not child._prefab_guid:
                        child._prefab_guid = root_entity._prefab_guid
                        child._prefab_source_id = sc_id
                    entity._scene.add_entity(child)
                    child.set_parent(entity)
                    prefab_guid = child._prefab_guid or root_entity._prefab_guid
                    Prefab._restore_source_children(child, sc.get("children", []), registry,
                                                     prefab_guid)

        revert_subtree(root_entity)

    @staticmethod
    def _restore_source_children(parent_entity, source_children, registry, prefab_guid):
        for sc in source_children:
            sc_id = sc.get("source_id", sc.get("id"))
            cd = copy.deepcopy(sc)
            cd["id"] = str(uuid.uuid4())
            cd.pop("parent", None)
            child = Entity.deserialize(cd, registry)
            if not child._prefab_guid:
                child._prefab_guid = prefab_guid
                child._prefab_source_id = sc_id
            parent_entity._scene.add_entity(child)
            child.set_parent(parent_entity)
            child_prefab_guid = child._prefab_guid or prefab_guid
            Prefab._restore_source_children(child, sc.get("children", []), registry, child_prefab_guid)

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
    def get_source(entity) -> Optional[str]:
        if isinstance(entity, Prefab):
            return entity.base_guid
        return entity._prefab_guid

    @staticmethod
    def get_original_source(entity, library=None) -> Optional[str]:
        if isinstance(entity, Prefab):
            p = entity
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
        return entity._prefab_guid


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
