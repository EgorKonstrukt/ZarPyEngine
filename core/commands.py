# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Optional, Callable, TYPE_CHECKING
if TYPE_CHECKING:
    from core.ecs import Entity, Scene
class Command:
    def execute(self): raise NotImplementedError
    def undo(self): raise NotImplementedError
    def redo(self):
        self.execute()
    def merge(self, other: Command) -> bool:
        return False
    @property
    def description(self) -> str:
        return self.__class__.__name__
class CompoundCommand(Command):
    def __init__(self, commands: list[Command], description: str = "Compound"):
        self._commands = commands
        self._desc = description
    def execute(self):
        for c in self._commands:
            c.execute()
    def undo(self):
        for c in reversed(self._commands):
            c.undo()
    @property
    def description(self) -> str:
        return self._desc
class CreateEntityCommand(Command):
    def __init__(self, scene: Scene, name: str = "Entity"):
        self._scene = scene
        self._name = name
        self._entity_id: Optional[str] = None
        self._entity_data: Optional[dict] = None
    def execute(self):
        from core.ecs import Entity
        e = self._scene.create_entity(self._name)
        self._entity_id = e.id
    def undo(self):
        if self._entity_id is not None:
            e = self._scene.get_entity(self._entity_id)
            if e:
                self._entity_data = e.serialize()
            self._scene.remove_entity(self._entity_id)
    def redo(self):
        if self._entity_data is not None:
            from core.engine import Engine
            from core.ecs import Entity
            reg = Engine.instance()._component_registry
            e = Entity.deserialize(self._entity_data, reg)
            self._scene.add_entity(e)
            self._entity_id = e.id
        else:
            self.execute()
    @property
    def description(self): return f"Create '{self._name}'"
class DeleteEntityCommand(Command):
    def __init__(self, scene: Scene, entity_id: str):
        self._scene = scene
        self._entity_id = entity_id
        self._entity_data: Optional[dict] = None
    def execute(self):
        e = self._scene.get_entity(self._entity_id)
        if e:
            self._entity_data = e.serialize()
            self._scene.remove_entity(self._entity_id)
    def undo(self):
        if self._entity_data:
            from core.engine import Engine
            from core.ecs import Entity
            reg = Engine.instance()._component_registry
            e = Entity.deserialize(self._entity_data, reg)
            self._scene.add_entity(e)
    @property
    def description(self): return f"Delete Entity #{self._entity_id}"
class SetComponentCommand(Command):
    def __init__(self, entity, component_type, prop_name: str, old_value, new_value):
        self._entity_id = entity.id
        self._scene = getattr(entity, '_scene', None)
        self._component_type = component_type
        self._prop = prop_name
        self._old = old_value
        self._new = new_value
    def _get_entity(self):
        if self._scene:
            e = self._scene.get_entity(self._entity_id)
            if e:
                return e
        from core.engine import Engine
        eng = Engine.instance()
        if eng and eng.scene:
            return eng.scene.get_entity(self._entity_id)
        return None
    def execute(self):
        e = self._get_entity()
        if e:
            comp = e.get_component(self._component_type)
            if comp and hasattr(comp, self._prop):
                setattr(comp, self._prop, self._new)
    def undo(self):
        e = self._get_entity()
        if e:
            comp = e.get_component(self._component_type)
            if comp and hasattr(comp, self._prop):
                setattr(comp, self._prop, self._old)
    @property
    def description(self):
        e = self._get_entity()
        ename = getattr(e, "name", "") or getattr(e, "id", "?")
        return f"{ename}: {self._component_type.__name__}.{self._prop}"
class AddComponentCommand(Command):
    def __init__(self, entity, component_cls, component_data: dict = None, key: str = None):
        self._entity = entity
        self._component_cls = component_cls
        self._component_data = component_data
        self._key = key
        self._added_key: Optional[str] = None
    def execute(self):
        comp = self._component_cls()
        if self._component_data:
            for k, v in self._component_data.items():
                if hasattr(comp, k):
                    setattr(comp, k, v)
        self._entity.add_component(comp, key=self._key)
        for k, c in self._entity._components.items():
            if c is comp:
                self._added_key = k
                break
    def undo(self):
        if self._added_key:
            self._entity.remove_component_by_key(self._added_key)
        else:
            self._entity.remove_component(self._component_cls)
        self._component_data = None
    @property
    def description(self): return f"Add {self._component_cls.__name__}"
class RemoveComponentCommand(Command):
    def __init__(self, entity, component_cls, component_key: str = None):
        self._entity = entity
        self._component_cls = component_cls
        self._component_key = component_key
        self._component_data = None
        self._removed_key: Optional[str] = None
    def execute(self):
        if self._component_key:
            comp = self._entity._components.get(self._component_key)
            if comp:
                self._component_data = {k: getattr(comp, k) for k in vars(comp) if not k.startswith("_")}
                self._removed_key = self._component_key
                self._entity.remove_component_by_key(self._component_key)
        else:
            comp = self._entity.get_component(self._component_cls)
            if comp:
                self._component_data = {k: getattr(comp, k) for k in vars(comp) if not k.startswith("_")}
                for k, c in list(self._entity._components.items()):
                    if c is comp:
                        self._removed_key = k
                        break
                self._entity.remove_component(self._component_cls)
    def undo(self):
        if self._component_data is not None:
            comp = self._component_cls()
            for k, v in self._component_data.items():
                if hasattr(comp, k):
                    setattr(comp, k, v)
            self._entity.add_component(comp, key=self._removed_key)
    @property
    def description(self):
        base = f"Remove {self._component_cls.__name__}"
        if self._component_key:
            base += f" ({self._component_key})"
        return base
class InstantiatePrefabCommand(Command):
    def __init__(self, scene, prefab, registry, parent=None, position=None):
        self._scene = scene
        self._prefab = prefab
        self._registry = registry
        self._parent = parent
        self._position = position
        self._spawned_ids: list[str] = []
    def execute(self):
        spawned = self._prefab.instantiate(self._scene, self._registry, self._parent)
        self._spawned_ids = [e.id for e in spawned]
    def undo(self):
        for eid in self._spawned_ids:
            e = self._scene.get_entity(eid)
            if e:
                self._scene.remove_entity(eid)
        self._spawned_ids.clear()
    def redo(self):
        self.execute()
    @property
    def description(self): return f"Instantiate '{self._prefab.name}'"
class RevertPrefabInstanceCommand(Command):
    def __init__(self, scene, root_entities: list, registry=None):
        self._scene = scene
        self._roots = root_entities
        self._registry = registry
        self._old_data: list[tuple[str, dict]] = []
    def _collect(self, entity) -> list[tuple[str, dict]]:
        items = [(entity.id, entity.serialize())]
        for c in entity.children:
            items.extend(self._collect(c))
        return items
    def execute(self):
        from core.prefab import Prefab
        if self._registry is None:
            from core.engine import Engine
            self._registry = Engine.instance()._component_registry
        self._old_data.clear()
        for root in self._roots:
            self._old_data.extend(self._collect(root))
        for root in self._roots:
            Prefab.revert_instance(root, self._registry)
    def undo(self):
        for eid, data in self._old_data:
            e = self._scene.get_entity(eid)
            if e:
                for ct in list(e._components.keys()):
                    c = e._components[ct]
                    c.on_destroy()
                e._components.clear()
                for cd in data.get("components", []):
                    ctype = cd.get("type")
                    comp_cls = self._registry.get(ctype)
                    if comp_cls:
                        comp = comp_cls.deserialize(cd)
                        e.add_component(comp)
                e._name = data.get("name", e._name)
                e._active = data.get("active", e._active)
                e._tags = set(data.get("tags", []))
                e._layer = data.get("layer", e._layer)
    @property
    def description(self): return "Revert Prefab"
class UnpackPrefabCommand(Command):
    def __init__(self, scene, root_entities: list):
        self._scene = scene
        self._roots = root_entities
        self._old_links: list[tuple[str, Optional[str], Optional[str]]] = []
    def execute(self):
        from core.prefab import Prefab
        self._old_links.clear()
        def unpack(e):
            self._old_links.append((e.id, e._prefab_guid, e._prefab_source_id))
            Prefab.unpack(e)
        for root in self._roots:
            unpack(root)
    def undo(self):
        for eid, guid, source_id in self._old_links:
            e = self._scene.get_entity(eid)
            if e:
                e._prefab_guid = guid
                e._prefab_source_id = source_id
    @property
    def description(self): return "Unpack Prefab"
class ApplyPrefabOverridesCommand(Command):
    def __init__(self, scene, root_entities: list, registry=None):
        self._scene = scene
        self._roots = root_entities
        self._registry = registry
    def execute(self):
        from core.prefab import Prefab
        if self._registry is None:
            from core.engine import Engine
            self._registry = Engine.instance()._component_registry
        for root in self._roots:
            Prefab.apply_overrides(root)
    def undo(self):
        pass
    @property
    def description(self): return "Apply Prefab Overrides"
class PasteEntitiesCommand(Command):
    def __init__(self, scene, clipboard_data: list[dict], registry):
        self._scene = scene
        self._clipboard_data = clipboard_data
        self._registry = registry
        self._spawned_ids: list[str] = []
        self._entity_datas: list[dict] = []
        self._entity_id: Optional[str] = None
    @property
    def spawned_ids(self) -> list[str]:
        return list(self._spawned_ids)
    def execute(self):
        import copy, uuid
        from core.ecs import Entity
        id_map: dict[str, str] = {}
        self._spawned_ids.clear()
        self._entity_datas.clear()
        self._entity_id = None
        for data in self._clipboard_data:
            d = copy.deepcopy(data)
            old_id = d["id"]
            new_id = str(uuid.uuid4())
            d["id"] = new_id
            id_map[old_id] = new_id
            e = Entity.deserialize(d, self._registry)
            self._scene.add_entity(e)
            self._spawned_ids.append(e.id)
        for data in self._clipboard_data:
            parent_id = data.get("parent")
            if parent_id and parent_id in id_map:
                child_id = id_map[data["id"]]
                new_parent_id = id_map[parent_id]
                child = self._scene.get_entity(child_id)
                new_parent = self._scene.get_entity(new_parent_id)
                if child and new_parent:
                    child.set_parent(new_parent)
        for eid in self._spawned_ids:
            e = self._scene.get_entity(eid)
            if e:
                self._entity_datas.append(e.serialize())
        if self._spawned_ids:
            self._entity_id = self._spawned_ids[0]
    def undo(self):
        for eid in self._spawned_ids:
            e = self._scene.get_entity(eid)
            if e:
                self._scene.remove_entity(eid)
    def redo(self):
        from core.ecs import Entity
        self._spawned_ids.clear()
        for d in self._entity_datas:
            e = Entity.deserialize(d, self._registry)
            self._scene.add_entity(e)
            self._spawned_ids.append(e.id)
        for d in self._entity_datas:
            parent_id = d.get("parent")
            if parent_id:
                child_id = d["id"]
                child = self._scene.get_entity(child_id)
                parent = self._scene.get_entity(parent_id)
                if child and parent:
                    child.set_parent(parent, preserve_world=False)
        if self._spawned_ids:
            self._entity_id = self._spawned_ids[0]
    @property
    def description(self):
        n = len(self._clipboard_data)
        return f"Paste {n} entit{'y' if n == 1 else 'ies'}"
class MoveComponentCommand(Command):
    def __init__(self, source_entity, target_entity, component_key, component_cls, component_data: dict):
        self._source = source_entity
        self._target = target_entity
        self._component_key = component_key
        self._component_cls = component_cls
        self._component_data = component_data
        self._target_key = None
    def execute(self):
        if self._component_key:
            self._source.remove_component_by_key(self._component_key)
        else:
            self._source.remove_component(self._component_cls)
        new_comp = self._component_cls.deserialize(self._component_data)
        key_to_use = self._component_key if not getattr(self._component_cls, '_allow_multiple', False) else None
        self._target.add_component(new_comp, key=key_to_use)
        for k, c in self._target._components.items():
            if c is new_comp:
                self._target_key = k
                break
    def undo(self):
        if self._target_key:
            self._target.remove_component_by_key(self._target_key)
        new_comp = self._component_cls.deserialize(self._component_data)
        self._source.add_component(new_comp, key=self._component_key)
        self._target_key = None
    @property
    def description(self):
        return f"Move {self._component_cls.__name__}"

class CopyComponentCommand(Command):
    def __init__(self, target_entity, component_cls, component_data: dict, source_key: str = None):
        self._target = target_entity
        self._component_cls = component_cls
        self._component_data = component_data
        self._source_key = source_key
        self._target_key = None
    def execute(self):
        new_comp = self._component_cls.deserialize(self._component_data)
        key_to_use = self._source_key if not getattr(self._component_cls, '_allow_multiple', False) else None
        self._target.add_component(new_comp, key=key_to_use)
        for k, c in self._target._components.items():
            if c is new_comp:
                self._target_key = k
                break
    def undo(self):
        if self._target_key:
            self._target.remove_component_by_key(self._target_key)
    @property
    def description(self):
        return f"Copy {self._component_cls.__name__}"

class CommandHistory:
    def __init__(self, max_size: int = 100):
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._undo_sel: list = []
        self._redo_sel: list = []
        self._max_size = max_size
        self._saved_index: int = 0
        self._last_affected_entity = None
        self._current_selection = None
        self._recording = True
        self._on_undo: Optional[Callable[[Command], None]] = None
        self._on_redo: Optional[Callable[[Command], None]] = None
    def set_on_undo(self, cb: Optional[Callable[[Command], None]]):
        self._on_undo = cb
    def set_on_redo(self, cb: Optional[Callable[[Command], None]]):
        self._on_redo = cb
    @property
    def can_undo(self) -> bool: return len(self._undo_stack) > 0
    @property
    def can_redo(self) -> bool: return len(self._redo_stack) > 0
    @property
    def is_dirty(self) -> bool:
        return len(self._undo_stack) != self._saved_index
    @property
    def last_affected_entity(self):
        return self._last_affected_entity
    @staticmethod
    def _extract_entity(cmd):
        e = getattr(cmd, "_entity", None)
        if e is not None:
            return e
        eid = getattr(cmd, "_entity_id", None)
        if eid is not None:
            scene = getattr(cmd, "_scene", None)
            if scene is None:
                from core.engine import Engine
                scene = Engine.instance().scene
            if scene:
                e = scene.get_entity(eid)
                if e:
                    return e
        cmds = getattr(cmd, "_commands", None)
        if cmds:
            for c in cmds:
                e = CommandHistory._extract_entity(c)
                if e is not None:
                    return e
        return None
    @property
    def undo_text(self) -> str:
        if self._undo_stack:
            return f"Undo {self._undo_stack[-1].description}"
        return "Can't Undo"
    @property
    def redo_text(self) -> str:
        if self._redo_stack:
            return f"Redo {self._redo_stack[-1].description}"
        return "Can't Redo"
    def set_current_selection(self, sel):
        self._current_selection = sel
    @property
    def current_selection(self):
        return self._current_selection
    def execute(self, command: Command):
        if self._recording:
            self._undo_sel.append(self._current_selection)
        command.execute()
        self._undo_stack.append(command)
        self._redo_stack.clear()
        self._redo_sel.clear()
        if len(self._undo_stack) > self._max_size:
            self._undo_stack.pop(0)
            if self._undo_sel:
                self._undo_sel.pop(0)
            if self._saved_index > 0:
                self._saved_index -= 1
    def undo(self):
        if not self._undo_stack:
            return
        was_recording = self._recording
        self._recording = False
        cmd = self._undo_stack.pop()
        pre_sel = self._undo_sel.pop() if self._undo_sel else None
        self._redo_sel.append(self._current_selection)
        cmd.undo()
        self._redo_stack.append(cmd)
        self._current_selection = pre_sel
        self._last_affected_entity = self._extract_entity(cmd) or pre_sel
        self._recording = was_recording
        if was_recording and self._on_undo:
            self._on_undo(cmd)
    def redo(self):
        if not self._redo_stack:
            return
        was_recording = self._recording
        self._recording = False
        cmd = self._redo_stack.pop()
        post_sel = self._redo_sel.pop() if self._redo_sel else None
        self._undo_sel.append(self._current_selection)
        cmd.redo()
        self._undo_stack.append(cmd)
        self._current_selection = post_sel
        self._last_affected_entity = self._extract_entity(cmd) or post_sel
        self._recording = was_recording
        if was_recording and self._on_redo:
            self._on_redo(cmd)
    @property
    def undo_count(self) -> int:
        return len(self._undo_stack)
    @property
    def redo_count(self) -> int:
        return len(self._redo_stack)
    def get_undo_descriptions(self) -> list[str]:
        return [c.description for c in self._undo_stack]
    def get_redo_descriptions(self) -> list[str]:
        return [c.description for c in self._redo_stack]
    def seek(self, target_head: int):
        total = len(self._undo_stack) + len(self._redo_stack)
        target_head = max(0, min(target_head, total))
        current = len(self._undo_stack)
        if current == target_head:
            return
        was_recording = self._recording
        self._recording = False
        if current > target_head:
            n = current - target_head
            for _ in range(n):
                cmd = self._undo_stack.pop()
                pre_sel = self._undo_sel.pop() if self._undo_sel else None
                self._redo_sel.append(self._current_selection)
                cmd.undo()
                self._redo_stack.append(cmd)
                self._current_selection = pre_sel
            self._last_affected_entity = (
                self._extract_entity(self._redo_stack[-1])
                if self._redo_stack else None
            ) or pre_sel
        else:
            n = target_head - current
            for _ in range(n):
                cmd = self._redo_stack.pop()
                post_sel = self._redo_sel.pop() if self._redo_sel else None
                self._undo_sel.append(self._current_selection)
                cmd.redo()
                self._undo_stack.append(cmd)
                self._current_selection = post_sel
            self._last_affected_entity = (
                self._extract_entity(self._undo_stack[-1])
                if self._undo_stack else None
            ) or post_sel
        self._recording = was_recording
    def mark_saved(self):
        self._saved_index = len(self._undo_stack)
    def clear(self):
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._undo_sel.clear()
        self._redo_sel.clear()
        self._saved_index = 0
_history: Optional[CommandHistory] = None
def get_history() -> CommandHistory:
    global _history
    if _history is None:
        _history = CommandHistory()
    return _history
