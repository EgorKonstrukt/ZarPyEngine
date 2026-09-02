# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
from typing import Optional
from core.ecs.ecs import Component, ComponentRegistry, Entity, Scene
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class NetworkManager(Component):
    _icon = "NetworkManager.png"
    _gizmo_icon_color = (255, 200, 50)
    _gizmo_icon_label = "M"
    _instance: Optional[NetworkManager] = None

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("port", "Port", FieldType.INT, min_val=1024, max_val=65535),
            InspectorField("max_players", "Max Players", FieldType.INT, min_val=1, max_val=64),
            InspectorField("server_name", "Server Name", FieldType.STRING),
            InspectorField("host_address", "Host Address", FieldType.STRING),
            InspectorField("player_prefab", "Player Prefab", FieldType.RESOURCE_PATH, file_filter="Prefab (*.zpep)"),
            InspectorField("auto_spawn_player", "Auto Spawn Player", FieldType.BOOL),
            InspectorField("tick_rate", "Tick Rate", FieldType.FLOAT, min_val=1.0, max_val=60.0),
            InspectorField("show_debug", "Show Debug", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.port: int = 7777
        self.max_players: int = 16
        self.server_name: str = "Zarin Server"
        self.host_address: str = "127.0.0.1"
        self.player_prefab: str = ""
        self.auto_spawn_player: bool = True
        self.tick_rate: float = 20.0
        self.show_debug: bool = False
        self._next_net_id: int = 1
        self._net_to_entity: dict[int, str] = {}
        self._entity_to_net: dict[str, int] = {}
        self._pending_spawn_requests: list[dict] = []

    @classmethod
    def get(cls) -> Optional[NetworkManager]:
        if cls._instance is not None:
            return cls._instance
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng and eng.scene:
                for e in eng.scene.get_all_entities():
                    c = e.get_component_by_name("NetworkManager")
                    if c is not None:
                        return c
        except Exception:
            pass
        return None

    @property
    def is_server(self) -> bool:
        try:
            from core.network.transport import get_transport
            return get_transport().is_server
        except Exception:
            return False

    @property
    def is_client(self) -> bool:
        try:
            from core.network.transport import get_transport
            return get_transport().is_client
        except Exception:
            return False

    @property
    def is_connected(self) -> bool:
        try:
            from core.network.transport import get_transport
            return get_transport().is_connected
        except Exception:
            return False

    @property
    def local_id(self) -> int:
        try:
            from core.network.transport import get_transport
            return get_transport().local_id
        except Exception:
            return -1

    @property
    def peer_count(self) -> int:
        try:
            from core.network.transport import get_transport
            return get_transport().peer_count
        except Exception:
            return 0

    def host(self, port: int | None = None, max_players: int | None = None):
        p = int(port) if port is not None else int(self.port)
        mp = int(max_players) if max_players is not None else int(self.max_players)
        mp = max(1, min(64, mp))
        self.max_players = mp
        try:
            from core.network.transport import get_transport
            from core.foundation.logger import Logger
            ok = get_transport().host("0.0.0.0", p, max_players=mp)
            if ok:
                self._next_net_id = 1
                self._net_to_entity.clear()
                self._entity_to_net.clear()
                self._register_existing()
                Logger.info(f"NetworkManager hosting on {p} max_players={mp}")
            else:
                from core.foundation.logger import Logger as _L
                _L.error(f"NetworkManager host failed on {p}")
            return bool(ok)
        except Exception as e:
            try:
                from core.foundation.logger import Logger as _L2
                _L2.error(f"NetworkManager host exception: {e}")
            except Exception:
                pass
            return False

    def connect(self, address: str | None = None, port: int | None = None, name: str | None = None):
        addr = str(address) if address is not None else str(self.host_address)
        p = int(port) if port is not None else int(self.port)
        n = str(name) if name is not None else "Player"
        try:
            from core.network.transport import get_transport
            ok = get_transport().connect(addr, p, n)
            self._net_to_entity.clear()
            self._entity_to_net.clear()
            return bool(ok)
        except Exception:
            return False

    def disconnect(self):
        try:
            from core.network.transport import get_transport
            get_transport().disconnect()
        except Exception:
            pass
        self._net_to_entity.clear()
        self._entity_to_net.clear()

    def _get_scene(self) -> Optional[Scene]:
        ent = self._entity
        if ent is not None and ent._scene is not None:
            return ent._scene
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng:
                return eng.scene
        except Exception:
            pass
        return None

    def _register_existing(self):
        scene = self._get_scene()
        if scene is None:
            return
        for e in scene.get_all_entities():
            ident = e.get_component_by_name("NetworkIdentity")
            if ident is not None and ident.net_id >= 0:
                self._net_to_entity[int(ident.net_id)] = e.id
                self._entity_to_net[e.id] = int(ident.net_id)
                if ident.net_id >= self._next_net_id:
                    self._next_net_id = int(ident.net_id) + 1

    def _alloc_net_id(self) -> int:
        nid = int(self._next_net_id)
        self._next_net_id += 1
        return nid

    def _find_entity_by_net_id(self, net_id: int) -> Optional[Entity]:
        eid = self._net_to_entity.get(int(net_id))
        if eid is not None:
            scene = self._get_scene()
            if scene:
                ent = scene.get_entity(eid)
                if ent is not None:
                    return ent
        scene = self._get_scene()
        if scene is None:
            return None
        for e in scene.get_all_entities():
            ident = e.get_component_by_name("NetworkIdentity")
            if ident is not None and int(ident.net_id) == int(net_id):
                self._net_to_entity[int(net_id)] = e.id
                self._entity_to_net[e.id] = int(net_id)
                return e
        return None

    def spawn_entity(self, entity: Entity, owner_id: int | None = None) -> int:
        scene = self._get_scene()
        if scene is None or entity is None:
            return -1
        if entity._scene is None:
            try:
                scene.add_entity(entity)
            except Exception:
                pass
        ident = entity.get_component_by_name("NetworkIdentity")
        if ident is None:
            from core.components.network.network_identity import NetworkIdentity
            ident = NetworkIdentity()
            entity.add_component(ident)
        if ident.net_id >= 0:
            return int(ident.net_id)
        if self.is_server:
            nid = self._alloc_net_id()
            ident.net_id = nid
            ident.network_id = nid
            ident.owner_id = int(owner_id) if owner_id is not None else int(self.local_id)
            ident._refresh_is_local()
            self._net_to_entity[nid] = entity.id
            self._entity_to_net[entity.id] = nid
            payload = {"net_id": nid, "owner_id": ident.owner_id, "prefab_id": ident.prefab_id, "entity": entity.serialize()}
            try:
                from core.network.transport import get_transport
                from core.network.protocol import MessageType
                get_transport().broadcast(MessageType.NET_SPAWN, payload)
            except Exception:
                pass
            return nid
        else:
            if not self.is_connected:
                return -1
            payload = {"prefab_id": ident.prefab_id, "entity": entity.serialize(), "requester": int(self.local_id)}
            try:
                from core.network.transport import get_transport
                from core.network.protocol import MessageType
                get_transport().send(MessageType.NET_SPAWN_REQUEST, payload)
            except Exception:
                pass
            return -1

    def spawn_prefab(self, prefab_path: str, pos=None, rot=None, owner_id: int | None = None):
        scene = self._get_scene()
        if scene is None:
            return None
        prefab_path = str(prefab_path).replace("\\", "/")
        if not self.is_server and self.is_connected:
            ident_tmp = None
            entity_tmp = None
            payload_entity = None
            if prefab_path.endswith(".zpep"):
                try:
                    from core.ecs.prefab import Prefab
                    from core.ecs.ecs import ComponentRegistry as CR
                    prefab = Prefab.load(prefab_path)
                    if prefab is not None:
                        spawned = prefab.instantiate(scene, CR)
                        if spawned:
                            entity_tmp = spawned[0]
                            for extra in spawned[1:]:
                                try:
                                    scene.remove_entity(extra.id)
                                except Exception:
                                    pass
                            payload_entity = entity_tmp.serialize()
                            scene.remove_entity(entity_tmp.id)
                except Exception:
                    pass
            if payload_entity is None:
                try:
                    from core.ecs.ecs import Entity as _E
                    from core.components.transform import Transform as _T
                    from core.maths.math3d import Vec3 as _V, Quat as _Q
                    tmp = _E("SpawnReq")
                    t = _T()
                    if pos is not None:
                        if isinstance(pos, _V):
                            t.local_position = pos
                        elif isinstance(pos, (list, tuple)):
                            t.local_position = _V(float(pos[0]), float(pos[1]), float(pos[2]))
                    payload_entity = tmp.serialize()
                except Exception:
                    payload_entity = {"id": "tmp"}
            try:
                from core.network.transport import get_transport
                from core.network.protocol import MessageType
                get_transport().send(MessageType.NET_SPAWN_REQUEST, {"prefab_id": prefab_path, "entity": payload_entity, "requester": int(self.local_id), "pos": list(pos.to_list()) if hasattr(pos, "to_list") else pos, "rot": list(rot.to_list()) if hasattr(rot, "to_list") else rot})
            except Exception:
                pass
            return None
        entity: Optional[Entity] = None
        if prefab_path.endswith(".zpep"):
            try:
                from core.ecs.prefab import Prefab
                from core.ecs.ecs import ComponentRegistry
                prefab = Prefab.load(prefab_path)
                if prefab is not None:
                    spawned = prefab.instantiate(scene, ComponentRegistry)
                    if spawned:
                        entity = spawned[0]
                        for extra in spawned[1:]:
                            pass
                else:
                    entity = scene.create_entity("Spawned")
                    from core.components.transform import Transform
                    from core.maths.math3d import Vec3, Quat
                    t = Transform()
                    if pos is not None:
                        if isinstance(pos, Vec3):
                            t.local_position = pos
                        else:
                            t.local_position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                    if rot is not None and hasattr(t, "local_rotation"):
                        if isinstance(rot, Quat):
                            t.local_rotation = rot
                        else:
                            try:
                                t.local_rotation = Quat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
                            except Exception:
                                pass
                    entity.add_component(t)
            except Exception:
                entity = scene.create_entity("Spawned")
                try:
                    from core.components.transform import Transform
                    from core.maths.math3d import Vec3, Quat
                    t = Transform()
                    if pos is not None:
                        if isinstance(pos, Vec3):
                            t.local_position = pos
                        else:
                            t.local_position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                    entity.add_component(t)
                except Exception:
                    pass
        else:
            entity = scene.create_entity("Spawned")
            try:
                from core.components.transform import Transform
                from core.maths.math3d import Vec3
                t = Transform()
                if pos is not None:
                    if isinstance(pos, Vec3):
                        t.local_position = pos
                    else:
                        t.local_position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                entity.add_component(t)
            except Exception:
                pass
        if entity is None:
            return None
        ident = entity.get_component_by_name("NetworkIdentity")
        if ident is None:
            from core.components.network.network_identity import NetworkIdentity
            ident = NetworkIdentity()
            ident.prefab_id = prefab_path
            entity.add_component(ident)
        else:
            ident.prefab_id = prefab_path
        if pos is not None and entity.transform is not None:
            try:
                from core.maths.math3d import Vec3, Quat
                tr = entity.transform
                if isinstance(pos, Vec3):
                    tr.local_position = pos
                elif isinstance(pos, (list, tuple)):
                    tr.local_position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                if rot is not None:
                    if isinstance(rot, Quat):
                        tr.local_rotation = rot
                    elif isinstance(rot, (list, tuple)):
                        tr.local_rotation = Quat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
            except Exception:
                pass
        nid = self.spawn_entity(entity, owner_id=owner_id)
        if nid < 0 and not self.is_server:
            try:
                scene.remove_entity(entity.id)
            except Exception:
                pass
            return None
        return entity

    def despawn(self, net_id: int):
        ident_nid = int(net_id)
        ent = self._find_entity_by_net_id(ident_nid)
        if ent is None:
            return
        scene = self._get_scene()
        ident = ent.get_component_by_name("NetworkIdentity")
        if ident is not None and not self.is_server and not ident.is_owner:
            return
        if self.is_server:
            try:
                from core.network.transport import get_transport
                from core.network.protocol import MessageType
                get_transport().broadcast(MessageType.NET_DESPAWN, {"net_id": ident_nid})
            except Exception:
                pass
        elif self.is_client:
            try:
                from core.network.transport import get_transport
                from core.network.protocol import MessageType
                get_transport().send(MessageType.NET_DESPAWN, {"net_id": ident_nid})
                return
            except Exception:
                pass
        self._net_to_entity.pop(ident_nid, None)
        self._entity_to_net.pop(ent.id, None)
        if scene:
            try:
                scene.remove_entity(ent.id)
            except Exception:
                pass

    def despawn_entity(self, entity: Entity):
        ident = entity.get_component_by_name("NetworkIdentity")
        if ident is None or ident.net_id < 0:
            scene = self._get_scene()
            if scene:
                try:
                    scene.remove_entity(entity.id)
                except Exception:
                    pass
            return
        self.despawn(int(ident.net_id))

    def send_rpc(self, net_id: int, method: str, args: dict | None = None, target: int | None = None):
        payload = {"net_id": int(net_id), "method": str(method), "args": dict(args) if args else {}, "t": time.time()}
        if target is not None:
            payload["target"] = int(target)
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            if target is not None and self.is_server:
                get_transport().send_to(int(target), MessageType.NET_RPC, payload)
            elif self.is_server or target is None:
                get_transport().broadcast(MessageType.NET_RPC, payload)
            else:
                get_transport().send(MessageType.NET_RPC, payload)
        except Exception:
            pass

    def _handle_spawn(self, data: dict):
        if self.is_server and int(data.get("_sender", -1)) != 0 and int(data.get("_sender", -1)) != self.local_id:
            return
        scene = self._get_scene()
        if scene is None:
            return
        net_id = int(data.get("net_id", -1))
        if net_id < 0:
            return
        if net_id in self._net_to_entity:
            return
        if net_id >= 1000000:
            return
        owner_id = int(data.get("owner_id", data.get("_sender", -1)))
        prefab_id = str(data.get("prefab_id", ""))[:256]
        entity_data = data.get("entity")
        ent: Optional[Entity] = None
        if entity_data:
            try:
                from core.ecs.ecs import ComponentRegistry
                ent = Entity.deserialize(dict(entity_data), ComponentRegistry)
                if ent.id in scene._entities:
                    return
                scene.add_entity(ent)
            except Exception:
                return
        else:
            return
        ident = ent.get_component_by_name("NetworkIdentity")
        if ident is None:
            from core.components.network.network_identity import NetworkIdentity
            ident = NetworkIdentity()
            ent.add_component(ident)
        ident.net_id = net_id
        ident.network_id = net_id
        ident.owner_id = owner_id
        ident.prefab_id = prefab_id
        ident._refresh_is_local()
        self._net_to_entity[net_id] = ent.id
        self._entity_to_net[ent.id] = net_id
        if net_id >= self._next_net_id:
            self._next_net_id = net_id + 1

    def _handle_despawn(self, data: dict):
        sender = int(data.get("_sender", -1))
        net_id = int(data.get("net_id", -1))
        ent = self._find_entity_by_net_id(net_id)
        if ent is None:
            return
        if not self.is_server and sender != 0 and sender != -1:
            ident = ent.get_component_by_name("NetworkIdentity")
            if ident is not None and ident.owner_id != sender:
                return
        scene = self._get_scene()
        self._net_to_entity.pop(net_id, None)
        self._entity_to_net.pop(ent.id, None)
        if scene:
            try:
                scene.remove_entity(ent.id)
            except Exception:
                pass

    def _handle_spawn_request(self, data: dict):
        if not self.is_server:
            return
        requester = int(data.get("requester", data.get("_sender", -1)))
        if requester < 0:
            return
        prefab_id = str(data.get("prefab_id", ""))[:256]
        entity_data = data.get("entity")
        scene = self._get_scene()
        if scene is None:
            return
        if not prefab_id and not entity_data:
            return
        ent: Optional[Entity] = None
        if entity_data:
            try:
                from core.ecs.ecs import ComponentRegistry
                ent = Entity.deserialize(dict(entity_data), ComponentRegistry)
                if ent.id in scene._entities:
                    ent._id = str(scene._generate_id()) if hasattr(scene, "_generate_id") else ent.id
                scene.add_entity(ent)
                try:
                    from core.maths.math3d import Vec3, Quat
                    pos = data.get("pos")
                    rot = data.get("rot")
                    if pos is not None and ent.transform is not None:
                        if isinstance(pos, (list, tuple)) and len(pos) == 3:
                            ent.transform.local_position = Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
                    if rot is not None and ent.transform is not None:
                        if isinstance(rot, (list, tuple)) and len(rot) == 4:
                            ent.transform.local_rotation = Quat(float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3]))
                except Exception:
                    pass
            except Exception:
                return
        else:
            ent = self.spawn_prefab(prefab_id, owner_id=requester)
            if ent is None:
                return
            return
        ident = ent.get_component_by_name("NetworkIdentity")
        if ident is None:
            from core.components.network.network_identity import NetworkIdentity
            ident = NetworkIdentity()
            ent.add_component(ident)
        nid = self._alloc_net_id()
        ident.net_id = nid
        ident.network_id = nid
        ident.owner_id = requester
        ident.prefab_id = prefab_id
        ident._refresh_is_local()
        self._net_to_entity[nid] = ent.id
        self._entity_to_net[ent.id] = nid
        payload = {"net_id": nid, "owner_id": requester, "prefab_id": prefab_id, "entity": ent.serialize()}
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            get_transport().broadcast(MessageType.NET_SPAWN, payload)
        except Exception:
            pass

    def _handle_rpc(self, data: dict):
        net_id = int(data.get("net_id", -1))
        method = str(data.get("method", ""))[:64]
        if not method:
            return
        args = data.get("args", {})
        if not isinstance(args, dict):
            return
        target = data.get("target", None)
        if target is not None:
            try:
                from core.network.transport import get_transport
                if int(target) != int(get_transport().local_id):
                    return
            except Exception:
                pass
        rpc_type = str(data.get("rpc_type", ""))
        if rpc_type == "server" and not self.is_server:
            ent = self._find_entity_by_net_id(net_id)
            if ent is None:
                return
            from core.network.rpc import invoke_rpc
            invoke_rpc(ent, method, args, sender=int(data.get("_sender", -1)))
            return
        if rpc_type == "client" and self.is_server:
            return
        ent = self._find_entity_by_net_id(net_id)
        if ent is None:
            return
        from core.network.rpc import invoke_rpc
        invoke_rpc(ent, method, args, sender=int(data.get("_sender", -1)))

    def _handle_owner_change(self, data: dict):
        net_id = int(data.get("net_id", -1))
        ent = self._find_entity_by_net_id(net_id)
        if ent is None:
            return
        ident = ent.get_component_by_name("NetworkIdentity")
        if ident is None:
            return
        sender = int(data.get("_sender", -1))
        if "requester" in data:
            if not self.is_server:
                return
            requester = int(data.get("requester", -1))
            ident.apply_owner_change(requester)
            try:
                from core.network.transport import get_transport
                from core.network.protocol import MessageType
                get_transport().broadcast(MessageType.NET_OWNER_CHANGE, {"net_id": net_id, "owner_id": requester})
            except Exception:
                pass
        else:
            if not self.is_server and sender != 0:
                return
            new_owner = int(data.get("owner_id", -1))
            ident.apply_owner_change(new_owner)

    def _sync_new_peer(self, peer_id: int):
        if not self.is_server:
            return
        scene = self._get_scene()
        if scene is None:
            return
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            t = get_transport()
            for nid, eid in list(self._net_to_entity.items()):
                ent = scene.get_entity(eid)
                if ent is None:
                    continue
                ident = ent.get_component_by_name("NetworkIdentity")
                owner = ident.owner_id if ident else -1
                payload = {"net_id": nid, "owner_id": owner, "prefab_id": ident.prefab_id if ident else "", "entity": ent.serialize()}
                t.send_to(peer_id, MessageType.NET_SPAWN, payload)
        except Exception:
            pass

    def on_awake(self):
        NetworkManager._instance = self
        self._register_existing()

    def on_destroy(self):
        if NetworkManager._instance is self:
            NetworkManager._instance = None

    def on_update(self, dt: float):
        if NetworkManager._instance is None:
            NetworkManager._instance = self
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            t = get_transport()
            msgs = t.poll()
            for msg_type, data in msgs:
                if msg_type == MessageType.NET_SPAWN:
                    self._handle_spawn(data)
                elif msg_type == MessageType.NET_DESPAWN:
                    self._handle_despawn(data)
                elif msg_type == MessageType.NET_SPAWN_REQUEST:
                    self._handle_spawn_request(data)
                elif msg_type == MessageType.NET_TRANSFORM:
                    nid = int(data.get("net_id", -1))
                    ent = self._find_entity_by_net_id(nid)
                    if ent:
                        comp = ent.get_component_by_name("NetworkTransform")
                        if comp:
                            comp.apply_snapshot(data)
                elif msg_type == MessageType.NET_RIGIDBODY:
                    nid = int(data.get("net_id", -1))
                    ent = self._find_entity_by_net_id(nid)
                    if ent:
                        comp = ent.get_component_by_name("NetworkRigidbody")
                        if comp:
                            comp.apply_snapshot(data)
                elif msg_type == MessageType.NET_ANIMATOR:
                    nid = int(data.get("net_id", -1))
                    ent = self._find_entity_by_net_id(nid)
                    if ent:
                        comp = ent.get_component_by_name("NetworkAnimator")
                        if comp:
                            comp.apply_snapshot(data)
                elif msg_type == MessageType.NET_RPC:
                    if "rpc" in data:
                        try:
                            from core.engine.engine import Engine
                            eng = Engine.instance()
                            if eng:
                                plug = eng.plugin_manager.get_plugin("NetworkPlugin") if hasattr(eng, "plugin_manager") else None
                                if plug and hasattr(plug, "_dispatch_rpc"):
                                    plug._dispatch_rpc(data)
                                    continue
                        except Exception:
                            pass
                    self._handle_rpc(data)
                elif msg_type == MessageType.NET_OWNER_CHANGE:
                    self._handle_owner_change(data)
                elif msg_type == MessageType.NET_VARIABLES:
                    nid = int(data.get("net_id", -1))
                    ent = self._find_entity_by_net_id(nid)
                    if ent:
                        comp = ent.get_component_by_name("NetworkVariables")
                        if comp:
                            comp.apply_snapshot(data)
                        else:
                            vars_data = data.get("vars", data.get("data", {}))
                            if isinstance(vars_data, dict):
                                for comp_key, props in vars_data.items():
                                    c = ent.get_component_by_name(comp_key)
                                    if c and isinstance(props, dict):
                                        for k, v in props.items():
                                            try:
                                                setattr(c, k, v)
                                            except Exception:
                                                pass
                elif msg_type == MessageType.NET_KICK:
                    self.disconnect()
                elif msg_type == MessageType.PEER_JOINED:
                    pid = int(data.get("id", -1))
                    if self.is_server and pid >= 0:
                        self._sync_new_peer(pid)
                        if self.auto_spawn_player and self.player_prefab:
                            pos = None
                            try:
                                from core.maths.math3d import Vec3
                                import random
                                pos = Vec3(float(random.uniform(-2, 2)), 0.0, float(random.uniform(-2, 2)))
                            except Exception:
                                pos = None
                            self.spawn_prefab(self.player_prefab, pos=pos, owner_id=pid)
                elif msg_type == MessageType.LEAVE:
                    nid = int(data.get("id", -1))
                    to_remove = []
                    for net_id, eid in list(self._net_to_entity.items()):
                        ent = self._find_entity_by_net_id(net_id)
                        if ent:
                            ident = ent.get_component_by_name("NetworkIdentity")
                            if ident and ident.owner_id == nid:
                                to_remove.append(net_id)
                    for net_id in to_remove:
                        self.despawn(net_id)
                elif msg_type == MessageType.JOINED:
                    pass
        except Exception:
            pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "port": self.port,
            "max_players": self.max_players,
            "server_name": self.server_name,
            "host_address": self.host_address,
            "player_prefab": self.player_prefab,
            "auto_spawn_player": self.auto_spawn_player,
            "tick_rate": self.tick_rate,
            "show_debug": self.show_debug,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkManager:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.port = int(data.get("port", 7777))
        inst.max_players = int(data.get("max_players", 16))
        inst.server_name = str(data.get("server_name", "Zarin Server"))
        inst.host_address = str(data.get("host_address", "127.0.0.1"))
        inst.player_prefab = str(data.get("player_prefab", ""))
        inst.auto_spawn_player = bool(data.get("auto_spawn_player", True))
        inst.tick_rate = float(data.get("tick_rate", 20.0))
        inst.show_debug = bool(data.get("show_debug", False))
        return inst
