# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from enum import Enum
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


class AuthorityMode(Enum):
    SERVER = "server"
    OWNER = "owner"
    SHARED = "shared"


@ComponentRegistry.register
class NetworkIdentity(Component):
    _icon = "NetworkIdentity.png"
    _gizmo_icon_color = (255, 100, 180)
    _gizmo_icon_label = "N"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("net_id", "Net ID", FieldType.INT, min_val=-1, max_val=999999, readonly=True),
            InspectorField("owner_id", "Owner ID", FieldType.INT, min_val=-1, max_val=999999, readonly=True),
            InspectorField("prefab_id", "Prefab ID", FieldType.STRING),
            InspectorField("authority", "Authority", FieldType.ENUM, enum_class=AuthorityMode),
            InspectorField("is_local_player", "Is Local Player", FieldType.BOOL),
            InspectorField("dont_destroy_on_load", "Dont Destroy", FieldType.BOOL),
        ]

    def __init__(self):
        super().__init__()
        self.net_id: int = -1
        self.owner_id: int = -1
        self.prefab_id: str = ""
        self.authority: AuthorityMode = AuthorityMode.SERVER
        self.is_local_player: bool = False
        self.dont_destroy_on_load: bool = False
        self.is_server: bool = False
        self.network_id: int = -1

    @property
    def is_spawned(self) -> bool:
        return self.net_id >= 0

    @property
    def is_owner(self) -> bool:
        try:
            from core.network.transport import get_transport
            return get_transport().local_id == self.owner_id
        except Exception:
            return False

    @property
    def has_authority(self) -> bool:
        if self.authority == AuthorityMode.SHARED:
            return True
        if self.authority == AuthorityMode.OWNER:
            return self.is_owner
        if self.authority == AuthorityMode.SERVER:
            try:
                from core.network.transport import get_transport
                return get_transport().is_server
            except Exception:
                return False
        return False

    def transfer_ownership(self, new_owner_id: int):
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            t = get_transport()
            if not t.is_server:
                return
            self.owner_id = int(new_owner_id)
            self._refresh_is_local()
            if t.is_connected:
                t.broadcast(MessageType.NET_OWNER_CHANGE, {"net_id": self.net_id, "owner_id": self.owner_id})
        except Exception:
            pass

    def request_ownership(self):
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            t = get_transport()
            if t.is_connected and not t.is_server:
                t.send(MessageType.NET_OWNER_CHANGE, {"net_id": self.net_id, "requester": t.local_id})
        except Exception:
            pass

    def apply_owner_change(self, new_owner_id: int):
        self.owner_id = int(new_owner_id)
        self._refresh_is_local()

    def _refresh_is_local(self):
        try:
            from core.network.transport import get_transport
            self.is_local_player = (self.owner_id == get_transport().local_id)
        except Exception:
            pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "net_id": self.net_id,
            "owner_id": self.owner_id,
            "prefab_id": self.prefab_id,
            "authority": self.authority.value if isinstance(self.authority, Enum) else str(self.authority),
            "is_local_player": self.is_local_player,
            "dont_destroy_on_load": self.dont_destroy_on_load,
            "network_id": self.net_id,
            "is_server": self.is_server,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkIdentity:
        ni = cls()
        ni.enabled = data.get("enabled", True)
        if "net_id" in data:
            ni.net_id = int(data.get("net_id", -1))
        else:
            ni.net_id = int(data.get("network_id", -1))
        ni.owner_id = int(data.get("owner_id", -1))
        ni.prefab_id = str(data.get("prefab_id", ""))
        raw_auth = data.get("authority", "server")
        try:
            ni.authority = AuthorityMode(raw_auth)
        except Exception:
            ni.authority = AuthorityMode.SERVER
        ni.is_local_player = bool(data.get("is_local_player", False))
        ni.dont_destroy_on_load = bool(data.get("dont_destroy_on_load", False))
        ni.is_server = bool(data.get("is_server", False))
        ni.network_id = ni.net_id
        return ni
