# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField


@ComponentRegistry.register
class NetworkPlayer(Component):
    _icon = "NetworkPlayer.png"
    _gizmo_icon_color = (255, 220, 80)
    _gizmo_icon_label = "P"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("player_name", "Player Name", FieldType.STRING),
            InspectorField("player_id", "Player ID", FieldType.INT, min_val=-1, max_val=9999, readonly=True),
            InspectorField("team", "Team", FieldType.INT, min_val=0, max_val=8),
            InspectorField("is_ready", "Is Ready", FieldType.BOOL),
            InspectorField("score", "Score", FieldType.INT, min_val=0, max_val=999999),
        ]

    def __init__(self):
        super().__init__()
        self.player_name: str = "Player"
        self.player_id: int = -1
        self.team: int = 0
        self.is_ready: bool = False
        self.score: int = 0
        self._is_local: bool = False

    @property
    def is_local(self) -> bool:
        ident = self._get_identity()
        if ident is not None:
            return ident.is_local_player or ident.is_owner
        return self._is_local

    def _get_identity(self):
        ent = self._entity
        if ent is None:
            return None
        return ent.get_component_by_name("NetworkIdentity")

    def _get_vars(self):
        ent = self._entity
        if ent is None:
            return None
        return ent.get_component_by_name("NetworkVariables")

    def set_ready(self, ready: bool):
        self.is_ready = bool(ready)
        self._sync()

    def add_score(self, delta: int):
        self.score = int(self.score + int(delta))
        self._sync()

    def set_team(self, team: int):
        self.team = int(team)
        self._sync()

    def set_name(self, name: str):
        self.player_name = str(name)
        self._sync()

    def _sync(self):
        ident = self._get_identity()
        if ident is None or ident.net_id < 0:
            return
        try:
            vars_c = self._get_vars()
            if vars_c is not None:
                vars_c.set_vars({
                    "player_name": self.player_name,
                    "player_id": self.player_id,
                    "team": self.team,
                    "is_ready": self.is_ready,
                    "score": self.score,
                })
                return
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            t = get_transport()
            if t.is_connected:
                t.broadcast(MessageType.NET_VARIABLES, {
                    "net_id": ident.net_id,
                    "vars": {
                        "NetworkPlayer/player_name": self.player_name,
                        "NetworkPlayer/team": self.team,
                        "NetworkPlayer/is_ready": self.is_ready,
                        "NetworkPlayer/score": self.score,
                    }
                })
        except Exception:
            pass

    def apply_vars(self, data: dict):
        if "player_name" in data:
            self.player_name = str(data["player_name"])
        if "player_id" in data:
            self.player_id = int(data["player_id"])
        if "team" in data:
            self.team = int(data["team"])
        if "is_ready" in data:
            self.is_ready = bool(data["is_ready"])
        if "score" in data:
            self.score = int(data["score"])

    def on_awake(self):
        ident = self._get_identity()
        if ident is not None:
            self.player_id = int(ident.owner_id)
            try:
                from core.network.transport import get_transport
                self._is_local = (ident.owner_id == get_transport().local_id)
            except Exception:
                pass

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "player_name": self.player_name,
            "player_id": self.player_id,
            "team": self.team,
            "is_ready": self.is_ready,
            "score": self.score,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> NetworkPlayer:
        inst = cls()
        inst.enabled = data.get("enabled", True)
        inst.player_name = str(data.get("player_name", "Player"))
        inst.player_id = int(data.get("player_id", -1))
        inst.team = int(data.get("team", 0))
        inst.is_ready = bool(data.get("is_ready", False))
        inst.score = int(data.get("score", 0))
        return inst
