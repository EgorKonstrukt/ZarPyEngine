# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Callable, Optional, Any
from core.foundation.plugin_manager import PluginBase
from core.foundation.logger import Logger
class NetworkMessage:
    def __init__(self, msg_type: str, payload: dict, sender_id: int = -1):
        self.msg_type = msg_type
        self.payload = payload
        self.sender_id = sender_id
class NetworkPlugin(PluginBase):
    NAME = "NetworkPlugin"
    VERSION = "0.1.0"
    DESCRIPTION = "Network stub for future multiplayer integration."
    SYSTEM = True
    def __init__(self):
        super().__init__()
        self._connected: bool = False
        self._server_mode: bool = False
        self._local_id: int = -1
        self._peers: dict[int, str] = {}
        self._handlers: dict[str, list[Callable]] = {}
        self._pending_messages: list[NetworkMessage] = []
    @property
    def is_connected(self) -> bool: return self._connected
    @property
    def is_server(self) -> bool: return self._server_mode
    @property
    def local_id(self) -> int: return self._local_id
    @property
    def peer_count(self) -> int: return len(self._peers)
    def host(self, port: int = 7777, max_players: int = 16):
        Logger.warning(f"[NetworkPlugin] host() called - stub only. Port={port}, MaxPlayers={max_players}")
        self._server_mode = True
        self._local_id = 0
        self._connected = True
    def connect(self, host: str, port: int = 7777):
        Logger.warning(f"[NetworkPlugin] connect() called - stub only. Host={host}:{port}")
        self._server_mode = False
        self._local_id = -1
        self._connected = False
    def disconnect(self):
        Logger.warning("[NetworkPlugin] disconnect() called - stub only.")
        self._connected = False
        self._peers.clear()
        self._local_id = -1
    def send(self, msg_type: str, payload: dict, target_id: int = -1):
        Logger.debug(f"[NetworkPlugin] send() stub: type={msg_type} target={target_id}")
    def broadcast(self, msg_type: str, payload: dict):
        Logger.debug(f"[NetworkPlugin] broadcast() stub: type={msg_type}")
    def on_message(self, msg_type: str, handler: Callable[[NetworkMessage], None]):
        self._handlers.setdefault(msg_type, []).append(handler)
    def poll(self):
        for msg in self._pending_messages:
            for h in self._handlers.get(msg.msg_type, []):
                try: h(msg)
                except Exception as e: Logger.error(f"Network handler error: {e}", e)
        self._pending_messages.clear()
    def initialize(self, engine):
        super().initialize(engine)
        Logger.info("[NetworkPlugin] Network stub initialized. Ready for real transport implementation.")
    def shutdown(self):
        self.disconnect()
        Logger.info("[NetworkPlugin] Network stub shutdown.")