# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from typing import Callable
from core.foundation.plugin_manager import PluginBase
from core.foundation.logger import Logger


class NetworkMessage:
    def __init__(self, msg_type: str, payload: dict, sender_id: int = -1):
        self.msg_type = msg_type
        self.payload = payload
        self.sender_id = sender_id


class NetworkPlugin(PluginBase):
    NAME = "NetworkPlugin"
    VERSION = "1.0.0"
    DESCRIPTION = "Multiplayer transport using GameServer/GameClient over TCP."
    SYSTEM = True

    def __init__(self):
        super().__init__()
        self._handlers: dict[str, list[Callable]] = {}
        self._pending_messages: list[NetworkMessage] = []
        self._subscribed = False

    @property
    def is_connected(self) -> bool:
        try:
            from core.network.transport import get_transport
            return get_transport().is_connected
        except Exception:
            return False

    @property
    def is_server(self) -> bool:
        try:
            from core.network.transport import get_transport
            return get_transport().is_server
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

    def host(self, port: int = 7777, max_players: int = 16):
        try:
            from core.network.transport import get_transport
            ok = get_transport().host("0.0.0.0", int(port), max_players=int(max_players))
            if ok:
                Logger.info(f"[NetworkPlugin] Hosting on port {port} max_players={max_players}")
            return ok
        except Exception as e:
            Logger.error(f"[NetworkPlugin] host failed: {e}")
            return False

    def connect(self, host: str, port: int = 7777):
        try:
            from core.network.transport import get_transport
            ok = get_transport().connect(str(host), int(port), "Player")
            if ok:
                Logger.info(f"[NetworkPlugin] Connected to {host}:{port}")
            return ok
        except Exception as e:
            Logger.error(f"[NetworkPlugin] connect failed: {e}")
            return False

    def disconnect(self):
        try:
            from core.network.transport import get_transport
            get_transport().disconnect()
            Logger.info("[NetworkPlugin] Disconnected")
        except Exception:
            pass

    def send(self, msg_type: str, payload: dict, target_id: int = -1):
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            t = get_transport()
            data = {"rpc": str(msg_type), "payload": dict(payload) if isinstance(payload, dict) else {"value": payload}}
            if int(target_id) >= 0 and t.is_server:
                t.send_to(int(target_id), MessageType.NET_RPC, data)
            else:
                t.broadcast(MessageType.NET_RPC, data)
        except Exception as e:
            Logger.error(f"[NetworkPlugin] send error: {e}")

    def broadcast(self, msg_type: str, payload: dict):
        try:
            from core.network.transport import get_transport
            from core.network.protocol import MessageType
            get_transport().broadcast(MessageType.NET_RPC, {"rpc": str(msg_type), "payload": dict(payload) if isinstance(payload, dict) else {"value": payload}})
        except Exception as e:
            Logger.error(f"[NetworkPlugin] broadcast error: {e}")

    def on_message(self, msg_type: str, handler: Callable[[NetworkMessage], None]):
        self._handlers.setdefault(str(msg_type), []).append(handler)

    def _dispatch_rpc(self, data: dict):
        try:
            from core.foundation.logger import Logger as _L
            rpc = str(data.get("rpc", ""))
            if not rpc:
                return False
            payload = data.get("payload", data)
            sender = int(data.get("_sender", data.get("sender_id", -1)))
            msg = NetworkMessage(rpc, payload if isinstance(payload, dict) else {}, sender)
            handled = False
            for h in self._handlers.get(rpc, []):
                try:
                    h(msg)
                    handled = True
                except Exception as e:
                    _L.error(f"Network handler error: {e}")
            if not handled:
                for h in self._handlers.get("*", []):
                    try:
                        h(msg)
                    except Exception as e:
                        _L.error(f"Network handler error: {e}")
            return True
        except Exception:
            return False

    def handle_transport_messages(self, msgs: list[tuple[int, dict]]) -> list[tuple[int, dict]]:
        try:
            from core.network.protocol import MessageType
            remaining: list[tuple[int, dict]] = []
            for mtype, data in msgs:
                if mtype == MessageType.NET_RPC and isinstance(data, dict) and "rpc" in data:
                    self._dispatch_rpc(data)
                else:
                    remaining.append((mtype, data))
            for msg in list(self._pending_messages):
                for h in self._handlers.get(msg.msg_type, []):
                    try:
                        h(msg)
                    except Exception as e:
                        Logger.error(f"Network handler error: {e}")
            self._pending_messages.clear()
            return remaining
        except Exception as e:
            Logger.error(f"[NetworkPlugin] dispatch error: {e}")
            return msgs

    def poll(self):
        try:
            from core.network.transport import get_transport
            msgs = get_transport().poll()
            remaining = self.handle_transport_messages(msgs)
            if remaining:
                try:
                    from core.network.transport import get_transport as _gt
                    t = _gt()
                    with t._lock:
                        for m in remaining:
                            t._incoming.appendleft(m)
                except Exception:
                    pass
        except Exception as e:
            Logger.error(f"[NetworkPlugin] poll error: {e}")

    def initialize(self, engine):
        super().initialize(engine)
        Logger.info("[NetworkPlugin] Network transport initialized")

    def shutdown(self):
        self.disconnect()
        Logger.info("[NetworkPlugin] Network shutdown")
