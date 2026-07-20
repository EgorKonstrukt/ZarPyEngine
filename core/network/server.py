# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import asyncio
import struct
import random
import time
import msgpack
from typing import Optional
from core.foundation.logger import Logger
from core.network.protocol import MessageType, make_msg, parse_msg, FRAME_HEADER_SIZE


class CollabClientInfo:
    __slots__ = ("reader", "writer", "peer_id", "name", "color", "addr")

    def __init__(self, reader, writer, peer_id: str, name: str, color: list[float], addr: str):
        self.reader = reader
        self.writer = writer
        self.peer_id = peer_id
        self.name = name
        self.color = color
        self.addr = addr


_COLORS = [
    [0.2, 0.6, 1.0],
    [1.0, 0.3, 0.3],
    [0.3, 1.0, 0.3],
    [1.0, 1.0, 0.2],
    [1.0, 0.5, 0.0],
    [0.8, 0.2, 1.0],
    [0.2, 1.0, 1.0],
    [1.0, 0.6, 0.6],
]


class CollabServer:
    def __init__(self, host: str = "0.0.0.0", port: int = 9876):
        self._host = host
        self._port = port
        self._clients: dict[str, CollabClientInfo] = {}
        self._server: Optional[asyncio.AbstractServer] = None
        self._color_idx = 0
        self._running = False

    @property
    def running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        return len(self._clients)

    async def start(self):
        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)
        self._running = True
        Logger.info(f"CollabServer listening on {self._host}:{self._port}")

    async def stop(self):
        self._running = False
        for c in list(self._clients.values()):
            try:
                c.writer.close()
            except Exception:
                pass
        self._clients.clear()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        Logger.info("CollabServer stopped")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername", ("unknown", 0))
        addr_str = f"{addr[0]}:{addr[1]}"
        peer_id = None
        try:
            header = await reader.readexactly(FRAME_HEADER_SIZE)
            payload_len = struct.unpack(">I", header)[0]
            payload = await reader.readexactly(payload_len)
            msg_type, data = parse_msg(payload)
            if msg_type != MessageType.JOIN:
                writer.close()
                return
            name = data.get("name", f"User_{len(self._clients)+1}")
            peer_id = f"peer_{int(time.time()*1000)}_{random.randint(1000,9999)}"
            color = _COLORS[self._color_idx % len(_COLORS)]
            self._color_idx += 1
            info = CollabClientInfo(reader, writer, peer_id, name, color, addr_str)
            self._clients[peer_id] = info
            join_data = {"id": peer_id, "name": name, "color": color}
            await self._broadcast_and_drain(MessageType.PEER_JOINED, join_data, exclude=peer_id)
            current_peers = [
                {"id": c.peer_id, "name": c.name, "color": c.color}
                for c in self._clients.values()
            ]
            stored_scene = getattr(self, "_scene_data", None) or data.get("scene")
            init_data = {
                "your_id": peer_id,
                "your_color": color,
                "peers": current_peers,
                "scene": stored_scene,
            }
            writer.write(make_msg(MessageType.JOINED, init_data))
            await writer.drain()
            if getattr(self, "_scene_data", None) is None and data.get("scene"):
                self._scene_data = data.get("scene")
            Logger.info(f"Collab peer joined: {name} ({addr_str}) as {peer_id}")
            await self._client_loop(info)
        except asyncio.IncompleteReadError:
            pass
        except ConnectionResetError:
            pass
        except Exception as e:
            Logger.error(f"Collab client error: {e}")
        finally:
            if peer_id:
                if peer_id in self._clients:
                    del self._clients[peer_id]
                self._broadcast(MessageType.LEAVE, {"id": peer_id}, exclude=peer_id)
            Logger.info(f"Collab peer left: {peer_id}")

    async def _client_loop(self, info: CollabClientInfo):
        reader = info.reader
        while True:
            try:
                header = await reader.readexactly(FRAME_HEADER_SIZE)
                payload_len = struct.unpack(">I", header)[0]
                payload = await reader.readexactly(payload_len)
                msg_type, data = parse_msg(payload)
                await self._route_message(info, msg_type, data)
            except ValueError:
                Logger.warning(f"Malformed message from {info.peer_id}, skipping")
                continue

    async def _route_message(self, info: CollabClientInfo, msg_type: int, data: dict):
        if msg_type == MessageType.CURSOR:
            data["id"] = info.peer_id
            self._broadcast(MessageType.CURSOR_UPDATE, data, exclude=info.peer_id)
        elif msg_type == MessageType.CAMERA:
            data["id"] = info.peer_id
            self._broadcast(MessageType.CAMERA_UPDATE, data, exclude=info.peer_id)
        elif msg_type == MessageType.ENTITY_CREATE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ENTITY_CREATED, data, exclude=info.peer_id)
        elif msg_type == MessageType.ENTITY_DELETE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ENTITY_DELETED, data, exclude=info.peer_id)
        elif msg_type == MessageType.TRANSFORM_UPDATE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.TRANSFORM_UPDATED, data, exclude=info.peer_id)
        elif msg_type == MessageType.SELECTION:
            data["id"] = info.peer_id
            self._broadcast(MessageType.SELECTION_UPDATE, data, exclude=info.peer_id)
        elif msg_type == MessageType.COMPONENT_UPDATE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.COMPONENT_UPDATED, data, exclude=info.peer_id)
        elif msg_type == MessageType.COMPONENT_ADD:
            data["id"] = info.peer_id
            self._broadcast(MessageType.COMPONENT_ADDED, data, exclude=info.peer_id)
        elif msg_type == MessageType.COMPONENT_REMOVE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.COMPONENT_REMOVED, data, exclude=info.peer_id)
        elif msg_type == MessageType.PING:
            timestamp = data.get("t", 0)
            info.writer.write(make_msg(MessageType.PONG, {"t": timestamp}))
            await info.writer.drain()
        elif msg_type == MessageType.PLAY_MODE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.PLAY_MODE, data, exclude=info.peer_id)
        elif msg_type == MessageType.GIZMO_STATE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.GIZMO_STATE_UPDATE, data, exclude=info.peer_id)
        elif msg_type == MessageType.SCENE_SNAPSHOT_REQ:
            scene = getattr(self, "_scene_data", None)
            if scene:
                snap_data = {"scene": scene, "requesting_id": info.peer_id}
                info.writer.write(make_msg(MessageType.SCENE_SNAPSHOT, snap_data))
                await info.writer.drain()
        elif msg_type == MessageType.ASSET_LIST_REQ:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ASSET_LIST_REQ, data, exclude=info.peer_id)
        elif msg_type == MessageType.ASSET_LIST:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ASSET_LIST, data, exclude=info.peer_id)
        elif msg_type == MessageType.ASSET_SYNC:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ASSET_SYNC, data, exclude=info.peer_id)
        elif msg_type == MessageType.ASSET_WATCH:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ASSET_WATCH, data, exclude=info.peer_id)
        elif msg_type == MessageType.ASSET_DELETE:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ASSET_DELETE, data, exclude=info.peer_id)
        elif msg_type == MessageType.ASSET_REQUEST:
            data["id"] = info.peer_id
            self._broadcast(MessageType.ASSET_REQUEST, data, exclude=info.peer_id)
        else:
            data["id"] = info.peer_id
            self._broadcast(msg_type, data, exclude=info.peer_id)

    def _broadcast(self, msg_type: int, data: dict, exclude: Optional[str] = None):
        msg = make_msg(msg_type, data)
        for pid, cinfo in list(self._clients.items()):
            if pid == exclude:
                continue
            try:
                cinfo.writer.write(msg)
            except Exception:
                pass

    async def _broadcast_and_drain(self, msg_type: int, data: dict, exclude: Optional[str] = None):
        self._broadcast(msg_type, data, exclude)
        for cinfo in self._clients.values():
            try:
                await cinfo.writer.drain()
            except Exception:
                pass

    def update_scene_data(self, scene: dict):
        self._scene_data = scene
