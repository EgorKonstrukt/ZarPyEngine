# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import asyncio
import struct
import threading
from typing import Optional, Callable
from collections import deque
from core.foundation.logger import Logger
from core.network.protocol import MessageType, make_msg, parse_msg, FRAME_HEADER_SIZE


class CollabClient:
    def __init__(self):
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._peer_id: Optional[str] = None
        self._name: str = ""
        self._color: list[float] = [0.5, 0.5, 0.5]
        self._incoming: deque[tuple[int, dict]] = deque()
        self._lock = threading.Lock()
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None
        self._stopped = False
        self._bytes_sent = 0
        self._bytes_received = 0

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def peer_id(self) -> Optional[str]:
        return self._peer_id

    @property
    def color(self) -> list[float]:
        return self._color

    @property
    def bytes_sent(self) -> int:
        return self._bytes_sent

    @property
    def bytes_received(self) -> int:
        return self._bytes_received

    def set_on_connected(self, cb: Callable):
        self._on_connected = cb

    def set_on_disconnected(self, cb: Callable):
        self._on_disconnected = cb

    def connect(self, host: str = "127.0.0.1", port: int = 9876, name: str = "User"):
        if self._connected:
            return
        self._name = name
        self._stopped = False
        self._thread = threading.Thread(
            target=self._run_loop, args=(host, port), daemon=True
        )
        self._thread.start()

    def disconnect(self):
        self._stopped = True
        if self._writer:
            try:
                self._writer.close()
            except Exception:
                pass
        self._connected = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    def _run_loop(self, host: str, port: int):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect(host, port))
        except Exception as e:
            Logger.error(f"CollabClient connect error: {e}")
            self._connected = False
            if self._on_disconnected:
                self._on_disconnected()

    async def _connect(self, host: str, port: int):
        try:
            self._reader, self._writer = await asyncio.open_connection(host, port)
            self._connected = True
            join_msg = make_msg(MessageType.JOIN, {"name": self._name})
            self._writer.write(join_msg)
            await self._writer.drain()
            header = await self._reader.readexactly(FRAME_HEADER_SIZE)
            payload_len = struct.unpack(">I", header)[0]
            payload = await self._reader.readexactly(payload_len)
            try:
                msg_type, data = parse_msg(payload)
            except ValueError:
                Logger.error("CollabClient: invalid join response")
                return
            if msg_type == MessageType.JOINED:
                self._peer_id = data.get("your_id", "")
                self._color = data.get("your_color", [0.5, 0.5, 0.5])
                with self._lock:
                    self._incoming.append((msg_type, data))
                if self._on_connected:
                    self._on_connected()
            await self._read_loop()
        except asyncio.IncompleteReadError:
            pass
        except ConnectionResetError:
            pass
        except Exception as e:
            Logger.error(f"CollabClient connection error: {e}")
        finally:
            self._connected = False
            if not self._stopped and self._on_disconnected:
                self._on_disconnected()

    async def _read_loop(self):
        while not self._stopped and self._reader:
            try:
                header = await self._reader.readexactly(FRAME_HEADER_SIZE)
                payload_len = struct.unpack(">I", header)[0]
                payload = await self._reader.readexactly(payload_len)
                self._bytes_received += FRAME_HEADER_SIZE + payload_len
                try:
                    msg_type, data = parse_msg(payload)
                except ValueError:
                    continue
                if msg_type == MessageType.PING:
                    self._writer.write(make_msg(MessageType.PONG, {"t": data.get("t", 0)}))
                    self._bytes_sent += FRAME_HEADER_SIZE
                else:
                    with self._lock:
                        self._incoming.append((msg_type, data))
            except (asyncio.IncompleteReadError, ConnectionResetError):
                break

    def send(self, msg_type: int, data: dict):
        if not self._connected or not self._writer:
            return
        try:
            msg = make_msg(msg_type, data)
            if self._loop and not self._loop.is_closed():
                asyncio.run_coroutine_threadsafe(
                    self._async_send(msg), self._loop
                )
        except Exception as e:
            Logger.error(f"CollabClient send error: {e}")

    async def _async_send(self, msg: bytes):
        try:
            self._writer.write(msg)
            self._bytes_sent += len(msg)
            await self._writer.drain()
        except Exception:
            pass

    def poll_messages(self) -> list[tuple[int, dict]]:
        with self._lock:
            msgs = list(self._incoming)
            self._incoming.clear()
            return msgs
