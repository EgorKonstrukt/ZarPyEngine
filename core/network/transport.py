# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import asyncio
import socket
import struct
import threading
import time
from collections import deque
from typing import Optional, Callable
from core.foundation.logger import Logger
from core.network.protocol import MessageType, make_msg, parse_msg, FRAME_HEADER_SIZE
from core.config.constants import MAX_MESSAGE_SIZE


class _ClientInfo:
    __slots__ = ("reader", "writer", "peer_id", "name", "addr")
    def __init__(self, reader, writer, peer_id: int, name: str, addr: str):
        self.reader = reader
        self.writer = writer
        self.peer_id = peer_id
        self.name = name
        self.addr = addr


class GameServer:
    def __init__(self, host: str, port: int, incoming: deque, lock: threading.Lock, max_players: int = 16):
        self._host = host
        self._port = port
        self._incoming = incoming
        self._lock = lock
        self._clients: dict[int, _ClientInfo] = {}
        self._clients_lock = threading.RLock()
        self._max_players = max(1, int(max_players))
        self._next_id: int = 1
        self._server: Optional[asyncio.AbstractServer] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()

    @property
    def running(self) -> bool:
        return self._running

    @property
    def client_count(self) -> int:
        with self._clients_lock:
            return len(self._clients)

    @property
    def clients(self) -> dict[int, _ClientInfo]:
        with self._clients_lock:
            return dict(self._clients)

    def set_max_players(self, n: int):
        self._max_players = max(1, int(n))

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5)
        return self._running

    def stop(self):
        self._running = False
        with self._clients_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for c in clients:
            try:
                c.writer.close()
            except Exception:
                pass
        if self._loop and self._loop.is_running():
            try:
                fut = asyncio.run_coroutine_threadsafe(self._async_stop(), self._loop)
                fut.result(timeout=2)
            except Exception:
                pass
            try:
                self._loop.call_soon_threadsafe(self._loop.stop)
            except Exception:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)

    async def _async_stop(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        Logger.info("GameServer stopped")

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
            self._ready.set()
            self._loop.run_forever()
        except Exception as e:
            Logger.error(f"GameServer error: {e}")
            self._ready.set()

    async def _async_start(self):
        self._server = await asyncio.start_server(self._handle_client, self._host, self._port)
        for s in self._server.sockets or []:
            try:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
        self._running = True
        Logger.info(f"GameServer listening on {self._host}:{self._port}")

    async def _read_frame(self, reader: asyncio.StreamReader) -> tuple[int, dict]:
        header = await reader.readexactly(FRAME_HEADER_SIZE)
        payload_len = struct.unpack(">I", header)[0]
        if payload_len == 0 or payload_len > MAX_MESSAGE_SIZE:
            raise ValueError(f"bad payload_len {payload_len}")
        payload = await reader.readexactly(payload_len)
        return parse_msg(payload)

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            sock = writer.get_extra_info("socket")
            if sock is not None:
                try:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                except Exception:
                    pass
        except Exception:
            pass
        addr = writer.get_extra_info("peername", ("unknown", 0))
        addr_str = f"{addr[0]}:{addr[1]}"
        peer_id: Optional[int] = None
        try:
            msg_type, data = await self._read_frame(reader)
            if msg_type != MessageType.JOIN:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
                return
            with self._clients_lock:
                if len(self._clients) >= self._max_players:
                    try:
                        writer.write(make_msg(MessageType.NET_KICK, {"reason": "server_full"}))
                        await writer.drain()
                    except Exception:
                        pass
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
                    return
                name = str(data.get("name", f"Player_{self._next_id}"))[:32]
                peer_id = self._next_id
                self._next_id += 1
                info = _ClientInfo(reader, writer, peer_id, name, addr_str)
                self._clients[peer_id] = info
            with self._lock:
                self._incoming.append((MessageType.PEER_JOINED, {"id": peer_id, "name": name}))
            with self._clients_lock:
                snapshot = list(self._clients.items())
            for pid, c in snapshot:
                if pid == peer_id:
                    continue
                try:
                    c.writer.write(make_msg(MessageType.PEER_JOINED, {"id": peer_id, "name": name}))
                except Exception:
                    pass
            with self._clients_lock:
                peers = [{"id": c.peer_id, "name": c.name} for c in self._clients.values()]
            writer.write(make_msg(MessageType.JOINED, {"your_id": peer_id, "peers": peers}))
            await writer.drain()
            with self._clients_lock:
                snapshot = list(self._clients.values())
            for c in snapshot:
                try:
                    await c.writer.drain()
                except Exception:
                    pass
            Logger.info(f"Game peer joined: {name} as {peer_id}")
            await self._client_loop(info)
        except (asyncio.IncompleteReadError, ConnectionResetError, ValueError):
            pass
        except Exception as e:
            Logger.error(f"GameServer client error: {e}")
        finally:
            if peer_id is not None:
                with self._clients_lock:
                    existed = peer_id in self._clients
                    if existed:
                        del self._clients[peer_id]
                if existed:
                    with self._lock:
                        self._incoming.append((MessageType.LEAVE, {"id": peer_id}))
                    msg = make_msg(MessageType.LEAVE, {"id": peer_id})
                    with self._clients_lock:
                        snapshot = list(self._clients.values())
                    for c in snapshot:
                        try:
                            c.writer.write(msg)
                        except Exception:
                            pass
                    for c in snapshot:
                        try:
                            await c.writer.drain()
                        except Exception:
                            pass
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            Logger.info(f"Game peer left: {peer_id}")

    async def _client_loop(self, info: _ClientInfo):
        reader = info.reader
        while True:
            try:
                msg_type, data = await self._read_frame(reader)
                if msg_type == MessageType.PING:
                    info.writer.write(make_msg(MessageType.PONG, {"t": data.get("t", 0)}))
                    await info.writer.drain()
                    continue
                if msg_type not in (
                    MessageType.NET_SPAWN_REQUEST,
                    MessageType.NET_TRANSFORM,
                    MessageType.NET_RIGIDBODY,
                    MessageType.NET_ANIMATOR,
                    MessageType.NET_RPC,
                    MessageType.NET_VARIABLES,
                    MessageType.NET_SNAPSHOT,
                    MessageType.NET_DESPAWN,
                    MessageType.NET_OWNER_CHANGE,
                    MessageType.NET_READY,
                ):
                    continue
                data["_sender"] = info.peer_id
                with self._lock:
                    self._incoming.append((msg_type, data))
                if msg_type in (MessageType.NET_TRANSFORM, MessageType.NET_RIGIDBODY, MessageType.NET_ANIMATOR, MessageType.NET_VARIABLES, MessageType.NET_RPC, MessageType.NET_SNAPSHOT):
                    msg = make_msg(msg_type, dict(data))
                    with self._clients_lock:
                        snapshot = list(self._clients.items())
                    for pid, c in snapshot:
                        if pid == info.peer_id:
                            continue
                        try:
                            c.writer.write(msg)
                        except Exception:
                            pass
                    for _, c in snapshot:
                        try:
                            await c.writer.drain()
                        except Exception:
                            pass
            except (asyncio.IncompleteReadError, ConnectionResetError):
                break
            except ValueError:
                continue

    def broadcast(self, msg_type: int, data: dict):
        if not self._running or not self._loop:
            return
        try:
            msg = make_msg(msg_type, dict(data))
        except Exception:
            return
        with self._clients_lock:
            snapshot = list(self._clients.values())
        for c in snapshot:
            try:
                c.writer.write(msg)
            except Exception:
                pass
        async def _drain():
            for c in snapshot:
                try:
                    await c.writer.drain()
                except Exception:
                    pass
        try:
            asyncio.run_coroutine_threadsafe(_drain(), self._loop)
        except Exception:
            pass

    def send_to(self, peer_id: int, msg_type: int, data: dict):
        if not self._running or not self._loop:
            return
        with self._clients_lock:
            c = self._clients.get(peer_id)
        if not c:
            return
        try:
            msg = make_msg(msg_type, data)
        except Exception:
            return
        try:
            c.writer.write(msg)
        except Exception:
            return
        async def _drain():
            try:
                await c.writer.drain()
            except Exception:
                pass
        try:
            asyncio.run_coroutine_threadsafe(_drain(), self._loop)
        except Exception:
            pass


class GameClient:
    def __init__(self, incoming: deque, lock: threading.Lock):
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._incoming = incoming
        self._lock = lock
        self._connected = False
        self._peer_id: int = -1
        self._name: str = ""
        self._stopped = False
        self._bytes_sent = 0
        self._bytes_received = 0
        self._on_connected: Optional[Callable] = None
        self._on_disconnected: Optional[Callable] = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def peer_id(self) -> int:
        return self._peer_id

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

    def connect(self, host: str, port: int, name: str):
        if self._connected:
            return
        self._name = name
        self._stopped = False
        self._thread = threading.Thread(target=self._run_loop, args=(host, port), daemon=True)
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
            Logger.error(f"GameClient connect error: {e}")
            self._connected = False
            if self._on_disconnected:
                try:
                    self._on_disconnected()
                except Exception:
                    pass

    async def _connect(self, host: str, port: int):
        try:
            self._reader, self._writer = await asyncio.open_connection(host, port)
            try:
                sock = self._writer.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass
            self._connected = True
            join_raw = make_msg(MessageType.JOIN, {"name": self._name})
            self._writer.write(join_raw)
            self._bytes_sent += len(join_raw)
            await self._writer.drain()
            header = await self._reader.readexactly(FRAME_HEADER_SIZE)
            payload_len = struct.unpack(">I", header)[0]
            if payload_len == 0 or payload_len > MAX_MESSAGE_SIZE:
                raise ValueError(f"bad payload_len {payload_len}")
            payload = await self._reader.readexactly(payload_len)
            self._bytes_received += FRAME_HEADER_SIZE + payload_len
            try:
                msg_type, data = parse_msg(payload)
            except ValueError:
                Logger.error("CollabClient: invalid join response")
                return
            if msg_type == MessageType.JOINED:
                self._peer_id = int(data.get("your_id", -1))
                with self._lock:
                    self._incoming.append((msg_type, data))
                if self._on_connected:
                    try:
                        self._on_connected()
                    except Exception:
                        pass
            elif msg_type == MessageType.NET_KICK:
                with self._lock:
                    self._incoming.append((msg_type, data))
                return
            await self._read_loop()
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        except Exception as e:
            Logger.error(f"GameClient connection error: {e}")
        finally:
            self._connected = False
            if self._writer:
                try:
                    self._writer.close()
                    await self._writer.wait_closed()
                except Exception:
                    pass
            if not self._stopped and self._on_disconnected:
                try:
                    self._on_disconnected()
                except Exception:
                    pass

    async def _read_loop(self):
        while not self._stopped and self._reader:
            try:
                header = await self._reader.readexactly(FRAME_HEADER_SIZE)
                payload_len = struct.unpack(">I", header)[0]
                if payload_len == 0 or payload_len > MAX_MESSAGE_SIZE:
                    raise ValueError(f"bad payload_len {payload_len}")
                payload = await self._reader.readexactly(payload_len)
                self._bytes_received += FRAME_HEADER_SIZE + payload_len
                try:
                    msg_type, data = parse_msg(payload)
                except ValueError:
                    continue
                if msg_type == MessageType.PING:
                    pong = make_msg(MessageType.PONG, {"t": data.get("t", 0)})
                    self._writer.write(pong)
                    self._bytes_sent += len(pong)
                    try:
                        await self._writer.drain()
                    except Exception:
                        pass
                elif msg_type == MessageType.NET_KICK:
                    with self._lock:
                        self._incoming.append((msg_type, data))
                    break
                else:
                    with self._lock:
                        self._incoming.append((msg_type, data))
            except (asyncio.IncompleteReadError, ConnectionResetError, ValueError):
                break

    def send(self, msg_type: int, data: dict):
        if not self._connected or not self._writer or not self._loop or self._loop.is_closed():
            return
        try:
            msg = make_msg(msg_type, data)
            asyncio.run_coroutine_threadsafe(self._async_send(msg), self._loop)
        except Exception as e:
            Logger.error(f"GameClient send error: {e}")

    async def _async_send(self, msg: bytes):
        try:
            self._writer.write(msg)
            self._bytes_sent += len(msg)
            await self._writer.drain()
        except Exception:
            pass


class Transport:
    _instance: Optional[Transport] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._incoming: deque[tuple[int, dict]] = deque()
        self._lock = threading.Lock()
        self._server: Optional[GameServer] = None
        self._client: Optional[GameClient] = None
        self._role: str = "none"
        self._local_id: int = -1
        self._connected = False

    @classmethod
    def instance(cls) -> Transport:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = Transport()
        return cls._instance

    @property
    def is_server(self) -> bool:
        return self._role == "server"

    @property
    def is_client(self) -> bool:
        return self._role == "client"

    @property
    def is_connected(self) -> bool:
        if self._role == "server":
            return self._server is not None and self._server.running
        if self._role == "client":
            return self._client is not None and self._client.connected
        return False

    @property
    def local_id(self) -> int:
        if self._role == "server":
            return 0
        if self._role == "client" and self._client:
            return self._client.peer_id
        return self._local_id

    @property
    def peer_count(self) -> int:
        if self._role == "server" and self._server:
            return self._server.client_count
        if self._role == "client":
            return 1 if self.is_connected else 0
        return 0

    @property
    def bytes_sent(self) -> int:
        if self._client:
            return self._client.bytes_sent
        return 0

    @property
    def bytes_received(self) -> int:
        if self._client:
            return self._client.bytes_received
        return 0

    def host(self, host: str = "0.0.0.0", port: int = 7777, max_players: int = 16) -> bool:
        self.disconnect()
        self._server = GameServer(host, port, self._incoming, self._lock, max_players=max_players)
        ok = self._server.start()
        if ok:
            self._role = "server"
            self._local_id = 0
            self._connected = True
        return bool(ok)

    def connect(self, host: str = "127.0.0.1", port: int = 7777, name: str = "Player") -> bool:
        self.disconnect()
        self._client = GameClient(self._incoming, self._lock)
        self._client.connect(host, port, name)
        self._role = "client"
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if self._client.connected:
                self._connected = True
                return True
            with self._lock:
                for mt, d in list(self._incoming):
                    if mt == MessageType.NET_KICK:
                        self.disconnect()
                        return False
            time.sleep(0.02)
        ok = self._client.connected
        self._connected = bool(ok)
        return bool(ok)

    def disconnect(self):
        if self._client:
            self._client.disconnect()
            self._client = None
        if self._server:
            self._server.stop()
            self._server = None
        self._role = "none"
        self._connected = False
        self._local_id = -1
        with self._lock:
            self._incoming.clear()

    def send(self, msg_type: int, data: dict):
        if self._role == "client" and self._client:
            self._client.send(msg_type, data)
        elif self._role == "server" and self._server:
            self._server.broadcast(msg_type, data)

    def send_to(self, peer_id: int, msg_type: int, data: dict):
        if self._role == "server" and self._server:
            self._server.send_to(peer_id, msg_type, data)
        elif self._role == "client" and self._client:
            self._client.send(msg_type, data)

    def broadcast(self, msg_type: int, data: dict):
        self.send(msg_type, data)

    def poll(self) -> list[tuple[int, dict]]:
        with self._lock:
            msgs = list(self._incoming)
            self._incoming.clear()
            return msgs

    def peek(self) -> list[tuple[int, dict]]:
        with self._lock:
            return list(self._incoming)

    def get_peers(self) -> list[dict]:
        if self._role == "server" and self._server:
            return [{"id": c.peer_id, "name": c.name} for c in self._server.clients.values()]
        return []


def get_transport() -> Transport:
    return Transport.instance()
