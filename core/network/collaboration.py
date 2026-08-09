# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
import time
import uuid
import hashlib
import threading
import asyncio
from typing import Optional, Callable
from core.foundation.logger import Logger
from core.network.protocol import MessageType
from core.network.server import CollabServer
from core.network.client import CollabClient
from core.ecs.ecs import Scene, Entity, ComponentRegistry
from core.config.config import get_global_config


def _compute_hash(path: str) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def _scan_assets_dir(assets_dir: str) -> dict[str, dict]:
    result = {}
    if not os.path.isdir(assets_dir):
        return result
    for root, dirs, files in os.walk(assets_dir):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, assets_dir)
            try:
                size = os.path.getsize(full)
                mtime = os.path.getmtime(full)
            except Exception:
                continue
            result[rel] = {"size": size, "mtime": mtime, "hash": ""}
    return result


class _AssetWatcher(threading.Thread):
    def __init__(self, assets_dir: str, interval: float, on_change: Callable, on_delete: Callable):
        super().__init__(daemon=True)
        self._assets_dir = assets_dir
        self._interval = interval
        self._on_change = on_change
        self._on_delete = on_delete
        self._stop_event = threading.Event()
        self._snapshots: dict[str, float] = {}
        self._refresh()

    def _refresh(self):
        self._snapshots.clear()
        if not os.path.isdir(self._assets_dir):
            return
        for root, dirs, files in os.walk(self._assets_dir):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, self._assets_dir)
                try:
                    self._snapshots[rel] = os.path.getmtime(full)
                except Exception:
                    pass

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if self._stop_event.is_set():
                return
            if not os.path.isdir(self._assets_dir):
                continue
            current: dict[str, float] = {}
            for root, dirs, files in os.walk(self._assets_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, self._assets_dir)
                    try:
                        current[rel] = os.path.getmtime(full)
                    except Exception:
                        pass
            for rel, mtime in current.items():
                old = self._snapshots.get(rel)
                if old is None:
                    self._on_change(rel)
                elif abs(mtime - old) > 0.1:
                    self._on_change(rel)
            for rel in list(self._snapshots.keys()):
                if rel not in current:
                    self._on_delete(rel)
            self._snapshots = current


class _PollThread(threading.Thread):
    def __init__(self, callback, interval_ms, stop_event):
        super().__init__(daemon=True)
        self._callback = callback
        self._interval = interval_ms / 1000.0
        self._stop_event = stop_event

    def run(self):
        while not self._stop_event.is_set():
            self._callback()
            self._stop_event.wait(self._interval)


class RemotePeer:
    __slots__ = ("peer_id", "name", "color", "current_tab",
                 "cursor_screen", "cursor_hit",
                 "camera_pos", "camera_fwd", "camera_up",
                 "selected_entity_ids", "transform_data", "transform_deltas", "last_seen",
                 "ping_ms", "ping_timestamp",
                 "gizmo_mode", "gizmo_hover_axis", "gizmo_dragging")

    def __init__(self, peer_id: str, name: str, color: list[float]):
        self.peer_id = peer_id
        self.name = name
        self.color = color
        self.current_tab: str = ""
        self.cursor_screen: tuple[float, float] = (0, 0)
        self.cursor_hit: Optional[list[float]] = None
        self.camera_pos: list[float] = [0, 0, 0]
        self.camera_fwd: list[float] = [0, 0, -1]
        self.camera_up: list[float] = [0, 1, 0]
        self.selected_entity_ids: list[str] = []
        self.transform_data: dict[str, dict] = {}
        self.transform_deltas: dict[str, dict] = {}
        self.last_seen: float = time.time()
        self.ping_ms: float = 0.0
        self.ping_timestamp: float = 0.0
        self.gizmo_mode: str = "none"
        self.gizmo_hover_axis: int = -1
        self.gizmo_dragging: bool = False


class CollabSettings:
    __slots__ = ("cursor_interval", "camera_interval", "transform_interval",
                 "gizmo_interval", "ping_interval", "poll_interval", "scene_sync_interval")

    def __init__(self):
        self.cursor_interval: float = 1.0 / 30.0
        self.camera_interval: float = 1.0 / 15.0
        self.transform_interval: float = 1.0 / 20.0
        self.gizmo_interval: float = 1.0 / 10.0
        self.ping_interval: float = 3.0
        self.poll_interval: int = 8
        self.scene_sync_interval: float = 1.0


class CollaborationManager:
    def __init__(self, engine):
        self._engine = engine
        self._server: Optional[CollabServer] = None
        self._client: Optional[CollabClient] = None
        self._peers: dict[str, RemotePeer] = {}
        self._own_name: str = "User"
        self._scene_snapshot_callback: Optional[Callable] = None
        self._on_scene_sync: Optional[Callable[[dict], bool]] = None
        self._peer_joined_callback: Optional[Callable] = None
        self._peer_left_callback: Optional[Callable] = None
        self._on_remote_script_open: Optional[Callable] = None
        self._on_remote_script_change: Optional[Callable] = None
        self._on_remote_script_cursor: Optional[Callable] = None
        self._on_remote_script_ops: Optional[Callable] = None
        self._entity_synced_ids: dict[str, str] = {}
        self._local_entity_ids: dict[str, str] = {}
        self._pending_scene_sync = False
        self._stop_events: list[threading.Event] = []
        self.settings = CollabSettings()
        self._load_settings_from_config()
        self._poll_thread = _PollThread(self._poll_messages, self.settings.poll_interval, self._make_stop_event())
        self._poll_thread.start()
        self._as_host = False
        self._server_ready: Optional[threading.Event] = None
        self._play_mode_active = False
        self._latency_ms: float = 0.0
        self._ping_timestamp: float = 0.0
        self._ping_thread = _PollThread(self._send_ping, int(self.settings.ping_interval * 1000), self._make_stop_event())
        self._ping_thread.start()
        self._asset_syncing = False
        self._asset_sync_progress: dict = {"total": 0, "current": 0, "current_file": "", "failed": 0}
        self._asset_checksums: dict[str, str] = {}
        self._asset_watcher: Optional[_AssetWatcher] = None
        self._on_asset_progress: Optional[Callable[[dict], None]] = None
        self._asset_write_lock = 0
        self._suppressed_paths: set[str] = set()
        self._custom_handlers: dict[int, Callable] = {}
        self._on_remote_scene_open: Optional[Callable] = None
        self._on_remote_tab_switch: Optional[Callable] = None
        self._on_remote_tab_close: Optional[Callable] = None
        self._current_tab: str = ""

    @property
    def current_tab(self) -> str:
        return self._current_tab

    def set_current_tab(self, tab_name: str):
        self._current_tab = tab_name

    def _make_stop_event(self) -> threading.Event:
        ev = threading.Event()
        self._stop_events.append(ev)
        return ev

    @property
    def peers(self) -> dict[str, RemotePeer]:
        return self._peers

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.connected

    @property
    def is_host(self) -> bool:
        return self._as_host

    @property
    def own_peer_id(self) -> Optional[str]:
        return self._client.peer_id if self._client else None

    @property
    def play_mode_active(self) -> bool:
        return self._play_mode_active

    @property
    def bytes_sent(self) -> int:
        return self._client.bytes_sent if self._client else 0

    @property
    def bytes_received(self) -> int:
        return self._client.bytes_received if self._client else 0

    @property
    def latency_ms(self) -> float:
        return self._latency_ms

    def set_scene_snapshot_callback(self, cb: Callable):
        self._scene_snapshot_callback = cb

    def set_on_scene_sync(self, cb: Callable[[dict], bool]):
        self._on_scene_sync = cb

    def set_peer_joined_callback(self, cb: Callable):
        self._peer_joined_callback = cb

    def set_peer_left_callback(self, cb: Callable):
        self._peer_left_callback = cb

    def set_on_remote_scene_open(self, cb: Callable):
        self._on_remote_scene_open = cb

    def set_on_remote_tab_switch(self, cb: Callable):
        self._on_remote_tab_switch = cb

    def set_on_remote_tab_close(self, cb: Callable):
        self._on_remote_tab_close = cb

    def set_on_remote_script_open(self, cb: Callable):
        self._on_remote_script_open = cb

    def set_on_remote_script_change(self, cb: Callable):
        self._on_remote_script_change = cb

    def set_on_remote_script_cursor(self, cb: Callable):
        self._on_remote_script_cursor = cb

    def set_on_remote_script_ops(self, cb: Callable):
        self._on_remote_script_ops = cb

    def send_script_open(self, path: str, content: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCRIPT_OPEN, {"path": path, "content": content})

    def send_script_change(self, path: str, content: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCRIPT_CHANGE, {"path": path, "content": content})

    def send_script_cursor(self, path: str, pos: int, sel_anchor: int, sel_end: int):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCRIPT_CURSOR, {
                "path": path, "pos": pos, "sel_anchor": sel_anchor, "sel_end": sel_end
            })

    def send_script_ops(self, path: str, ops: list):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCRIPT_OPS, {"path": path, "ops": ops})

    def send_scene_open(self, name: str, path: str, data: dict):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCENE_OPEN, {"name": name, "path": path, "data": data})
            print(f"[COLLAB] send_scene_open called: name={name}, path={path}, data_size={len(str(data))}", flush=True)
        else:
            print(f"[COLLAB] send_scene_open skipped: client={self._client is not None}, connected={self._client.connected if self._client else False}", flush=True)

    def send_scene_tab_switch(self, name: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCENE_TAB_SWITCH, {"name": name})

    def send_scene_tab_close(self, name: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCENE_TAB_CLOSE, {"name": name})

    def get_peer(self, peer_id: str) -> Optional[RemotePeer]:
        return self._peers.get(peer_id)

    def start_server(self, host: str = "0.0.0.0", port: int = 9876):
        if self._client:
            self.stop()
        self._as_host = True
        self._server = CollabServer(host, port)
        self._server_thread_task = None

        self._server_ready = threading.Event()

        def _run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._server.start())
            self._server_ready.set()
            loop.run_forever()

        self._server_thread = threading.Thread(target=_run, daemon=True)
        self._server_thread.start()
        self._server_ready.wait(timeout=5)
        self._start_asset_watcher()
        Logger.info(f"Collab server started on {host}:{port}")

    def connect(self, host: str = "127.0.0.1", port: int = 9876, name: str = "User"):
        if self._client:
            self.stop()
        if self._server and self._engine.scene:
            self.update_server_scene(self._engine.scene)
        self._own_name = name
        if not self._server:
            self._as_host = False
        self._client = CollabClient()
        self._client.set_on_disconnected(self._on_disconnected)
        self._client.connect(host, port, name)

    def stop(self):
        self._stop_asset_watcher()
        for ev in self._stop_events:
            ev.set()
        self._stop_events.clear()
        if self._client:
            self._client.disconnect()
            self._client = None
        if self._server:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._server.stop())
                loop.close()
            except Exception:
                pass
            self._server = None
        self._peers.clear()
        self._entity_synced_ids.clear()
        self._local_entity_ids.clear()
        self._as_host = False
        self._asset_syncing = False
        Logger.info("Collab stopped")

    def _on_disconnected(self):
        self._peers.clear()
        Logger.info("Collab disconnected")

    def _poll_messages(self):
        if not self._client or not self._client.connected:
            return
        msgs = self._client.poll_messages()
        for msg_type, data in msgs:
            self._handle_message(msg_type, data)
        self._sync_server_scene()

    def _sync_server_scene(self):
        if not self._as_host or not self._server:
            return
        now = time.time()
        if now - getattr(self, "_last_scene_sync", 0) < 1.0:
            return
        self._last_scene_sync = now
        scene = self._engine.scene
        if scene:
            self.update_server_scene(scene)

    def _handle_message(self, msg_type: int, data: dict):
        if msg_type == MessageType.JOINED:
            peers_raw = data.get("peers", [])
            for p in peers_raw:
                pid = p["id"]
                if pid not in self._peers and pid != self._client.peer_id:
                    self._add_peer(pid, p["name"], p["color"])
            if data.get("scene"):
                self._apply_scene_snapshot(data["scene"])
            if self._scene_snapshot_callback:
                self._scene_snapshot_callback(data)
            if not self._as_host:
                self.request_asset_list()
            self._start_asset_watcher()
        elif msg_type == MessageType.PEER_JOINED:
            pid = data.get("id", "")
            if pid and pid not in self._peers and pid != self._client.peer_id:
                self._add_peer(pid, data.get("name", ""), data.get("color", [0.5, 0.5, 0.5]))
        elif msg_type == MessageType.LEAVE:
            pid = data.get("id", "")
            if pid in self._peers:
                peer = self._peers.pop(pid)
                if self._peer_left_callback:
                    self._peer_left_callback(peer)
        elif msg_type == MessageType.CURSOR_UPDATE:
            pid = data.get("id", "")
            peer = self._peers.get(pid)
            if peer:
                peer.cursor_screen = (data.get("x", 0), data.get("y", 0))
                peer.cursor_hit = data.get("hit")
                peer.last_seen = time.time()
        elif msg_type == MessageType.CAMERA_UPDATE:
            pid = data.get("id", "")
            peer = self._peers.get(pid)
            if peer:
                peer.camera_pos = data.get("pos", [0, 0, 0])
                peer.camera_fwd = data.get("fwd", [0, 0, -1])
                peer.camera_up = data.get("up", [0, 1, 0])
        elif msg_type == MessageType.ENTITY_CREATED:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._tab_matches(data):
                entity_data = data.get("entity", {})
                self._create_remote_entity(entity_data)
        elif msg_type == MessageType.ENTITY_DELETED:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._tab_matches(data):
                entity_id = data.get("entity_id", "")
                self._delete_remote_entity(entity_id)
        elif msg_type == MessageType.TRANSFORM_UPDATED:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._tab_matches(data):
                self._apply_remote_transform(data)
        elif msg_type == MessageType.SELECTION_UPDATE:
            pid = data.get("id", "")
            peer = self._peers.get(pid)
            if peer:
                peer.selected_entity_ids = data.get("entity_ids", [])
        elif msg_type == MessageType.COMPONENT_UPDATED:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._tab_matches(data):
                self._apply_remote_component(data)
        elif msg_type == MessageType.COMPONENT_ADDED:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._tab_matches(data):
                self._apply_remote_component_add(data)
        elif msg_type == MessageType.COMPONENT_REMOVED:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._tab_matches(data):
                self._apply_remote_component_remove(data)
        elif msg_type == MessageType.PONG:
            now = time.time()
            self._latency_ms = (now - self._ping_timestamp) * 500
        elif msg_type == MessageType.PLAY_MODE:
            self._play_mode_active = data.get("active", False)
        elif msg_type == MessageType.COMPONENT_SYNC:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._tab_matches(data):
                self._apply_remote_component_sync(data)
        elif msg_type == MessageType.GIZMO_STATE_UPDATE:
            pid = data.get("id", "")
            peer = self._peers.get(pid)
            if peer:
                peer.gizmo_mode = data.get("mode", "none")
                peer.gizmo_hover_axis = data.get("hover", -1)
                peer.gizmo_dragging = data.get("dragging", False)
        elif msg_type == MessageType.SCENE_SNAPSHOT:
            scene_data = data.get("scene")
            req_id = data.get("requesting_id", "")
            if req_id and req_id != self._client.peer_id:
                pass
            elif scene_data:
                self._apply_scene_snapshot(scene_data)
        elif msg_type == MessageType.ASSET_LIST_REQ:
            pid = data.get("id", "")
            if self._as_host and pid:
                self.send_asset_list(pid)
        elif msg_type == MessageType.ASSET_LIST:
            pid = data.get("id", "")
            if pid != self._client.peer_id:
                return
            self._handle_asset_list(data)
        elif msg_type == MessageType.ASSET_SYNC:
            pid = data.get("id", "")
            if pid != self._client.peer_id:
                self._handle_asset_sync(data)
        elif msg_type == MessageType.ASSET_WATCH:
            pid = data.get("id", "")
            if pid != self._client.peer_id:
                self._handle_asset_watch(data)
        elif msg_type == MessageType.ASSET_DELETE:
            pid = data.get("id", "")
            if pid != self._client.peer_id:
                self._handle_asset_delete(data)
        elif msg_type == MessageType.ASSET_REQUEST:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._as_host:
                self._handle_asset_request(data)
        elif msg_type == MessageType.SCENE_OPEN:
            pid = data.get("id", "")
            pname = data.get("name", "")
            print(f"[COLLAB] Received SCENE_OPEN: from={pid}, name={pname}, has_callback={self._on_remote_scene_open is not None}", flush=True)
            if pid != self._client.peer_id:
                if pid in self._peers:
                    self._peers[pid].current_tab = pname
                if self._on_remote_scene_open:
                    self._on_remote_scene_open(data)
                    print(f"[COLLAB] Forwarded SCENE_OPEN to callback for tab '{pname}'", flush=True)
        elif msg_type == MessageType.SCENE_TAB_SWITCH:
            pid = data.get("id", "")
            tab = data.get("name", "")
            if pid != self._client.peer_id:
                if pid in self._peers:
                    self._peers[pid].current_tab = tab
                if tab and tab == self._current_tab:
                    self._send_full_scene_sync(pid)
                if self._on_remote_tab_switch:
                    self._on_remote_tab_switch(pid, tab)
        elif msg_type == MessageType.SCENE_TAB_CLOSE:
            pid = data.get("id", "")
            if pid != self._client.peer_id:
                if pid in self._peers and self._peers[pid].current_tab == data.get("name", ""):
                    self._peers[pid].current_tab = ""
                if self._on_remote_tab_close:
                    self._on_remote_tab_close(data.get("name", ""))
        elif msg_type == MessageType.SCRIPT_OPEN:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._on_remote_script_open:
                self._on_remote_script_open(data)
        elif msg_type == MessageType.SCRIPT_CHANGE:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._on_remote_script_change:
                self._on_remote_script_change(data)
        elif msg_type == MessageType.SCRIPT_CURSOR:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._on_remote_script_cursor:
                self._on_remote_script_cursor(pid, data)
        elif msg_type == MessageType.SCRIPT_OPS:
            pid = data.get("id", "")
            if pid != self._client.peer_id and self._on_remote_script_ops:
                self._on_remote_script_ops(pid, data)
        else:
            handler = self._custom_handlers.get(msg_type)
            if handler is not None:
                handler(data)

    def _handle_asset_list(self, data: dict):
        paths = data.get("assets", [])
        checksums = data.get("checksums", {})
        assets_dir = self._get_assets_dir()
        os.makedirs(assets_dir, exist_ok=True)
        need_sync = []
        for rel in paths:
            full = os.path.join(assets_dir, rel)
            expected = checksums.get(rel, "")
            if os.path.exists(full):
                actual = _compute_hash(full)
                if actual == expected:
                    continue
            need_sync.append(rel)
        self._asset_syncing = True
        self._asset_sync_progress = {"total": len(need_sync), "current": 0, "current_file": "", "failed": 0}
        for rel in need_sync:
            self._asset_sync_progress["current_file"] = rel
            if self._on_asset_progress:
                self._on_asset_progress(self._asset_sync_progress)
            self.send_asset_request(rel)
        if not need_sync:
            self._asset_syncing = False
            self._asset_sync_progress["current_file"] = ""
            self._asset_sync_progress["current"] = 0
            if self._on_asset_progress:
                self._on_asset_progress(self._asset_sync_progress)

    def _handle_asset_sync(self, data: dict):
        rel = data.get("path", "")
        raw = data.get("data")
        checksum = data.get("checksum", "")
        if not rel or raw is None:
            return
        assets_dir = self._get_assets_dir()
        full = os.path.join(assets_dir, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        self._asset_write_lock += 1
        self._suppressed_paths.add(rel)
        try:
            with open(full, "wb") as f:
                f.write(raw)
        except Exception as e:
            Logger.error(f"Collab: failed to write asset '{rel}': {e}")
            self._asset_sync_progress["failed"] += 1
            self._asset_write_lock -= 1
            return
        self._asset_write_lock -= 1
        if checksum:
            actual = _compute_hash(full)
            if actual != checksum:
                Logger.warning(f"Collab: asset '{rel}' checksum mismatch")
        self._asset_sync_progress["current"] += 1
        if self._asset_sync_progress["current"] >= self._asset_sync_progress["total"]:
            self._asset_syncing = False
            self._asset_sync_progress["current_file"] = ""
        if self._on_asset_progress:
            self._on_asset_progress(self._asset_sync_progress)

    def _handle_asset_watch(self, data: dict):
        rel = data.get("path", "")
        raw = data.get("data")
        if not rel or raw is None:
            return
        assets_dir = self._get_assets_dir()
        full = os.path.join(assets_dir, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        self._asset_write_lock += 1
        self._suppressed_paths.add(rel)
        try:
            with open(full, "wb") as f:
                f.write(raw)
        except Exception as e:
            Logger.error(f"Collab: failed to write watched asset '{rel}': {e}")
        finally:
            self._asset_write_lock -= 1

    def _handle_asset_delete(self, data: dict):
        rel = data.get("path", "")
        if not rel:
            return
        assets_dir = self._get_assets_dir()
        full = os.path.join(assets_dir, rel)
        self._asset_write_lock += 1
        self._suppressed_paths.add(rel)
        try:
            if os.path.exists(full):
                os.remove(full)
                Logger.info(f"Collab: removed asset '{rel}'")
        except Exception as e:
            Logger.error(f"Collab: failed to remove asset '{rel}': {e}")
        finally:
            self._asset_write_lock -= 1

    def send_cursor(self, screen_x: float, screen_y: float, hit_pos: Optional[list[float]] = None):
        if self._client and self._client.connected:
            self._client.send(MessageType.CURSOR, {
                "x": screen_x, "y": screen_y, "hit": hit_pos
            })

    def send_camera(self, pos: list[float], fwd: list[float], up: list[float]):
        if self._client and self._client.connected:
            self._client.send(MessageType.CAMERA, {
                "pos": pos, "fwd": fwd, "up": up
            })

    def _with_tab(self, d: dict) -> dict:
        if self._current_tab:
            d["tab"] = self._current_tab
        return d

    def _tab_matches(self, data: dict) -> bool:
        msg_tab = data.get("tab", "")
        return not msg_tab or msg_tab == self._current_tab

    def _send_full_scene_sync(self, target_peer_id: str):
        if not self._current_tab:
            return
        scene = self._engine.scene
        if not scene:
            return
        for eid, entity in scene._entities.items():
            entity_data = entity.serialize()
            if self._client and self._client.connected:
                self._client.send(MessageType.ENTITY_CREATE, self._with_tab({"entity": entity_data}))
            if entity.transform:
                t = entity.transform
                pos = t.local_position.to_list()
                rot = t.local_rotation.to_list()
                scl = t.local_scale.to_list()
                if self._client and self._client.connected:
                    self._client.send(MessageType.TRANSFORM_UPDATE, self._with_tab({
                        "entity_id": eid, "p": pos, "r": rot, "s": scl
                    }))

    def send_entity_create(self, entity_data: dict):
        if self._client and self._client.connected:
            self._client.send(MessageType.ENTITY_CREATE, self._with_tab({"entity": entity_data}))

    def send_entity_delete(self, entity_id: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.ENTITY_DELETE, self._with_tab({"entity_id": entity_id}))

    def send_transform(self, entity_id: str, pos: list[float], rot: list[float], scale: list[float]):
        if self._client and self._client.connected:
            self._client.send(MessageType.TRANSFORM_UPDATE, self._with_tab({
                "entity_id": entity_id, "p": pos, "r": rot, "s": scale
            }))

    def send_selection(self, entity_ids: list[str]):
        if self._client and self._client.connected:
            self._client.send(MessageType.SELECTION, self._with_tab({"entity_ids": entity_ids}))

    @staticmethod
    def _collapse_value(v):
        if hasattr(v, 'to_list'):
            return v.to_list()
        if hasattr(v, '__iter__') and not isinstance(v, (str, bytes, dict)):
            return list(v)
        if isinstance(v, dict):
            return {k: CollaborationManager._collapse_value(v) for k, v in v.items()}
        return v

    def send_component_update(self, entity_id: str, component_key: str,
                              prop: str, value):
        if self._client and self._client.connected:
            self._client.send(MessageType.COMPONENT_UPDATE, self._with_tab({
                "entity_id": entity_id, "component_key": component_key,
                "prop": prop, "value": self._collapse_value(value)
            }))

    def send_component_sync(self, entity_id: str, component_key: str, data: dict):
        if self._client and self._client.connected:
            self._client.send(MessageType.COMPONENT_SYNC, self._with_tab({
                "entity_id": entity_id, "component_key": component_key,
                "data": self._collapse_value(data)
            }))

    def send_component_add(self, entity_id: str, component_key: str, comp_data: dict):
        if self._client and self._client.connected:
            self._client.send(MessageType.COMPONENT_ADD, self._with_tab({
                "entity_id": entity_id, "component_key": component_key,
                "data": self._collapse_value(comp_data)
            }))

    def send_component_remove(self, entity_id: str, component_key: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.COMPONENT_REMOVE, self._with_tab({
                "entity_id": entity_id, "component_key": component_key
            }))

    def send_play_mode(self, active: bool):
        if self._client and self._client.connected:
            self._client.send(MessageType.PLAY_MODE, {"active": active})
            self._play_mode_active = active

    def apply_settings(self):
        for ev in self._stop_events:
            ev.set()
        self._stop_events.clear()
        self._poll_thread = _PollThread(self._poll_messages, self.settings.poll_interval, self._make_stop_event())
        self._poll_thread.start()
        self._ping_thread = _PollThread(self._send_ping, int(self.settings.ping_interval * 1000), self._make_stop_event())
        self._ping_thread.start()

    def _load_settings_from_config(self):
        cfg = get_global_config()
        s = self.settings
        s.cursor_interval = 1.0 / max(1, cfg.get("collab.cursor_rate", 30.0))
        s.camera_interval = 1.0 / max(1, cfg.get("collab.camera_rate", 15.0))
        s.transform_interval = 1.0 / max(1, cfg.get("collab.transform_rate", 20.0))
        s.gizmo_interval = 1.0 / max(1, cfg.get("collab.gizmo_rate", 10.0))
        s.ping_interval = cfg.get("collab.ping_interval", 3.0)
        s.poll_interval = cfg.get("collab.poll_interval", 8)

    def save_settings_to_config(self):
        cfg = get_global_config()
        s = self.settings
        cfg.set("collab.cursor_rate", 1.0 / max(0.001, s.cursor_interval))
        cfg.set("collab.camera_rate", 1.0 / max(0.001, s.camera_interval))
        cfg.set("collab.transform_rate", 1.0 / max(0.001, s.transform_interval))
        cfg.set("collab.gizmo_rate", 1.0 / max(0.001, s.gizmo_interval))
        cfg.set("collab.ping_interval", s.ping_interval)
        cfg.set("collab.poll_interval", s.poll_interval)
        cfg.save()

    def register_handler(self, msg_type: int, callback: Callable[[dict], None]):
        self._custom_handlers[msg_type] = callback

    def unregister_handler(self, msg_type: int):
        self._custom_handlers.pop(msg_type, None)

    def set_asset_progress_callback(self, cb: Callable[[dict], None]):
        self._on_asset_progress = cb

    @property
    def asset_syncing(self) -> bool:
        return self._asset_syncing

    @property
    def asset_sync_progress(self) -> dict:
        return dict(self._asset_sync_progress)

    def _get_assets_dir(self) -> str:
        root = self._engine.project_root if self._engine.project_root else os.getcwd()
        return os.path.join(root, "assets")

    def _scan_assets(self) -> dict[str, dict]:
        return _scan_assets_dir(self._get_assets_dir())

    def _load_checksums(self, manifest: dict[str, dict]) -> dict[str, str]:
        cache = {}
        for rel, info in manifest.items():
            full = os.path.join(self._get_assets_dir(), rel)
            if os.path.exists(full):
                cache[rel] = _compute_hash(full)
            else:
                cache[rel] = ""
        return cache

    def request_asset_list(self):
        if self._client and self._client.connected:
            self._client.send(MessageType.ASSET_LIST_REQ, {})

    def send_asset_list(self, target_peer_id: str):
        manifest = self._scan_assets()
        self._asset_checksums = self._load_checksums(manifest)
        paths = list(manifest.keys())
        self._client.send(MessageType.ASSET_LIST, {
            "target": target_peer_id,
            "assets": paths,
            "checksums": {p: self._asset_checksums.get(p, "") for p in paths},
        })

    def send_asset_file(self, relative_path: str):
        full = os.path.join(self._get_assets_dir(), relative_path)
        if not os.path.exists(full):
            return
        try:
            with open(full, "rb") as f:
                raw = f.read()
        except Exception as e:
            Logger.error(f"Collab: failed to read asset '{relative_path}': {e}")
            return
        checksum = _compute_hash(full)
        self._client.send(MessageType.ASSET_SYNC, {
            "path": relative_path,
            "data": raw,
            "checksum": checksum,
            "size": len(raw),
        })

    def send_asset_request(self, relative_path: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.ASSET_REQUEST, {
                "path": relative_path,
            })

    def _handle_asset_request(self, data: dict):
        rel = data.get("path", "")
        if not rel:
            return
        self.send_asset_file(rel)

    def send_asset_delete(self, relative_path: str):
        if self._client and self._client.connected:
            self._client.send(MessageType.ASSET_DELETE, {
                "path": relative_path,
            })

    def _send_watch_update(self, relative_path: str):
        if not self._client or not self._client.connected:
            return
        if self._asset_syncing or self._asset_write_lock > 0:
            return
        full = os.path.join(self._get_assets_dir(), relative_path)
        if not os.path.exists(full):
            return
        try:
            with open(full, "rb") as f:
                raw = f.read()
        except Exception:
            return
        self._client.send(MessageType.ASSET_WATCH, {
            "path": relative_path,
            "data": raw,
            "size": len(raw),
        })

    def _start_asset_watcher(self):
        self._stop_asset_watcher()
        ad = self._get_assets_dir()
        if not os.path.isdir(ad):
            return
        def _on_watch_change(rel):
            if rel in self._suppressed_paths:
                self._suppressed_paths.discard(rel)
                return
            self._send_watch_update(rel)
        def _on_watch_delete(rel):
            if rel in self._suppressed_paths:
                self._suppressed_paths.discard(rel)
                return
            self.send_asset_delete(rel)
        self._asset_watcher = _AssetWatcher(
            ad, 2.0,
            on_change=_on_watch_change,
            on_delete=_on_watch_delete,
        )
        self._asset_watcher.start()

    def _stop_asset_watcher(self):
        if self._asset_watcher:
            self._asset_watcher.stop()
            self._asset_watcher = None

    def send_gizmo_state(self, mode: str, hover_axis: int, dragging: bool):
        if self._client and self._client.connected:
            self._client.send(MessageType.GIZMO_STATE, {
                "mode": mode, "hover": hover_axis, "dragging": dragging
            })

    def _send_ping(self):
        if self._client and self._client.connected:
            self._ping_timestamp = time.time()
            self._client.send(MessageType.PING, {"t": self._ping_timestamp})

    def request_scene_snapshot(self):
        if self._client and self._client.connected:
            self._client.send(MessageType.SCENE_SNAPSHOT_REQ, {})

    def update_server_scene(self, scene: Scene):
        if self._server:
            data = scene.serialize()
            self._server.update_scene_data(data)

    def _add_peer(self, peer_id: str, name: str, color: list[float]):
        if peer_id not in self._peers:
            peer = RemotePeer(peer_id, name, color)
            self._peers[peer_id] = peer
            if self._peer_joined_callback:
                self._peer_joined_callback(peer)

    def _apply_scene_snapshot(self, scene_data: dict):
        tab_name = scene_data.get("name", "SyncedScene")
        path = scene_data.get("path", "")
        if self._on_remote_scene_open:
            self._on_remote_scene_open({"name": tab_name, "path": path, "data": scene_data})

    def _create_remote_entity(self, entity_data: dict):
        scene = self._engine.scene
        if not scene:
            return
        eid = entity_data.get("id", "")
        if eid and eid not in scene._entities:
            e = Entity.deserialize(entity_data, ComponentRegistry)
            scene.add_entity(e)
            pid = entity_data.get("parent")
            if pid and pid in scene._entities:
                parent = scene._entities.get(pid)
                if parent:
                    e.set_parent(parent)
        elif eid and eid in scene._entities:
            e = scene._entities[eid]
            for cd in entity_data.get("components", []):
                ctype = cd.get("type")
                comp_cls = ComponentRegistry.get(ctype)
                if comp_cls:
                    existing = e.get_component(comp_cls)
                    if existing:
                        for k, v in cd.items():
                            if k not in ("type", "_key"):
                                setattr(existing, k, v)
                    else:
                        comp = comp_cls.deserialize(cd)
                        e.add_component(comp)

    def _delete_remote_entity(self, entity_id: str):
        scene = self._engine.scene
        if scene and entity_id in scene._entities:
            scene.remove_entity(entity_id)

    def _apply_remote_transform(self, data: dict):
        scene = self._engine.scene
        if not scene:
            return
        entity_id = data.get("entity_id", "")
        e = scene.get_entity(entity_id)
        if not e:
            return
        t = e.transform
        if not t:
            return
        p = data.get("p")
        r = data.get("r")
        s = data.get("s")
        if p:
            from core.maths.math3d import Vec3
            old_local = Vec3(t.local_position.x, t.local_position.y, t.local_position.z)
            t.local_position = p
            pid = data.get("id", "")
            peer = self._peers.get(pid)
            if peer:
                dx = p[0] - old_local.x
                dy = p[1] - old_local.y
                dz = p[2] - old_local.z
                if abs(dx) > 0.0001 or abs(dy) > 0.0001 or abs(dz) > 0.0001:
                    peer.transform_deltas[entity_id] = {"pos": [dx, dy, dz], "time": time.time()}
        if r:
            from core.maths.math3d import Quat
            t.local_rotation = Quat(r[0], r[1], r[2], r[3])
        if s:
            t.local_scale = s

    def _apply_remote_component_sync(self, data: dict):
        scene = self._engine.scene
        if not scene:
            return
        entity_id = data.get("entity_id", "")
        e = scene.get_entity(entity_id)
        if not e:
            return
        comp_key = data.get("component_key", "")
        comp_data = data.get("data", {})
        comp = e.get_component_by_name(comp_key)
        if comp:
            for k, v in comp_data.items():
                if not hasattr(comp, k):
                    continue
                try:
                    current = getattr(comp, k)
                    if hasattr(current, 'from_list') and isinstance(v, list):
                        current.from_list(v)
                    elif hasattr(current, 'x') and isinstance(v, (list, tuple)) and len(v) == 3:
                        from core.maths.math3d import Vec3
                        setattr(comp, k, Vec3(v[0], v[1], v[2]))
                    elif hasattr(current, 'x') and isinstance(v, (list, tuple)) and len(v) == 4:
                        from core.maths.math3d import Quat
                        setattr(comp, k, Quat(v[0], v[1], v[2], v[3]))
                    else:
                        setattr(comp, k, v)
                except Exception:
                    pass
        else:
            registry = ComponentRegistry
            if comp_key in registry._component_types:
                comp_cls = registry._component_types[comp_key]
                comp = comp_cls.deserialize(comp_data)
                if comp:
                    e.add_component(comp)

    def _apply_remote_component(self, data: dict):
        scene = self._engine.scene
        if not scene:
            return
        entity_id = data.get("entity_id", "")
        e = scene.get_entity(entity_id)
        if not e:
            return
        comp_key = data.get("component_key", "")
        comp = e.get_component_by_name(comp_key)
        if not comp:
            return
        prop = data.get("prop", "")
        value = data.get("value")
        if not hasattr(comp, prop):
            return
        try:
            current = getattr(comp, prop)
            if hasattr(current, 'from_list') and isinstance(value, list):
                current.from_list(value)
            elif hasattr(current, 'x') and isinstance(value, (list, tuple)) and len(value) == 3:
                from core.maths.math3d import Vec3
                setattr(comp, prop, Vec3(value[0], value[1], value[2]))
            elif hasattr(current, 'x') and isinstance(value, (list, tuple)) and len(value) == 4:
                from core.maths.math3d import Quat
                setattr(comp, prop, Quat(value[0], value[1], value[2], value[3]))
            else:
                setattr(comp, prop, value)
        except Exception as e:
            Logger.warning(f"Collab: apply remote component error: {e}")

    def _apply_remote_component_add(self, data: dict):
        scene = self._engine.scene
        if not scene:
            return
        entity_id = data.get("entity_id", "")
        e = scene.get_entity(entity_id)
        if not e:
            return
        comp_key = data.get("component_key", "")
        comp_data = data.get("data", {})
        from core.ecs.ecs import ComponentRegistry
        registry = ComponentRegistry
        comp_cls = registry.get(comp_key)
        if not comp_cls:
            return
        can_multiple = getattr(comp_cls, '_allow_multiple', False)
        if not can_multiple and e.get_component_by_name(comp_key):
            return
        comp = comp_cls.deserialize(comp_data)
        if comp:
            e.add_component(comp)

    def _apply_remote_component_remove(self, data: dict):
        scene = self._engine.scene
        if not scene:
            return
        entity_id = data.get("entity_id", "")
        e = scene.get_entity(entity_id)
        if not e:
            return
        comp_key = data.get("component_key", "")
        for key in list(e._components.keys()):
            if key == comp_key or key.startswith(comp_key + "."):
                e.remove_component_by_key(key)
                return
