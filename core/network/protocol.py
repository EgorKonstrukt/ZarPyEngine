# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import struct
import msgpack
from enum import IntEnum

from core.config.constants import FRAME_HEADER_SIZE, MAX_MESSAGE_SIZE

class MessageType(IntEnum):
    JOIN = 1
    JOINED = 2
    LEAVE = 3
    CURSOR = 4
    CURSOR_UPDATE = 5
    CAMERA = 6
    CAMERA_UPDATE = 7
    ENTITY_CREATE = 8
    ENTITY_CREATED = 9
    ENTITY_DELETE = 10
    ENTITY_DELETED = 11
    TRANSFORM_UPDATE = 12
    TRANSFORM_UPDATED = 13
    SELECTION = 14
    SELECTION_UPDATE = 15
    SCENE_SNAPSHOT_REQ = 16
    SCENE_SNAPSHOT = 17
    COMPONENT_UPDATE = 18
    COMPONENT_UPDATED = 19
    PEER_JOINED = 20
    COMPONENT_SYNC = 21
    PLAY_MODE = 22
    PING = 23
    PONG = 24
    GIZMO_STATE = 25
    GIZMO_STATE_UPDATE = 26
    COMPONENT_ADD = 27
    COMPONENT_REMOVE = 28
    COMPONENT_ADDED = 29
    COMPONENT_REMOVED = 30
    ASSET_LIST_REQ = 31
    ASSET_LIST = 32
    ASSET_SYNC = 33
    ASSET_SYNC_END = 34
    ASSET_WATCH = 35
    ASSET_DELETE = 36
    ASSET_REQUEST = 37
    GRAPH_NODE_MOVE = 38
    GRAPH_PORT_CONNECT = 39
    GRAPH_PORT_DISCONNECT = 40
    GRAPH_PARAM_CHANGE = 41
    GRAPH_NODE_ADD = 42
    GRAPH_NODE_DELETE = 43
    GRAPH_CURSOR = 44
    GRAPH_SYNC_REQ = 45
    GRAPH_SYNC = 46
    SCENE_OPEN = 47
    SCENE_TAB_SWITCH = 48
    SCENE_TAB_CLOSE = 49
    SCRIPT_OPEN = 50
    SCRIPT_CHANGE = 51
    SCRIPT_CURSOR = 53
    SCRIPT_OPS = 55
    NET_SPAWN = 100
    NET_DESPAWN = 101
    NET_SPAWN_REQUEST = 102
    NET_RPC = 103
    NET_TRANSFORM = 104
    NET_RIGIDBODY = 105
    NET_ANIMATOR = 106
    NET_OWNER_CHANGE = 107
    NET_VARIABLES = 108
    NET_READY = 109
    NET_KICK = 110
    NET_SNAPSHOT = 111


def make_msg(msg_type: int, data: dict) -> bytes:
    payload = msgpack.packb({"t": int(msg_type), "d": data}, use_bin_type=True)
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ValueError(f"payload too large: {len(payload)}")
    return struct.pack(">I", len(payload)) + payload


def parse_msg(data: bytes) -> tuple[int, dict]:
    try:
        obj = msgpack.unpackb(data, raw=False)
        return int(obj["t"]), dict(obj["d"])
    except Exception as e:
        hexdump = data[:64].hex()
        raise ValueError(f"Invalid message payload: {e} (raw: {hexdump})") from e
