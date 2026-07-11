# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import struct
import msgpack
from enum import IntEnum

from core.config.constants import FRAME_HEADER_SIZE

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


def make_msg(msg_type: int, data: dict) -> bytes:
    payload = msgpack.packb({"t": msg_type, "d": data})
    return struct.pack(">I", len(payload)) + payload


def parse_msg(data: bytes) -> tuple[int, dict]:
    try:
        obj = msgpack.unpackb(data, raw=False)
        return obj["t"], obj["d"]
    except Exception as e:
        hexdump = data[:64].hex()
        raise ValueError(f"Invalid message payload: {e} (raw: {hexdump})") from e
