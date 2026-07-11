# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import hashlib
import json
import time as _time
from collections import deque
from typing import Optional, Any, TYPE_CHECKING
from core.foundation.logger import Logger

if TYPE_CHECKING:
    from core.ecs.ecs import Scene, ComponentRegistry


class FrameSnapshot:
    __slots__ = (
        "frame_number", "timestamp", "entity_count", "data",
        "frame_time_ms", "camera_data", "bookmarked", "hash",
        "breakpoint_hit",
    )

    def __init__(self, frame_number: int, timestamp: float,
                 entity_count: int, data: dict,
                 frame_time_ms: float = 0.0):
        self.frame_number: int = frame_number
        self.timestamp: float = timestamp
        self.entity_count: int = entity_count
        self.data: dict = data
        self.frame_time_ms: float = frame_time_ms
        self.camera_data: Optional[dict] = None
        self.bookmarked: bool = False
        self.hash: str = _compute_hash(data)
        self.breakpoint_hit: bool = False


def _compute_hash(data: dict) -> str:
    return hashlib.md5(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def diff_scenes(old_data: dict, new_data: dict) -> dict:
    old_entities: dict = old_data.get("entities", {})
    new_entities: dict = new_data.get("entities", {})
    old_ids = set(old_entities)
    new_ids = set(new_entities)
    created = sorted(new_ids - old_ids)
    deleted = sorted(old_ids - new_ids)
    changed: dict[str, list] = {}
    for eid in sorted(old_ids & new_ids):
        oe = old_entities[eid]
        ne = new_entities[eid]
        changes = _diff_entity(oe, ne)
        if changes:
            changed[eid] = changes
    return {"created": created, "deleted": deleted, "changed": changed}


def _diff_entity(old: dict, new: dict) -> list:
    changes = []
    old_name = old.get("name", "")
    new_name = new.get("name", "")
    if old_name != new_name:
        changes.append(("name", old_name, new_name))
    old_comps = {c.get("_key", ""): c for c in old.get("components", [])}
    new_comps = {c.get("_key", ""): c for c in new.get("components", [])}
    for key in sorted(set(old_comps) | set(new_comps)):
        if key not in old_comps:
            changes.append(("+comp", key, new_comps[key]))
        elif key not in new_comps:
            changes.append(("-comp", key, old_comps[key]))
        else:
            oc = old_comps[key]
            nc = new_comps[key]
            comp_diff = _diff_component(oc, nc)
            if comp_diff:
                changes.append(("~comp", key, comp_diff))
    return changes


def _diff_component(old: dict, new: dict) -> list:
    diff = []
    skip_keys = {"_key", "type"}
    for k in sorted(set(old.keys()) | set(new.keys())):
        if k in skip_keys:
            continue
        ov = old.get(k)
        nv = new.get(k)
        if ov != nv:
            diff.append((k, ov, nv))
    return diff


def find_entity_frame_changes(frames: list[FrameSnapshot],
                              entity_id: str) -> list[int]:
    changed = []
    prev_data: Optional[dict] = None
    for idx, snap in enumerate(frames):
        e_data = snap.data.get("entities", {}).get(entity_id)
        if e_data != prev_data:
            changed.append(idx)
        prev_data = e_data
    return changed


class SnapshotRecorder:
    def __init__(self, max_frames: int = 600):
        self._buffer: deque[FrameSnapshot] = deque(maxlen=max_frames)
        self._max_frames: int = max_frames
        self._recording: bool = False
        self._capture_interval: int = 1
        self._frame_counter: int = 0
        self._total_frames: int = 0
        self._breakpoint_expr: str = ""
        self._breakpoint_ns: dict = {}
        self._filter_unchanged: bool = False
        self._last_hash: str = ""
        self._last_frame_time: float = 0.0

    @property
    def is_recording(self) -> bool: return self._recording
    @property
    def num_frames(self) -> int: return len(self._buffer)
    @property
    def max_frames(self) -> int: return self._max_frames
    @property
    def total_frames(self) -> int: return self._total_frames
    @property
    def capture_interval(self) -> int: return self._capture_interval
    @capture_interval.setter
    def capture_interval(self, n: int): self._capture_interval = max(1, n)
    @property
    def breakpoint_expr(self) -> str: return self._breakpoint_expr
    @breakpoint_expr.setter
    def breakpoint_expr(self, expr: str): self._breakpoint_expr = expr
    @property
    def filter_unchanged(self) -> bool: return self._filter_unchanged
    @filter_unchanged.setter
    def filter_unchanged(self, v: bool): self._filter_unchanged = v

    def start(self):
        self._recording = True
        self._frame_counter = 0
        self._total_frames = 0
        self._last_hash = ""
        self._last_frame_time = _time.perf_counter()
        Logger.info("Time-travel recording started.")

    def stop(self):
        self._recording = False
        Logger.info(f"Time-travel stopped. {len(self._buffer)} frames.")

    def clear(self):
        self._buffer.clear()
        self._total_frames = 0
        self._frame_counter = 0
        self._last_hash = ""

    def set_breakpoint_ns(self, **kwargs):
        self._breakpoint_ns.update(kwargs)

    def capture(self, scene: Scene, frame_time_ms: float = 0.0,
                camera_data: Optional[dict] = None) -> Optional[int]:
        if not self._recording or not scene:
            return None
        self._frame_counter += 1
        if self._frame_counter < self._capture_interval:
            return None
        self._frame_counter = 0
        try:
            data = scene.serialize()
        except Exception as e:
            Logger.error(f"Snapshot serialize failed: {e}")
            return None
        h = _compute_hash(data)
        if self._filter_unchanged and self._last_hash == h:
            return None
        self._last_hash = h
        ec = len(scene.get_all_entities())
        snap = FrameSnapshot(
            frame_number=self._total_frames,
            timestamp=_time.time(),
            entity_count=ec,
            data=data,
            frame_time_ms=frame_time_ms,
        )
        snap.hash = h
        snap.camera_data = camera_data
        if self._breakpoint_expr:
            try:
                bp_ns = {
                    "scene_data": data,
                    "entity_count": ec,
                    "frame": self._total_frames,
                    "time": snap.timestamp,
                    **self._breakpoint_ns,
                }
                result = eval(self._breakpoint_expr, {"__builtins__": {}}, bp_ns)
                if result:
                    snap.breakpoint_hit = True
                    Logger.info(f"Breakpoint hit at frame {self._total_frames}")
                    self._recording = False
            except Exception:
                pass
        self._buffer.append(snap)
        idx = len(self._buffer) - 1
        self._total_frames += 1
        return idx

    def get_frame(self, index: int) -> Optional[FrameSnapshot]:
        if 0 <= index < len(self._buffer):
            return self._buffer[index]
        return None

    def restore(self, registry: ComponentRegistry,
                index: int) -> Optional[Scene]:
        snap = self.get_frame(index)
        if snap is None:
            return None
        try:
            from core.ecs.ecs import Scene
            return Scene.deserialize(snap.data, registry)
        except Exception as e:
            Logger.error(f"Snapshot restore failed: {e}")
            return None

    def frames(self) -> list[FrameSnapshot]:
        return list(self._buffer)

    def toggle_bookmark(self, index: int) -> bool:
        snap = self.get_frame(index)
        if snap:
            snap.bookmarked = not snap.bookmarked
            return snap.bookmarked
        return False

    def bookmarked_indices(self) -> list[int]:
        return [i for i, s in enumerate(self._buffer) if s.bookmarked]

    def clear_bookmarks(self):
        for s in self._buffer:
            s.bookmarked = False

    def export_frame(self, index: int, path: str) -> bool:
        snap = self.get_frame(index)
        if not snap:
            return False
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(snap.data, f, indent=2)
            Logger.info(f"Frame {snap.frame_number} exported to {path}")
            return True
        except Exception as e:
            Logger.error(f"Export failed: {e}")
            return False

    def find_entity_changes(self, entity_id: str) -> list[int]:
        return find_entity_frame_changes(self.frames(), entity_id)
