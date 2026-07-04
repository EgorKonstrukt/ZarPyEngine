# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import threading
import time
from collections import deque

_perf_counter = time.perf_counter

_PROFILER_COLORS = {
    "frame": "#aaaaaa", "tick": "#ff5252", "render_scene": "#ff5252",
    "fixed_update": "#448aff", "update": "#448aff",
    "scene_update": "#69f0ae", "scene_fixed_update": "#69f0ae",
    "animation_ms": "#ffd740", "audio_ms": "#e040fb",
    "ai_ms": "#40c4ff", "particle_ms": "#ff6e40",
    "ui_ms": "#b2ff59", "load_mesh": "#ff8a65",
    "load_obj": "#ff8a65", "scene_start": "#69f0ae",
    "scene_serialize": "#69f0ae", "scene_deserialize": "#69f0ae",
    "render_ms": "#ff5252", "physics_ms": "#448aff",
    "PhysicsPlugin": "#448aff",
    "physics_scan_bodies": "#64b5f6", "physics_drain_results": "#42a5f5",
    "physics_snapshot": "#1e88e5", "physics_worker_send": "#1565c0",
    "physics_scene_step": "#0d47a1",
    "phys_register": "#81c784", "phys_sync_to_solver": "#66bb6a",
    "phys_step_sim": "#4caf50", "phys_constrain_2d": "#43a047",
    "phys_sync_to_ecs": "#388e3c", "phys_collision_events": "#2e7d32",
    "input_handling": "#7e57c2", "logic_update": "#42a5f5",
    "render_widget": "#66bb6a", "frame_overhead": "#78909c",
    "editor_particles": "#ab47bc", "collab_camera": "#26c6da",
    "cam_update": "#29b6f6", "gl_setup": "#ff8a65",
    "gizmo_lines": "#ffd740", "gizmo_wireframes": "#ffaa00",
    "gizmo_icons": "#ffab40", "gizmo_collab": "#ff6e40",
    "overlay_draw": "#40c4ff",
}


class ProfileSample:
    __slots__ = ('name', 'depth', 'start_ms', 'duration_ms', 'color')
    def __init__(self, name: str, depth: int, start_ms: float, duration_ms: float, color: str = "#aaaaaa"):
        self.name = name; self.depth = depth; self.start_ms = start_ms; self.duration_ms = duration_ms; self.color = color


class FrameProfile:
    __slots__ = ('samples', 'frame_time_ms', 'frame_number', 'flat_data')
    def __init__(self):
        self.samples: list[ProfileSample] = []
        self.frame_time_ms: float = 0.0
        self.frame_number: int = 0
        self.flat_data: dict[str, float] = {}


class Profiler:
    __slots__ = ('_frames', '_current_frame', '_stack', '_frame_start',
                 '_max_frames', '_flat_data', '_enabled', '_frame_number',
                 '_capture_frames', '_lock')

    def __init__(self, max_frames: int = 300):
        self._lock = threading.Lock()
        self._frames: deque[FrameProfile] = deque(maxlen=max_frames)
        self._current_frame: FrameProfile | None = None
        self._stack: list[tuple[str, float]] = []
        self._frame_start: float = 0.0
        self._max_frames = max_frames
        self._flat_data: dict[str, float] = {}
        self._enabled: bool = False
        self._frame_number: int = 0
        self._capture_frames: bool = False

    def start(self, key: str):
        if not self._enabled: return
        with self._lock:
            self._stack.append((key, _perf_counter()))

    def stop(self, key: str):
        if not self._enabled: return
        with self._lock:
            stack = self._stack
            if not stack: return
            if stack[-1][0] == key:
                name, t0 = stack.pop()
            else:
                for i in range(len(stack) - 1, -1, -1):
                    if stack[i][0] == key:
                        name, t0 = stack.pop(i)
                        break
                else:
                    return
            now = _perf_counter()
            duration_ms = (now - t0) * 1000.0
            self._flat_data[name] = self._flat_data.get(name, 0.0) + duration_ms
            if self._capture_frames:
                cf = self._current_frame
                if cf is not None and self._frame_start > 0:
                    start_ms = (t0 - self._frame_start) * 1000.0
                    depth = len(stack)
                    c = _PROFILER_COLORS.get(name, "#aaaaaa")
                    cf.samples.append(ProfileSample(name, depth, start_ms, duration_ms, c))
                    cf.flat_data[name] = cf.flat_data.get(name, 0.0) + duration_ms

    def set_value(self, key: str, value_ms: float):
        if not self._enabled: return
        with self._lock:
            self._flat_data[key] = value_ms
            if self._capture_frames:
                cf = self._current_frame
                if cf is not None:
                    cf.flat_data[key] = value_ms

    def capture_frame(self):
        if not self._enabled: return
        with self._lock:
            now = _perf_counter()
            cf = self._current_frame
            frame_start = self._frame_start
            if self._stack:
                for name, t0 in self._stack:
                    duration_ms = (now - t0) * 1000.0
                    self._flat_data[name] = self._flat_data.get(name, 0.0) + duration_ms
                    if self._capture_frames and cf is not None and frame_start > 0:
                        start_ms = (t0 - frame_start) * 1000.0
                        c = _PROFILER_COLORS.get(name, "#aaaaaa")
                        depth = len(self._stack)
                        cf.samples.append(ProfileSample(name, depth, start_ms, duration_ms, c))
                        cf.flat_data[name] = cf.flat_data.get(name, 0.0) + duration_ms
                self._stack.clear()
            if cf is not None:
                cf.frame_time_ms = (now - frame_start) * 1000.0
                cf.frame_number = self._frame_number
                self._frames.append(cf)
                self._frame_number += 1
            self._current_frame = FrameProfile() if self._capture_frames else None
            self._frame_start = now

    def reset(self):
        if not self._enabled: return
        with self._lock:
            self._frames.clear()
            self._current_frame = None
            self._stack.clear()
            self._flat_data.clear()
            self._frame_start = 0.0
            self._frame_number = 0

    @property
    def data(self) -> dict[str, float]:
        return dict(self._flat_data)

    @property
    def frames(self) -> list[FrameProfile]:
        return list(self._frames)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, v: bool):
        self._enabled = v
        if v:
            self._capture_frames = True

    @property
    def capture_frames(self) -> bool:
        return self._capture_frames

    @capture_frames.setter
    def capture_frames(self, v: bool):
        self._capture_frames = v
