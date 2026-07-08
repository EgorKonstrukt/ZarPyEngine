# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
from typing import Callable
from core.ecs import Component, ComponentRegistry
from core.math3d import Vec3
from core.components.inspector_meta import FieldType, InspectorField
from core.audio_system import AudioSystem, AudioSourceManager
from core.logger import Logger


@ComponentRegistry.register
class AudioSource(Component):
    _icon = "AudioSource.png"
    _gizmo_icon_color = (80, 220, 80)
    _gizmo_icon_label = "A"
    _gizmo_pass = "audio"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("clip_path", "Clip", FieldType.RESOURCE_PATH, file_filter="Audio (*.wav *.mp3 *.ogg)"),
            InspectorField("volume", "Volume", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("pitch", "Pitch", FieldType.FLOAT, min_val=-3.0, max_val=3.0, step=0.01),
            InspectorField("loop", "Loop", FieldType.BOOL),
            InspectorField("play_on_awake", "Play On Awake", FieldType.BOOL),
            InspectorField("spatial_blend", "Spatial Blend", FieldType.FLOAT, min_val=0.0, max_val=1.0, step=0.01),
            InspectorField("volume_rolloff", "Volume Rolloff", FieldType.CURVE),
            InspectorField("min_distance", "Min Distance", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=0.5, decimals=2),
            InspectorField("max_distance", "Max Distance", FieldType.FLOAT, min_val=0.0, max_val=10000.0, step=1.0, decimals=2),
            InspectorField("offset", "Offset (sec)", FieldType.FLOAT, min_val=0.0, max_val=3600.0, step=0.01, decimals=2),
            InspectorField("fade_in_time", "Fade In Time", FieldType.FLOAT, min_val=0.0, max_val=60.0, step=0.1, decimals=2),
            InspectorField("fade_out_time", "Fade Out Time", FieldType.FLOAT, min_val=0.0, max_val=60.0, step=0.1, decimals=2),
        ]

    def __init__(self):
        super().__init__()
        self.clip_path: str = ""
        self.volume: float = 1.0
        self.pitch: float = 1.0
        self.loop: bool = False
        self.play_on_awake: bool = False
        self.spatial_blend: float = 1.0
        self.volume_rolloff: list[list[float]] = [[0, 1], [1, 0]]
        self.min_distance: float = 1.0
        self.max_distance: float = 50.0
        self.offset: float = 0.0
        self.fade_in_time: float = 0.0
        self.fade_out_time: float = 0.0
        self._source_id: int = 0
        self._playing: bool = False
        self._fade_volume: float = 1.0
        self._fade_duration: float = 0.0
        self._fade_elapsed: float = 0.0
        self._fade_from: float = 1.0
        self._fade_to: float = 1.0
        self._fade_active: bool = False
        self._fade_stop_requested: bool = False
        self._callback_finished_fired: bool = False
        self._prev_pos: Vec3 | None = None
        self.on_finished: Callable[[], None] | None = None
        self.on_stopped: Callable[[], None] | None = None

    def on_start(self):
        if self.play_on_awake and self.clip_path and not self._playing:
            self.play()

    def on_enable(self):
        if self.clip_path and not self._playing:
            self.play()

    def on_disable(self):
        if self._source_id:
            self._stop_immediate()

    def on_destroy(self):
        if self._source_id:
            self._stop_immediate()

    @staticmethod
    def linear_preset() -> list[list[float]]:
        return [[0, 1], [1, 0]]

    @staticmethod
    def logarithmic_preset() -> list[list[float]]:
        keys = [[0.0, 1.0]]
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            keys.append([t, 1.0 / (1.0 + 2.0 * t)])
        keys.append([1.0, 0.0])
        return keys

    def fade_in(self, duration: float):
        self._fade_from = 0.0
        self._fade_to = 1.0
        self._fade_duration = duration
        self._fade_elapsed = 0.0
        self._fade_volume = 0.0
        self._fade_active = True

    def fade_out(self, duration: float):
        self._fade_from = self._fade_volume
        self._fade_to = 0.0
        self._fade_duration = duration
        self._fade_elapsed = 0.0
        self._fade_active = True

    def _process_fade(self, dt: float):
        if not self._fade_active:
            return
        self._fade_elapsed += dt
        t = min(self._fade_elapsed / self._fade_duration, 1.0) if self._fade_duration > 0 else 1.0
        self._fade_volume = self._fade_from + (self._fade_to - self._fade_from) * t
        if t >= 1.0:
            self._fade_volume = self._fade_to
            self._fade_active = False
            if self._fade_to == 0.0:
                self._stop_immediate()
                if self.on_stopped:
                    try:
                        self.on_stopped()
                    except Exception as e:
                        Logger.error(f"AudioSource.on_stopped error: {e}")

    def _stop_immediate(self):
        if self._source_id:
            mgr = AudioSourceManager.instance()
            if mgr:
                mgr.stop(self._source_id)
                self._source_id = 0
        self._playing = False
        self._fade_active = False
        self._fade_stop_requested = False

    def play(self):
        if not self.clip_path or self._playing: return
        audio_sys = AudioSystem.instance()
        if not audio_sys: return
        mgr = AudioSourceManager.instance()
        if not mgr: return
        source = mgr.play(
            clip_path=self.clip_path,
            loop=self.loop,
            volume=self.volume,
            pitch=self.pitch,
            spatial_blend=self.spatial_blend,
            min_distance=self.min_distance,
            max_distance=self.max_distance,
            volume_rolloff=self.volume_rolloff,
            offset=self.offset,
        )
        if source:
            self._source_id = source
            self._playing = True
            self._callback_finished_fired = False
            self._fade_stop_requested = False
            self._prev_pos = None
            if self.fade_in_time > 0:
                import openal as al
                al.alSourcef(self._source_id, al.AL_GAIN, 0.0)
                self.fade_in(self.fade_in_time)
            else:
                self._fade_volume = 1.0
                self._fade_active = False

    def stop(self):
        if not self._source_id or not self._playing: return
        if self.fade_out_time > 0 and not self._fade_stop_requested:
            self._fade_stop_requested = True
            self.fade_out(self.fade_out_time)
            return
        self._stop_immediate()
        if self.on_stopped:
            try:
                self.on_stopped()
            except Exception as e:
                Logger.error(f"AudioSource.on_stopped error: {e}")

    @property
    def is_playing(self) -> bool: return self._playing

    def on_update(self, dt: float):
        if not self._source_id or not self._playing: return
        self._process_fade(dt)
        try:
            import openal as al
            state = al.ctypes.c_int()
            al.alGetSourcei(self._source_id, al.AL_SOURCE_STATE, state)
            if state.value != al.AL_PLAYING and self.clip_path:
                was_looping = self.loop
                was_fade_requested = self._fade_stop_requested
                self._stop_immediate()
                if was_looping:
                    self.play()
                else:
                    if was_fade_requested:
                        if self.on_stopped:
                            try:
                                self.on_stopped()
                            except Exception as e:
                                Logger.error(f"AudioSource.on_stopped error: {e}")
                    else:
                        if self.on_finished and not self._callback_finished_fired:
                            self._callback_finished_fired = True
                            try:
                                self.on_finished()
                            except Exception as e:
                                Logger.error(f"AudioSource.on_finished error: {e}")
                return

            mgr = AudioSourceManager.instance()
            if not mgr: return

            tr = self.transform
            pos = (0.0, 0.0, 0.0)
            if tr:
                p = tr.position
                pos = (p.x, p.y, p.z)

            if self._prev_pos is not None and dt > 0:
                vel = (
                    (pos[0] - self._prev_pos[0]) / dt,
                    (pos[1] - self._prev_pos[1]) / dt,
                    (pos[2] - self._prev_pos[2]) / dt,
                )
            else:
                vel = (0.0, 0.0, 0.0)
            self._prev_pos = pos

            effective_volume = self.volume * self._fade_volume
            mgr.update_source(self._source_id, effective_volume, self.pitch, pos, self.spatial_blend, vel)
        except Exception as e:
            err_str = str(e)
            if "AL_INVALID_NAME" in err_str or "invalid name" in err_str.lower():
                self._source_id = 0
                self._playing = False
                self._fade_active = False
                self._fade_stop_requested = False
            else:
                Logger.error(f"AudioSource.on_update error: {e}")

    def gizmo_lines(self) -> list[tuple[Vec3, Vec3, list[float]]]:
        tr = self.transform
        if not tr:
            return []
        pos = tr.position
        min_dist = self.min_distance
        max_dist = self.max_distance
        lines: list[tuple[Vec3, Vec3, list[float]]] = []
        segments = 24
        min_color = [0.2, 1.0, 0.2, 0.5]
        max_color = [1.0, 0.7, 0.1, 0.4]
        if min_dist > 0.01:
            for axis_idx in range(3):
                pts = []
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    if axis_idx == 0:
                        pt = Vec3(0, math.cos(theta) * min_dist, math.sin(theta) * min_dist)
                    elif axis_idx == 1:
                        pt = Vec3(math.cos(theta) * min_dist, 0, math.sin(theta) * min_dist)
                    else:
                        pt = Vec3(math.cos(theta) * min_dist, math.sin(theta) * min_dist, 0)
                    pts.append(pos + pt)
                for i in range(segments):
                    lines.append((pts[i], pts[i + 1], min_color))
        if max_dist > 0.01 and max_dist > min_dist:
            for axis_idx in range(3):
                pts = []
                for i in range(segments + 1):
                    theta = 2.0 * math.pi * i / segments
                    if axis_idx == 0:
                        pt = Vec3(0, math.cos(theta) * max_dist, math.sin(theta) * max_dist)
                    elif axis_idx == 1:
                        pt = Vec3(math.cos(theta) * max_dist, 0, math.sin(theta) * max_dist)
                    else:
                        pt = Vec3(math.cos(theta) * max_dist, math.sin(theta) * max_dist, 0)
                    pts.append(pos + pt)
                for i in range(segments):
                    lines.append((pts[i], pts[i + 1], max_color))
        return lines

    def gizmo(self):
        lines = self.gizmo_lines()
        if not lines:
            return []
        from core.ecs import GizmoPrimitive, GizmoStyle
        inner = GizmoPrimitive.from_lines(lines, GizmoStyle(pulsating=True, glow=True, pulse_speed=1.5, pulse_min_alpha=0.4))
        return [inner]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "clip_path": self.clip_path, "volume": self.volume, "pitch": self.pitch,
            "loop": self.loop, "play_on_awake": self.play_on_awake,
            "spatial_blend": self.spatial_blend,
            "volume_rolloff": self.volume_rolloff or [[0, 1], [1, 0]],
            "min_distance": self.min_distance, "max_distance": self.max_distance,
            "offset": self.offset,
            "fade_in_time": self.fade_in_time,
            "fade_out_time": self.fade_out_time,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> AudioSource:
        a = cls()
        a.enabled = data.get("enabled", True)
        a.clip_path = data.get("clip_path", "")
        a.volume = data.get("volume", 1.0)
        a.pitch = data.get("pitch", 1.0)
        a.loop = data.get("loop", False)
        a.play_on_awake = data.get("play_on_awake", False)
        a.spatial_blend = data.get("spatial_blend", 1.0)
        a.volume_rolloff = data.get("volume_rolloff", [[0, 1], [1, 0]])
        a.min_distance = data.get("min_distance", 1.0)
        a.max_distance = data.get("max_distance", 50.0)
        a.offset = data.get("offset", 0.0)
        a.fade_in_time = data.get("fade_in_time", 0.0)
        a.fade_out_time = data.get("fade_out_time", 0.0)
        return a
