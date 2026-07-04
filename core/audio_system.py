# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
import subprocess
import io
import wave
from typing import Optional, Dict, Any, Tuple
from core.pool import audio as _get_audio_pool

try:
    import openal as al
    _openal_available = True
except Exception:
    _openal_available = False
    al = None


class AudioRolloffCurve:
    @staticmethod
    def evaluate(distance: float, min_dist: float, max_dist: float,
                 curve_data: list[list[float]]) -> float:
        if not curve_data or min_dist >= max_dist or max_dist <= 0:
            return 1.0
        if distance <= min_dist:
            return 1.0
        if distance >= max_dist:
            return 0.0

        t = (distance - min_dist) / (max_dist - min_dist)
        keys = sorted(curve_data, key=lambda k: k[0])

        if t <= keys[0][0]:
            return keys[0][1]
        if t >= keys[-1][0]:
            return keys[-1][1]

        for i in range(len(keys) - 1):
            t0, v0 = keys[i]
            t1, v1 = keys[i + 1]
            if t0 <= t <= t1:
                dt = t1 - t0
                if dt == 0:
                    return v0
                return v0 + (v1 - v0) * (t - t0) / dt

        return 1.0


class AudioClip:
    def __init__(self):
        self._sample_rate: int = 0
        self._channels: int = 0
        self._data: memoryview | None = None
        self._format: int = 0
        self._buffer: Optional[al.Buffer] = None
        self._source: Optional[Any] = None

    @property
    def sample_rate(self) -> int: return self._sample_rate
    @property
    def channels(self) -> int: return self._channels
    @property
    def data(self): return self._data
    @property
    def buffer(self): return self._buffer

    def _detect_format(self, num_channels: int, sample_width: int) -> int:
        if num_channels == 1:
            if sample_width == 2: return al.AL_FORMAT_MONO16
        elif num_channels == 2:
            if sample_width == 2: return al.AL_FORMAT_STEREO16
        raise ValueError(f"Unsupported audio format: {num_channels}ch, {sample_width}bytes")

    def load_from_file(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".ogg", ".vorbis"):
            self._load_ogg(path)
        elif ext in (".wav", ".wave"):
            self._load_wav(path)
        elif ext == ".mp3":
            self._load_mp3(path)
        else:
            raise ValueError(f"Unsupported audio format: {ext}")

    def _load_wav(self, path: str):
        with wave.open(path, "rb") as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            raw_data = wf.readframes(frames)

        self._sample_rate = sample_rate
        self._channels = num_channels
        self._format = self._detect_format(num_channels, sample_width)
        self._data = memoryview(bytearray(raw_data))

    def _load_ogg(self, path: str):
        al.oalInit()
        source = al.oalOpen(path)
        if not source:
            raise RuntimeError(f"Failed to load OGG file '{path}'")
        self._source = source
        self._buffer = source.buffer

    def _load_mp3(self, path: str):
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError("pydub is required for MP3 support")
        segment = AudioSegment.from_mp3(path)
        raw_data = segment.raw_data
        self._sample_rate = segment.frame_rate
        self._channels = segment.channels
        self._format = self._detect_format(segment.channels, segment.sample_width)
        self._data = memoryview(bytearray(raw_data))

    def create_buffer(self):
        if not self._data: return
        al.oalInit()
        self._buffer = al.Buffer(self._format, self._data.tobytes(), len(self._data), self._sample_rate)

    def destroy_buffer(self):
        if self._source:
            try:
                self._source.destroy()
            except Exception:
                pass
            self._source = None
        elif self._buffer:
            try:
                self._buffer.destroy()
            except Exception:
                pass
            self._buffer = None


class AudioSystem:
    _instance: Optional[AudioSystem] = None

    def __init__(self):
        AudioSystem._instance = self
        self._clips: Dict[str, AudioClip] = {}
        self._initialized: bool = False
        self._listener_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def instance(cls) -> Optional[AudioSystem]: return cls._instance

    def initialize(self):
        if not _openal_available:
            from core.logger import Logger
            Logger.warning("OpenAL not available, audio disabled.")
            self._initialized = False
            return
        try:
            al.oalInit()
            if not al.oalGetInit():
                raise RuntimeError("OpenAL oalInit returned not initialized")
            al.alDistanceModel(al.AL_INVERSE_DISTANCE_CLAMPED)
            al.alDopplerFactor(1.0)
            al.alSpeedOfSound(343.3)
            AudioSourceManager()
            self._initialized = True
        except Exception as e:
            from core.logger import Logger
            Logger.error(f"OpenAL initialization failed: {e}")
            self._initialized = False

    def shutdown(self):
        if not _openal_available or not al:
            return
        mgr = AudioSourceManager.instance()
        if mgr:
            for src in list(mgr._active_sources.keys()):
                try:
                    al.alSourceStop(src)
                    al.alDeleteSources(1, al.ctypes.pointer(al.ctypes.c_uint(src)))
                except Exception:
                    pass
            mgr._active_sources.clear()
            mgr._source_info.clear()
        for clip in list(self._clips.values()):
            if clip.buffer:
                try:
                    clip.buffer.destroy()
                except Exception:
                    pass
        self._clips.clear()
        try:
            al._buffers.clear()
            al.oalQuit()
        except Exception:
            pass
        self._initialized = False

    def load_clip(self, path: str) -> AudioClip:
        abs_path = os.path.abspath(path)
        if abs_path in self._clips:
            return self._clips[abs_path]
        clip = AudioClip()
        try:
            clip.load_from_file(abs_path)
            clip.create_buffer()
            self._clips[abs_path] = clip
        except Exception as e:
            from core.logger import Logger
            Logger.error(f"Failed to load audio clip '{path}': {e}")
        return clip

    def load_clip_async(self, path: str, callback):
        abs_path = os.path.abspath(path)
        if abs_path in self._clips:
            callback(self._clips[abs_path])
            return
        def _load():
            clip = AudioClip()
            try:
                clip.load_from_file(abs_path)
                clip.create_buffer()
                self._clips[abs_path] = clip
            except Exception as e:
                from core.logger import Logger
                Logger.error(f"Failed to load audio clip '{path}': {e}")
                clip = None
            callback(clip)
        _get_audio_pool().submit(_load)

    def get_clip(self, path: str) -> Optional[AudioClip]:
        abs_path = os.path.abspath(path)
        return self._clips.get(abs_path)

    def set_listener_position(self, pos: tuple[float, float, float]):
        self._listener_pos = pos
        mgr = AudioSourceManager.instance()
        if mgr:
            mgr.update_listener_position(pos)
        if not self._initialized or not al.oalGetInit(): return
        al.alListener3f(al.AL_POSITION, *pos)

    def set_listener_velocity(self, vel: tuple[float, float, float]):
        if not self._initialized or not al.oalGetInit(): return
        al.alListener3f(al.AL_VELOCITY, *vel)

    def set_listener_orientation(self, at: tuple[float, float, float], up: tuple[float, float, float]):
        if not self._initialized or not al.oalGetInit(): return
        arr = (al.ALfloat * 6)(at[0], at[1], at[2], up[0], up[1], up[2])
        al.alListenerfv(al.AL_ORIENTATION, arr)

    def set_master_volume(self, volume: float):
        if not self._initialized or not al.oalGetInit(): return
        al.alListenerf(al.AL_GAIN, volume)

    def set_doppler_factor(self, factor: float):
        if not self._initialized or not al.oalGetInit(): return
        al.alDopplerFactor(factor)

    def set_speed_of_sound(self, speed: float):
        if not self._initialized or not al.oalGetInit(): return
        al.alSpeedOfSound(speed)


_AL_AUX_SEND_FILTER = None


def _get_aux_send_filter() -> int:
    global _AL_AUX_SEND_FILTER
    if _AL_AUX_SEND_FILTER is None:
        _AL_AUX_SEND_FILTER = al.alGetEnumValue(b"AL_AUXILIARY_SEND_FILTER") if al else 0x1604
    return _AL_AUX_SEND_FILTER


class AudioSourceManager:
    _instance: Optional[AudioSourceManager] = None

    def __init__(self):
        AudioSourceManager._instance = self
        self._active_sources: dict[int, int] = {}
        self._source_info: dict[int, dict] = {}
        self._source_positions: dict[int, tuple[float, float, float]] = {}
        self._source_aux_slot: dict[int, int] = {}
        self._listener_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def instance(cls) -> Optional[AudioSourceManager]: return cls._instance

    def _distance_to_listener(self, position: Tuple[float, float, float]) -> float:
        dx = position[0] - self._listener_pos[0]
        dy = position[1] - self._listener_pos[1]
        dz = position[2] - self._listener_pos[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def play(self, clip_path: str, loop: bool = False, volume: float = 1.0, pitch: float = 1.0,
             spatial_blend: float = 1.0, min_distance: float = 1.0, max_distance: float = 50.0,
             volume_rolloff: list[list[float]] = None, offset: float = 0.0,
             velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> int | None:
        if not al.oalGetInit(): return None
        audio_sys = AudioSystem.instance()
        if not audio_sys: return None

        clip = audio_sys.load_clip(clip_path)
        if not clip or not clip.buffer: return None

        source_id = al.ctypes.c_uint()
        al.alGenSources(1, al.ctypes.pointer(source_id))
        src_val = source_id.value

        al.alSourcei(src_val, al.AL_BUFFER, clip.buffer._geti())
        al.alSourcef(src_val, al.AL_PITCH, pitch)
        al.alSourcef(src_val, al.AL_GAIN, volume)
        al.alSourcei(src_val, al.AL_LOOPING, 1 if loop else 0)

        try:
            al.alSourcei(src_val, 0x1214, 0x0001)
        except Exception:
            pass

        al.alSource3f(src_val, al.AL_POSITION, 0.0, 0.0, 0.0)
        al.alSource3f(src_val, al.AL_VELOCITY, *velocity)
        al.alSourcef(src_val, al.AL_REFERENCE_DISTANCE, min_distance)
        al.alSourcef(src_val, al.AL_MAX_DISTANCE, max_distance)
        al.alSourcef(src_val, al.AL_ROLLOFF_FACTOR, 0.0)
        al.alSourcei(src_val, al.AL_SOURCE_RELATIVE, 0 if spatial_blend > 0 else 1)

        if offset > 0.0:
            al.alSourcef(src_val, 0x1024, offset)

        self._active_sources[src_val] = src_val
        self._source_info[src_val] = {
            "min_distance": min_distance,
            "max_distance": max_distance,
            "spatial_blend": spatial_blend,
            "volume_rolloff": volume_rolloff or [[0, 1], [1, 0]],
            "offset": offset,
            "velocity": velocity,
        }
        self._source_positions[src_val] = (0.0, 0.0, 0.0)
        self._source_aux_slot[src_val] = 0
        al.alSourcePlay(src_val)
        return src_val

    def update_source(self, source: int, volume: float, pitch: float, position: tuple[float, float, float],
                      spatial_blend: float | None = None,
                      velocity: tuple[float, float, float] | None = None):
        if not source or not al.oalGetInit(): return
        try:
            state = al.ctypes.c_int()
            al.alGetSourcei(source, al.AL_SOURCE_STATE, state)
            if state.value in (al.AL_PLAYING, al.AL_PAUSED):
                al.alSourcef(source, al.AL_PITCH, pitch)

                if velocity is not None:
                    al.alSource3f(source, al.AL_VELOCITY, *velocity)
                    info = self._source_info.get(source)
                    if info:
                        info["velocity"] = velocity

                info = self._source_info.get(source)
                if spatial_blend is not None and info:
                    info["spatial_blend"] = spatial_blend
                    if spatial_blend > 0:
                        al.alSource3f(source, al.AL_POSITION, *position)
                        al.alSourcei(source, al.AL_SOURCE_RELATIVE, 0)
                        self._source_positions[source] = position
                        dist = self._distance_to_listener(position)
                        atten = AudioRolloffCurve.evaluate(
                            dist,
                            info["min_distance"],
                            info["max_distance"],
                            info.get("volume_rolloff", [[0, 1], [1, 0]])
                        )
                        final_volume = volume * atten
                        al.alSourcef(source, al.AL_GAIN, final_volume)
                    else:
                        al.alSource3f(source, al.AL_POSITION, 0.0, 0.0, 0.0)
                        al.alSourcei(source, al.AL_SOURCE_RELATIVE, 1)
                        al.alSourcef(source, al.AL_GAIN, volume)
                else:
                    al.alSource3f(source, al.AL_POSITION, *position)
                    al.alSourcef(source, al.AL_GAIN, volume)
        except Exception as e:
            from core.logger import Logger
            Logger.error(f"update_source error: {e}")

    def update_listener_position(self, pos: Tuple[float, float, float]):
        self._listener_pos = pos

    def get_active_source_ids(self) -> list[int]:
        return list(self._active_sources.keys())

    def get_active_sound_count(self) -> int:
        if not _openal_available or not al.oalGetInit():
            return 0
        count = 0
        for src in list(self._active_sources.keys()):
            try:
                state = al.ctypes.c_int()
                al.alGetSourcei(src, al.AL_SOURCE_STATE, state)
                if state.value == al.AL_PLAYING:
                    count += 1
            except Exception:
                pass
        return count

    def get_total_sound_count(self) -> int:
        return len(self._active_sources)

    def get_dsp_load(self) -> float:
        return 0.0

    def get_source_position(self, source_id: int) -> Optional[tuple[float, float, float]]:
        return self._source_positions.get(source_id)

    def get_source_aux_slot(self, source_id: int) -> int:
        return self._source_aux_slot.get(source_id, 0)

    def set_source_aux_send(self, source_id: int, slot_id: int):
        if not source_id or not al.oalGetInit():
            return
        current = self._source_aux_slot.get(source_id, 0)
        if current == slot_id:
            return
        try:
            filter_enum = _get_aux_send_filter()
            al.alSource3i(source_id, filter_enum, slot_id, 0, 0)
            self._source_aux_slot[source_id] = slot_id
        except Exception:
            pass

    def pause(self, source: int):
        if not source or not al.oalGetInit(): return
        try:
            state = al.ctypes.c_int()
            al.alGetSourcei(source, al.AL_SOURCE_STATE, state)
            if state.value == al.AL_PLAYING:
                al.alSourcePause(source)
        except Exception:
            pass

    def resume(self, source: int):
        if not source or not al.oalGetInit(): return
        try:
            state = al.ctypes.c_int()
            al.alGetSourcei(source, al.AL_SOURCE_STATE, state)
            if state.value == al.AL_PAUSED:
                al.alSourcePlay(source)
        except Exception:
            pass

    def stop(self, source: int):
        if not source or not al.oalGetInit(): return
        try:
            state = al.ctypes.c_int()
            al.alGetSourcei(source, al.AL_SOURCE_STATE, state)
            if state.value in (al.AL_PLAYING, al.AL_PAUSED):
                al.alSourceStop(source)
            al.alDeleteSources(1, al.ctypes.pointer(al.ctypes.c_uint(source)))
        except Exception:
            pass
        self._active_sources.pop(source, None)
        self._source_info.pop(source, None)
        self._source_positions.pop(source, None)
        self._source_aux_slot.pop(source, None)

    def stop_all(self):
        for src in list(self._active_sources.keys()):
            try:
                state = al.ctypes.c_int()
                al.alGetSourcei(src, al.AL_SOURCE_STATE, state)
                if state.value in (al.AL_PLAYING, al.AL_PAUSED):
                    al.alSourceStop(src)
                al.alDeleteSources(1, al.ctypes.pointer(al.ctypes.c_uint(src)))
            except Exception:
                pass
        self._active_sources.clear()
        self._source_info.clear()
        self._source_positions.clear()
        self._source_aux_slot.clear()
