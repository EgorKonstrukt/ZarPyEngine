# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import math
import os
import numpy as np
from typing import Optional, Dict, Any, Tuple
from core.ecs.pool import audio as _get_audio_pool
from core.foundation.progress import task_complete, task_start
from core.audio.miniaudio_decoder import (
    decode_audio, decode_audio_stereo, decode_audio_channels,
    miniaudio_available, AUDIO_EXTENSIONS,
    _detect_openal_format,
)

try:
    import openal as al
    from openal import alc
    _openal_available = True
except Exception:
    _openal_available = False
    al = None
    alc = None

_DISTANCE_MODEL_MAP = {
    "none": al.AL_NONE if al else 0,
    "inverse_distance": al.AL_INVERSE_DISTANCE if al else 0xD001,
    "inverse_distance_clamped": al.AL_INVERSE_DISTANCE_CLAMPED if al else 0xD002,
    "linear_distance": al.AL_LINEAR_DISTANCE if al else 0xD003,
    "linear_distance_clamped": al.AL_LINEAR_DISTANCE_CLAMPED if al else 0xD004,
    "exponent_distance": al.AL_EXPONENT_DISTANCE if al else 0xD005,
    "exponent_distance_clamped": al.AL_EXPONENT_DISTANCE_CLAMPED if al else 0xD006,
}

_AUDIO_CATEGORIES = ("master", "sfx", "music", "voice", "ambient")


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
        self._path: str = ""
        self._pcm_mono: Any = None
        self._pcm_stereo: Any = None
        self._vis_rate: int = 0

    @property
    def sample_rate(self) -> int: return self._sample_rate
    @property
    def channels(self) -> int: return self._channels
    @property
    def data(self): return self._data
    @property
    def buffer(self): return self._buffer

    def ensure_pcm(self):
        if self._pcm_mono is not None:
            return self._pcm_mono
        if self._data is not None:
            raw = np.frombuffer(self._data, dtype=np.int16)
            pcm = raw.astype(np.float32) / 32768.0
            if self._channels == 2:
                if pcm.size % 2:
                    pcm = pcm[:pcm.size - pcm.size % 2]
                pcm = pcm.reshape(-1, 2).mean(axis=1)
            self._pcm_mono = np.ascontiguousarray(pcm, dtype=np.float32)
            self._vis_rate = self._sample_rate
            return self._pcm_mono
        if self._path:
            ext = os.path.splitext(self._path)[1].lower()
            if ext in (".wav", ".wave"):
                try:
                    import wave
                    with wave.open(self._path, "rb") as wf:
                        nch = wf.getnchannels()
                        sw = wf.getsampwidth()
                        sr = wf.getframerate()
                        raw = wf.readframes(wf.getnframes())
                    if sw == 2:
                        d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sw == 4:
                        d = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        return None
                    if nch > 1:
                        d = d.reshape(-1, nch).mean(axis=1)
                    self._sample_rate = sr
                    self._channels = nch
                    self._pcm_mono = np.ascontiguousarray(d, dtype=np.float32)
                    self._vis_rate = sr
                    return self._pcm_mono
                except Exception:
                    return None
            if miniaudio_available():
                decoded = decode_audio(self._path)
                if decoded is not None:
                    raw = np.frombuffer(decoded.pcm_int16, dtype=np.int16)
                    pcm = raw.astype(np.float32) / 32768.0
                    self._sample_rate = decoded.sample_rate
                    self._channels = decoded.channels
                    self._pcm_mono = np.ascontiguousarray(pcm, dtype=np.float32)
                    self._vis_rate = decoded.sample_rate
                    return self._pcm_mono
        return None

    def ensure_stereo_pcm(self):
        if self._pcm_stereo is not None:
            return self._pcm_stereo
        if self._data is not None:
            raw = np.frombuffer(self._data, dtype=np.int16)
            pcm = raw.astype(np.float32) / 32768.0
            if self._channels == 2:
                if pcm.size % 2:
                    pcm = pcm[:pcm.size - pcm.size % 2]
                pcm = pcm.reshape(-1, 2)
            else:
                pcm = np.column_stack([pcm, pcm])
            self._pcm_stereo = np.ascontiguousarray(pcm, dtype=np.float32)
            self._vis_rate = self._sample_rate
            return self._pcm_stereo
        if self._path:
            ext = os.path.splitext(self._path)[1].lower()
            if ext in (".wav", ".wave"):
                try:
                    import wave
                    with wave.open(self._path, "rb") as wf:
                        nch = wf.getnchannels()
                        sw = wf.getsampwidth()
                        sr = wf.getframerate()
                        raw = wf.readframes(wf.getnframes())
                    if sw == 2:
                        d = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    elif sw == 4:
                        d = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
                    else:
                        return None
                    if nch >= 2:
                        d = d.reshape(-1, nch)[:, :2]
                    else:
                        d = np.column_stack([d, d])
                    self._sample_rate = sr
                    self._channels = nch
                    self._pcm_stereo = np.ascontiguousarray(d, dtype=np.float32)
                    self._vis_rate = sr
                    return self._pcm_stereo
                except Exception:
                    return None
            if miniaudio_available():
                decoded = decode_audio_stereo(self._path)
                if decoded is not None:
                    raw = np.frombuffer(decoded.pcm_int16, dtype=np.int16)
                    pcm = raw.astype(np.float32) / 32768.0
                    if decoded.channels == 2:
                        if pcm.size % 2:
                            pcm = pcm[:pcm.size - pcm.size % 2]
                        pcm = pcm.reshape(-1, 2)
                    else:
                        pcm = np.column_stack([pcm, pcm])
                    self._sample_rate = decoded.sample_rate
                    self._channels = decoded.channels
                    self._pcm_stereo = np.ascontiguousarray(pcm, dtype=np.float32)
                    self._vis_rate = decoded.sample_rate
                    return self._pcm_stereo
        return None

    def load_from_file(self, path: str):
        import json
        quality = None
        self._path = path
        import_path = path + ".import"
        if os.path.exists(import_path):
            try:
                with open(import_path) as f:
                    settings = json.load(f)
                quality = settings.get("quality")
            except Exception:
                pass

        ext = os.path.splitext(path)[1].lower()
        if ext not in AUDIO_EXTENSIONS:
            raise ValueError(f"Unsupported audio format: {ext}")

        if ext in (".wav", ".wave"):
            self._load_wav(path)
        elif miniaudio_available():
            self._load_with_miniaudio(path)
        else:
            raise RuntimeError(
                f"miniaudio not available; cannot load '{ext}' files. "
                "Install miniaudio: pip install miniaudio"
            )

        if quality is not None:
            self._resample(quality)

    def _load_wav(self, path: str):
        import wave
        with wave.open(path, "rb") as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            raw_data = wf.readframes(frames)
        self._sample_rate = sample_rate
        self._channels = num_channels
        self._format = _detect_openal_format(num_channels, sample_width)
        self._data = memoryview(bytearray(raw_data))

    def _load_with_miniaudio(self, path: str):
        decoded = decode_audio_channels(path, 1)
        if decoded is None:
            raise RuntimeError(f"Failed to decode audio file '{path}' with miniaudio")
        self._sample_rate = decoded.sample_rate
        self._channels = decoded.channels
        self._format = _detect_openal_format(decoded.channels, 2)
        self._data = memoryview(bytearray(decoded.pcm_int16))

    def _resample(self, quality: int):
        if quality >= 100 or self._data is None:
            return
        old_rate = self._sample_rate
        new_rate = max(8000, int(old_rate * quality / 100))
        if new_rate >= old_rate:
            return
        ratio = old_rate / new_rate
        raw = np.frombuffer(self._data, dtype=np.int16)
        if self._channels == 2:
            raw = raw.reshape(-1, 2)
        n = len(raw)
        new_n = max(1, int(n / ratio))
        indices = np.linspace(0, n - 1, new_n)
        if self._channels == 2:
            resampled = np.column_stack([
                np.interp(indices, np.arange(n), raw[:, 0]),
                np.interp(indices, np.arange(n), raw[:, 1])
            ]).astype(np.int16)
        else:
            resampled = np.interp(indices, np.arange(n), raw).astype(np.int16)
        self._data = memoryview(resampled.tobytes())
        self._sample_rate = new_rate

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
        self._device_name: str = ""

    @classmethod
    def instance(cls) -> Optional[AudioSystem]: return cls._instance

    def initialize(self):
        if not _openal_available:
            from core.foundation.logger import Logger
            Logger.warning("OpenAL not available, audio disabled.")
            self._initialized = False
            return
        try:
            al.oalInit()
            if not al.oalGetInit():
                raise RuntimeError("OpenAL oalInit returned not initialized")
            try:
                from core.audio.audio_efx import invalidate_efx_cache, ensure_efx
                invalidate_efx_cache()
                ensure_efx()
            except Exception:
                pass

            try:
                default_name = alc.alcGetString(None, alc.ALC_DEFAULT_DEVICE_SPECIFIER)
                if default_name:
                    self._device_name = default_name.decode("utf-8", errors="replace") if isinstance(default_name, bytes) else str(default_name)
                else:
                    self._device_name = "System Default"
            except Exception:
                self._device_name = "System Default"

            from core.foundation.logger import Logger
            Logger.info(f"Audio device: {self._device_name}")

            al.alDistanceModel(al.AL_INVERSE_DISTANCE_CLAMPED)
            al.alDopplerFactor(1.0)
            al.alSpeedOfSound(343.3)

            AudioSourceManager()
            self._initialized = True
        except Exception as e:
            from core.foundation.logger import Logger
            Logger.error(f"OpenAL initialization failed: {e}")
            self._initialized = False

    def shutdown(self):
        if not _openal_available or not al.oalGetInit():
            try:
                from core.audio.audio_efx import invalidate_efx_cache
                invalidate_efx_cache()
            except Exception:
                pass
            self._initialized = False
            return
        mgr = AudioSourceManager.instance()
        if mgr:
            try:
                mgr.stop_all()
            except Exception:
                pass
            try:
                mgr._active_sources.clear()
                mgr._source_info.clear()
                mgr._source_positions.clear()
                mgr._source_clips.clear()
                mgr._source_gains.clear()
                if hasattr(mgr, "_source_aux_slot"):
                    mgr._source_aux_slot.clear()
                if hasattr(mgr, "_source_aux_filter"):
                    try:
                        from core.audio.audio_efx import delete_filter
                        for _fid in list(mgr._source_aux_filter.values()):
                            try:
                                delete_filter(int(_fid))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    mgr._source_aux_filter.clear()
                if hasattr(mgr, "_source_aux_gain"):
                    mgr._source_aux_gain.clear()
                if hasattr(mgr, "_source_direct_filter"):
                    try:
                        from core.audio.audio_efx import delete_filter as _del_f
                        for _fid in list(mgr._source_direct_filter.values()):
                            try:
                                _del_f(int(_fid))
                            except Exception:
                                pass
                    except Exception:
                        pass
                    mgr._source_direct_filter.clear()
            except Exception:
                pass
        for _, clip in list(self._clips.values()):
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
        try:
            from core.audio.audio_efx import invalidate_efx_cache as _inv2
            _inv2()
        except Exception:
            pass
        self._initialized = False

    @property
    def device_name(self) -> str:
        return self._device_name

    def _import_mtime(self, path: str) -> float:
        import_path = path + ".import"
        try:
            return os.path.getmtime(import_path)
        except OSError:
            return 0.0

    def load_clip(self, path: str) -> AudioClip:
        abs_path = os.path.abspath(path)
        imtime = self._import_mtime(abs_path)
        cached = self._clips.get(abs_path)
        if cached is not None:
            c_mtime, clip = cached
            if abs(imtime - c_mtime) < 0.001:
                return clip
            try:
                clip.destroy_buffer()
            except Exception:
                pass
            self._clips.pop(abs_path, None)
        task_id = "audio_load:" + abs_path
        try:
            file_size = os.path.getsize(abs_path)
        except OSError:
            file_size = 0
        task_start(task_id, f"Loading audio {os.path.basename(abs_path)}...",
                   total=float(file_size) if file_size else None, units="bytes")
        try:
            clip = AudioClip()
            try:
                clip.load_from_file(abs_path)
                clip.create_buffer()
                self._clips[abs_path] = (imtime, clip)
            except Exception as e:
                from core.foundation.logger import Logger
                Logger.error(f"Failed to load audio clip '{path}': {e}")
            return clip
        finally:
            task_complete(task_id)

    def load_clip_async(self, path: str, callback):
        abs_path = os.path.abspath(path)
        imtime = self._import_mtime(abs_path)
        cached = self._clips.get(abs_path)
        if cached is not None:
            c_mtime, clip = cached
            if abs(imtime - c_mtime) < 0.001:
                callback(clip)
                return
            try:
                clip.destroy_buffer()
            except Exception:
                pass
            self._clips.pop(abs_path, None)
        task_id = "audio_load:" + abs_path
        try:
            file_size = os.path.getsize(abs_path)
        except OSError:
            file_size = 0
        task_start(task_id, f"Loading audio {os.path.basename(abs_path)}...",
                   total=float(file_size) if file_size else None, units="bytes")
        def _load():
            clip = AudioClip()
            try:
                clip.load_from_file(abs_path)
                clip.create_buffer()
                self._clips[abs_path] = (imtime, clip)
            except Exception as e:
                from core.foundation.logger import Logger
                Logger.error(f"Failed to load audio clip '{path}': {e}")
                clip = None
                self._clips.pop(abs_path, None)
            task_complete(task_id)
            callback(clip)
        _get_audio_pool().submit(_load)

    def get_clip(self, path: str) -> Optional[AudioClip]:
        abs_path = os.path.abspath(path)
        cached = self._clips.get(abs_path)
        if cached is not None:
            return cached[1]
        return None

    def set_listener_position(self, pos: tuple[float, float, float]):
        self._listener_pos = pos
        mgr = AudioSourceManager.instance()
        if mgr:
            mgr.update_listener_position(pos)
        if not self._initialized: return
        al.alListener3f(al.AL_POSITION, *pos)

    def set_listener_velocity(self, vel: tuple[float, float, float]):
        if not self._initialized: return
        al.alListener3f(al.AL_VELOCITY, *vel)

    def set_listener_orientation(self, at: tuple[float, float, float], up: tuple[float, float, float]):
        if not self._initialized: return
        arr = (al.ALfloat * 6)(at[0], at[1], at[2], up[0], up[1], up[2])
        al.alListenerfv(al.AL_ORIENTATION, arr)

    def set_master_volume(self, volume: float):
        if not self._initialized: return
        al.alListenerf(al.AL_GAIN, max(0.0, min(1.0, volume)))

    def set_doppler_factor(self, factor: float):
        if not self._initialized: return
        al.alDopplerFactor(factor)

    def set_speed_of_sound(self, speed: float):
        if not self._initialized: return
        al.alSpeedOfSound(speed)

    def set_distance_model(self, model: str):
        if not self._initialized: return
        al.alDistanceModel(_DISTANCE_MODEL_MAP.get(model, al.AL_INVERSE_DISTANCE_CLAMPED))

    def set_category_volume(self, category: str, volume: float):
        if category not in _AUDIO_CATEGORIES:
            return
        from core.config.config import get_global_config
        cfg = get_global_config()
        cfg.set(f"audio.{category}_volume", max(0.0, min(1.0, volume)), notify=False)
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng and eng._project_path:
                from core.config.config import get_project_config
                pcfg = get_project_config(eng._project_path)
                pcfg.set(f"audio.{category}_volume", max(0.0, min(1.0, volume)), notify=False)
        except Exception:
            pass

    def get_category_volume(self, category: str) -> float:
        if category not in _AUDIO_CATEGORIES:
            return 1.0
        from core.config.config import get_global_config
        cfg = get_global_config()
        global_val = cfg.get(f"audio.{category}_volume", 1.0)
        if category == "master":
            return global_val
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if eng and eng._project_path:
                from core.config.config import get_project_config
                pcfg = get_project_config(eng._project_path)
                proj_val = pcfg.get(f"audio.{category}_volume", 1.0)
                return global_val * proj_val
        except Exception:
            pass
        return global_val

    def get_effective_volume(self, category: str, base_volume: float) -> float:
        return base_volume * self.get_category_volume(category) * self.get_category_volume("master")

    def apply_project_audio_config(self):
        if not self._initialized:
            return
        try:
            from core.engine.engine import Engine
            eng = Engine.instance()
            if not eng or not eng._project_path:
                return
            from core.config.config import get_project_config
            pcfg = get_project_config(eng._project_path)
        except Exception:
            return
        distance_model = pcfg.get("audio.distance_model", "inverse_distance_clamped")
        al.alDistanceModel(_DISTANCE_MODEL_MAP.get(distance_model, al.AL_INVERSE_DISTANCE_CLAMPED))
        al.alDopplerFactor(pcfg.get("audio.doppler_factor", 1.0))
        al.alSpeedOfSound(pcfg.get("audio.speed_of_sound", 343.3))

    @staticmethod
    def get_available_devices() -> list[str]:
        if not alc:
            return []
        devices = []
        try:
            raw = alc.alcGetString(None, alc.ALC_DEVICE_SPECIFIER)
            if raw:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                for dev in raw.split("\x00"):
                    if dev.strip():
                        devices.append(dev.strip())
        except Exception:
            pass
        return devices

    @staticmethod
    def get_default_device_name() -> str:
        if not alc:
            return ""
        try:
            raw = alc.alcGetString(None, alc.ALC_DEFAULT_DEVICE_SPECIFIER)
            if raw:
                return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
        except Exception:
            pass
        return ""


_AL_AUX_SEND_FILTER = None


def _get_aux_send_filter() -> int:
    global _AL_AUX_SEND_FILTER
    if _AL_AUX_SEND_FILTER is None:
        try:
            from core.audio.audio_efx import get_aux_send_enum
            _AL_AUX_SEND_FILTER = int(get_aux_send_enum())
        except Exception:
            _AL_AUX_SEND_FILTER = al.alGetEnumValue(b"AL_AUXILIARY_SEND_FILTER") if al else 0x20006
    return int(_AL_AUX_SEND_FILTER)


def _read_audio_value(key: str, default):
    try:
        from core.engine.engine import Engine
        eng = Engine.instance()
        if eng is not None and getattr(eng, "_project_path", None):
            try:
                from core.config.config import get_project_config
                pcfg = get_project_config(eng._project_path)
                value = pcfg.get(key, None)
                if value is not None:
                    return value
            except Exception:
                pass
    except Exception:
        pass
    try:
        from core.config.config import get_global_config
        cfg = get_global_config()
        return cfg.get(key, default)
    except Exception:
        return default


class AudioSourceManager:
    _instance: Optional[AudioSourceManager] = None

    def __init__(self):
        AudioSourceManager._instance = self
        self._active_sources: dict[int, int] = {}
        self._source_info: dict[int, dict] = {}
        self._source_positions: dict[int, tuple[float, float, float]] = {}
        self._source_aux_slot: dict[int, int] = {}
        self._source_aux_filter: dict[int, int] = {}
        self._source_aux_gain: dict[int, float] = {}
        self._source_direct_filter: dict[int, int] = {}
        self._source_clips: dict[int, AudioClip] = {}
        self._source_gains: dict[int, float] = {}
        self._listener_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @classmethod
    def instance(cls) -> Optional[AudioSourceManager]: return cls._instance

    def _distance_to_listener(self, position: Tuple[float, float, float]) -> float:
        dx = position[0] - self._listener_pos[0]
        dy = position[1] - self._listener_pos[1]
        dz = position[2] - self._listener_pos[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _clear_al_errors():
        if not al:
            return
        try:
            while al.alGetError() != al.AL_NO_ERROR:
                pass
        except Exception:
            pass

    @staticmethod
    def _delete_al_source(src_val: int):
        if not al or not src_val:
            return
        try:
            al.alDeleteSources(1, al.ctypes.pointer(al.ctypes.c_uint(src_val)))
        except Exception:
            pass

    def _max_sources_limit(self) -> int:
        try:
            value = _read_audio_value("audio.max_sources", 32)
            return max(1, min(256, int(value)))
        except Exception:
            return 32

    def _priority_threshold(self) -> float:
        try:
            value = _read_audio_value("audio.priority_threshold", 0.1)
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.1

    def _spatialization_on(self) -> bool:
        try:
            value = _read_audio_value("audio.enable_spatialization", True)
            return bool(value)
        except Exception:
            return True

    def _enforce_source_limit(self):
        try:
            limit = self._max_sources_limit()
            if len(self._active_sources) < limit:
                return True
            scored = []
            for src in list(self._active_sources.keys()):
                pos = self._source_positions.get(src, self._listener_pos)
                dist = self._distance_to_listener(pos) if pos is not None else 0.0
                info = self._source_info.get(src, {})
                looping = bool(info.get("looping", False))
                gain = float(self._source_gains.get(src, 1.0))
                scored.append((looping, gain, -dist, src))
            scored.sort(key=lambda item: (item[0], item[1], item[2]))
            victim = scored[0][3] if scored else None
            if victim is not None:
                try:
                    self.stop(int(victim))
                except Exception:
                    pass
            return len(self._active_sources) < limit
        except Exception:
            return True

    def play(self, clip_path: str, loop: bool = False, volume: float = 1.0, pitch: float = 1.0,
             spatial_blend: float = 1.0, min_distance: float = 1.0, max_distance: float = 50.0,
             volume_rolloff: list[list[float]] = None, offset: float = 0.0,
             velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> int | None:
        if not al.oalGetInit(): return None
        audio_sys = AudioSystem.instance()
        if not audio_sys: return None
        if not self._enforce_source_limit():
            return None
        clip = audio_sys.load_clip(clip_path)
        if not clip or not clip.buffer: return None
        if not self._spatialization_on():
            spatial_blend = 0.0
        src_val = 0
        try:
            self._clear_al_errors()
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
            al.alSourcePlay(src_val)
        except Exception:
            self._delete_al_source(src_val)
            self._clear_al_errors()
            return None
        self._active_sources[src_val] = src_val
        self._source_clips[src_val] = clip
        self._source_gains[src_val] = volume
        self._source_info[src_val] = {
            "min_distance": min_distance,
            "max_distance": max_distance,
            "spatial_blend": spatial_blend,
            "volume_rolloff": volume_rolloff or [[0, 1], [1, 0]],
            "offset": offset,
            "velocity": velocity,
            "looping": loop,
        }
        self._source_positions[src_val] = (0.0, 0.0, 0.0)
        self._source_aux_slot[src_val] = 0
        if not hasattr(self, "_source_aux_filter"):
            self._source_aux_filter = {}
        if not hasattr(self, "_source_aux_gain"):
            self._source_aux_gain = {}
        if not hasattr(self, "_source_direct_filter"):
            self._source_direct_filter = {}
        self._source_aux_filter.pop(src_val, None)
        self._source_aux_gain[src_val] = 0.0
        self._source_direct_filter.pop(src_val, None)
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
                if not self._spatialization_on():
                    spatial_blend = 0.0
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
                    if self._spatialization_on():
                        al.alSource3f(source, al.AL_POSITION, *position)
                    else:
                        al.alSource3f(source, al.AL_POSITION, 0.0, 0.0, 0.0)
                        try:
                            al.alSourcei(source, al.AL_SOURCE_RELATIVE, 1)
                        except Exception:
                            pass
                    al.alSourcef(source, al.AL_GAIN, volume)
        except Exception as e:
            from core.foundation.logger import Logger
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

    def get_source_aux_filter(self, source_id: int) -> int:
        if not hasattr(self, "_source_aux_filter"):
            self._source_aux_filter = {}
        return self._source_aux_filter.get(source_id, 0)

    def get_source_aux_gain(self, source_id: int) -> float:
        if not hasattr(self, "_source_aux_gain"):
            self._source_aux_gain = {}
        return float(self._source_aux_gain.get(source_id, 0.0))

    def _ensure_aux_filter(self, source_id: int) -> int:
        if not hasattr(self, "_source_aux_filter"):
            self._source_aux_filter = {}
        existing = self._source_aux_filter.get(source_id, 0)
        if existing:
            try:
                from core.audio.audio_efx import is_filter
                if is_filter(int(existing)):
                    return int(existing)
            except Exception:
                pass
            self._source_aux_filter.pop(source_id, None)
        try:
            from core.audio.audio_efx import create_filter, set_filter_type, AL_FILTER_LOWPASS
            fid = int(create_filter())
            set_filter_type(int(fid), int(AL_FILTER_LOWPASS))
            self._source_aux_filter[int(source_id)] = int(fid)
            return int(fid)
        except Exception:
            return 0

    def _delete_aux_filter(self, source_id: int):
        if not hasattr(self, "_source_aux_filter"):
            return
        fid = self._source_aux_filter.pop(int(source_id), 0)
        if fid:
            try:
                from core.audio.audio_efx import delete_filter
                delete_filter(int(fid))
            except Exception:
                pass
        if hasattr(self, "_source_aux_gain"):
            self._source_aux_gain.pop(int(source_id), None)

    def _delete_direct_filter(self, source_id: int):
        if not hasattr(self, "_source_direct_filter"):
            return
        fid = self._source_direct_filter.pop(int(source_id), 0)
        if fid:
            try:
                from core.audio.audio_efx import delete_filter, set_source_direct_filter
                try:
                    set_source_direct_filter(int(source_id), 0)
                except Exception:
                    pass
                delete_filter(int(fid))
            except Exception:
                pass

    def set_source_aux_send(self, source_id: int, slot_id: int, filter_id: int = 0, send_index: int = 0):
        if not source_id or not al.oalGetInit():
            return
        if int(slot_id) != 0:
            try:
                from core.audio.audio_efx import reverb_enabled
                if not bool(reverb_enabled()):
                    slot_id = 0
                    filter_id = 0
            except Exception:
                pass
        current = self._source_aux_slot.get(int(source_id), 0)
        if hasattr(self, "_source_aux_filter"):
            tracked = self._source_aux_filter.get(int(source_id), 0)
            if int(filter_id) == 0 and int(tracked) != 0:
                filter_id = int(tracked)
        else:
            self._source_aux_filter = {}
        if int(current) == int(slot_id) and int(filter_id) == int(self._source_aux_filter.get(int(source_id), 0)):
            if int(slot_id) == 0:
                return
            try:
                if float(self._source_aux_gain.get(int(source_id), 1.0)) > 0.999 and int(filter_id) != 0:
                    pass
                else:
                    raise ValueError("gain")
            except ValueError:
                pass
            except Exception:
                return
        try:
            from core.audio.audio_efx import set_source_aux_send as _efx_send
            ok = bool(_efx_send(int(source_id), int(slot_id), int(filter_id), int(send_index)))
            if ok:
                self._source_aux_slot[int(source_id)] = int(slot_id)
                if int(filter_id) != 0:
                    self._source_aux_filter[int(source_id)] = int(filter_id)
                if int(slot_id) == 0 and int(filter_id) == 0:
                    if hasattr(self, "_source_aux_gain"):
                        self._source_aux_gain[int(source_id)] = 0.0
            else:
                if int(slot_id) == 0:
                    self._source_aux_slot[int(source_id)] = 0
        except Exception:
            pass

    def set_source_aux_send_weighted(self, source_id: int, slot_id: int, weight: float, send_index: int = 0):
        if not source_id or not al.oalGetInit():
            return
        try:
            w = max(0.0, min(1.0, float(weight)))
        except Exception:
            w = 0.0
        if w <= 0.001 or int(slot_id) == 0:
            try:
                current = int(self._source_aux_slot.get(int(source_id), 0))
            except Exception:
                current = 0
            if current == 0:
                if hasattr(self, "_source_aux_gain"):
                    self._source_aux_gain[int(source_id)] = 0.0
                return
            self.set_source_aux_send(int(source_id), 0, 0, int(send_index))
            if hasattr(self, "_source_aux_gain"):
                self._source_aux_gain[int(source_id)] = 0.0
            return
        fid = 0
        try:
            from core.audio.audio_efx import efx_available, set_filter_param_f, AL_LOWPASS_GAIN, AL_LOWPASS_GAINHF
            if bool(efx_available()):
                fid = int(self._ensure_aux_filter(int(source_id)))
                if fid:
                    set_filter_param_f(int(fid), int(AL_LOWPASS_GAIN), float(w))
                    set_filter_param_f(int(fid), int(AL_LOWPASS_GAINHF), 1.0)
                else:
                    fid = 0
        except Exception:
            fid = 0
        try:
            prev_slot = int(self._source_aux_slot.get(int(source_id), 0))
            prev_gain = float(self._source_aux_gain.get(int(source_id), -1.0)) if hasattr(self, "_source_aux_gain") else -1.0
            if prev_slot == int(slot_id) and abs(prev_gain - float(w)) < 0.01:
                return
        except Exception:
            pass
        self.set_source_aux_send(int(source_id), int(slot_id), int(fid), int(send_index))
        if hasattr(self, "_source_aux_gain"):
            self._source_aux_gain[int(source_id)] = float(w)

    def set_source_occlusion(self, source_id: int, occlusion: float):
        if not source_id or not al.oalGetInit():
            return
        try:
            occ = max(0.0, min(1.0, float(occlusion)))
        except Exception:
            occ = 0.0
        try:
            from core.audio.audio_efx import occlusion_enabled
            if not bool(occlusion_enabled()):
                occ = 0.0
        except Exception:
            pass
        if occ <= 0.001:
            if hasattr(self, "_source_direct_filter"):
                existing = int(self._source_direct_filter.get(int(source_id), 0))
                if existing:
                    self._delete_direct_filter(int(source_id))
            return
        try:
            from core.audio.audio_efx import efx_available, create_filter, set_filter_type, set_filter_param_f, set_source_direct_filter, AL_FILTER_LOWPASS, AL_LOWPASS_GAIN, AL_LOWPASS_GAINHF
            if not bool(efx_available()):
                return
            if not hasattr(self, "_source_direct_filter"):
                self._source_direct_filter = {}
            fid = int(self._source_direct_filter.get(int(source_id), 0))
            valid = False
            if fid:
                try:
                    from core.audio.audio_efx import is_filter as _is_f
                    valid = bool(_is_f(int(fid)))
                except Exception:
                    valid = False
            if not valid:
                fid = int(create_filter())
                set_filter_type(int(fid), int(AL_FILTER_LOWPASS))
                self._source_direct_filter[int(source_id)] = int(fid)
            gain = max(0.05, 1.0 - float(occ) * 0.85)
            gain_hf = max(0.05, 1.0 - float(occ) * 0.9)
            set_filter_param_f(int(fid), int(AL_LOWPASS_GAIN), float(gain))
            set_filter_param_f(int(fid), int(AL_LOWPASS_GAINHF), float(gain_hf))
            set_source_direct_filter(int(source_id), int(fid))
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
        if not source:
            return
        try:
            if al.oalGetInit():
                try:
                    from core.audio.audio_efx import set_source_aux_send as _efx_clear, set_source_direct_filter as _efx_clear_direct
                    try:
                        _efx_clear(int(source), 0, 0, 0)
                    except Exception:
                        pass
                    try:
                        _efx_clear_direct(int(source), 0)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    state = al.ctypes.c_int()
                    al.alGetSourcei(source, al.AL_SOURCE_STATE, state)
                    if state.value in (al.AL_PLAYING, al.AL_PAUSED):
                        al.alSourceStop(source)
                    al.alDeleteSources(1, al.ctypes.pointer(al.ctypes.c_uint(source)))
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._delete_aux_filter(int(source))
        except Exception:
            pass
        try:
            self._delete_direct_filter(int(source))
        except Exception:
            pass
        self._active_sources.pop(source, None)
        self._source_info.pop(source, None)
        self._source_positions.pop(source, None)
        self._source_aux_slot.pop(source, None)
        self._source_clips.pop(source, None)
        self._source_gains.pop(source, None)
        if hasattr(self, "_source_aux_gain"):
            self._source_aux_gain.pop(source, None)

    def stop_all(self):
        if not al.oalGetInit():
            try:
                self._active_sources.clear()
                self._source_info.clear()
                self._source_positions.clear()
                self._source_aux_slot.clear()
                self._source_clips.clear()
                self._source_gains.clear()
                if hasattr(self, "_source_aux_filter"):
                    self._source_aux_filter.clear()
                if hasattr(self, "_source_aux_gain"):
                    self._source_aux_gain.clear()
                if hasattr(self, "_source_direct_filter"):
                    self._source_direct_filter.clear()
            except Exception:
                pass
            return
        for src in list(self._active_sources.keys()):
            try:
                try:
                    from core.audio.audio_efx import set_source_aux_send as _efx_c, set_source_direct_filter as _efx_cd
                    try:
                        _efx_c(int(src), 0, 0, 0)
                    except Exception:
                        pass
                    try:
                        _efx_cd(int(src), 0)
                    except Exception:
                        pass
                except Exception:
                    pass
                state = al.ctypes.c_int()
                al.alGetSourcei(src, al.AL_SOURCE_STATE, state)
                if state.value in (al.AL_PLAYING, al.AL_PAUSED):
                    al.alSourceStop(src)
                al.alDeleteSources(1, al.ctypes.pointer(al.ctypes.c_uint(src)))
            except Exception:
                pass
            try:
                self._delete_aux_filter(int(src))
            except Exception:
                pass
            try:
                self._delete_direct_filter(int(src))
            except Exception:
                pass
        self._active_sources.clear()
        self._source_info.clear()
        self._source_positions.clear()
        self._source_aux_slot.clear()
        self._source_clips.clear()
        self._source_gains.clear()
        if hasattr(self, "_source_aux_filter"):
            try:
                from core.audio.audio_efx import delete_filter as _df
                for _fid in list(self._source_aux_filter.values()):
                    try:
                        _df(int(_fid))
                    except Exception:
                        pass
            except Exception:
                pass
            self._source_aux_filter.clear()
        if hasattr(self, "_source_aux_gain"):
            self._source_aux_gain.clear()
        if hasattr(self, "_source_direct_filter"):
            try:
                from core.audio.audio_efx import delete_filter as _df2
                for _fid in list(self._source_direct_filter.values()):
                    try:
                        _df2(int(_fid))
                    except Exception:
                        pass
            except Exception:
                pass
            self._source_direct_filter.clear()
