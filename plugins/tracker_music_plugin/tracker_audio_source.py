# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

import os

from core.components.audio.audio_source import AudioSource
from core.components.inspector_meta import FieldType, InspectorField
from core.ecs.ecs import ComponentRegistry
from core.foundation.logger import Logger
from core.foundation.plugin_manager import _zarin_user_dir
from .renderer import TrackerRenderService

TRACKER_EXTS = (".mod", ".xm", ".s3m", ".it", ".669", ".mtm", ".stm", ".ult")


def _is_tracker(path: str) -> bool:
    return bool(path) and os.path.splitext(path)[1].lower() in TRACKER_EXTS


def _cache_dir() -> str:
    return _zarin_user_dir("tracker_cache")


def render_to_wav(module_path: str, quality: str = "software") -> str | None:
    try:
        service = TrackerRenderService(_cache_dir())
        wav = service.render_wav(module_path, quality)
        if wav and os.path.isfile(wav):
            return wav
    except Exception as e:
        Logger.error(f"TrackerAudioSource render error: {e}", e)
    return None


def patch_audio_source_play():
    from core.ecs.ecs import ComponentRegistry
    comp_cls = ComponentRegistry.get("AudioSource")
    if comp_cls is None:
        return None
    from core.foundation.patcher import ComponentPatcher
    patcher = ComponentPatcher()

    def before_play(self, *args, **kwargs):
        if getattr(self, "_tracker_swap_active", False):
            return None
        clip = getattr(self, "clip_path", "") or ""
        if not _is_tracker(clip):
            return None
        wav = render_to_wav(clip, getattr(self, "render_quality", "software"))
        if wav is None:
            return False
        self._tracker_swap_orig = clip
        self.clip_path = wav
        self._tracker_swap_active = True
        return None

    def after_play(self, *args, **kwargs):
        if getattr(self, "_tracker_swap_active", False):
            self.clip_path = self._tracker_swap_orig
            self._tracker_swap_active = False
        return None

    patcher.wrap_method(comp_cls, "play", before=before_play, after=after_play)
    return patcher


@ComponentRegistry.register
class TrackerAudioSource(AudioSource):
    _icon = "TrackerAudioSource.png"
    _gizmo_icon_color = (80, 220, 160)
    _gizmo_icon_label = "T"

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        fields = [
            InspectorField("clip_path", "Module", FieldType.RESOURCE_PATH,
                           file_filter="Tracker (*.mod *.xm *.s3m *.it)"),
            InspectorField("render_quality", "Render Quality", FieldType.ENUM,
                           enum_options=["software", "external"]),
        ]
        for field in super()._inspector_fields():
            if field.name == "clip_path":
                continue
            fields.append(field)
        fields.append(InspectorField("song_length_sec", "Song Length (sec)", FieldType.FLOAT,
                                     readonly=True, decimals=2))
        fields.append(InspectorField("channels_count", "Channels", FieldType.INT, readonly=True))
        return fields

    def __init__(self):
        super().__init__()
        self.render_quality: str = "software"
        self.song_length_sec: float = 0.0
        self.channels_count: int = 0

    def play(self):
        if not self.clip_path or self._playing:
            return
        if _is_tracker(self.clip_path):
            wav = render_to_wav(self.clip_path, self.render_quality)
            if wav is None:
                Logger.error(f"TrackerAudioSource: failed to render '{self.clip_path}'")
                return
            orig = self.clip_path
            self.clip_path = wav
            try:
                super().play()
            finally:
                self.clip_path = orig
            self._refresh_info(self.clip_path)
        else:
            super().play()

    def _refresh_info(self, module_path: str):
        try:
            from .renderer import load_module
            song = load_module(module_path)
            if song is None:
                return
            self.channels_count = int(getattr(song, "n_channels", 0) or 0)
            try:
                info = song.get_song_info() or {}
            except Exception:
                info = {}
            self.song_length_sec = float(info.get("duration_seconds", 0.0) or 0.0)
        except Exception as e:
            Logger.error(f"TrackerAudioSource info error: {e}")

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "render_quality": self.render_quality,
            "song_length_sec": self.song_length_sec,
            "channels_count": self.channels_count,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> TrackerAudioSource:
        a = super().deserialize(data)
        a.render_quality = data.get("render_quality", "software")
        a.song_length_sec = data.get("song_length_sec", 0.0)
        a.channels_count = data.get("channels_count", 0)
        return a