# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from core.ecs.ecs import Component, ComponentRegistry
from core.components.inspector_meta import FieldType, InspectorField
@ComponentRegistry.register
class VideoRenderer(Component):
    _icon = "VideoRenderer.png"
    _gizmo_icon_color = (200, 120, 80)
    _gizmo_icon_label = "V"
    _show_gizmo_icon: bool = False
    _updates: bool = True

    @classmethod
    def _inspector_fields(cls) -> list[InspectorField]:
        return [
            InspectorField("video_path", "Video", FieldType.RESOURCE_PATH, file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.webm)"),
            InspectorField("audio_source_entity_id", "Audio Source", FieldType.GAMEOBJECT),
            InspectorField("color", "Color", FieldType.COLOR),
            InspectorField("flip_x", "Flip X", FieldType.BOOL),
            InspectorField("flip_y", "Flip Y", FieldType.BOOL),
            InspectorField("play_on_start", "Play on Start", FieldType.BOOL),
            InspectorField("loop", "Loop", FieldType.BOOL),
            InspectorField("volume", "Volume", FieldType.FLOAT, min_val=0.0, max_val=1.0),
            InspectorField("offset", "Offset (s)", FieldType.FLOAT, min_val=0.0),
        ]

    def __init__(self):
        super().__init__()
        self.video_path: str = ""
        self.audio_source_entity_id: str = ""
        self._color: list[float] = [1, 1, 1, 1]
        self.flip_x: bool = False
        self.flip_y: bool = False
        self.play_on_start: bool = True
        self.loop: bool = True
        self.volume: float = 1.0
        self.offset: float = 0.0

    @property
    def color(self) -> list[float]:
        return self._color

    @color.setter
    def color(self, val):
        if val is None:
            self._color = [1, 1, 1, 1]
        elif len(val) == 3:
            self._color = [val[0], val[1], val[2], 1.0]
        elif len(val) >= 4:
            self._color = [val[0], val[1], val[2], val[3]]
        else:
            self._color = [1, 1, 1, 1]

    def serialize(self) -> dict:
        d = super().serialize()
        d.update({
            "video_path": self.video_path,
            "audio_source_entity_id": self.audio_source_entity_id,
            "color": self.color,
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "play_on_start": self.play_on_start,
            "loop": self.loop,
            "volume": self.volume,
            "offset": self.offset,
        })
        return d

    @classmethod
    def deserialize(cls, data: dict) -> VideoRenderer:
        vr = cls()
        vr.enabled = data.get("enabled", True)
        vr.video_path = data.get("video_path", "") or ""
        vr.audio_source_entity_id = data.get("audio_source_entity_id", "") or ""
        vr.color = data.get("color", [1, 1, 1, 1])
        vr.flip_x = data.get("flip_x", False)
        vr.flip_y = data.get("flip_y", False)
        vr.play_on_start = data.get("play_on_start", True)
        vr.loop = data.get("loop", True)
        vr.volume = data.get("volume", 1.0)
        vr.offset = data.get("offset", 0.0)
        return vr
