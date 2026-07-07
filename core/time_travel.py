# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
from collections import deque
from typing import Optional, TYPE_CHECKING
from core.logger import Logger

if TYPE_CHECKING:
    from core.ecs import Scene, ComponentRegistry


class FrameSnapshot:
    __slots__ = ("frame_number", "timestamp", "entity_count", "data")

    def __init__(self, frame_number: int, timestamp: float,
                 entity_count: int, data: dict):
        self.frame_number: int = frame_number
        self.timestamp: float = timestamp
        self.entity_count: int = entity_count
        self.data: dict = data


class SnapshotRecorder:
    """Ring-buffered scene snapshot recorder for time-travel debugging.

    Captures full serialized scene state every N frames so the user can
    scrub back in time and inspect any previous frame.
    """

    def __init__(self, max_frames: int = 600):
        self._buffer: deque[FrameSnapshot] = deque(maxlen=max_frames)
        self._max_frames: int = max_frames
        self._recording: bool = False
        self._capture_interval: int = 1
        self._frame_counter: int = 0
        self._total_frames: int = 0

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def num_frames(self) -> int:
        return len(self._buffer)

    @property
    def max_frames(self) -> int:
        return self._max_frames

    @property
    def total_frames(self) -> int:
        """Total frames captured since recording started (may exceed buffer)."""
        return self._total_frames

    @property
    def capture_interval(self) -> int:
        return self._capture_interval

    @capture_interval.setter
    def capture_interval(self, n: int):
        self._capture_interval = max(1, n)

    def start(self):
        self._recording = True
        self._frame_counter = 0
        self._total_frames = 0
        Logger.info("Time-travel recording started.")

    def stop(self):
        self._recording = False
        Logger.info(f"Time-travel recording stopped. {len(self._buffer)} frames in buffer.")

    def clear(self):
        self._buffer.clear()
        self._total_frames = 0
        self._frame_counter = 0

    def capture(self, scene: Scene) -> Optional[int]:
        """Capture a snapshot of the current scene state.

        Returns the frame index in the buffer, or None if not captured.
        """
        if not self._recording or not scene:
            return None
        self._frame_counter += 1
        if self._frame_counter < self._capture_interval:
            return None
        self._frame_counter = 0

        try:
            data = scene.serialize()
            import time
            snapshot = FrameSnapshot(
                frame_number=self._total_frames,
                timestamp=time.time(),
                entity_count=len(scene.get_all_entities()),
                data=data,
            )
            self._buffer.append(snapshot)
            idx = len(self._buffer) - 1
            self._total_frames += 1
            return idx
        except Exception as e:
            Logger.error(f"Snapshot capture failed: {e}")
            return None

    def get_frame(self, index: int) -> Optional[FrameSnapshot]:
        """Get a snapshot by buffer index (0 = oldest)."""
        if 0 <= index < len(self._buffer):
            return self._buffer[index]
        return None

    def restore(self, registry: ComponentRegistry,
                index: int) -> Optional[Scene]:
        """Restore a snapshot frame into a new Scene object.

        Returns the restored Scene, or None on failure.
        """
        snap = self.get_frame(index)
        if snap is None:
            return None
        try:
            from core.ecs import Scene
            restored = Scene.deserialize(snap.data, registry)
            return restored
        except Exception as e:
            Logger.error(f"Snapshot restore failed: {e}")
            return None

    def frames(self) -> list[FrameSnapshot]:
        """Returns a copy of all frames for UI display."""
        return list(self._buffer)
