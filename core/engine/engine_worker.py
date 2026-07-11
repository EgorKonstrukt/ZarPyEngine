# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import threading
import time
from typing import TYPE_CHECKING
from PyQt6.QtCore import QThread

if TYPE_CHECKING:
    from core.engine.engine import Engine

_MAX_FIXED_STEPS = 5


class GameWorker(QThread):
    """Runs Engine tick logic in a background thread with staged locking.

    Staged execution so the renderer (main thread) can acquire
    ``engine._scene_lock`` between stages:
        в†’ flush transforms (lock held briefly)
        в†’ fixed update (physics, at capped rate)
        в†’ script update (at ``update_rate``)
    """

    def __init__(self, engine: Engine, update_rate: float = 120.0,
                 fixed_rate: float = 60.0):
        super().__init__()
        self._engine = engine
        self._update_dt = 1.0 / max(update_rate, 1.0)
        self._fixed_dt = 1.0 / max(fixed_rate, 1.0)
        self._stop_event = threading.Event()

    def run(self):
        engine = self._engine
        update_dt = self._update_dt
        next_update = time.perf_counter()

        while not self._stop_event.is_set():
            now = time.perf_counter()

            # Stage 1: flush transforms, calc dt
            with engine._scene_lock:
                dt = engine.tick_begin()
            # Lock released вЂ” renderer can read transforms

            # Stage 2: fixed update steps (physics, at capped rate)
            for _ in range(_MAX_FIXED_STEPS):
                with engine._scene_lock:
                    if not engine.tick_fixed_step():
                        break
                # Lock released between each fixed step
            # Lock released вЂ” renderer can read physics results

            # Stage 3: script update (at full update_rate)
            with engine._scene_lock:
                engine.tick_update(dt)
            # Lock released вЂ” renderer can read script results

            next_update += update_dt
            sleep_time = max(0, next_update - time.perf_counter())
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

            # If we fell behind, skip frames to catch up
            if time.perf_counter() - next_update > update_dt * 10:
                next_update = time.perf_counter()

    def stop(self, timeout_ms: int = 2000):
        self._stop_event.set()
        self.wait(timeout_ms)
