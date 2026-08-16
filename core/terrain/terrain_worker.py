# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os
import queue
import threading
from dataclasses import dataclass

import moderngl
import numpy as np

from core.foundation.logger import Logger
from core.foundation.progress import task_complete, task_start, task_update
from core.shaders.compute_shader import compile_compute_shader

_SHADER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "shaders", "terrain_gen.compute")


@dataclass
class _Job:
    settings: dict
    resolution: int
    token: int


class TerrainGenWorker:
    def __init__(self):
        self._req: "queue.Queue[_Job]" = queue.Queue()
        self._result: "queue.Queue[tuple[int, np.ndarray]]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._token = 0
        self._lock = threading.Lock()
        self._ctx: moderngl.Context | None = None
        self._program: moderngl.ComputeShader | None = None
        self._buf: moderngl.Buffer | None = None
        self._last_res = 0
        self._float_uniforms: list[str] = []
        self._int_uniforms: list[str] = []
        self._batch_active: bool = False
        self._batch_done: int = 0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._req.put_nowait(None)
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    def request(self, settings: dict, resolution: int) -> int:
        with self._lock:
            self._token += 1
            tok = self._token
        self._req.put_nowait(_Job(dict(settings), int(resolution), tok))
        return tok

    def consume_result(self):
        try:
            return self._result.get_nowait()
        except queue.Empty:
            return None

    def _load_program(self):
        if self._program is not None:
            return
        if not os.path.exists(_SHADER_PATH):
            Logger.error(f"TerrainGenWorker: shader not found: {_SHADER_PATH}")
            return
        with open(_SHADER_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        start = content.find("GLSLPROGRAM")
        end = content.find("ENDGLSL", start)
        if start < 0 or end < 0:
            Logger.error("TerrainGenWorker: invalid shader file")
            return
        source = content[start + len("GLSLPROGRAM"):end].strip()
        self._program = compile_compute_shader(self._ctx, source, _SHADER_PATH)

    def _cache_uniform_names(self):
        if self._float_uniforms or self._int_uniforms:
            return
        prog = self._program
        if prog is None:
            return
        from core.terrain.terrain_generator import _FLOAT_KEYS, _INT_KEYS, _DEFAULTS
        float_names = set("u_" + k for k in _FLOAT_KEYS)
        int_names = set("u_" + k for k in _INT_KEYS)
        for name in prog:
            if name in float_names:
                self._float_uniforms.append(name)
            elif name in int_names:
                self._int_uniforms.append(name)

    def _run(self):
        try:
            self._ctx = moderngl.create_standalone_context(require=460)
            self._ctx.pixel_alignment = 1
        except Exception as e:
            Logger.error(f"TerrainGenWorker: cannot create GL context: {e}", e)
            return
        self._load_program()
        if self._program is None:
            return
        self._cache_uniform_names()
        from core.terrain.terrain_generator import _DEFAULTS
        while not self._stop.is_set():
            try:
                job = self._req.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            if not self._batch_active and self._req.qsize() >= 1:
                self._batch_active = True
                self._batch_done = 0
                task_start("terrain:gen", "Generating terrain…", fraction=0.0, total=float(1 + self._req.qsize()))
            try:
                arr = self._generate(job, _DEFAULTS)
                if arr is not None:
                    self._result.put((job.token, arr))
            except Exception as e:
                Logger.error(f"TerrainGenWorker: generation error: {e}", e)
            if self._batch_active:
                self._batch_done += 1
                total = self._batch_done + self._req.qsize()
                task_update(
                    "terrain:gen",
                    fraction=min(1.0, self._batch_done / total),
                    detail=f"Generated {self._batch_done}/{total} chunks",
                )
                if self._req.qsize() == 0:
                    task_complete("terrain:gen")
                    self._batch_active = False
        try:
            if self._buf is not None:
                self._buf.release()
            if self._program is not None:
                self._program.release()
            if self._ctx is not None:
                self._ctx.release()
        except Exception:
            pass

    def _generate(self, job: _Job, defaults: dict) -> np.ndarray | None:
        res = job.resolution
        res = max(16, min(2048, res))
        res = (res // 16) * 16
        if res < 16:
            res = 16
        n = res * res
        if self._buf is None or self._last_res != res:
            self._buf = self._ctx.buffer(reserve=n * 4)
            self._last_res = res
        d = job.settings
        for name in self._float_uniforms:
            key = name[2:]
            try:
                self._program[name].value = float(d.get(key, defaults.get(key, 0.0)))
            except Exception:
                pass
        for name in self._int_uniforms:
            key = name[2:]
            try:
                self._program[name].value = int(d.get(key, defaults.get(key, 0)))
            except Exception:
                pass
        self._buf.bind_to_storage_buffer(0)
        groups = (res + 15) // 16
        self._program.run(groups, groups, 1)
        self._ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        raw = self._buf.read()
        arr = np.frombuffer(raw, dtype=np.float32).reshape(res, res).copy()
        return arr


_worker: TerrainGenWorker | None = None


def get_worker() -> TerrainGenWorker:
    global _worker
    if _worker is None:
        _worker = TerrainGenWorker()
        _worker.start()
    return _worker
