# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import time
import json
import os
import threading
from typing import Optional, Any, TYPE_CHECKING
from core.ecs.ecs import Scene, ComponentRegistry
from core.foundation.plugin_manager import PluginManager
from core.foundation.logger import Logger
from core.config.config import get_global_config
from core.config.constants import PATH_FIELDS as _PATH_FIELDS
from core.ecs.pool import general as _get_pool
from core.ecs.embedded_resources import embed_scene_resources as _embed_scene_resources
from core.ecs.embedded_resources import extract_embedded_resources as _extract_embedded_resources
from core.foundation.profiler import Profiler as _Profiler
from core.foundation.progress import task_complete, task_start, task_update

if TYPE_CHECKING:
    from core.engine.engine_worker import GameWorker

class Engine:
    _instance: Optional[Engine] = None
    def __init__(self):
        Engine._instance = self
        self._plugin_manager: PluginManager = PluginManager()
        self._plugin_manager.set_engine(self)
        self._scene: Optional[Scene] = None
        self._running: bool = False
        self._play_mode: bool = False
        self._time_scale: float = 1.0
        self._fixed_dt: float = 0.02
        self._fixed_accum: float = 0.0
        self._last_time: float = 0.0
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._fps_accum: float = 0.0
        self._fps_frames: int = 0
        self._tps: float = 0.0
        self._tps_accum: float = 0.0
        self._tps_frames: int = 0
        self._scene_lock = threading.RLock()
        self._profiler = _Profiler()
        self._time_travel_recorder: Optional[Any] = None
        self._event_listeners: dict[str, list] = {}
        self._component_registry = ComponentRegistry
        self._collab_manager: Optional[Any] = None
        self._game_worker: Optional[GameWorker] = None
        self._plugin_ui_registry: dict[str, list] = {
            "docks": [],
            "toolbar_actions": [],
            "menu_items": [],
        }
        self._project_path: Optional[str] = None
        self._project_settings_path: Optional[str] = None
    @classmethod
    def instance(cls) -> Optional[Engine]: return cls._instance

    _debug_no_qt_overlay: bool = False
    @property
    def debug_no_qt_overlay(self) -> bool:
        """When True, hides all Qt child widgets from SceneViewport
        (overlay, toolbar, labels) to isolate QOpenGLWidget compositor overhead."""
        return self._debug_no_qt_overlay
    @debug_no_qt_overlay.setter
    def debug_no_qt_overlay(self, value: bool):
        self._debug_no_qt_overlay = value
    @property
    def plugin_manager(self) -> PluginManager: return self._plugin_manager
    @property
    def scene(self) -> Optional[Scene]: return self._scene
    @property
    def play_mode(self) -> bool: return self._play_mode
    @property
    def fps(self) -> float: return self._fps
    @property
    def tps(self) -> float: return self._tps
    @property
    def frame_count(self) -> int: return self._frame_count
    @property
    def time_scale(self) -> float: return self._time_scale
    @time_scale.setter
    def time_scale(self, v: float): self._time_scale = max(0.0, v)
    @property
    def fixed_dt(self) -> float: return self._fixed_dt
    @fixed_dt.setter
    def fixed_dt(self, v: float): self._fixed_dt = max(0.001, v)
    @property
    def viewport(self):
        return getattr(self, '_viewport', None)
    @viewport.setter
    def viewport(self, v):
        self._viewport = v
        for p in self._plugin_manager.get_all():
            try: p.on_viewport_ready(v)
            except Exception as e: Logger.error(f"Plugin on_viewport_ready error: {e}", e)
    @property
    def profiler(self):
        return self._profiler
    @property
    def profiler_data(self) -> dict: return self._profiler.data
    def get_profiler_data(self, key: str, default: float = 0.0) -> float:
        return self._profiler.data.get(key, default)
    def reset_profiler(self):
        self._profiler.reset()
    @property
    def project_root(self) -> str:
        return getattr(self, '_project_path', os.getcwd())
    @project_root.setter
    def project_root(self, path: str):
        self._project_path = path
        try:
            from core.audio.audio_system import AudioSystem
            audio = AudioSystem.instance()
            if audio:
                audio.apply_project_audio_config()
        except Exception:
            pass
    def _embedded_cache_mode(self) -> str:
        try:
            from core.config.config import get_global_config
            return get_global_config().get("engine.embedded_cache_mode", "project")
        except Exception:
            return "project"
    def _compress_level(self) -> int:
        try:
            scene_level = getattr(self._scene, "compress_level", None)
            if scene_level is not None:
                return self._resolve_compress_level(scene_level)
            return self._resolve_compress_level(
                get_global_config().get("engine.embedded_compression_level", "balanced"))
        except Exception:
            return 6
    @staticmethod
    def _resolve_compress_level(val) -> int:
        if isinstance(val, bool) or not isinstance(val, int):
            return {"fast": 1, "balanced": 6, "max": 9}.get(str(val).lower(), 6)
        return max(0, min(9, val))
    def _defer_gui(self, fn):
        poster = getattr(self, "_gui_poster", None)
        if poster is not None:
            poster(fn)
        else:
            fn()
    def _persist_scene_data(self, data: dict, path: str, level: int, task_id: str,
                            existing_storage: Optional[dict] = None) -> dict:
        def _cb(done: int, total_: int, name: str) -> None:
            frac = None if total_ <= 0 else done / max(1, total_)
            task_update(task_id, fraction=frac, detail=name)
        _embed_scene_resources(data, self.project_root, existing_storage,
                               compress_level=level, progress_cb=_cb)
        storage = data.get("embedded_resources", {})
        self.relativize_scene_paths(data)
        build_path = path + ".build"
        with open(build_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(build_path, path)
        return storage
    def save_scene_async(self, path: Optional[str] = None, on_done: Optional[Any] = None):
        scene = self._scene
        if scene is None:
            return
        save_path = path or scene.path
        if not save_path:
            Logger.warning("No path for scene save.")
            return
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        level = self._compress_level()
        task_id = "scene:save"
        task_start(task_id, f"Saving scene {os.path.basename(save_path)}...", fraction=0.0, total=1.0)
        snapshot = scene.serialize()
        if scene.embedded_resources:
            snapshot["embedded_resources"] = dict(scene.embedded_resources)
        existing = scene.embedded_resources
        def _worker():
            ok = True
            storage = {}
            try:
                storage = self._persist_scene_data(snapshot, save_path, level, task_id, existing)
            except Exception as e:
                ok = False
                Logger.error(f"Failed to save scene: {e}", e)
            def _finish():
                if ok:
                    scene.embedded_resources = storage
                    scene.path = save_path
                    scene.mark_clean()
                    self._emit_event("scene_saved", scene)
                task_complete(task_id)
                if callable(on_done):
                    on_done(ok)
            self._defer_gui(_finish)
        threading.Thread(target=_worker, name="scene-save-worker", daemon=True).start()
    def load_scene_async(self, path: str, data: Optional[dict] = None,
                         on_done: Optional[Any] = None):
        task_id = "scene:load"
        task_start(task_id, f"Loading scene {os.path.basename(path)}...", fraction=0.0, total=1.0)
        def _cb(done: int, total_: int, name: str) -> None:
            frac = None if total_ <= 0 else done / max(1, total_)
            task_update(task_id, fraction=frac, detail=name)
        def _worker(snapshot: dict):
            try:
                embedded = _extract_embedded_resources(snapshot, self.project_root,
                                                       self._embedded_cache_mode(), progress_cb=_cb)
                self.resolve_scene_paths(snapshot)
            except Exception as ex:
                msg = str(ex)
                Logger.error(f"Failed to load scene '{path}': {ex}", ex)
                def _err():
                    from core.foundation import progress
                    progress.notify_error(f"Failed to load scene: {msg}")
                    task_complete(task_id)
                self._defer_gui(_err)
                return
            self._defer_gui(lambda: self._apply_loaded(snapshot, embedded, path, task_id, on_done))
        if data is None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                Logger.error(f"Failed to load scene '{path}': {e}", e)
                task_complete(task_id)
                return
        threading.Thread(target=_worker, args=(data,), name="scene-load-worker", daemon=True).start()
    def _apply_loaded(self, data: dict, embedded: dict, path: str, task_id: str,
                      on_done: Optional[Any] = None):
        try:
            if self._scene:
                self._plugin_manager.notify_scene_unloaded(self._scene)
            from core.components.rendering.postfx.graphics_effect import GraphicsEffect
            GraphicsEffect.cleanup_registry()
            self._scene = Scene.deserialize(data, self._component_registry)
            self._scene.embedded_resources = embedded
            self._scene.path = path
            self._scene.mark_clean()
            self._plugin_manager.notify_scene_loaded(self._scene)
            Logger.info(f"Scene loaded: {path}")
            self._emit_event("scene_loaded", self._scene)
        except Exception as e:
            Logger.error(f"Failed to load scene '{path}': {e}", e)
        finally:
            task_complete(task_id)
        if callable(on_done):
            on_done(self._scene)
    def resolve_scene_paths(self, data: dict):
        root = self.project_root
        entities = data.get("entities", {})
        for eid, edata in entities.items():
            for comp in edata.get("components", []):
                self._resolve_component_paths(comp, root)
    def relativize_scene_paths(self, data: dict):
        """Convert absolute paths to project-relative in scene JSON data."""
        root = self.project_root
        entities = data.get("entities", {})
        for eid, edata in entities.items():
            for comp in edata.get("components", []):
                self._relativize_component_paths(comp, root)
    @staticmethod
    def _resolve_component_paths(comp: dict, root: str):
        for key, val in comp.items():
            if key in _PATH_FIELDS and val and isinstance(val, str):
                comp[key] = Engine._resolve_path(val, root)
        mats = Engine._copy_materials(comp.get("materials"))
        if mats is not None:
            for mat in mats.values() if isinstance(mats, dict) else mats:
                if isinstance(mat, dict) and "path" in mat:
                    mat["path"] = Engine._resolve_path(mat["path"], root)
            comp["materials"] = mats
    @staticmethod
    def _relativize_component_paths(comp: dict, root: str):
        for key, val in comp.items():
            if key in _PATH_FIELDS and val and isinstance(val, str):
                comp[key] = Engine._relativize_path(val, root)
        mats = Engine._copy_materials(comp.get("materials"))
        if mats is not None:
            for mat in mats.values() if isinstance(mats, dict) else mats:
                if isinstance(mat, dict) and "path" in mat:
                    mat["path"] = Engine._relativize_path(mat["path"], root)
            comp["materials"] = mats
    @staticmethod
    def _copy_materials(mats):
        """Detach materials entries so path rewriting never mutates live components."""
        if isinstance(mats, list):
            return [dict(e) if isinstance(e, dict) else e for e in mats]
        if isinstance(mats, dict):
            return {k: dict(e) if isinstance(e, dict) else e for k, e in mats.items()}
        return None
    @staticmethod
    def _resolve_path(val: str, root: str) -> str:
        if not val or os.path.exists(val):
            return val
        # Try resolving relative to project root
        candidate = os.path.normpath(os.path.join(root, val))
        if os.path.exists(candidate):
            return candidate.replace("\\", "/")
        # Windows absolute path (C:\...) on Linux вЂ” extract subpath after project name
        if len(val) > 1 and val[1] == ":":
            parts = val.replace("\\", "/").split("/")
            # Try each suffix from longest to shortest
            for i in range(len(parts)):
                sub = "/".join(parts[i:])
                if sub:
                    c = os.path.normpath(os.path.join(root, sub))
                    if os.path.exists(c):
                        return c.replace("\\", "/")
        return val
    @staticmethod
    def _relativize_path(val: str, root: str) -> str:
        if not val:
            return ""
        if not os.path.isabs(val):
            return val.replace("\\", "/")
        try:
            return os.path.relpath(val, root).replace("\\", "/")
        except ValueError:
            return val
    def initialize(self):
        import core.components

        cfg = get_global_config()
        self._time_scale = cfg.get("engine.time_scale", 1.0)
        self._fixed_dt = max(0.001, cfg.get("engine.fixed_update_dt", 0.02))

        try:
            from core.audio.audio_system import AudioSystem
            audio_sys = AudioSystem()
            audio_sys.initialize()
            if self._project_path:
                audio_sys.apply_project_audio_config()
        except Exception as e:
            Logger.error(f"Audio system init failed: {e}")

        # Initialize BuildSettings
        from core.config.build_settings import BuildSettings
        bs = BuildSettings()
        if self._project_path:
            bs.load(os.path.join(self._project_path, "BuildSettings.json"))

        Logger.info("Zarin Engine initialized.")
    def load_scene(self, path: str) -> Optional[Scene]:
        task_start("scene:load", f"Loading scene {os.path.basename(path)}...")
        try:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                data["_source"] = path
                embedded = _extract_embedded_resources(data, self.project_root, self._embedded_cache_mode())
                self.resolve_scene_paths(data)
                if self._scene:
                    self._plugin_manager.notify_scene_unloaded(self._scene)
                from core.components.rendering.postfx.graphics_effect import GraphicsEffect
                GraphicsEffect.cleanup_registry()
                self._scene = Scene.deserialize(data, self._component_registry)
                self._scene.embedded_resources = embedded
                self._scene.path = path
                self._scene.mark_clean()
                self._plugin_manager.notify_scene_loaded(self._scene)
                Logger.info(f"Scene loaded: {path}")
                self._emit_event("scene_loaded", self._scene)
                return self._scene
            except Exception as e:
                Logger.error(f"Failed to load scene '{path}': {e}", e)
                return None
        finally:
            task_complete("scene:load")
    def load_scene_from_data(self, data: dict) -> Optional[Scene]:
        task_start("scene:load_data", "Loading scene data...")
        try:
            try:
                if self._scene:
                    self._plugin_manager.notify_scene_unloaded(self._scene)
                from core.components.rendering.postfx.graphics_effect import GraphicsEffect
                GraphicsEffect.cleanup_registry()
                self._scene = Scene.deserialize(data, self._component_registry)
                self._scene.mark_clean()
                self._plugin_manager.notify_scene_loaded(self._scene)
                Logger.info(f"Scene synced: {self._scene.name}")
                self._emit_event("scene_loaded", self._scene)
                return self._scene
            except Exception as e:
                Logger.error(f"Failed to load synced scene: {e}", e)
                return None
        finally:
            task_complete("scene:load_data")
    def save_scene(self, path: Optional[str] = None):
        if not self._scene: return
        save_path = path or self._scene.path
        if not save_path:
            Logger.warning("No path for scene save.")
            return
        level = self._compress_level()
        task_start("scene:save", f"Saving scene {os.path.basename(save_path)}...")
        try:
            data = self._scene.serialize()
            _embed_scene_resources(data, self.project_root, self._scene.embedded_resources, compress_level=level)
            self._scene.embedded_resources = data.get("embedded_resources", {})
            self.relativize_scene_paths(data)
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            self._scene.path = save_path
            self._scene.mark_clean()
            Logger.info(f"Scene saved: {save_path}")
            self._emit_event("scene_saved", self._scene)
        except Exception as e:
            Logger.error(f"Failed to save scene: {e}", e)
        finally:
            task_complete("scene:save")
    def new_scene(self, name: str = "NewScene") -> Scene:
        if self._scene:
            self._plugin_manager.notify_scene_unloaded(self._scene)
        from core.components.rendering.postfx.graphics_effect import GraphicsEffect
        GraphicsEffect.cleanup_registry()
        self._scene = Scene(name)
        self._add_default_scene_objects(self._scene)
        self._plugin_manager.notify_scene_loaded(self._scene)
        self._emit_event("scene_loaded", self._scene)
        Logger.info(f"New scene created: {name}")
        return self._scene
    def _add_default_scene_objects(self, scene):
        pass
    def start_play(self):
        if self._play_mode: return
        self._play_mode = True
        self._fixed_accum = 0.0
        self._last_time = time.perf_counter()
        from core.input.input_system import Input
        Input.LoadProjectBindings(self.project_root)
        if self._scene: self._scene.start()
        self._plugin_manager.notify_play_start()
        self._emit_event("play_start", None)
        Logger.info("Play mode started.")
        from core.engine.engine_worker import GameWorker
        cfg = get_global_config()
        update_rate = cfg.get("rendering.tick_rate", 120.0)
        fixed_rate = cfg.get("rendering.fixed_tick_rate", 60.0)
        self._game_worker = GameWorker(self, update_rate, fixed_rate)
        self._game_worker.start()
    def stop_play(self):
        if not self._play_mode: return
        if self._game_worker:
            self._game_worker.stop()
            self._game_worker = None
        from core.audio.audio_system import AudioSourceManager
        mgr = AudioSourceManager.instance()
        if mgr: mgr.stop_all()
        self._play_mode = False
        self._plugin_manager.notify_play_stop()
        self._emit_event("play_stop", None)
        Logger.info("Play mode stopped.")
    def tick(self):
        if not self._play_mode: return
        dt = self.tick_begin()
        MAX_FIXED_STEPS = 5
        for _ in range(MAX_FIXED_STEPS):
            if not self.tick_fixed_step():
                break
        self.tick_update(dt)

    def tick_begin(self) -> float:
        sc = self._scene
        if sc is not None:
            sc.flush_transforms()
        now = time.perf_counter()
        raw_dt = now - self._last_time
        self._last_time = now
        dt = raw_dt * self._time_scale
        self._profiler.start("tick")
        self._fixed_accum += dt
        return dt

    def tick_fixed_step(self) -> bool:
        fd = self._fixed_dt
        if self._fixed_accum < fd:
            return False
        prof = self._profiler
        prof.start("fixed_update")
        sys_plugins = self._plugin_manager.get_system_plugins()
        for p in sys_plugins:
            prof.start(p.NAME)
            try:
                p.pre_step(fd)
            except Exception as e:
                Logger.error(f"Plugin {p.NAME} pre_step exception: {e}")
            prof.stop(p.NAME)
        sc = self._scene
        if sc is not None:
            try:
                sc.fixed_update(fd)
            except Exception as e:
                Logger.error(f"FixedUpdate exception: {e}", e)
        for p in sys_plugins:
            prof.start(p.NAME)
            try:
                p.step(fd)
            except Exception as e:
                Logger.error(f"Plugin {p.NAME} exception: {e}")
            prof.stop(p.NAME)
        self._fixed_accum -= fd
        if self._fixed_accum < 0:
            self._fixed_accum = 0.0
        prof.stop("fixed_update")
        return True

    def tick_update(self, dt: float):
        prof = self._profiler
        prof.start("update")
        sc = self._scene
        if sc is not None:
            try:
                sc.update(dt)
            except Exception as e:
                Logger.error(f"Update exception: {e}", e)
        prof.stop("update")
        self._frame_count += 1
        ts = self._time_scale if self._time_scale > 0.001 else 0.001
        inv = dt / ts
        self._fps_accum += inv
        self._fps_frames += 1
        self._tps_accum += inv
        self._tps_frames += 1
        if self._fps_accum >= 0.5:
            self._fps = self._fps_frames / self._fps_accum
            self._fps_accum = 0.0
            self._fps_frames = 0
            self._tps = self._tps_frames / self._tps_accum
            self._tps_accum = 0.0
            self._tps_frames = 0
        prof.stop("tick")
        rec = self._time_travel_recorder
        if rec is not None and rec.is_recording:
            rec.capture(sc)
    def set_profiler_data(self, key: str, value_ms: float):
        self._profiler.set_value(key, value_ms)
    def capture_profiler_frame(self):
        self._profiler.capture_frame()
    @property
    def profiler_enabled(self) -> bool: return self._profiler.enabled
    @profiler_enabled.setter
    def profiler_enabled(self, v: bool): self._profiler.enabled = v
    def on(self, event: str, callback):
        self._event_listeners.setdefault(event, []).append(callback)
    def off(self, event: str, callback):
        if event in self._event_listeners:
            try: self._event_listeners[event].remove(callback)
            except ValueError: pass
    def _emit_event(self, event: str, data: Any):
        cbs = self._event_listeners.get(event, [])
        from concurrent.futures import as_completed
        if len(cbs) >= 10:
            futures = [_get_pool().submit(cb, data) for cb in cbs]
            for f in as_completed(futures):
                try: f.result()
                except Exception as e: Logger.error(f"Event callback error '{event}': {e}")
        else:
            for cb in cbs:
                try: cb(data)
                except Exception as e: Logger.error(f"Event callback error '{event}': {e}", e)
    @property
    def collab_manager(self):
        return self._collab_manager
    @collab_manager.setter
    def collab_manager(self, v):
        self._collab_manager = v
    @property
    def plugin_ui_registry(self) -> dict:
        return self._plugin_ui_registry

    def shutdown(self):
        try:
            from core.audio.audio_system import AudioSystem
            audio_sys = AudioSystem.instance()
            if audio_sys: audio_sys.shutdown()
        except Exception:
            pass
        self._plugin_manager.shutdown_all()
        Logger.info("Zarin Engine shutdown.")