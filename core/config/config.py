# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Optional
from core.foundation.logger import Logger
class Config:
    def __init__(self, path: str, defaults: Optional[dict] = None):
        self._path = path
        self._data: dict = dict(defaults or {})
        self._defaults: dict = dict(defaults or {})
        self._restart_keys: set = set()
        self._listeners: list = []
        self._load()
    @property
    def path(self) -> str: return self._path
    def mark_restart(self, key: str):
        self._restart_keys.add(key)
    def is_restart_key(self, key: str) -> bool:
        return key in self._restart_keys
    def on_changed(self, callback):
        self._listeners.append(callback)
    def _notify(self, key: str, value: Any):
        for cb in self._listeners:
            try:
                cb(key, value)
            except Exception:
                pass
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        d = self._data
        for k in keys:
            if isinstance(d, dict):
                d = d.get(k)
            else:
                return default
        return d if d is not None else default
    def set(self, key: str, value: Any, notify: bool = True):
        keys = key.split(".")
        d = self._data
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
        if notify:
            self._notify(key, value)
    def has(self, key: str) -> bool:
        return self.get(key, self) is not self
    def reset(self, key: Optional[str] = None):
        if key is None:
            self._data = dict(self._defaults)
            return
        keys = key.split(".")
        d = self._data
        dd = self._defaults
        for k in keys[:-1]:
            if isinstance(d, dict):
                d = d.get(k, {})
            else:
                return
            if isinstance(dd, dict):
                dd = dd.get(k, {})
            else:
                dd = {}
        if keys[-1] in dd:
            d[keys[-1]] = dd[keys[-1]]
        elif keys[-1] in d:
            del d[keys[-1]]
    def to_dict(self) -> dict:
        return dict(self._data)
    def update(self, data: dict):
        self._deep_update(self._data, data)
    def _deep_update(self, target: dict, source: dict):
        for k, v in source.items():
            if k in target and isinstance(target[k], dict) and isinstance(v, dict):
                self._deep_update(target[k], v)
            else:
                target[k] = v
    def save(self):
        try:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._data, f, indent=2)
        except Exception as e:
            Logger.error(f"Failed to save config to {self._path}: {e}")
    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path) as f:
                    loaded = json.load(f)
                    self._deep_update(self._data, loaded)
                    self._cleanup_old_keys()
        except Exception as e:
            Logger.warning(f"Failed to load config from {self._path}: {e}")
    _DEPRECATED_KEYS: set = {
        "ambient_r", "ambient_g", "ambient_b",
        "selection_outline_r", "selection_outline_g", "selection_outline_b", "selection_outline_a",
        "selection_bounds_color_r", "selection_bounds_color_g", "selection_bounds_color_b",
        "clear_r", "clear_g", "clear_b",
        "no_scene_r", "no_scene_g", "no_scene_b",
        "bg_r", "bg_g", "bg_b", "bg_a",
        "tri_r", "tri_g", "tri_b", "tri_a",
        "wire_r", "wire_g", "wire_b", "wire_a",
        "sky_top_r", "sky_top_g", "sky_top_b",
        "sky_bottom_r", "sky_bottom_g", "sky_bottom_b",
        "sky_horizon_r", "sky_horizon_g", "sky_horizon_b", "sky_horizon_power",
        "editor.selection_bounds", "editor.selection_bounds_speed",
    }
    def _cleanup_old_keys(self):
        simple = set()
        dotted = {}
        for k in self._DEPRECATED_KEYS:
            if "." in k:
                prefix, leaf = k.split(".", 1)
                dotted.setdefault(prefix, set()).add(leaf)
            else:
                simple.add(k)
        def _clean(d):
            to_del = [k for k in d if k in simple]
            for k in to_del:
                del d[k]
            for prefix, leaves in dotted.items():
                if prefix in d and isinstance(d[prefix], dict):
                    for leaf in leaves:
                        d[prefix].pop(leaf, None)
            for v in d.values():
                if isinstance(v, dict):
                    _clean(v)
        _clean(self._data)
_global_config: Optional[Config] = None
_project_config: Optional[Config] = None
def get_global_config() -> Config:
    global _global_config
    if _global_config is None:
        path = os.path.join(str(Path.home()), ".zarin", "settings.json")
        _global_config = Config(path, {
            "editor": {
                "theme": "dark",
                "font_size": 12,
                "ui_scale": 75,
                "language": "en",
                "auto_save": True,
                "auto_save_interval": 300,
                "thumb_cache_mode": "metadata",
                "thumb_resolution": 512,
            },
            "camera": {
                "fov": 70.0,
                "near": 0.01,
                "far": 1000.0,
                "move_speed": 5.0,
                "fast_mult": 3.0,
                "rotate_speed": 0.3,
                "zoom_speed": 4.0,
                "pan_speed": 0.01,
                "zoom_strength": 0.3,
                "damping": 8.0,
                "acceleration": 12.0,
                "transition_speed": 2.5,
                "zoom_smooth_speed": 15.0,
                "use_ortho_in_2d": True,
                "speed_boost_enabled": True,
                "speed_boost_mult": 3.0,
                "speed_boost_ramp_time": 2.0
            },
            "profiler": {
                "enabled": True,
                "update_interval": 0.5,
                "max_samples": 200,
                "refresh_interval": 200
            },
            "rendering": {
                "vsync": True,
                "target_fps": 60,
                "shadow_resolution": 4096,
                "show_grid": True,
                "grid_size": 10.0,
                "grid_world_size": 2000.0,
                "grid_2d_mode": False,
                "grid_zoom_distance": 5.0,
                "ambient": [0.26, 0.28, 0.34],
                "selection_outline": [0.8, 0.5, 0.1, 1.0],
                "selection_outline_thickness": 0.03,
                "tick_rate": 120.0,
                "fixed_tick_rate": 60.0,
                "max_lights": 8,
                "light_scale": 1.0,
                "play_viewport_throttle": "editor",
                "play_viewport_throttle_step": 2,
                "bvh_build_mode": "fast",
            },
            "gizmo": {
                "handle_size": 0.1,
                "base_axis_length": 1.0,
                "plane_handle_size": 0.22,
                "pick_threshold": 30.0,
                "arrow_size_ratio": 0.2,
                "center_handle_size": 0.14,
                "screen_axis_length": 100.0,
                "line_width": 2.5,
                "show_delta_label": True,
                "smooth_snap": True,
                "smooth_snap_speed": 0.25,
                "show_icons": True,
                "icon_scale": 5.0,
                "selection_bounds": True,
                "selection_bounds_speed": 13.0,
                "selection_bounds_color": [0.25, 0.55, 1.0]
            },
            "console": {
                "font_size": 10,
                "font_family": "Segoe UI",
                "max_blocks": 2000,
                "refresh_interval": 100
            },
            "terminal": {
                "font_size": 10,
                "font_family": "Segoe UI"
            },
            "viewport": {
                "clear": [0.18, 0.18, 0.18],
                "no_scene": [0.12, 0.12, 0.12],
                "update_interval": 16,
                "grid_step": 10.0,
                "overlay_fps": 60
            },
            "collab": {
                "cursor_rate": 30.0,
                "camera_rate": 15.0,
                "transform_rate": 20.0,
                "gizmo_rate": 10.0,
                "ping_interval": 3.0,
                "poll_interval": 8
            },
            "undo": {
                "max_stack": 200
            },
            "engine": {
                "time_scale": 1.0,
                "fixed_update_dt": 0.02,
                "python_jit": False,
                "python_optimize": 0,
                "python_unbuffered": False,
                "python_no_bytecode": False,
            },
            "hierarchy": {
                "refresh_interval": 500
            },
            "inspector": {
                "refresh_interval": 100
            },
            "project": {
                "thumb_size": 64,
                "dual_pane": True
            },
            "plotter": {
                "refresh_interval": 100,
                "history_limit": 2000
            },
            "audio": {
                "enable_audio": True,
                "device_name": "",
                "sample_rate": 48000,
                "master_volume": 1.0,
                "sfx_volume": 1.0,
                "music_volume": 0.8,
                "voice_volume": 1.0,
                "ambient_volume": 1.0,
                "max_sources": 32,
                "stream_buffer_size": 4096,
                "distance_model": "inverse_distance_clamped",
                "doppler_factor": 1.0,
                "speed_of_sound": 343.3,
            },
            "file_assoc": {
                "registered_extensions": ""
            },
            "mesh_preview": {
                "camera_rot_x": 30.0,
                "camera_rot_y": -45.0,
                "bg": [0.0, 0.0, 0.0, 0.0],
                "tri": [0.39, 0.63, 0.86, 0.16],
                "wire": [0.71, 0.82, 0.94, 0.78],
                "wire_width": 1.0,
            }
        })
        _global_config._data.pop("physics", None)
        for _rk in ["editor.theme", "editor.font_size", "editor.ui_scale", "editor.language",
                     "engine.python_jit", "engine.python_optimize", "engine.python_unbuffered",
                     "engine.python_no_bytecode"]:
            _global_config.mark_restart(_rk)
    return _global_config
def get_project_config(project_path: str) -> Config:
    global _project_config
    if _project_config is None or _project_config.path != os.path.join(project_path, "ProjectSettings.json"):
        path = os.path.join(project_path, "ProjectSettings.json")
        _project_config = Config(path, {
            "project": {
                "name": "Untitled",
                "version": "1.0.0",
                "default_scene": ""
            },
            "input": {
                "control_scheme": "fps",
                "horizontal": "a,d",
                "vertical": "w,s",
                "jump": "space",
                "fire": "mouse 0",
                "crouch": "left ctrl",
                "sprint": "left shift",
                "interact": "e",
                "reload": "r",
                "mouse_axis_x": "mouse x",
                "mouse_axis_y": "mouse y",
                "mouse_sensitivity": 1.0,
                "invert_mouse_x": False,
                "invert_mouse_y": False,
                "axis_gravity": 3.0,
                "axis_sensitivity": 1.0,
                "axis_dead": 0.001
            },
            "rendering": {
                "render_pipeline": "forward",
                "anti_aliasing": "none",
                "shadow_distance": 50.0
            },
            "physics": {
                "solver": "culverin",
                "physx_device": "cpu",
                "gravity_x": 0.0,
                "gravity_y": -9.81,
                "gravity_z": 0.0,
                "fixed_time_step": 0.02,
                "num_sub_steps": 2,
                "solver_iterations": 10,
                "erp": 0.2,
                "contact_erp": 0.2,
                "friction_erp": 0.0,
                "contact_breaking_threshold": 0.02,
                "restitution": 0.0,
                "linear_damping": 0.04,
                "angular_damping": 0.04,
                "max_contacts_per_body": 64,
                "simulation_mode": "multi_threaded",
                "culverin_max_bodies": 65536,
                "culverin_max_pairs": 65536,
                "culverin_max_contact_constraints": 65536,
                "culverin_temp_allocator_size": 16777216,
                "culverin_max_physics_jobs": 0,
                "culverin_max_physics_barriers": 0,
                "culverin_num_threads": 0,
                "culverin_penetration_slop": 0.02,
                "culverin_enable_ccd": False,
                "culverin_enable_sleeping": False,
                "layer_names": ["Default","TransparentFX","Ignore Raycast","Water","UI","Player","Enemy","Projectile","Trigger","Ground","Layer10","Layer11","Layer12","Layer13","Layer14","Layer15"],
                "collision_matrix": [65535,65535,0,65535,16,65535,65535,65535,65535,65535,65535,65535,65535,65535,65535,65535]
            },
            "audio": {
                "master_volume": 1.0,
                "sfx_volume": 1.0,
                "music_volume": 0.8,
                "voice_volume": 1.0,
                "ambient_volume": 1.0,
                "enable_spatialization": True,
                "enable_reverb": True,
                "enable_occlusion": True,
                "distance_model": "inverse_distance_clamped",
                "doppler_factor": 1.0,
                "speed_of_sound": 343.3,
                "priority_threshold": 0.1,
            }
        })
    return _project_config
