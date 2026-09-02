# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations

from plugins.zarin_mcp.main_thread import run_on_main_thread


def register(registry, engine):

    @registry.tool(
        "engine_get_status",
        "Get engine runtime status (play mode, fps, scene info)",
        {"type": "object", "properties": {}},
    )
    def engine_get_status():
        scene = engine.scene
        return {
            "play_mode": engine.play_mode,
            "fps": engine.fps,
            "frame_count": engine.frame_count,
            "time_scale": engine.time_scale,
            "fixed_dt": engine.fixed_dt,
            "scene_name": scene.name if scene else None,
            "scene_path": scene.path if scene else None,
            "scene_dirty": scene.dirty if scene else None,
            "entity_count": len(scene.get_all_entities()) if scene else 0,
        }

    @registry.tool(
        "engine_play",
        "Start play mode",
        {"type": "object", "properties": {}},
    )
    def engine_play():
        if engine is None:
            return {"error": "Engine not available"}
        run_on_main_thread(engine.start_play)
        return {"message": "Play mode started"}

    @registry.tool(
        "engine_stop",
        "Stop play mode",
        {"type": "object", "properties": {}},
    )
    def engine_stop():
        if engine is None:
            return {"error": "Engine not available"}
        run_on_main_thread(engine.stop_play)
        return {"message": "Play mode stopped"}

    @registry.tool(
        "engine_set_time_scale",
        "Set engine time scale for slow-motion / fast-forward",
        {
            "type": "object",
            "properties": {
                "scale": {
                    "type": "number",
                    "description": "Time scale (0 = pause, 1 = normal, 2 = double speed)",
                }
            },
            "required": ["scale"],
        },
    )
    def engine_set_time_scale(scale=1.0):
        if engine is None:
            return {"error": "Engine not available"}
        engine.time_scale = max(0.0, scale)
        return {"message": f"Time scale set to {engine.time_scale}"}

    @registry.tool(
        "engine_get_profiler_data",
        "Get engine profiler timing data",
        {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Optional specific profiler key (e.g. 'render_ms', 'physics_ms')",
                    "default": "",
                }
            },
        },
    )
    def engine_get_profiler_data(key=""):
        if key:
            return {"profiler": {key: engine.get_profiler_data(key)}}
        return {"profiler": engine.profiler_data}

    @registry.tool(
        "engine_get_performance_snapshot",
        "Get a comprehensive performance snapshot",
        {"type": "object", "properties": {}},
    )
    def engine_get_performance_snapshot():
        return {
            "fps": engine.fps,
            "frame_count": engine.frame_count,
            "play_mode": engine.play_mode,
            "time_scale": engine.time_scale,
            "profiler": engine.profiler_data,
            "entity_count": len(engine.scene.get_all_entities()) if engine.scene else 0,
        }

    @registry.tool(
        "engine_get_events",
        "List registered engine event listeners",
        {"type": "object", "properties": {}},
    )
    def engine_get_events():
        if engine is None:
            return {"error": "Engine not available"}
        events = {}
        for ev, cbs in engine._event_listeners.items():
            events[ev] = len(cbs)
        return {"events": events}


