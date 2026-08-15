# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import traceback

from plugins.zarin_mcp.main_thread import run_on_main_thread


def register(registry, engine):

    @registry.tool(
        "console_get_log",
        "Get recent console log entries",
        {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": ["", "DEBUG", "INFO", "WARNING", "ERROR"],
                    "description": "Filter by log level (empty = all)",
                    "default": "",
                },
                "max_entries": {
                    "type": "integer",
                    "description": "Maximum entries to return",
                    "default": 50,
                },
            },
        },
    )
    def console_get_log(level="", max_entries=50):
        from core.foundation.logger import Logger
        entries = Logger.get_entries()
        if level:
            entries = [e for e in entries if e.level.name == level.upper()]
        entries = entries[-max_entries:]
        return {
            "entries": [
                {
                    "level": e.level.name,
                    "message": e.message,
                    "timestamp": e.timestamp,
                    "has_traceback": hasattr(e, 'traceback_str') and bool(e.traceback_str),
                }
                for e in entries
            ],
            "count": len(entries),
            "total": len(Logger.get_entries()),
        }

    @registry.tool(
        "console_clear",
        "Clear the console log",
        {"type": "object", "properties": {}},
    )
    def console_clear():
        from core.foundation.logger import Logger
        Logger.clear()
        return {"message": "Console cleared"}

    @registry.tool(
        "console_execute_code",
        "Execute Python code in the engine context",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "context": {
                    "type": "string",
                    "enum": ["", "engine", "scene"],
                    "description": "Execution context",
                    "default": "",
                },
            },
            "required": ["code"],
        },
    )
    def console_execute_code(code="", context=""):
        try:
            loc = {}
            if context == "engine":
                loc["engine"] = engine
            elif context == "scene":
                loc["scene"] = engine.scene if engine else None
            else:
                loc["engine"] = engine
                loc["scene"] = engine.scene if engine else None
                import core.ecs.ecs as ecs
                loc["Entity"] = ecs.Entity
                loc["Component"] = ecs.Component
                loc["ComponentRegistry"] = ecs.ComponentRegistry

            def _run():
                exec(code, globals(), loc)
                return loc.get("_result", "Code executed successfully (no _result variable set)")

            result = run_on_main_thread(_run)
            return {"result": str(result)}
        except Exception as e:
            return {"error": f"Execution error: {e}\n{traceback.format_exc()}"}
