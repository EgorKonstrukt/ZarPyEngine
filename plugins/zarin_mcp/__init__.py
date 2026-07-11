# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import os

from core.foundation.plugin_manager import PluginBase
from core.foundation.logger import Logger

from plugins.zarin_mcp.registry import Registry
from plugins.zarin_mcp.mcp_server import McpServer


class ZarinMCPPlugin(PluginBase):
    NAME = "ZarinMCP"
    VERSION = "2.0.0"
    DESCRIPTION = "MCP server providing LLMs full access to the engine, scene, assets, editor, and more. 60+ tools, 10+ resources, prompts."
    SYSTEM = False

    def __init__(self):
        super().__init__()
        self._server: McpServer = None
        self._registry = Registry()

    def initialize(self, engine):
        super().initialize(engine)
        self._discover_handlers(engine)
        self._server = McpServer(self._registry, port=self.get_config("port", 9100))
        mcp_mode = os.environ.get("ZARIN_MCP_MODE", "") or getattr(engine, "_mcp_mode", "")
        if mcp_mode == "stdio":
            Logger.info("[ZarinMCP] Running in stdio mode")
            self._server.run_stdio_forever()
        else:
            self._register_ui()
            self._server.start_sse()
            Logger.info(f"[ZarinMCP] SSE server on http://127.0.0.1:{self.get_config('port', 9100)}/sse")

    def shutdown(self):
        if self._server:
            self._server.stop()
        Logger.info("[ZarinMCP] Shutdown.")

    def _register_ui(self):
        self.add_menu_item("ZarinMCP", "Open MCP Server...", lambda: Logger.info(f"ZarinMCP on port {self.get_config('port', 9100)}"))
        self.add_toolbar_button("MCP", lambda: Logger.info(f"ZarinMCP on port {self.get_config('port', 9100)}"),
                                tooltip="ZarinMCP Server Status")

    def _discover_handlers(self, engine):
        import plugins.zarin_mcp.handlers.scene as _scene
        import plugins.zarin_mcp.handlers.components as _components
        import plugins.zarin_mcp.handlers.project as _project
        import plugins.zarin_mcp.handlers.engine as _engine
        import plugins.zarin_mcp.handlers.assets as _assets
        import plugins.zarin_mcp.handlers.editor as _editor
        import plugins.zarin_mcp.handlers.console as _console
        import plugins.zarin_mcp.handlers.resources as _resources

        _scene.register(self._registry, engine)
        _components.register(self._registry, engine)
        _project.register(self._registry, engine)
        _engine.register(self._registry, engine)
        _assets.register(self._registry, engine)
        _editor.register(self._registry, engine)
        _console.register(self._registry, engine)
        _resources.register(self._registry, engine)

        t = len(self._registry.tools)
        r = len(self._registry.resources)
        rt = len(self._registry.resource_templates)
        p = len(self._registry.prompts)
        Logger.info(f"[ZarinMCP] Registered {t} tools, {r} resources, {rt} templates, {p} prompts")


def get_plugin():
    return ZarinMCPPlugin()
