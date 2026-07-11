# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import threading
import traceback
from typing import Optional

import anyio
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, Resource, ResourceTemplate, Prompt, PromptArgument
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request

from core.foundation.logger import Logger


class McpServer:
    def __init__(self, registry, host: str = "127.0.0.1", port: int = 9100):
        self._host = host
        self._port = port
        self._registry = registry
        self._app = Server("zarin-engine")
        self._thread: Optional[threading.Thread] = None
        self._uvicorn_server: Optional[uvicorn.Server] = None
        self._register_handlers()

    def _register_handlers(self):
        tools = self._registry.tools
        resources = self._registry.resources
        templates = self._registry.resource_templates
        prompts = self._registry.prompts

        @self._app.list_tools()
        async def list_tools():
            return [
                Tool(
                    name=name,
                    description=tdef.get("description", ""),
                    inputSchema=tdef.get("inputSchema", {"type": "object", "properties": {}}),
                )
                for name, tdef in tools.items()
            ]

        @self._app.call_tool()
        async def call_tool(name: str, arguments: dict) -> list[TextContent]:
            tdef = tools.get(name)
            if tdef is None:
                return [TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]
            try:
                result = tdef["handler"](**arguments)
                return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]
            except Exception as e:
                return [TextContent(type="text", text=json.dumps(
                    {"error": str(e), "traceback": traceback.format_exc()},
                    ensure_ascii=False,
                ))]

        @self._app.list_resources()
        async def list_resources():
            return [
                Resource(
                    uri=uri,
                    name=rdef.get("name", uri),
                    description=rdef.get("description", ""),
                    mimeType=rdef.get("mimeType", "application/json"),
                )
                for uri, rdef in resources.items()
            ]

        @self._app.read_resource()
        async def read_resource(uri: str):
            rdef = resources.get(uri)
            if rdef is None:
                for tmpl_uri, tmpl_def in templates.items():
                    pattern = tmpl_uri.replace("{", "").replace("}", "")
                    if pattern in uri:
                        rdef = tmpl_def
                        param = uri.split("/")[-1]
                        break
                if rdef is None:
                    raise ValueError(f"Unknown resource: {uri}")
                content = rdef["handler"](param)
            else:
                content = rdef["handler"]()
            return json.dumps(content, ensure_ascii=False, default=str)

        if templates:
            @self._app.list_resource_templates()
            async def list_resource_templates():
                return [
                    ResourceTemplate(
                        uriTemplate=uri,
                        name=tdef.get("name", uri),
                        description=tdef.get("description", ""),
                        mimeType=tdef.get("mimeType", "application/json"),
                    )
                    for uri, tdef in templates.items()
                ]

        if prompts:
            @self._app.list_prompts()
            async def list_prompts():
                return [
                    Prompt(
                        name=name,
                        description=pdef.get("description", ""),
                        arguments=[
                            PromptArgument(
                                name=a["name"],
                                description=a.get("description", ""),
                                required=a.get("required", False),
                            )
                            for a in pdef.get("arguments", [])
                        ],
                    )
                    for name, pdef in prompts.items()
                ]

            @self._app.get_prompt()
            async def get_prompt(name: str, arguments: dict = None):
                pdef = prompts.get(name)
                if pdef is None:
                    raise ValueError(f"Unknown prompt: {name}")
                result = pdef["handler"](**(arguments or {}))
                return result

    async def _run_stdio(self):
        async with stdio_server() as (read_stream, write_stream):
            await self._app.run(
                read_stream, write_stream,
                self._app.create_initialization_options(),
            )

    def run_stdio_forever(self):
        Logger.info("ZarinMCP: stdio MCP server running")
        anyio.run(self._run_stdio, backend="asyncio")

    async def _run_sse(self):
        sse = SseServerTransport("/messages/")

        async def handle_sse(request: Request):
            async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                await self._app.run(
                    streams[0], streams[1],
                    self._app.create_initialization_options(),
                )

        starlette_app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ]
        )

        config = uvicorn.Config(starlette_app, host=self._host, port=self._port, log_level="warning")
        self._uvicorn_server = uvicorn.Server(config)
        await self._uvicorn_server.serve()

    def start_sse(self):
        self._thread = threading.Thread(target=self._run_sse_in_thread, daemon=True, name="MCP-SSE")
        self._thread.start()
        Logger.info(f"ZarinMCP: SSE server on http://{self._host}:{self._port}/sse")

    def _run_sse_in_thread(self):
        anyio.run(self._run_sse, backend="asyncio")

    def stop(self):
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        Logger.info("ZarinMCP: server stopped")
