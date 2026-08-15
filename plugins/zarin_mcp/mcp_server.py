# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
#
# Copyright (c) 2026 Zarrakun

from __future__ import annotations
import json
import threading
import time as _time
import traceback
from collections import deque
from typing import Optional

import anyio
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import (
    Tool,
    TextContent,
    TextResourceContents,
    Resource,
    ResourceTemplate,
    Prompt,
    PromptArgument,
    PromptMessage,
    ListToolsResult,
    CallToolResult,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ReadResourceResult,
    ListPromptsResult,
    GetPromptResult,
    PaginatedRequestParams,
    CallToolRequestParams,
    ReadResourceRequestParams,
    GetPromptRequestParams,
)
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
        self._activity: deque = deque(maxlen=300)
        self._activity_listeners: list = []
        self._register_handlers()

    def add_activity_listener(self, fn):
        self._activity_listeners.append(fn)

    def remove_activity_listener(self, fn):
        if fn in self._activity_listeners:
            self._activity_listeners.remove(fn)

    def recent_actions(self, limit=100):
        return list(self._activity)[-limit:]

    def clear_activity(self):
        self._activity.clear()

    def _record_activity(self, kind, name, **fields):
        rec = {"kind": kind, "name": name, "timestamp": _time.time(), **fields}
        self._activity.append(rec)
        for fn in list(self._activity_listeners):
            try:
                fn(rec)
            except Exception:
                pass
        return rec

    @property
    def host(self):
        return self._host

    @property
    def port(self):
        return self._port

    @port.setter
    def port(self, value):
        self._port = value

    @property
    def is_running(self):
        return bool(self._thread and self._thread.is_alive())

    def restart(self):
        self.stop()
        self.start_sse()

    @staticmethod
    def _prompt_messages(result):
        if isinstance(result, str):
            return [PromptMessage(role="user", content=TextContent(type="text", text=result))]
        messages = []
        for m in (result or {}).get("messages", []):
            content = m.get("content", {})
            if isinstance(content, dict) and content.get("type") == "text":
                text = content.get("text", "")
            elif isinstance(content, str):
                text = content
            else:
                text = str(content)
            messages.append(
                PromptMessage(role=m.get("role", "user"), content=TextContent(type="text", text=text))
            )
        return messages

    def _register_handlers(self):
        tools = self._registry.tools
        resources = self._registry.resources
        templates = self._registry.resource_templates
        prompts = self._registry.prompts

        async def list_tools(ctx, params):
            return ListToolsResult(tools=[
                Tool(
                    name=name,
                    description=tdef.get("description", ""),
                    inputSchema=tdef.get("inputSchema", {"type": "object", "properties": {}}),
                )
                for name, tdef in tools.items()
            ])

        async def call_tool(ctx, params):
            tdef = tools.get(params.name)
            if tdef is None:
                args = params.arguments or {}
                Logger.warning(f"[ZarinMCP] Agent called unknown tool {params.name}("
                               f"{json.dumps(args, ensure_ascii=False, default=str)})")
                self._record_activity("tool", params.name, args=args, status="error", error="Unknown tool")
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {params.name}"}))],
                    isError=True,
                )
            args = params.arguments or {}
            compact = json.dumps(args, ensure_ascii=False, default=str)
            if len(compact) > 500:
                compact = compact[:500] + "..."
            started = _time.perf_counter()
            Logger.info(f"[ZarinMCP] Agent calls {params.name}({compact})")
            try:
                result = tdef["handler"](**args)
                ms = (_time.perf_counter() - started) * 1000.0
                text = json.dumps(result, ensure_ascii=False, default=str)
                summary = text if len(text) <= 400 else text[:400] + "..."
                Logger.info(f"[ZarinMCP] {params.name} OK in {ms:.0f} ms: {summary}")
                self._record_activity("tool", params.name, args=args, status="ok",
                                      ms=round(ms, 1), summary=summary)
                return CallToolResult(content=[TextContent(type="text", text=text)])
            except Exception as e:
                ms = (_time.perf_counter() - started) * 1000.0
                tb = traceback.format_exc()
                Logger.error(f"[ZarinMCP] {params.name} ERROR in {ms:.0f} ms: {e}")
                self._record_activity("tool", params.name, args=args, status="error",
                                      ms=round(ms, 1), error=str(e))
                return CallToolResult(
                    content=[TextContent(type="text", text=json.dumps(
                        {"error": str(e), "traceback": tb}, ensure_ascii=False))],
                    isError=True,
                )

        async def list_resources(ctx, params):
            return ListResourcesResult(resources=[
                Resource(
                    uri=uri,
                    name=rdef.get("name", uri),
                    description=rdef.get("description", ""),
                    mimeType=rdef.get("mimeType", "application/json"),
                )
                for uri, rdef in resources.items()
            ])

        async def read_resource(ctx, params):
            uri = params.uri
            started = _time.perf_counter()
            rdef = resources.get(uri)
            param = None
            if rdef is None:
                for tmpl_uri, tmpl_def in templates.items():
                    marker = tmpl_uri.find("{")
                    pattern = tmpl_uri[:marker] if marker != -1 else tmpl_uri
                    if pattern in uri:
                        rdef = tmpl_def
                        param = uri.split("/")[-1]
                        break
                if rdef is None:
                    self._record_activity("resource", uri, status="error", error="Unknown resource")
                    raise ValueError(f"Unknown resource: {uri}")
            try:
                if param is not None:
                    content = rdef["handler"](param)
                else:
                    content = rdef["handler"]()
            except Exception as e:
                self._record_activity("resource", uri, status="error", error=str(e))
                Logger.debug(f"[ZarinMCP] Agent reads {uri} ERROR: {e}")
                raise
            ms = (_time.perf_counter() - started) * 1000.0
            Logger.debug(f"[ZarinMCP] Agent reads {uri} in {ms:.0f} ms")
            self._record_activity("resource", uri, status="ok", ms=round(ms, 1))
            text = json.dumps(content, ensure_ascii=False, default=str)
            return ReadResourceResult(contents=[TextResourceContents(uri=uri, text=text)])

        async def list_resource_templates(ctx, params):
            return ListResourceTemplatesResult(resourceTemplates=[
                ResourceTemplate(
                    uriTemplate=uri,
                    name=tdef.get("name", uri),
                    description=tdef.get("description", ""),
                    mimeType=tdef.get("mimeType", "application/json"),
                )
                for uri, tdef in templates.items()
            ])

        async def list_prompts(ctx, params):
            return ListPromptsResult(prompts=[
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
            ])

        async def get_prompt(ctx, params):
            pdef = prompts.get(params.name)
            if pdef is None:
                raise ValueError(f"Unknown prompt: {params.name}")
            args = params.arguments or {}
            started = _time.perf_counter()
            try:
                result = pdef["handler"](**args)
                ms = (_time.perf_counter() - started) * 1000.0
                Logger.debug(f"[ZarinMCP] Agent gets prompt {params.name} in {ms:.0f} ms")
                self._record_activity("prompt", params.name, args=args, status="ok", ms=round(ms, 1))
                return GetPromptResult(messages=self._prompt_messages(result))
            except Exception as e:
                ms = (_time.perf_counter() - started) * 1000.0
                self._record_activity("prompt", params.name, args=args, status="error",
                                      ms=round(ms, 1), error=str(e))
                Logger.debug(f"[ZarinMCP] Agent gets prompt {params.name} ERROR: {e}")
                raise

        self._app.add_request_handler("tools/list", PaginatedRequestParams, list_tools)
        self._app.add_request_handler("tools/call", CallToolRequestParams, call_tool)
        self._app.add_request_handler("resources/list", PaginatedRequestParams, list_resources)
        self._app.add_request_handler("resources/read", ReadResourceRequestParams, read_resource)

        if templates:
            self._app.add_request_handler("resources/templates/list", PaginatedRequestParams, list_resource_templates)

        if prompts:
            self._app.add_request_handler("prompts/list", PaginatedRequestParams, list_prompts)
            self._app.add_request_handler("prompts/get", GetPromptRequestParams, get_prompt)

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
