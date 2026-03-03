"""Lazy Tool Discovery MCP server framework.

Ported from shared/lazy_discovery.py with additions:
- build_mcp_mount() for embedding into FastAPI apps (EcosystemService uses this)
- ResilientSessionManager for container-restart resilience
- run() still works for standalone MCP-only services (backward compat)

Agents see only 3 meta-tools: discover_tools, get_tool_details, invoke_tool.
"""

import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import anyio
from anyio.abc import TaskStatus
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http import (
    StreamableHTTPServerTransport,
)
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

type ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]


# ─────────────────────────────────────────────────────────────
# ResilientSessionManager — survives container restarts
# ─────────────────────────────────────────────────────────────


class ResilientSessionManager(StreamableHTTPSessionManager):
    """Session manager that auto-recreates sessions for unknown session IDs.

    When a Docker container restarts, in-memory session state is lost.
    The MCP client still sends the cached session ID from before the restart.
    The default SDK behavior returns 404 "Session not found", which the client
    treats as fatal.

    This subclass intercepts the unknown-session-ID case and transparently
    recreates the session:
    1. Creates a new transport with the client's original session ID
    2. Starts the MCP server with stateless=True (skips initialize handshake)
    3. Processes the original request — the client never sees an error

    Sessions are preserved (not removed) for future SSE and server-initiated
    notification support.
    """

    async def _handle_stateful_request(
        self,
        scope: Any,
        receive: Any,
        send: Any,
    ) -> None:
        """Override to recover unknown sessions instead of returning 404."""
        from mcp.server.streamable_http import MCP_SESSION_ID_HEADER
        from starlette.requests import Request

        request = Request(scope, receive)
        request_session_id = request.headers.get(MCP_SESSION_ID_HEADER)

        # Known session or no session — delegate to parent (normal flow)
        if request_session_id is None or request_session_id in self._server_instances:
            await super()._handle_stateful_request(scope, receive, send)
            return

        # Unknown session ID — recover instead of 404
        logger.info("Recovering unknown session: %s", request_session_id)

        async with self._session_creation_lock:
            # Double-check under lock (another request may have recreated it)
            if request_session_id in self._server_instances:
                transport = self._server_instances[request_session_id]
                await transport.handle_request(scope, receive, send)
                return

            # Create transport with the CLIENT's original session ID
            http_transport = StreamableHTTPServerTransport(
                mcp_session_id=request_session_id,
                is_json_response_enabled=self.json_response,
                event_store=self.event_store,
                security_settings=self.security_settings,
                retry_interval=self.retry_interval,
            )
            self._server_instances[request_session_id] = http_transport

            # Start server with stateless=True — sets ServerSession to
            # Initialized state, skipping the initialize handshake.
            # The client already initialized before the container restart.
            async def run_recovered_server(
                *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED,
            ) -> None:
                async with http_transport.connect() as streams:
                    read_stream, write_stream = streams
                    task_status.started()
                    try:
                        await self.app.run(
                            read_stream,
                            write_stream,
                            self.app.create_initialization_options(),
                            stateless=True,
                        )
                    except Exception as exc:
                        logger.error(
                            "Recovered session %s crashed: %s",
                            request_session_id,
                            exc,
                            exc_info=True,
                        )
                    finally:
                        if (
                            request_session_id in self._server_instances
                            and not http_transport.is_terminated
                        ):
                            logger.info(
                                "Cleaning up crashed recovered session %s",
                                request_session_id,
                            )
                            del self._server_instances[request_session_id]

            assert self._task_group is not None
            await self._task_group.start(run_recovered_server)
            await http_transport.handle_request(scope, receive, send)


class ToolEntry:
    """Internal representation of a registered tool."""

    __slots__ = ("name", "description", "category", "input_schema", "handler", "examples")

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        input_schema: dict[str, Any],
        handler: ToolHandler,
        examples: list[str] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.category = category
        self.input_schema = input_schema
        self.handler = handler
        self.examples = examples or []


class LazyMCPServer:
    """MCP server with lazy tool discovery.

    Instead of exposing N tool definitions to the agent, exposes 3 meta-tools
    that let the agent discover, inspect, and invoke tools on demand.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._server = Server(name)
        self._registry: dict[str, ToolEntry] = {}
        self._asgi_app: Any = None
        self._session_manager: Any = None
        self._setup_meta_tools()

    def tool(
        self,
        name: str,
        description: str,
        category: str,
        input_schema: dict[str, Any] | None = None,
        examples: list[str] | None = None,
    ) -> Callable[[ToolHandler], ToolHandler]:
        """Decorator to register a tool handler."""
        schema = input_schema or {"type": "object", "properties": {}}

        def decorator(fn: ToolHandler) -> ToolHandler:
            self._registry[name] = ToolEntry(
                name=name,
                description=description,
                category=category,
                input_schema=schema,
                handler=fn,
                examples=examples or [],
            )
            return fn

        return decorator

    def register(
        self,
        name: str,
        description: str,
        category: str,
        handler: ToolHandler,
        input_schema: dict[str, Any] | None = None,
        examples: list[str] | None = None,
    ) -> None:
        """Imperative tool registration (alternative to decorator)."""
        schema = input_schema or {"type": "object", "properties": {}}
        self._registry[name] = ToolEntry(
            name=name,
            description=description,
            category=category,
            input_schema=schema,
            handler=handler,
            examples=examples or [],
        )

    @property
    def categories(self) -> list[str]:
        return sorted({t.category for t in self._registry.values()})

    @property
    def tool_count(self) -> int:
        return len(self._registry)

    def _build_catalog(self, category: str | None = None) -> list[dict[str, str]]:
        entries = list(self._registry.values())
        if category:
            entries = [t for t in entries if t.category == category]
        return [
            {"name": t.name, "category": t.category, "description": t.description}
            for t in sorted(entries, key=lambda t: (t.category, t.name))
        ]

    def _setup_meta_tools(self) -> None:
        """Wire up the 3 meta-tools to the MCP server."""
        categories_desc = "Categories are dynamically listed in discover_tools output."

        meta_tools = [
            Tool(
                name="discover_tools",
                description=(
                    f"List available tools on the {self.name} server. "
                    "Returns compact catalog: name, category, one-line description. "
                    "Optionally filter by category. Call this FIRST."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": f"Optional category filter. {categories_desc}",
                        },
                    },
                },
            ),
            Tool(
                name="get_tool_details",
                description=(
                    "Get full input schema, description, and usage examples for a specific tool. "
                    "Call BEFORE invoke_tool to understand what arguments are needed."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Tool name from discover_tools",
                        },
                    },
                    "required": ["tool_name"],
                },
            ),
            Tool(
                name="invoke_tool",
                description=(
                    "Execute a registered tool by name with the given arguments. "
                    "Use discover_tools + get_tool_details first."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Tool name to invoke",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Tool arguments (see get_tool_details for schema)",
                            "default": {},
                        },
                    },
                    "required": ["tool_name"],
                },
            ),
        ]

        @self._server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
        async def list_tools() -> list[Tool]:
            return meta_tools

        @self._server.call_tool()  # type: ignore[untyped-decorator]
        async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            text = await self._handle_meta(name, arguments)
            return [TextContent(type="text", text=text)]

    async def _handle_meta(self, name: str, args: dict[str, Any]) -> str:
        if name == "discover_tools":
            return await self._meta_discover(args)
        elif name == "get_tool_details":
            return await self._meta_get_details(args)
        elif name == "invoke_tool":
            return await self._meta_invoke(args)
        return json.dumps({"error": f"Unknown meta-tool: {name}"})

    async def _meta_discover(self, args: dict[str, Any]) -> str:
        category = args.get("category")
        catalog = self._build_catalog(category)
        categories = self.categories

        result: dict[str, Any] = {
            "server": self.name,
            "total_tools": self.tool_count,
            "categories": categories,
        }

        if category and category not in categories:
            result["error"] = f"Unknown category '{category}'. Valid: {categories}"
        else:
            if category:
                result["filter"] = category
            result["tools"] = catalog
            result["hint"] = "Use get_tool_details to see full schema before invoking."

        return json.dumps(result, indent=2)

    async def _meta_get_details(self, args: dict[str, Any]) -> str:
        tool_name = args.get("tool_name", "")
        entry = self._registry.get(tool_name)

        if not entry:
            return json.dumps(
                {
                    "error": f"Unknown tool: '{tool_name}'",
                    "available_tools": sorted(self._registry.keys()),
                },
                indent=2,
            )

        return json.dumps(
            {
                "name": entry.name,
                "description": entry.description,
                "category": entry.category,
                "input_schema": entry.input_schema,
                "examples": entry.examples,
                "hint": "Use invoke_tool with this tool_name and arguments to execute.",
            },
            indent=2,
        )

    async def _meta_invoke(self, args: dict[str, Any]) -> str:
        tool_name = args.get("tool_name", "")
        tool_args = args.get("arguments", {})

        entry = self._registry.get(tool_name)
        if not entry:
            return json.dumps(
                {
                    "error": f"Unknown tool: '{tool_name}'",
                    "available_tools": sorted(self._registry.keys()),
                    "hint": "Use discover_tools to see what's available.",
                },
                indent=2,
            )

        required = entry.input_schema.get("required", [])
        missing = [r for r in required if r not in tool_args]
        if missing:
            return json.dumps(
                {
                    "error": f"Missing required arguments: {missing}",
                    "input_schema": entry.input_schema,
                },
                indent=2,
            )

        return await entry.handler(tool_args)

    # ─────────────────────────────────────────────────────────────
    # Mount: embed MCP into a parent ASGI app (used by EcosystemService)
    # ─────────────────────────────────────────────────────────────

    def build_mcp_mount(self) -> tuple[Any, Any]:
        """Return (mcp_asgi_app, session_manager) for mounting into a parent app.

        The parent app is responsible for running the session manager lifespan.

        Usage:
            mcp_app, session_mgr = mcp.build_mcp_mount()
            app = FastAPI(lifespan=combined_lifespan)
            app.mount("/mcp", mcp_app)
        """
        from mcp.server.fastmcp.server import StreamableHTTPASGIApp

        if self._session_manager is None:
            self._session_manager = ResilientSessionManager(
                app=self._server,
                json_response=False,
                stateless=False,
            )
        return StreamableHTTPASGIApp(self._session_manager), self._session_manager

    # ─────────────────────────────────────────────────────────────
    # Standalone: run as MCP-only server (backward compat)
    # ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Start standalone MCP server. Transport selected by MCP_TRANSPORT env var.

        - "stdio" (default): stdin/stdout subprocess for Claude Code
        - "http": Streamable HTTP for Docker deployment
        """
        transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()
        if transport == "http":
            self._run_http()
        else:
            asyncio.run(self._run_stdio())

    async def _run_stdio(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            init_options = self._server.create_initialization_options()
            await self._server.run(read_stream, write_stream, init_options)

    def _run_http(self) -> None:
        import uvicorn

        host = os.environ.get("MCP_HOST", "0.0.0.0")  # nosec B104
        port = int(os.environ.get("MCP_PORT", "8000"))
        app = self.build_asgi_app()
        uvicorn.run(app, host=host, port=port)

    def build_asgi_app(self) -> Any:
        """Build standalone Starlette ASGI app with /mcp and /health.

        Used for MCP-only services (backward compat). EcosystemService uses
        build_mcp_mount() instead to embed MCP into FastAPI.
        """
        if self._asgi_app is not None:
            return self._asgi_app

        from mcp.server.fastmcp.server import StreamableHTTPASGIApp
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        if self._session_manager is None:
            self._session_manager = ResilientSessionManager(
                app=self._server,
                json_response=False,
                stateless=False,
            )

        mcp_asgi = StreamableHTTPASGIApp(self._session_manager)

        server_name = self.name
        tool_count_ref = self

        async def health(request: Request) -> JSONResponse:
            return JSONResponse(
                {
                    "status": "ok",
                    "service": server_name,
                    "tools": tool_count_ref.tool_count,
                    "transport": "streamable-http",
                }
            )

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            async with self._session_manager.run():
                yield

        self._asgi_app = Starlette(
            routes=[
                Mount("/mcp", app=mcp_asgi),
                Route("/health", endpoint=health, methods=["GET"]),
            ],
            lifespan=lifespan,
        )

        return self._asgi_app
