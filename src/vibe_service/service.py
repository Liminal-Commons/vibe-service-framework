"""EcosystemService — the one right way to create an ecosystem service.

Creates a dual-protocol server: FastAPI (REST at /api/) + LazyMCPServer (MCP at /mcp).
Both interfaces share the same domain logic layer.
"""

import contextlib
import functools
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any

from fastapi import FastAPI

from vibe_service.auth import make_auth_dependency
from vibe_service.config import BaseServiceConfig
from vibe_service.errors import ServiceError
from vibe_service.errors import error as _error
from vibe_service.errors import ok as _ok
from vibe_service.health import build_health_router
from vibe_service.logging import configure_logging
from vibe_service.mcp import LazyMCPServer, ToolHandler

_logger = logging.getLogger(__name__)


class EcosystemService:
    """Dual-protocol ecosystem service: REST + MCP + Health.

    Usage:
        service = EcosystemService("my-service", Config())

        @service.tool(name="do_thing", ...)
        async def do_thing(args): ...

        @service.api.get("/api/things")
        async def list_things(): ...

        service.run()
    """

    def __init__(self, name: str, config: BaseServiceConfig) -> None:
        self.name = name
        self.config = config
        self._start_time = time.monotonic()

        # MCP layer
        self.mcp = LazyMCPServer(name)

        # Logging
        configure_logging(level=config.log_level, service_name=name)

        # MCP mount (for embedding into FastAPI)
        self._mcp_app, self._session_manager = self.mcp.build_mcp_mount()

        # FastAPI app with lifespan that manages MCP session manager
        @contextlib.asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[None]:
            async with self._session_manager.run():
                yield

        self.api = FastAPI(title=name, version=config.service_version, lifespan=lifespan)

        # Mount MCP at /mcp
        self.api.mount("/mcp", self._mcp_app)

        # Health endpoint
        health_router = build_health_router(
            service_name=name,
            service_version=config.service_version,
            tool_count_fn=lambda: self.mcp.tool_count,
            start_time=self._start_time,
        )
        self.api.include_router(health_router)

        # Auth dependency (available for services that want it on their routes)
        self.auth = make_auth_dependency(config.service_key)

    def tool(
        self,
        name: str,
        description: str,
        category: str,
        input_schema: dict[str, Any] | None = None,
        examples: list[str] | None = None,
        auto_envelope: bool = False,
    ) -> Callable[..., Any]:
        """Register an MCP tool. Delegates to LazyMCPServer.

        When auto_envelope=True, the handler returns plain data and raises
        ServiceError for errors. The framework handles json.dumps + ok/error:

            @service.tool(name="my_tool", ..., auto_envelope=True)
            async def my_tool(args):
                if not args.get("id"):
                    raise ServiceError("VALIDATION", "id is required")
                return {"result": "data"}  # auto-wrapped in ok() + json.dumps
        """
        mcp_decorator = self.mcp.tool(
            name=name,
            description=description,
            category=category,
            input_schema=input_schema,
            examples=examples,
        )

        if not auto_envelope:
            return mcp_decorator

        def decorator(fn: Callable[..., Any]) -> ToolHandler:
            @functools.wraps(fn)
            async def wrapper(args: dict[str, Any]) -> str:
                try:
                    result = await fn(args)
                    return json.dumps(_ok(result))
                except ServiceError as e:
                    return json.dumps(_error(e.code, str(e), e.details))

            return mcp_decorator(wrapper)

        return decorator

    @staticmethod
    def ok(data: Any) -> dict[str, Any]:
        """Standard success response envelope."""
        return _ok(data)

    @staticmethod
    def error(code: str, message: str, details: Any = None) -> dict[str, Any]:
        """Standard error response envelope."""
        return _error(code, message, details)

    def run(self) -> None:
        """Start the dual-protocol server."""
        import uvicorn

        uvicorn.run(self.api, host=self.config.host, port=self.config.port)
