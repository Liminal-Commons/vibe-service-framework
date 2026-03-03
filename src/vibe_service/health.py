"""Standard health endpoint for ecosystem services.

Response shape: {status, service, version, tools, uptime}
Mounted at GET /health.
"""

import time

from fastapi import APIRouter
from fastapi.responses import JSONResponse


def build_health_router(
    service_name: str,
    service_version: str,
    tool_count_fn: object,
    start_time: float,
) -> APIRouter:
    """Create a health check router.

    Args:
        service_name: Name of the service.
        service_version: Version string.
        tool_count_fn: Callable returning the number of MCP tools registered.
        start_time: time.monotonic() at service startup for uptime calculation.

    Returns:
        FastAPI APIRouter with GET /health.
    """
    router = APIRouter()

    @router.get("/health")
    async def health() -> JSONResponse:
        uptime = time.monotonic() - start_time
        return JSONResponse(
            {
                "status": "ok",
                "service": service_name,
                "version": service_version,
                "tools": tool_count_fn(),  # type: ignore[operator]
                "uptime": round(uptime, 1),
            }
        )

    return router
