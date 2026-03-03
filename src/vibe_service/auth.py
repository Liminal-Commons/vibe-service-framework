"""Service-to-service authentication middleware.

X-Service-Key header check. Disabled when service_key is empty (dev mode).
Applied as a FastAPI dependency on REST routes that need protection.
MCP endpoint at /mcp goes through mcp-hub which handles its own auth.
"""

from collections.abc import Awaitable, Callable

from fastapi import Header, HTTPException


def make_auth_dependency(service_key: str) -> Callable[..., Awaitable[str]]:
    """Create an auth dependency bound to the service's configured key.

    Args:
        service_key: The expected key value. Empty string disables auth.

    Returns:
        A FastAPI dependency function.
    """

    async def verify_service_key(
        x_service_key: str = Header(default=""),
    ) -> str:
        if not service_key:
            return x_service_key  # dev mode: no auth required
        if x_service_key != service_key:
            raise HTTPException(status_code=403, detail="Invalid service key")
        return x_service_key

    return verify_service_key
