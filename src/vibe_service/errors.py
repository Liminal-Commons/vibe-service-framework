"""Standard response envelope for ecosystem services.

Success: {"success": true, "data": <payload>}
Error:   {"success": false, "error": {"code": "<CODE>", "message": "...", "details": ...}}
"""

from typing import Any


class ServiceError(Exception):
    """Raise in tool handlers to return an error envelope automatically.

    Usage with auto_envelope=True:
        @service.tool(name="my_tool", ..., auto_envelope=True)
        async def my_tool(args):
            if not args.get("id"):
                raise ServiceError("VALIDATION", "id is required")
            return {"result": "data"}  # auto-wrapped in ok()
    """

    def __init__(self, code: str, message: str, details: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


def ok(data: Any) -> dict[str, Any]:
    """Standard success response."""
    return {"success": True, "data": data}


def error(code: str, message: str, details: Any = None) -> dict[str, Any]:
    """Standard error response."""
    err: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        err["details"] = details
    return {"success": False, "error": err}
