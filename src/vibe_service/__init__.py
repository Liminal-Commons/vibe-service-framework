"""vibe-service-framework — shared framework for vibe ecosystem services.

Every ecosystem service uses EcosystemService as its entry point:

    from vibe_service import EcosystemService, BaseServiceConfig

    class Config(BaseServiceConfig):
        model_config = {"env_prefix": "MY_SERVICE_"}

    service = EcosystemService("my-service", Config())

    @service.tool(name="do_thing", description="...", category="ops", input_schema={...})
    async def do_thing(args: dict) -> str:
        return service.ok(await domain.do_thing(args["param"]))

    @service.api.get("/api/things")
    async def list_things():
        return service.ok(await domain.list_things())

    if __name__ == "__main__":
        service.run()
"""

def __getattr__(name: str) -> object:
    """Lazy imports — allows `from vibe_service.mcp import LazyMCPServer`
    without pulling in fastapi/pydantic-settings (which EcosystemService needs)."""
    if name == "EcosystemService":
        from vibe_service.service import EcosystemService
        return EcosystemService
    if name == "BaseServiceConfig":
        from vibe_service.config import BaseServiceConfig
        return BaseServiceConfig
    if name == "ResilientSessionManager":
        from vibe_service.mcp import ResilientSessionManager
        return ResilientSessionManager
    if name == "ServiceError":
        from vibe_service.errors import ServiceError
        return ServiceError
    raise AttributeError(f"module 'vibe_service' has no attribute {name!r}")

__all__ = [
    "EcosystemService",
    "BaseServiceConfig",
    "ResilientSessionManager",
    "ServiceError",
]
