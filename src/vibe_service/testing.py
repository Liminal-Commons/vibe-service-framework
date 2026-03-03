"""Testing helpers for ecosystem services.

Provides fixtures and utilities for testing services built on the framework.
"""

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from vibe_service.config import BaseServiceConfig
from vibe_service.service import EcosystemService


class TestServiceConfig(BaseServiceConfig):
    """Config for test services with sensible defaults."""

    model_config = {"env_prefix": "TEST_SERVICE_"}


def create_test_service(
    name: str = "test-service",
    config: BaseServiceConfig | None = None,
) -> EcosystemService:
    """Create an EcosystemService instance for testing."""
    if config is None:
        config = TestServiceConfig()
    return EcosystemService(name, config)


@pytest.fixture
def test_service() -> EcosystemService:
    """Pytest fixture providing a fresh test service."""
    return create_test_service()


@pytest.fixture
async def test_client(test_service: EcosystemService) -> AsyncClient:  # type: ignore[misc]
    """Pytest fixture providing an async HTTP client for the test service."""
    transport = ASGITransport(app=test_service.api)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def make_test_client(service: EcosystemService) -> AsyncClient:
    """Create an httpx AsyncClient for any EcosystemService.

    Usage in a service's test file:
        from vibe_service.testing import make_test_client
        from myservice.server import service

        @pytest.fixture
        def client():
            return make_test_client(service)
    """
    transport = ASGITransport(app=service.api)
    return AsyncClient(transport=transport, base_url="http://test")


async def invoke_mcp_tool(
    service: EcosystemService,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Invoke an MCP tool directly (bypass HTTP, useful for unit tests)."""
    return await service.mcp._meta_invoke(
        {"tool_name": tool_name, "arguments": arguments or {}}
    )
