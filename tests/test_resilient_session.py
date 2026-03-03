"""Tests for ResilientSessionManager — session recovery after container restart."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import pytest
from httpx import ASGITransport, AsyncClient

from vibe_service import BaseServiceConfig, EcosystemService
from vibe_service.mcp import LazyMCPServer, ResilientSessionManager


# ── Unit tests: verify branching logic of the override ──


class TestResilientSessionManagerUnit:
    """Unit tests for _handle_stateful_request override."""

    def _make_manager(self) -> ResilientSessionManager:
        server = MagicMock()
        server.create_initialization_options.return_value = {}
        return ResilientSessionManager(
            app=server,
            json_response=False,
            stateless=False,
        )

    def _make_scope(self, session_id: str | None = None) -> dict:  # type: ignore[type-arg]
        headers = [(b"accept", b"application/json, text/event-stream")]
        if session_id is not None:
            headers.append((b"mcp-session-id", session_id.encode()))
        return {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "headers": headers,
            "query_string": b"",
            "root_path": "",
            "scheme": "http",
            "server": ("127.0.0.1", 8000),
        }

    async def test_known_session_delegates_to_parent(self) -> None:
        """Known session IDs delegate to the parent class."""
        mgr = self._make_manager()
        session_id = "known123"
        mock_transport = MagicMock()
        mock_transport.handle_request = AsyncMock()
        mgr._server_instances[session_id] = mock_transport

        scope = self._make_scope(session_id)
        receive = AsyncMock()
        send = AsyncMock()

        with patch.object(
            type(mgr).__bases__[0],
            "_handle_stateful_request",
            new_callable=AsyncMock,
        ) as parent_mock:
            await mgr._handle_stateful_request(scope, receive, send)
            parent_mock.assert_called_once_with(scope, receive, send)

    async def test_no_session_delegates_to_parent(self) -> None:
        """No session ID delegates to the parent class (new session flow)."""
        mgr = self._make_manager()
        scope = self._make_scope(session_id=None)
        receive = AsyncMock()
        send = AsyncMock()

        with patch.object(
            type(mgr).__bases__[0],
            "_handle_stateful_request",
            new_callable=AsyncMock,
        ) as parent_mock:
            await mgr._handle_stateful_request(scope, receive, send)
            parent_mock.assert_called_once_with(scope, receive, send)

    async def test_unknown_session_does_not_delegate_to_parent(self) -> None:
        """Unknown session IDs should NOT delegate to parent (would return 404)."""
        mgr = self._make_manager()
        unknown_id = "unknown456"
        scope = self._make_scope(unknown_id)
        receive = AsyncMock()
        send = AsyncMock()

        # app.run blocks forever in real use; simulate with an Event
        run_started = asyncio.Event()

        async def blocking_run(*args: object, **kwargs: object) -> None:
            run_started.set()
            await asyncio.sleep(3600)  # Block until cancelled

        async with anyio.create_task_group() as tg:
            mgr._task_group = tg

            with patch.object(
                type(mgr).__bases__[0],
                "_handle_stateful_request",
                new_callable=AsyncMock,
            ) as parent_mock:
                with patch(
                    "vibe_service.mcp.StreamableHTTPServerTransport",
                ) as mock_transport_cls:
                    mock_transport = MagicMock()
                    mock_transport.mcp_session_id = unknown_id
                    mock_transport.is_terminated = False
                    mock_transport.handle_request = AsyncMock()

                    mock_streams = (AsyncMock(), AsyncMock())
                    mock_connect = MagicMock()
                    mock_connect.__aenter__ = AsyncMock(return_value=mock_streams)
                    mock_connect.__aexit__ = AsyncMock(return_value=False)
                    mock_transport.connect.return_value = mock_connect
                    mock_transport_cls.return_value = mock_transport

                    mgr.app.run = blocking_run

                    await mgr._handle_stateful_request(scope, receive, send)

                    # Parent should NOT have been called (no 404)
                    parent_mock.assert_not_called()

                    # Transport created with the original session ID
                    mock_transport_cls.assert_called_once()
                    call_kwargs = mock_transport_cls.call_args
                    assert call_kwargs.kwargs["mcp_session_id"] == unknown_id

                    # Transport registered in _server_instances (still running)
                    assert unknown_id in mgr._server_instances

                    # handle_request called on the new transport
                    mock_transport.handle_request.assert_called_once()

            tg.cancel_scope.cancel()

    async def test_unknown_session_starts_server_with_stateless_true(self) -> None:
        """Recovered sessions start the server with stateless=True (skip init handshake)."""
        mgr = self._make_manager()
        unknown_id = "stale789"
        scope = self._make_scope(unknown_id)
        receive = AsyncMock()
        send = AsyncMock()

        captured_kwargs: dict[str, object] = {}

        async def capture_run(*args: object, **kwargs: object) -> None:
            captured_kwargs.update(kwargs)
            # Check positional args too (stateless is 4th positional)
            if len(args) >= 4:
                captured_kwargs["_stateless_positional"] = args[3]
            await asyncio.sleep(3600)

        async with anyio.create_task_group() as tg:
            mgr._task_group = tg

            with patch(
                "vibe_service.mcp.StreamableHTTPServerTransport",
            ) as mock_transport_cls:
                mock_transport = MagicMock()
                mock_transport.mcp_session_id = unknown_id
                mock_transport.is_terminated = False
                mock_transport.handle_request = AsyncMock()

                mock_streams = (AsyncMock(), AsyncMock())
                mock_connect = MagicMock()
                mock_connect.__aenter__ = AsyncMock(return_value=mock_streams)
                mock_connect.__aexit__ = AsyncMock(return_value=False)
                mock_transport.connect.return_value = mock_connect
                mock_transport_cls.return_value = mock_transport

                mgr.app.run = capture_run

                await mgr._handle_stateful_request(scope, receive, send)

                # Verify stateless=True was passed
                assert captured_kwargs.get("stateless") is True or (
                    captured_kwargs.get("_stateless_positional") is True
                )

            tg.cancel_scope.cancel()

    async def test_concurrent_recovery_uses_lock(self) -> None:
        """When two requests arrive with the same unknown session ID,
        only one transport should be created (double-check under lock)."""
        mgr = self._make_manager()
        unknown_id = "concurrent123"

        async def blocking_run(*args: object, **kwargs: object) -> None:
            await asyncio.sleep(3600)

        async with anyio.create_task_group() as tg:
            mgr._task_group = tg

            with patch(
                "vibe_service.mcp.StreamableHTTPServerTransport",
            ) as mock_transport_cls:
                mock_transport = MagicMock()
                mock_transport.mcp_session_id = unknown_id
                mock_transport.is_terminated = False
                mock_transport.handle_request = AsyncMock()

                mock_streams = (AsyncMock(), AsyncMock())
                mock_connect = MagicMock()
                mock_connect.__aenter__ = AsyncMock(return_value=mock_streams)
                mock_connect.__aexit__ = AsyncMock(return_value=False)
                mock_transport.connect.return_value = mock_connect
                mock_transport_cls.return_value = mock_transport

                mgr.app.run = blocking_run

                # First request creates the session
                scope1 = self._make_scope(unknown_id)
                await mgr._handle_stateful_request(scope1, AsyncMock(), AsyncMock())

                # Second request should find the session (double-check path)
                scope2 = self._make_scope(unknown_id)
                await mgr._handle_stateful_request(scope2, AsyncMock(), AsyncMock())

                # Transport constructor should have been called only once
                assert mock_transport_cls.call_count == 1
                # But handle_request should have been called twice
                assert mock_transport.handle_request.call_count == 2

            tg.cancel_scope.cancel()


# ── Integration tests: framework uses ResilientSessionManager ──


class TestFrameworkIntegration:
    """Verify EcosystemService and LazyMCPServer use ResilientSessionManager."""

    def test_ecosystem_service_uses_resilient_manager(self) -> None:
        config = BaseServiceConfig(service_version="0.1.0")
        svc = EcosystemService("test", config)
        assert isinstance(svc._session_manager, ResilientSessionManager)

    def test_lazy_mcp_build_mcp_mount_uses_resilient_manager(self) -> None:
        mcp = LazyMCPServer("test")
        _app, mgr = mcp.build_mcp_mount()
        assert isinstance(mgr, ResilientSessionManager)

    def test_lazy_mcp_build_asgi_app_uses_resilient_manager(self) -> None:
        mcp = LazyMCPServer("test")
        _app = mcp.build_asgi_app()
        assert isinstance(mcp._session_manager, ResilientSessionManager)


# ── Health endpoint still works (no session needed) ──


class SvcConfig(BaseServiceConfig):
    model_config = {"env_prefix": "RSM_TEST_"}


@pytest.fixture
def service() -> EcosystemService:
    config = SvcConfig(service_version="0.1.0")
    svc = EcosystemService("rsm-test", config)

    @svc.tool(
        name="echo",
        description="Echo back input",
        category="test",
        input_schema={
            "type": "object",
            "properties": {"msg": {"type": "string"}},
            "required": ["msg"],
        },
    )
    async def echo(args: dict) -> str:  # type: ignore[type-arg]
        return json.dumps(svc.ok(args["msg"]))

    return svc


@pytest.fixture
async def client(service: EcosystemService) -> AsyncClient:  # type: ignore[misc]
    transport = ASGITransport(app=service.api)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c  # type: ignore[misc]


class TestHealthWithResilientManager:
    async def test_health_endpoint_works(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "rsm-test"
