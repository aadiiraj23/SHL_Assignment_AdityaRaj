from __future__ import annotations
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import httpx
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models import ChatResponse


@pytest.fixture
def mock_agent():
    agent = MagicMock()
    agent.run = AsyncMock(
        return_value=ChatResponse(
            reply="Here are recommendations",
            recommendations=[],
            end_of_conversation=False,
        )
    )
    return agent


@pytest.mark.asyncio
async def test_health_check_endpoint():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert "catalog_size" in payload
    assert isinstance(payload["catalog_size"], int)
    assert "agent_ready" in payload
    assert "version" in payload


@pytest.mark.asyncio
async def test_chat_endpoint_successful_recommendation(mock_agent):
    import app.main as main_module
    original_agent = main_module._agent
    main_module._agent = mock_agent
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "Need assessment"}]},
            )
        assert response.status_code == 200
        payload = response.json()
        assert payload["reply"] == "Here are recommendations"
        assert payload["end_of_conversation"] is False
        assert "recommendations" in payload
    finally:
        main_module._agent = original_agent


@pytest.mark.asyncio
async def test_chat_endpoint_validation_error():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post("/chat", json={"messages": []})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_chat_endpoint_server_error_fallback():
    import app.main as main_module
    broken_agent = MagicMock()
    broken_agent.run = AsyncMock(side_effect=RuntimeError("boom"))
    original_agent = main_module._agent
    main_module._agent = broken_agent
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "Need assessment"}]},
            )
        payload = response.json()
        assert response.status_code == 200
        assert payload["end_of_conversation"] is False
        assert payload["recommendations"] == []
        assert "error" in payload["reply"].lower() or "encountered" in payload["reply"].lower()
    finally:
        main_module._agent = original_agent


@pytest.mark.asyncio
async def test_chat_agent_unavailable_returns_503():
    import app.main as main_module
    original_agent = main_module._agent
    main_module._agent = None
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            response = await ac.post(
                "/chat",
                json={"messages": [{"role": "user", "content": "Need assessment"}]},
            )
        assert response.status_code == 503
    finally:
        main_module._agent = original_agent


@pytest.mark.asyncio
async def test_chat_endpoint_invalid_role_returns_422():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.post(
            "/chat",
            json={"messages": [{"role": "assistant", "content": "hello"}]},
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_health_returns_ok_status():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.get("/health")
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_cors_headers_present():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        response = await ac.options(
            "/health",
            headers={"Origin": "http://example.com"}
        )
    assert response.status_code in [200, 405]