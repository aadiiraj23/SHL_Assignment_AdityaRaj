from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.models import ChatResponse


client = TestClient(app)


def test_health_check_endpoint() -> None:
    with patch("app.main.get_catalog_store") as get_catalog_store:
        catalog = MagicMock()
        catalog.size.return_value = 7
        get_catalog_store.return_value = catalog

        response = client.get("/health")

    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "healthy"
    assert payload["catalog_size"] == 7
    assert isinstance(payload["timestamp"], float)


def test_chat_endpoint_successful_recommendation() -> None:
    with patch("app.main.get_catalog_store") as get_catalog_store, patch(
        "app.main.get_llm_client"
    ) as get_llm_client, patch("app.main.SHLAgent") as agent_class:
        get_catalog_store.return_value = MagicMock()
        get_llm_client.return_value = MagicMock()
        agent_instance = agent_class.return_value
        agent_instance.run = AsyncMock(
            return_value=ChatResponse(
                reply="Here are recommendations",
                recommendations=[],
                end_of_conversation=False,
            )
        )

        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Need assessment"}]},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["reply"] == "Here are recommendations"
    assert payload["end_of_conversation"] is False


def test_chat_endpoint_validation_error() -> None:
    response = client.post("/api/chat", json={"messages": []})

    assert response.status_code == 422


def test_chat_endpoint_server_error_fallback() -> None:
    with patch("app.main.get_catalog_store") as get_catalog_store, patch(
        "app.main.get_llm_client"
    ) as get_llm_client, patch("app.main.SHLAgent") as agent_class:
        get_catalog_store.return_value = MagicMock()
        get_llm_client.return_value = MagicMock()
        agent_instance = agent_class.return_value
        agent_instance.run = AsyncMock(side_effect=RuntimeError("boom"))

        response = client.post(
            "/api/chat",
            json={"messages": [{"role": "user", "content": "Need assessment"}]},
        )

    payload = response.json()
    assert response.status_code == 500
    assert payload["end_of_conversation"] is True
    assert payload["recommendations"] == []
    assert "internal server error" in payload["reply"].lower()
