from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import app.llm_client as llm_client


def _build_client() -> llm_client.GeminiClient:
    with patch.object(llm_client.genai, "configure"), patch.object(
        llm_client.genai, "GenerativeModel"
    ) as model_class:
        model_instance = MagicMock()
        model_class.return_value = model_instance
        return llm_client.GeminiClient(api_key="test-key")


def test_build_prompt_includes_system() -> None:
    client = _build_client()
    prompt = client._build_prompt("System rules", [{"role": "user", "content": "Hi"}])
    assert "SYSTEM:\nSystem rules" in prompt


def test_build_prompt_includes_user_content() -> None:
    client = _build_client()
    prompt = client._build_prompt("System", [{"role": "user", "content": "Need Java"}])
    assert "User: Need Java" in prompt


def test_build_prompt_json_mode_adds_instruction() -> None:
    client = _build_client()
    prompt = client._build_prompt(
        "System", [{"role": "user", "content": "Need Java"}], json_mode=True
    )
    assert "Respond ONLY with valid JSON" in prompt


def test_build_prompt_truncates_long_history() -> None:
    client = _build_client()
    messages = []
    for idx in range(10):
        role = "user" if idx % 2 == 0 else "assistant"
        messages.append({"role": role, "content": f"message-{idx}"})

    prompt = client._build_prompt("System", messages)

    assert "message-0" in prompt
    assert "message-1" not in prompt
    assert "message-9" in prompt


def test_get_llm_client_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    llm_client._client_instance = None

    with patch.object(llm_client.genai, "configure"), patch.object(
        llm_client.genai, "GenerativeModel"
    ):
        with pytest.raises(ValueError):
            llm_client.get_llm_client()


@pytest.mark.asyncio
async def test_complete_with_fallback_returns_fallback_on_error() -> None:
    client = _build_client()
    client.model.generate_content = MagicMock(side_effect=Exception("boom"))

    with pytest.raises(llm_client.LLMException):
        await client.complete("System", [{"role": "user", "content": "Hi"}])


@pytest.mark.asyncio
async def test_complete_times_out_fast() -> None:
    client = _build_client()
    client.request_timeout_seconds = 0.01

    def _slow_generate_content(*args, **kwargs):
        time.sleep(1)
        return MagicMock(text='{"reply":"ok"}')

    client.model.generate_content = MagicMock(side_effect=_slow_generate_content)

    with pytest.raises(llm_client.LLMException):
        await client.complete("System", [{"role": "user", "content": "Hi"}])
