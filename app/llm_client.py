from __future__ import annotations

import asyncio
import os
import re
from typing import Optional

import httpx
import structlog

from app import utils


class LLMException(Exception):
    pass


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required to use the LLM client")

        self.api_key = key
        self.endpoint_url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-1.5-flash:generateContent"
        )
        self.request_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
        self.logger = structlog.get_logger(__name__)

    def _build_prompt(self, system: str, messages: list, json_mode: bool = False) -> str:
        original_count = len(messages)
        truncated_messages = utils.truncate_messages(messages, keep_last_n=6)
        truncated_count = len(truncated_messages)
        if truncated_count < original_count:
            self.logger.info(
                "llm_history_truncated",
                original=original_count,
                truncated=truncated_count,
            )

        conversation_lines: list[str] = []
        for message in truncated_messages:
            role = message.get("role") if isinstance(message, dict) else getattr(
                message, "role", "user"
            )
            content = message.get("content") if isinstance(message, dict) else getattr(
                message, "content", ""
            )
            role_label = "User" if role == "user" else "Assistant"
            conversation_lines.append(f"{role_label}: {content}")

        prompt = "SYSTEM:\n" + system + "\n\nCONVERSATION:\n" + "\n".join(
            conversation_lines
        )

        if json_mode:
            prompt += "\nRespond ONLY with valid JSON string representing the output schema."

        return prompt

    async def complete(self, system: str, messages: list, json_mode: bool = False) -> str:
        prompt = self._build_prompt(system, messages, json_mode=json_mode)
        try:
            text = await asyncio.wait_for(
                self._generate_content(prompt, json_mode=json_mode),
                timeout=self.request_timeout_seconds,
            )
            cleaned = self._strip_json_fences(text)
            return cleaned.strip()
        except asyncio.TimeoutError as exc:
            self.logger.warning(
                "llm_completion_timeout", timeout_seconds=self.request_timeout_seconds
            )
            raise LLMException("LLM completion timed out") from exc
        except httpx.HTTPError as exc:
            self.logger.exception("llm_completion_failed", error=str(exc))
            raise LLMException("LLM completion failed") from exc
        except Exception as exc:
            self.logger.exception("llm_completion_failed", error=str(exc))
            raise LLMException("LLM completion failed") from exc

    async def _generate_content(self, prompt: str, json_mode: bool = False) -> str:
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt,
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 1000,
            },
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        url = f"{self.endpoint_url}?key={self.api_key}"
        async with httpx.AsyncClient(timeout=self.request_timeout_seconds) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        return self._extract_text(data)

    @staticmethod
    def _extract_text(data: dict) -> str:
        candidates = data.get("candidates") if isinstance(data, dict) else None
        if not candidates:
            raise LLMException("LLM completion failed")

        first_candidate = candidates[0] or {}
        content = first_candidate.get("content", {}) if isinstance(first_candidate, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list):
            raise LLMException("LLM completion failed")

        text_chunks: list[str] = []
        for part in parts:
            if isinstance(part, dict):
                text_chunks.append(str(part.get("text", "")))

        text = "".join(text_chunks).strip()
        if not text:
            raise LLMException("LLM completion failed")
        return text

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        if not text:
            return ""
        # Using hex character escapes (\x60 is a backtick) avoids literal triple backticks in code blocks
        pattern = r"\x60{3}(?:json)?\s*(.*?)\s*\x60{3}"
        fenced_match = re.search(pattern, text, re.DOTALL)
        return fenced_match.group(1) if fenced_match else text


_client_instance: Optional[GeminiClient] = None


def get_llm_client() -> GeminiClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance