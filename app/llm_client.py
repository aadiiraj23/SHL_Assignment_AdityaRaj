from __future__ import annotations

import os
import re
from typing import Optional

import google.generativeai as genai
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app import utils


class LLMException(Exception):
    pass


class GeminiClient:
    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is required to use the LLM client")

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
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
            prompt += "\nRespond ONLY with valid JSON. No markdown fences. No explanation."

        return prompt

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=1, max=8),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def complete(self, system: str, messages: list, json_mode: bool = False) -> str:
        prompt = self._build_prompt(system, messages, json_mode=json_mode)
        try:
            response = await self.model.generate_content_async(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1000,
                },
            )
            text = getattr(response, "text", "") or ""
            cleaned = self._strip_json_fences(text)
            return cleaned.strip()
        except Exception as exc:
            self.logger.exception("llm_completion_failed", error=str(exc))
            raise LLMException("LLM completion failed") from exc

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        if not text:
            return ""
        fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        return fenced_match.group(1) if fenced_match else text


_client_instance: Optional[GeminiClient] = None


def get_llm_client() -> GeminiClient:
    global _client_instance
    if _client_instance is None:
        _client_instance = GeminiClient()
    return _client_instance
