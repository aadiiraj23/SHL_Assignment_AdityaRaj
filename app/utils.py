from __future__ import annotations

import json
import re
import secrets
from typing import Optional

import structlog

from .models import Message

logger = structlog.get_logger(__name__)


def count_total_messages(messages: list[Message]) -> int:
    """Return the total number of messages in the list."""
    return len(messages)


def count_turns(messages: list[Message]) -> int:
    """Return the number of complete user plus assistant message pairs."""
    turns = 0
    waiting_for_assistant = False
    for message in messages:
        if message.role == "user":
            waiting_for_assistant = True
        elif message.role == "assistant" and waiting_for_assistant:
            turns += 1
            waiting_for_assistant = False
    return turns


def messages_remaining(messages: list[Message], max_total: int = 8) -> int:
    """Return how many more messages can be added before hitting max_total."""
    remaining = max_total - len(messages)
    return remaining if remaining > 0 else 0


def get_last_user_message(messages: list[Message]) -> Optional[str]:
    """Return the content of the most recent user message, or None."""
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return None


def format_history_for_llm(messages: list[Message]) -> list[dict]:
    """Convert messages into a list of role/content dictionaries for an LLM."""
    return [{"role": message.role, "content": message.content} for message in messages]


def truncate_messages(messages: list[Message], keep_last_n: int = 6) -> list[Message]:
    """Keep the first message plus the last keep_last_n messages."""
    if len(messages) <= 2:
        return messages

    last_n = max(1, keep_last_n)
    if len(messages) <= last_n + 1:
        return messages

    return [messages[0]] + messages[-last_n:]


def safe_json_parse(text: str) -> Optional[dict]:
    """Parse a JSON string, stripping optional markdown fences."""
    if not text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    cleaned = fenced_match.group(1) if fenced_match else text

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.exception("json_parse_failed", error=str(exc))
        return None

    return payload if isinstance(payload, dict) else None


def validate_recommendations(recs: list[dict], valid_urls: set[str]) -> list[dict]:
    """Filter out recommendations whose URLs are not in the valid set."""
    cleaned: list[dict] = []
    # normalize valid urls for comparison
    norm_valid = {str(u).strip().rstrip("/").lower() for u in valid_urls}
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        url = str(rec.get("url", "")).strip()
        norm_url = url.rstrip("/").lower()
        if norm_url in norm_valid:
            cleaned.append(rec)
            continue
        logger.warning(
            "invalid_recommendation_url",
            url=_sanitize_for_log(url),
            name=_sanitize_for_log(str(rec.get("name", ""))),
        )
    return cleaned


def _sanitize_for_log(text: str, max_len: int = 120) -> str:
    cleaned = text.replace("\n", " ").replace("\r", " ")
    return cleaned[:max_len]


def build_request_id() -> str:
    """Return a short unique 8-character hexadecimal request id."""
    return secrets.token_hex(4)
