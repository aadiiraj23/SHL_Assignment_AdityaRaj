from __future__ import annotations

from typing import Iterable

import structlog

logger = structlog.get_logger(__name__)

INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions",
    "ignore all instructions",
    "you are now",
    "pretend you are",
    "forget your",
    "new instructions:",
    "override instructions",
    "system prompt:",
    "disregard",
    "jailbreak",
    "act as if",
    "your real instructions",
]

OFF_TOPIC_KEYWORDS: list[str] = [
    "salary",
    "compensation",
    "pay range",
    "visa",
    "immigration",
    "legal advice",
    "lawsuit",
    "hogan",
    "korn ferry",
    "mercer",
    "talentplus",
    "hire someone",
    "fire someone",
    "medical",
    "disability accommodation",
    "background check policy",
]


def is_prompt_injection(text: str) -> bool:
    """Return True if prompt injection patterns are detected."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in INJECTION_PATTERNS)


def is_off_topic(text: str) -> bool:
    """Return True if off-topic keywords are detected."""
    lowered = text.lower()
    return any(keyword in lowered for keyword in OFF_TOPIC_KEYWORDS)


def validate_recommendations(recs: list[dict], valid_urls: set[str]) -> list[dict]:
    """Filter out recommendations whose URLs are not in the valid set."""
    cleaned: list[dict] = []
    for rec in recs:
        url = str(rec.get("url", "")).strip()
        if url in valid_urls:
            cleaned.append(rec)
            continue
        logger.warning(
            "invalid_recommendation_url",
            url=sanitize_for_log(url),
            name=sanitize_for_log(str(rec.get("name", ""))),
        )
    return cleaned


def sanitize_for_log(text: str, max_len: int = 120) -> str:
    """Trim text for safe logging by removing newlines and truncating."""
    cleaned = text.replace("\n", " ").replace("\r", " ")
    return cleaned[:max_len]
