from __future__ import annotations

import re

from app import prompts


def test_system_core_instructions_rules() -> None:
    text = prompts.SYSTEM_CORE_INSTRUCTIONS.lower()
    assert "catalog" in text
    assert "never" in text
    assert "refuse" in text or "off-topic" in text


def test_catalog_context_template_placeholders() -> None:
    assert "{catalog_items}" in prompts.CATALOG_CONTEXT_TEMPLATE
    assert "{history}" in prompts.CATALOG_CONTEXT_TEMPLATE


def test_clarify_prompt_template_placeholders() -> None:
    assert "{history}" in prompts.CLARIFY_PROMPT_TEMPLATE


def test_compare_prompt_template_placeholders() -> None:
    assert "{assessment_details}" in prompts.COMPARE_PROMPT_TEMPLATE
    assert "{history}" in prompts.COMPARE_PROMPT_TEMPLATE


def test_refusal_off_topic_formatting() -> None:
    formatted = prompts.REFUSAL_OFF_TOPIC.format(topic="pricing")
    assert "pricing" in formatted
    assert re.search(r"outside", formatted.lower())


def test_force_recommend_addendum_not_empty() -> None:
    assert prompts.FORCE_RECOMMEND_ADDENDUM.strip()
