from __future__ import annotations

import pytest

from app.models import ChatRequest, Message


@pytest.fixture()
def sample_assessments() -> list[dict]:
    return [
        {
            "name": "Verify - Numerical Reasoning",
            "url": "https://www.shl.com/solutions/products/verify-numerical-reasoning/",
            "test_type": "A",
            "description": "Measures numerical reasoning and data interpretation.",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 36,
            "languages": ["en", "fr"],
        },
        {
            "name": "OPQ32r",
            "url": "https://www.shl.com/solutions/products/opq32r/",
            "test_type": "P",
            "description": "Personality questionnaire for workplace preferences.",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 25,
            "languages": ["en"],
        },
        {
            "name": "Java 8 (New)",
            "url": "https://www.shl.com/solutions/products/java-8-new/",
            "test_type": "K",
            "description": "Assesses core Java 8 programming knowledge.",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 45,
            "languages": ["en"],
        },
        {
            "name": "MQ",
            "url": "https://www.shl.com/solutions/products/mq/",
            "test_type": "A",
            "description": "Measures managerial and leadership potential.",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 30,
            "languages": ["en", "es"],
        },
        {
            "name": "Situational Judgement",
            "url": "https://www.shl.com/solutions/products/situational-judgement/",
            "test_type": "S",
            "description": "Evaluates judgment in realistic workplace scenarios.",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 35,
            "languages": ["en"],
        },
    ]


@pytest.fixture()
def sample_messages_vague() -> list[Message]:
    return [Message(role="user", content="I need an assessment")]


@pytest.fixture()
def sample_messages_with_context() -> list[Message]:
    return [
        Message(
            role="user",
            content="I am hiring a mid-level Java developer who needs to work with stakeholders",
        ),
        Message(role="assistant", content="What seniority level and key skills matter most?"),
        Message(
            role="user",
            content="Around 4 years experience, needs problem solving and communication",
        ),
    ]


@pytest.fixture()
def sample_chat_request_vague(sample_messages_vague: list[Message]) -> ChatRequest:
    return ChatRequest(messages=sample_messages_vague)


@pytest.fixture()
def sample_chat_request_context(
    sample_messages_with_context: list[Message],
) -> ChatRequest:
    return ChatRequest(messages=sample_messages_with_context)
