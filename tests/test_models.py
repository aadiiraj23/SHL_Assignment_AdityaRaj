from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models import Assessment, ChatRequest, ChatResponse, Message, Recommendation


def test_assessment_valid() -> None:
    assessment = Assessment(
        name="Verify - Numerical Reasoning",
        url="https://www.shl.com/solutions/products/verify-numerical-reasoning/",
        test_type="A",
        description="Measures numerical reasoning and data interpretation.",
        remote_testing=True,
        adaptive=False,
        duration_minutes=36,
        languages=["en"],
    )

    assert assessment.name == "Verify - Numerical Reasoning"
    assert assessment.test_type == "A"
    assert assessment.remote_testing is True
    assert assessment.duration_minutes == 36


def test_assessment_defaults() -> None:
    assessment = Assessment(
        name="OPQ32r",
        url="https://www.shl.com/solutions/products/opq32r/",
        test_type="P",
    )

    assert assessment.description == ""
    assert assessment.remote_testing is False
    assert assessment.adaptive is False
    assert assessment.duration_minutes is None
    assert assessment.languages == []


def test_message_valid_user() -> None:
    message = Message(role="user", content="Hello")
    assert message.role == "user"
    assert message.content == "Hello"


def test_message_valid_assistant() -> None:
    message = Message(role="assistant", content="Hi there")
    assert message.role == "assistant"


def test_message_empty_content_raises() -> None:
    with pytest.raises(ValidationError):
        Message(role="user", content="")


def test_message_too_long_raises() -> None:
    with pytest.raises(ValidationError):
        Message(role="user", content="a" * 2001)


def test_chat_request_valid() -> None:
    request = ChatRequest(messages=[Message(role="user", content="Need assessments")])
    assert len(request.messages) == 1


def test_chat_request_must_start_with_user() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[Message(role="assistant", content="Hi")])


def test_chat_request_empty_raises() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(messages=[])


def test_recommendation_valid() -> None:
    rec = Recommendation(
        name="Verify - Numerical Reasoning",
        url="https://www.shl.com/solutions/products/verify-numerical-reasoning/",
        test_type="A",
    )
    assert rec.name.startswith("Verify")


def test_chat_response_defaults() -> None:
    response = ChatResponse()
    assert response.recommendations == []
    assert response.end_of_conversation is False


def test_chat_response_too_many_recommendations() -> None:
    recs = [
        Recommendation(name=f"Test {idx}", url=f"https://example.com/{idx}", test_type="A")
        for idx in range(11)
    ]
    with pytest.raises(ValidationError):
        ChatResponse(reply="Too many", recommendations=recs)


def test_chat_response_serializes_to_json() -> None:
    response = ChatResponse(
        reply="Ok",
        recommendations=[
            Recommendation(
                name="OPQ32r",
                url="https://www.shl.com/solutions/products/opq32r/",
                test_type="P",
            )
        ],
    )
    payload = response.model_dump()
    assert payload["reply"] == "Ok"
    assert payload["recommendations"][0]["name"] == "OPQ32r"
