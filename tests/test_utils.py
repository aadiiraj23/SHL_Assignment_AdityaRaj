from __future__ import annotations

from app.models import Message
from app.utils import (
    build_request_id,
    count_total_messages,
    count_turns,
    format_history_for_llm,
    get_last_user_message,
    messages_remaining,
    safe_json_parse,
    truncate_messages,
)


def test_count_total_messages() -> None:
    messages = [Message(role="user", content="Hi"), Message(role="assistant", content="Hello")]
    assert count_total_messages(messages) == 2


def test_count_turns_even() -> None:
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello"),
        Message(role="user", content="More"),
        Message(role="assistant", content="Sure"),
    ]
    assert count_turns(messages) == 2


def test_count_turns_odd() -> None:
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello"),
        Message(role="user", content="More"),
    ]
    assert count_turns(messages) == 1


def test_messages_remaining_under_limit() -> None:
    messages = [Message(role="user", content="Hi")]
    assert messages_remaining(messages, max_total=4) == 3


def test_messages_remaining_at_limit() -> None:
    messages = [
        Message(role="user", content="1"),
        Message(role="assistant", content="2"),
    ]
    assert messages_remaining(messages, max_total=2) == 0


def test_messages_remaining_over_limit() -> None:
    messages = [
        Message(role="user", content="1"),
        Message(role="assistant", content="2"),
        Message(role="user", content="3"),
    ]
    assert messages_remaining(messages, max_total=2) == 0


def test_get_last_user_message_found() -> None:
    messages = [
        Message(role="user", content="First"),
        Message(role="assistant", content="Reply"),
        Message(role="user", content="Second"),
    ]
    assert get_last_user_message(messages) == "Second"


def test_get_last_user_message_not_found() -> None:
    assert get_last_user_message([]) is None


def test_format_history_for_llm() -> None:
    messages = [Message(role="user", content="Hi")]
    formatted = format_history_for_llm(messages)
    assert formatted[0]["role"] == "user"
    assert "content" in formatted[0]


def test_truncate_messages_keeps_first_and_last() -> None:
    messages = [
        Message(role="user", content="Anchor"),
        Message(role="assistant", content="One"),
        Message(role="user", content="Two"),
        Message(role="assistant", content="Three"),
        Message(role="user", content="Four"),
    ]
    truncated = truncate_messages(messages, keep_last_n=2)
    assert truncated[0].content == "Anchor"
    assert truncated[-1].content == "Four"
    assert len(truncated) == 3


def test_truncate_messages_short_list_unchanged() -> None:
    messages = [
        Message(role="user", content="Anchor"),
        Message(role="assistant", content="One"),
    ]
    assert truncate_messages(messages, keep_last_n=4) == messages


def test_safe_json_parse_valid_json() -> None:
    payload = safe_json_parse('{"a": 1}')
    assert payload == {"a": 1}


def test_safe_json_parse_with_fences() -> None:
    payload = safe_json_parse("```json\n{\"a\": 2}\n```")
    assert payload == {"a": 2}


def test_safe_json_parse_invalid_returns_none() -> None:
    assert safe_json_parse("not json") is None


def test_build_request_id_length() -> None:
    request_id = build_request_id()
    assert len(request_id) == 8


def test_build_request_id_unique() -> None:
    assert build_request_id() != build_request_id()
