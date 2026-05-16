from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.catalog import CatalogStore
from app.models import Assessment
from app.retriever import Retriever


@pytest.fixture()
def mock_catalog() -> CatalogStore:
    catalog = CatalogStore.__new__(CatalogStore)
    catalog.assessments = [
        Assessment(
            name="Java 8 (New)",
            url="https://example.com/java-8-new",
            test_type="K",
            description="Java coding assessment for developers.",
            remote_testing=True,
            adaptive=False,
            duration_minutes=30,
            languages=["English"],
        ),
        Assessment(
            name="OPQ32r",
            url="https://example.com/opq32r",
            test_type="P",
            description="Personality questionnaire for workplace preferences.",
            remote_testing=True,
            adaptive=False,
            duration_minutes=25,
            languages=["English"],
        ),
    ]

    def _hybrid_search(query: str, k: int = 10, alpha: float = 0.7):
        return catalog.assessments[:k]

    catalog.hybrid_search = MagicMock(side_effect=_hybrid_search)
    return catalog


def test_extract_search_query_basic(mock_catalog: CatalogStore) -> None:
    retriever = Retriever(mock_catalog)
    messages = [{"role": "user", "content": "Hiring a Java developer"}]

    query = retriever.extract_search_query(messages)

    assert query == "Java developer"


def test_extract_search_query_strips_stopwords(mock_catalog: CatalogStore) -> None:
    retriever = Retriever(mock_catalog)
    messages = [{"role": "user", "content": "I am hiring a Java developer"}]

    query = retriever.extract_search_query(messages)

    assert query == "Java developer"


def test_extract_search_query_multi_turn(mock_catalog: CatalogStore) -> None:
    retriever = Retriever(mock_catalog)
    messages = [
        {"role": "user", "content": "Need a Java developer"},
        {"role": "assistant", "content": "What skills?"},
        {"role": "user", "content": "Problem solving and communication"},
    ]

    query = retriever.extract_search_query(messages)

    assert query == "Java developer Problem solving and communication"


def test_search_for_conversation_returns_assessments(mock_catalog: CatalogStore) -> None:
    retriever = Retriever(mock_catalog)
    messages = [{"role": "user", "content": "Java developer"}]

    results = retriever.search_for_conversation(messages, k=1)

    assert results
    mock_catalog.hybrid_search.assert_called_once()


def test_format_for_prompt_numbered(mock_catalog: CatalogStore) -> None:
    retriever = Retriever(mock_catalog)

    formatted = retriever.format_for_prompt(mock_catalog.assessments)

    lines = formatted.splitlines()
    assert lines[0].startswith("1. ")
    assert lines[1].startswith("2. ")


def test_format_for_prompt_truncates_description(mock_catalog: CatalogStore) -> None:
    retriever = Retriever(mock_catalog)
    long_description = "A" * 200
    assessment = Assessment(
        name="Long Description",
        url="https://example.com/long",
        test_type="A",
        description=long_description,
        remote_testing=True,
        adaptive=False,
        duration_minutes=20,
        languages=["English"],
    )

    formatted = retriever.format_for_prompt([assessment])
    description = formatted.split(": ", maxsplit=1)[1]

    assert description.endswith("...")
    assert len(description) <= 150
