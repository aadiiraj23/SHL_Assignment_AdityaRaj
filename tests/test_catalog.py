from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import app.catalog as catalog_module
from app.catalog import CatalogStore


def _write_catalog(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


@pytest.fixture()
def dummy_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sentence_transformers")

    class DummySentenceTransformer:
        def __init__(self, name: str) -> None:
            self.name = name

        def encode(self, texts, show_progress_bar: bool = False):
            if isinstance(texts, str):
                texts = [texts]
            vectors = []
            for text in texts:
                length = len(text)
                word_count = len(text.split())
                checksum = sum(ord(char) for char in text)
                vectors.append(
                    [float(length % 5 + 1), float(word_count % 5 + 1), float(checksum % 5 + 1)]
                )
            return np.array(vectors, dtype="float32")

    module.SentenceTransformer = DummySentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


@pytest.fixture()
def sample_assessments() -> list[dict]:
    return [
        {
            "name": "Java 8 (New)",
            "url": "https://example.com/java-8-new",
            "test_type": "K",
            "description": "Java skills assessment",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 30,
            "languages": ["English"],
        },
        {
            "name": "OPQ32r",
            "url": "https://example.com/opq32r",
            "test_type": "P",
            "description": "Personality questionnaire",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 25,
            "languages": ["English"],
        },
        {
            "name": "Verify Numerical",
            "url": "https://example.com/verify-numerical",
            "test_type": "A",
            "description": "Numerical reasoning assessment",
            "remote_testing": True,
            "adaptive": True,
            "duration_minutes": 36,
            "languages": ["English"],
        },
        {
            "name": "Situational Judgement",
            "url": "https://example.com/sjt",
            "test_type": "S",
            "description": "Judgement scenarios",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 40,
            "languages": ["English"],
        },
        {
            "name": "Competency Lead",
            "url": "https://example.com/competency",
            "test_type": "C",
            "description": "Leadership competency review",
            "remote_testing": True,
            "adaptive": False,
            "duration_minutes": 20,
            "languages": ["English"],
        },
    ]


@pytest.fixture()
def catalog_format_a(tmp_path: Path, sample_assessments: list[dict]) -> Path:
    path = tmp_path / "catalog_a.json"
    return _write_catalog(path, sample_assessments)


@pytest.fixture()
def catalog_format_b(tmp_path: Path, sample_assessments: list[dict]) -> Path:
    path = tmp_path / "catalog_b.json"
    payload = {"assessments": sample_assessments, "total": 5, "scraped_at": "2026-05-16T10:00:00Z"}
    return _write_catalog(path, payload)


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_catalog_load_format_a(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    assert store.size() == 5


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_catalog_load_format_b(catalog_format_b: Path) -> None:
    store = CatalogStore(str(catalog_format_b))
    assert store.size() == 5


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_semantic_search_returns_results(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    results = store.semantic_search("java", k=3)
    assert len(results) > 0


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_semantic_search_k_limit(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    results = store.semantic_search("assessment", k=50)
    assert len(results) == store.size()


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_bm25_search_returns_results(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    results = store.bm25_search("numerical", k=3)
    assert len(results) > 0


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_hybrid_search_returns_results(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    results = store.hybrid_search("leadership competency", k=3)
    assert len(results) > 0


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_hybrid_search_deduplicates(tmp_path: Path, sample_assessments: list[dict]) -> None:
    duplicated = list(sample_assessments)
    duplicated.append({**sample_assessments[0], "url": "https://example.com/java-8-new-dup"})
    path = _write_catalog(tmp_path / "catalog_dup.json", duplicated)
    store = CatalogStore(str(path))

    results = store.hybrid_search("java", k=10)
    names = [assessment.name for assessment in results]
    assert len(names) == len(set(names))


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_get_by_name_exact_match(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    match = store.get_by_name("OPQ32r")
    assert match is not None
    assert match.name == "OPQ32r"


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_get_by_name_case_insensitive(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    match = store.get_by_name("verify numerical")
    assert match is not None
    assert match.name == "Verify Numerical"


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_get_by_name_not_found(catalog_format_a: Path) -> None:
    store = CatalogStore(str(catalog_format_a))
    assert store.get_by_name("Nonexistent") is None


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_valid_urls_populated(catalog_format_a: Path, sample_assessments: list[dict]) -> None:
    store = CatalogStore(str(catalog_format_a))
    expected = {item["url"] for item in sample_assessments}
    assert store.valid_urls == expected


@pytest.mark.usefixtures("dummy_sentence_transformer")
def test_singleton_get_catalog_store(
    monkeypatch: pytest.MonkeyPatch, catalog_format_a: Path
) -> None:
    monkeypatch.setenv("CATALOG_PATH", str(catalog_format_a))
    catalog_module._CATALOG_STORE = None

    first = catalog_module.get_catalog_store()
    second = catalog_module.get_catalog_store()

    assert first is second
