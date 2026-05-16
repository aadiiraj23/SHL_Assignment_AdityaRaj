from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import structlog
from rank_bm25 import BM25Okapi

from app.models import Assessment

logger = structlog.get_logger(__name__)
_CATALOG_STORE: "CatalogStore | None" = None


class CatalogStore:
    def __init__(self, catalog_path: str = "data/catalog.json") -> None:
        start_time = time.time()
        path = Path(catalog_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Catalog file not found at '{path}'. "
                "Set CATALOG_PATH or generate data/catalog.json first."
            )

        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            items = payload.get("assessments", [])
        else:
            items = payload or []

        assessments: list[Assessment] = []
        for item in items:
            try:
                assessments.append(Assessment.model_validate(item))
            except Exception as exc:
                logger.warning("catalog_item_invalid", error=str(exc))

        self.assessments = assessments
        self.valid_urls = {assessment.url for assessment in self.assessments}
        self.embeddings: np.ndarray | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.bm25: BM25Okapi | None = None
        self.model = None

        self._build_index()
        logger.info(
            "catalog_loaded",
            total=len(self.assessments),
            build_seconds=round(time.time() - start_time, 2),
        )

    def _make_document(self, a: Assessment) -> str:
        return (
            f"Assessment: {a.name}. Type: {a.test_type}. {a.description}. "
            f"Remote testing: {a.remote_testing}. Adaptive: {a.adaptive}. "
            f"Languages: {', '.join(a.languages)}."
        )

    def _build_index(self) -> None:
        start_time = time.time()
        from sentence_transformers import SentenceTransformer

        if not self.assessments:
            self.embeddings = np.empty((0, 0), dtype="float32")
            self.index = None
            self.bm25 = None
            self.model = None
            logger.info("catalog_index_empty")
            return

        documents = [self._make_document(assessment) for assessment in self.assessments]
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vectors = model.encode(documents, show_progress_bar=False).astype("float32")
        vectors = self._l2_normalize(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        tokenized = [doc.lower().split() for doc in documents]
        bm25 = BM25Okapi(tokenized)

        self.embeddings = vectors
        self.index = index
        self.bm25 = bm25
        self.model = model

        logger.info("catalog_index_built", seconds=round(time.time() - start_time, 2))

    def semantic_search(self, query: str, k: int = 10) -> list[Assessment]:
        if not self.index or not self.model or not self.assessments:
            return []
        limit = min(k, len(self.assessments))
        if limit <= 0:
            return []

        query_vector = self.model.encode([query], show_progress_bar=False).astype(
            "float32"
        )
        query_vector = self._l2_normalize(query_vector)
        scores, indices = self.index.search(query_vector, limit)
        result_indices = indices[0].tolist()
        return [self.assessments[idx] for idx in result_indices if idx >= 0]

    def bm25_search(self, query: str, k: int = 10) -> list[Assessment]:
        if not self.bm25 or not self.assessments:
            return []
        limit = min(k, len(self.assessments))
        if limit <= 0:
            return []

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )
        return [self.assessments[idx] for idx, _ in ranked[:limit]]

    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.7) -> list[Assessment]:
        if not self.assessments:
            return []
        limit = min(k, len(self.assessments))
        if limit <= 0:
            return []

        candidate_k = min(len(self.assessments), limit * 2)
        semantic_candidates = self._semantic_scores(query, candidate_k)
        bm25_candidates = self._bm25_scores(query, candidate_k)

        semantic_scores = self._normalize_scores([score for _, score in semantic_candidates])
        bm25_scores = self._normalize_scores([score for _, score in bm25_candidates])

        sem_map: dict[str, float] = {}
        sem_assessments: dict[str, Assessment] = {}
        for (assessment, score), normalized in zip(semantic_candidates, semantic_scores):
            name_key = assessment.name.lower()
            sem_assessments.setdefault(name_key, assessment)
            sem_map[name_key] = max(sem_map.get(name_key, 0.0), normalized)

        bm_map: dict[str, float] = {}
        bm_assessments: dict[str, Assessment] = {}
        for (assessment, score), normalized in zip(bm25_candidates, bm25_scores):
            name_key = assessment.name.lower()
            bm_assessments.setdefault(name_key, assessment)
            bm_map[name_key] = max(bm_map.get(name_key, 0.0), normalized)

        combined: list[tuple[Assessment, float]] = []
        for name_key in set(sem_map) | set(bm_map):
            assessment = sem_assessments.get(name_key) or bm_assessments.get(name_key)
            if not assessment:
                continue
            score = alpha * sem_map.get(name_key, 0.0) + (1 - alpha) * bm_map.get(
                name_key, 0.0
            )
            combined.append((assessment, score))

        combined.sort(key=lambda item: item[1], reverse=True)
        results = [assessment for assessment, _ in combined[:limit]]
        logger.info("hybrid_search", query=query, results=len(results))
        return results

    def get_by_name(self, name: str) -> Optional[Assessment]:
        if not name:
            return None
        name_lower = name.lower()
        for assessment in self.assessments:
            if assessment.name.lower() == name_lower:
                return assessment
        for assessment in self.assessments:
            if name_lower in assessment.name.lower():
                return assessment
        return None

    def get_all(self) -> list[Assessment]:
        return list(self.assessments)

    def size(self) -> int:
        return len(self.assessments)

    @staticmethod
    def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def _semantic_scores(self, query: str, k: int) -> list[tuple[Assessment, float]]:
        if not self.index or not self.model or not self.assessments:
            return []
        query_vector = self.model.encode([query], show_progress_bar=False).astype(
            "float32"
        )
        query_vector = self._l2_normalize(query_vector)
        scores, indices = self.index.search(query_vector, k)
        candidates: list[tuple[Assessment, float]] = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if idx >= 0:
                candidates.append((self.assessments[idx], float(score)))
        return candidates

    def _bm25_scores(self, query: str, k: int) -> list[tuple[Assessment, float]]:
        if not self.bm25 or not self.assessments:
            return []
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            enumerate(scores), key=lambda item: item[1], reverse=True
        )[:k]
        return [(self.assessments[idx], float(score)) for idx, score in ranked]

    @staticmethod
    def _normalize_scores(scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]


def get_catalog_store() -> CatalogStore:
    global _CATALOG_STORE
    if _CATALOG_STORE is not None:
        return _CATALOG_STORE

    catalog_path = os.getenv("CATALOG_PATH", "data/catalog.json")
    if not Path(catalog_path).exists():
        raise FileNotFoundError(
            f"Catalog file not found at '{catalog_path}'. "
            "Set CATALOG_PATH or run the scraper to generate the catalog."
        )

    _CATALOG_STORE = CatalogStore(catalog_path=catalog_path)
    return _CATALOG_STORE
