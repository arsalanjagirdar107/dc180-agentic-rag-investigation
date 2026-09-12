"""Semantic retrieval for Phase 4 JSONL documents using sentence embeddings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from retrieval.lexical import load_documents


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class SearchResult:
    document: dict[str, object]
    score: float


class SemanticRetriever:
    """Lazily keep normalized document embeddings in memory and rank by cosine similarity."""

    def __init__(self, documents: list[dict[str, object]], model_name: str = DEFAULT_MODEL) -> None:
        if not documents:
            raise ValueError("The corpus contains no documents.")
        self.documents = documents
        self.model_name = model_name
        self._model: Any | None = None
        self._document_embeddings: np.ndarray | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether the embedding model and corpus vectors have been initialized."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        """Import and initialize the model only for a semantic-dependent request."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            # Keep this import local: importing sentence-transformers imports PyTorch.
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(self.model_name)
            texts = [str(document["text"]) for document in self.documents]
            # Unit-normalized vectors make a dot product exactly equal cosine similarity.
            embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
            self._model = model
            self._document_embeddings = embeddings

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the highest cosine-similarity documents for a natural-language query."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._ensure_loaded()
        assert self._model is not None and self._document_embeddings is not None
        query_embedding = self._model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
        scores = self._document_embeddings @ query_embedding
        ranked_indices = np.argsort(-scores, kind="stable")[:top_k]
        return [SearchResult(self.documents[index], float(scores[index])) for index in ranked_indices]


def format_result(result: SearchResult) -> str:
    metadata = result.document.get("metadata", {})
    subject = metadata.get("subject") if isinstance(metadata, dict) else None
    body = str(result.document.get("body", "")).replace("\n", " ")
    snippet = body[:180] + ("..." if len(body) > 180 else "")
    return f"[{result.score:.3f}] {result.document.get('document_id')} | {subject or '(no subject)'}\n  {snippet}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Phase 4 JSONL with sentence-embedding semantic retrieval.")
    parser.add_argument("--corpus", type=Path, required=True, help="Phase 4 JSONL corpus file.")
    parser.add_argument("--query", action="append", required=True, help="Search query; repeat for multiple queries.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query (default: 5).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Sentence-transformer model (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()
    retriever = SemanticRetriever(load_documents(args.corpus), args.model)
    for query in args.query:
        print(f"Query: {query}")
        print("\n".join(format_result(result) for result in retriever.search(query, args.top_k)))
        print()


if __name__ == "__main__":
    main()
