"""Transparent Phase 7 hybrid retrieval over the existing lexical and semantic modules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from retrieval.lexical import TfidfRetriever, load_documents
from retrieval.semantic import DEFAULT_MODEL, SemanticRetriever


@dataclass(frozen=True)
class HybridResult:
    document_id: str
    score: float
    lexical_score: float
    semantic_score: float
    lexical_normalized_score: float
    semantic_normalized_score: float
    retrieval_methods: tuple[str, ...]
    document: dict[str, object]


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Put one retriever's per-query scores on a 0–1 scale for combination."""
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if low == high:
        return {document_id: 1.0 if score else 0.0 for document_id, score in scores.items()}
    return {document_id: (score - low) / (high - low) for document_id, score in scores.items()}


class HybridRetriever:
    """Combine unchanged TF-IDF and sentence-embedding rankings in memory."""

    def __init__(
        self,
        documents: list[dict[str, object]],
        lexical_weight: float = 0.5,
        model_name: str = DEFAULT_MODEL,
        semantic_retriever: SemanticRetriever | None = None,
    ) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between 0 and 1")
        self.documents = documents
        self.lexical_weight = lexical_weight
        self.lexical = TfidfRetriever(documents)
        self.semantic = semantic_retriever or SemanticRetriever(documents, model_name)

    def search(self, query: str, top_k: int = 5) -> list[HybridResult]:
        """Return a weighted combination of normalized lexical and semantic scores."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        lexical_results = self.lexical.search(query, top_k=len(self.documents))
        semantic_results = self.semantic.search(query, top_k=len(self.documents))
        lexical_scores = {str(result.document["document_id"]): result.score for result in lexical_results}
        semantic_scores = {str(result.document["document_id"]): result.score for result in semantic_results}
        lexical_normalized = min_max_normalize(lexical_scores)
        semantic_normalized = min_max_normalize(semantic_scores)

        results: list[HybridResult] = []
        for document in self.documents:
            document_id = str(document["document_id"])
            lexical_score = lexical_scores.get(document_id, 0.0)
            semantic_score = semantic_scores.get(document_id, 0.0)
            lexical_component = lexical_normalized.get(document_id, 0.0)
            semantic_component = semantic_normalized.get(document_id, 0.0)
            score = self.lexical_weight * lexical_component + (1 - self.lexical_weight) * semantic_component
            methods = tuple(
                method
                for method, raw_score in (("lexical_tfidf", lexical_score), ("semantic_embedding", semantic_score))
                if raw_score != 0.0
            )
            results.append(
                HybridResult(
                    document_id=document_id,
                    score=score,
                    lexical_score=lexical_score,
                    semantic_score=semantic_score,
                    lexical_normalized_score=lexical_component,
                    semantic_normalized_score=semantic_component,
                    retrieval_methods=methods,
                    document=document,
                )
            )
        return sorted(results, key=lambda result: (-result.score, result.document_id))[:top_k]


def format_result(result: HybridResult) -> str:
    metadata = result.document.get("metadata", {})
    subject = metadata.get("subject") if isinstance(metadata, dict) else None
    methods = ", ".join(result.retrieval_methods) or "none"
    return (
        f"{result.document_id} | {subject or '(no subject)'}\n"
        f"  hybrid={result.score:.3f} | lexical={result.lexical_score:.3f} (normalized={result.lexical_normalized_score:.3f}) "
        f"| semantic={result.semantic_score:.3f} (normalized={result.semantic_normalized_score:.3f})\n"
        f"  methods: {methods}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Phase 4 JSONL with transparent hybrid retrieval.")
    parser.add_argument("--corpus", type=Path, required=True, help="Phase 4 JSONL corpus file.")
    parser.add_argument("--query", action="append", required=True, help="Search query; repeat for multiple queries.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query (default: 5).")
    parser.add_argument("--lexical-weight", type=float, default=0.5, help="TF-IDF weight from 0 to 1 (default: 0.5).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Sentence-transformer model (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()
    retriever = HybridRetriever(load_documents(args.corpus), args.lexical_weight, args.model)
    for query in args.query:
        print(f"Query: {query}")
        print("\n".join(format_result(result) for result in retriever.search(query, args.top_k)))
        print()


if __name__ == "__main__":
    main()
