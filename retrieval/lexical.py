"""A small, dependency-free TF-IDF lexical retriever for Phase 4 JSONL data."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


TOKEN_PATTERN = re.compile(r"\b[\w']+\b", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into searchable word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class SearchResult:
    document: dict[str, object]
    score: float


class TfidfRetriever:
    """Index Phase 4 documents in memory and retrieve by TF-IDF cosine score."""

    def __init__(self, documents: list[dict[str, object]]) -> None:
        if not documents:
            raise ValueError("The corpus contains no documents.")
        self.documents = documents
        term_counts = [Counter(tokenize(str(document.get("text", "")))) for document in documents]
        document_frequency: Counter[str] = Counter()
        for counts in term_counts:
            document_frequency.update(counts.keys())

        total_documents = len(documents)
        self.idf = {
            term: math.log((1 + total_documents) / (1 + frequency)) + 1
            for term, frequency in document_frequency.items()
        }
        self.document_vectors = [self._weighted_vector(counts) for counts in term_counts]
        self.document_norms = [self._norm(vector) for vector in self.document_vectors]

    def _weighted_vector(self, counts: Counter[str]) -> dict[str, float]:
        # Log-scaled term frequency prevents repeated words dominating a result.
        return {term: (1 + math.log(count)) * self.idf[term] for term, count in counts.items()}

    @staticmethod
    def _norm(vector: dict[str, float]) -> float:
        return math.sqrt(sum(weight * weight for weight in vector.values()))

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Return the highest-scoring documents that share terms with *query*."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        query_counts = Counter(tokenize(query))
        query_vector = {
            term: (1 + math.log(count)) * self.idf[term]
            for term, count in query_counts.items()
            if term in self.idf
        }
        query_norm = self._norm(query_vector)
        if not query_norm:
            return []

        results: list[SearchResult] = []
        for document, vector, document_norm in zip(self.documents, self.document_vectors, self.document_norms):
            if not document_norm:
                continue
            dot_product = sum(query_weight * vector.get(term, 0.0) for term, query_weight in query_vector.items())
            if dot_product:
                results.append(SearchResult(document, dot_product / (query_norm * document_norm)))
        return sorted(results, key=lambda result: (-result.score, str(result.document.get("document_id", ""))))[:top_k]


def load_documents(corpus_file: Path) -> list[dict[str, object]]:
    """Load non-empty JSONL records produced by the Phase 4 loader."""
    documents: list[dict[str, object]] = []
    with corpus_file.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            document = json.loads(line)
            if not isinstance(document, dict) or not isinstance(document.get("text"), str):
                raise ValueError(f"Invalid Phase 4 document at line {line_number}: expected an object with text.")
            documents.append(document)
    return documents


def format_result(result: SearchResult) -> str:
    metadata = result.document.get("metadata", {})
    subject = metadata.get("subject") if isinstance(metadata, dict) else None
    body = str(result.document.get("body", "")).replace("\n", " ")
    snippet = body[:180] + ("..." if len(body) > 180 else "")
    return f"[{result.score:.3f}] {result.document.get('document_id')} | {subject or '(no subject)'}\n  {snippet}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Phase 4 JSONL with lexical TF-IDF retrieval.")
    parser.add_argument("--corpus", type=Path, required=True, help="Phase 4 JSONL corpus file.")
    parser.add_argument("--query", action="append", required=True, help="Search query; repeat for multiple queries.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results per query (default: 5).")
    args = parser.parse_args()
    retriever = TfidfRetriever(load_documents(args.corpus))
    for query in args.query:
        print(f"Query: {query}")
        results = retriever.search(query, args.top_k)
        print("\n".join(format_result(result) for result in results) if results else "No matching terms found.")
        print()


if __name__ == "__main__":
    main()
