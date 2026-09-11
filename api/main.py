"""Thin FastAPI adapter over the existing evidence, investigation, and graph modules."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from evidence_graph.builder import EvidenceGraph
from fact_checking.fact_checker import FactCheckReport, FactChecker, OpenAIFactCheckPolicy, RuleBasedFactCheckPolicy
from investigation.investigator import (
    InvestigationReport,
    Investigator,
    OpenAIReasoningPolicy,
    RuleBasedReasoningPolicy,
)
from preprocessing.enron_loader import build_corpus
from retrieval.hybrid import HybridRetriever
from retrieval.lexical import TfidfRetriever, load_documents
from retrieval.semantic import DEFAULT_MODEL, SemanticRetriever


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    method: Literal["lexical", "semantic", "hybrid"] = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)


class EvidenceResult(BaseModel):
    document_id: str
    score: float
    source_path: str
    subject: str | None
    retrieval_methods: list[str]
    metadata: dict[str, Any]
    excerpt: str
    verification: dict[str, str]


class SearchResponse(BaseModel):
    query: str
    method: str
    results: list[EvidenceResult]


class InvestigateRequest(BaseModel):
    question: str = Field(min_length=1)
    max_retries: int = Field(default=2, ge=0, le=5)
    evidence_limit: int = Field(default=3, ge=1, le=10)
    policy: Literal["openai", "rule"] = "openai"
    llm_model: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-5-mini"))


class FactCheckRequest(BaseModel):
    investigation: InvestigationReport
    evidence_limit: int = Field(default=3, ge=1, le=10)
    policy: Literal["openai", "rule"] = "openai"
    llm_model: str = Field(default_factory=lambda: os.environ.get("OPENAI_MODEL", "gpt-5-mini"))


class IngestRequest(BaseModel):
    input_dir: Path
    output_file: Path
    max_documents: int = Field(default=200, ge=1)
    include_path: str | None = None


class IngestResponse(BaseModel):
    documents_written: int
    corpus_path: str


class VerdictRequest(BaseModel):
    prediction: str = Field(min_length=1)
    rationale: str | None = None
    related_document_ids: list[str] = Field(default_factory=list)


class VerdictResponse(BaseModel):
    received: bool
    evaluation_status: Literal["not_evaluated"]
    message: str
    prediction: str


class Backend:
    def __init__(self, corpus_path: Path) -> None:
        self.reload(corpus_path)

    def reload(self, corpus_path: Path) -> None:
        self.corpus_path = corpus_path
        self.documents = load_documents(corpus_path)
        self.lexical = TfidfRetriever(self.documents)
        self.semantic = SemanticRetriever(self.documents)
        self.hybrid = HybridRetriever(self.documents)
        self.graph = EvidenceGraph.from_documents(self.documents)

    @staticmethod
    def _excerpt(document: dict[str, object], limit: int = 240) -> str:
        text = " ".join(str(document.get("body", "")).split())
        return text[:limit] + ("..." if len(text) > limit else "")

    def search(self, request: SearchRequest) -> SearchResponse:
        if request.method == "lexical":
            raw = self.lexical.search(request.query, request.top_k)
            rows = [(item.document, item.score, ["lexical_tfidf"]) for item in raw]
        elif request.method == "semantic":
            raw = self.semantic.search(request.query, request.top_k)
            rows = [(item.document, item.score, ["semantic_embedding"]) for item in raw]
        else:
            raw = self.hybrid.search(request.query, request.top_k)
            rows = [(item.document, item.score, list(item.retrieval_methods)) for item in raw]
        return SearchResponse(query=request.query, method=request.method, results=[
            EvidenceResult(
                document_id=str(document["document_id"]), score=score,
                source_path=str(document.get("source_path", "")),
                subject=(document.get("metadata", {}) or {}).get("subject"),
                retrieval_methods=methods, metadata=document.get("metadata", {}) or {},
                excerpt=self._excerpt(document), verification=document.get("verification", {}) or {
                    "status": "unverified", "reason": "Corpus evidence is not independently verified."
                },
            ) for document, score, methods in rows
        ])


def create_app(corpus_path: Path | str | None = None) -> FastAPI:
    path = Path(corpus_path or os.environ.get("EVIDENCE_CORPUS", "examples/phase5_sample.jsonl"))
    backend = Backend(path)
    app = FastAPI(title="Agentic RAG Investigation API", version="0.1.0")
    app.state.backend = backend

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "documents": len(backend.documents), "corpus_path": str(backend.corpus_path)}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(request: IngestRequest) -> IngestResponse:
        count = build_corpus(request.input_dir, request.output_file, request.max_documents, request.include_path)
        backend.reload(request.output_file)
        return IngestResponse(documents_written=count, corpus_path=str(request.output_file))

    @app.post("/search_evidence", response_model=SearchResponse)
    def search_evidence(request: SearchRequest) -> SearchResponse:
        return backend.search(request)

    @app.post("/interrogate", response_model=SearchResponse)
    def interrogate(request: SearchRequest) -> SearchResponse:
        return backend.search(request)

    @app.post("/investigate", response_model=InvestigationReport)
    def investigate(request: InvestigateRequest) -> InvestigationReport:
        try:
            policy = OpenAIReasoningPolicy(request.llm_model) if request.policy == "openai" else RuleBasedReasoningPolicy()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return Investigator(backend.hybrid, policy, request.max_retries, request.evidence_limit).investigate(request.question)

    @app.post("/fact_check", response_model=FactCheckReport)
    def fact_check(request: FactCheckRequest) -> FactCheckReport:
        try:
            policy = OpenAIFactCheckPolicy(request.llm_model) if request.policy == "openai" else RuleBasedFactCheckPolicy()
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        return FactChecker(backend.hybrid, request.evidence_limit).check(request.investigation, policy)

    @app.post("/submit_verdict", response_model=VerdictResponse)
    def submit_verdict(request: VerdictRequest) -> VerdictResponse:
        return VerdictResponse(received=True, evaluation_status="not_evaluated",
            message="Prediction recorded; it has not been automatically judged correct.", prediction=request.prediction)

    @app.get("/graph/{node_id}")
    def graph_neighbors(node_id: str) -> dict[str, object]:
        try:
            return {"node_id": node_id, "relationships": backend.graph.neighbors(node_id)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error

    return app


app = create_app()
