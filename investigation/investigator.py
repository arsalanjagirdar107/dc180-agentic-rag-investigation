"""A bounded, LLM-reasoned and evidence-grounded Investigator loop."""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel, Field

from retrieval.hybrid import HybridResult, HybridRetriever
from retrieval.lexical import load_documents
from retrieval.semantic import DEFAULT_MODEL


class Evidence(BaseModel):
    document_id: str
    source_path: str
    subject: str | None
    hybrid_score: float
    lexical_score: float
    semantic_score: float
    retrieval_methods: list[str]
    excerpt: str
    verification_status: Literal["unverified", "misleading", "verified"] = "unverified"
    verification_reason: str = "Corpus evidence is not independently verified."


class Theory(BaseModel):
    statement: str
    supporting_document_ids: list[str]


class SufficiencyAssessment(BaseModel):
    sufficient: bool
    reason: str


class InvestigationAttempt(BaseModel):
    attempt_number: int = Field(ge=1)
    query: str
    evidence: list[Evidence]
    theory: Theory
    assessment: SufficiencyAssessment


class InvestigationState(BaseModel):
    question: str
    max_retries: int = Field(ge=0)
    attempts: list[InvestigationAttempt] = Field(default_factory=list)
    status: Literal["searching", "sufficient", "exhausted"] = "searching"


class GroundedClaim(BaseModel):
    claim: str
    supporting_document_ids: list[str]
    verification_status: Literal["unverified", "misleading", "verified"] = "unverified"
    verification_reason: str = "Corpus evidence is not independently verified."


class InvestigationReport(BaseModel):
    state: InvestigationState
    final_theory: Theory
    grounded_claims: list[GroundedClaim]


class ClaimDraft(BaseModel):
    claim: str
    supporting_document_ids: list[str]


class ReasoningDecision(BaseModel):
    """LLM output; Python validates it before acting."""
    theory: str
    theory_document_ids: list[str]
    sufficient: bool
    sufficiency_reason: str
    next_query: str | None = None
    claims: list[ClaimDraft] = Field(default_factory=list)


class ReasoningPolicy(Protocol):
    def decide(self, question: str, query: str, evidence: list[Evidence]) -> ReasoningDecision: ...


class OpenAIReasoningPolicy:
    """Structured OpenAI Responses API policy; the SDK reads OPENAI_API_KEY."""
    def __init__(self, model: str = "gpt-5-mini") -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI reasoning policy.")
        self.client, self.model = OpenAI(), model

    def decide(self, question: str, query: str, evidence: list[Evidence]) -> ReasoningDecision:
        response = self.client.responses.create(
            model=self.model,
            instructions=(
                "You are an investigation reasoning policy. Use only supplied evidence. "
                "Return a cautious theory with valid document IDs; decide sufficiency; if insufficient, "
                "give a more specific next_query. Claims must be exact verbatim excerpts from evidence."
            ),
            input=f"Question: {question}\nCurrent query: {query}\nEvidence: {[item.model_dump() for item in evidence]}",
            text={"format": {"type": "json_schema", "name": "investigation_decision",
                  "strict": True, "schema": ReasoningDecision.model_json_schema()}},
            store=False,
        )
        return ReasoningDecision.model_validate_json(response.output_text)


class RuleBasedReasoningPolicy:
    """Optional offline fallback, retained only for local demos/tests."""
    HINTS = {"attorney": "legal", "clearance": "approval", "transaction": "contract", "deal": "contract"}

    def decide(self, question: str, query: str, evidence: list[Evidence]) -> ReasoningDecision:
        if not evidence:
            return ReasoningDecision(theory="No evidence was retrieved.", theory_document_ids=[], sufficient=False,
                sufficiency_reason="No evidence was returned.", next_query=f"{query} email evidence")
        top = evidence[0]
        sufficient = top.semantic_score >= 0.55 or top.lexical_score >= 0.50
        tokens = re.findall(r"\b[\w']+\b", query.lower())
        hints = list(dict.fromkeys(self.HINTS[t] for t in tokens if t in self.HINTS and self.HINTS[t] not in tokens))
        return ReasoningDecision(
            theory=f"The retrieved evidence most strongly concerns: {top.subject or top.document_id}.",
            theory_document_ids=[top.document_id], sufficient=sufficient,
            sufficiency_reason="Evidence meets local score thresholds." if sufficient else "Evidence is below local score thresholds.",
            next_query=None if sufficient else f"{query} {' '.join(hints or ['email evidence'])}",
            claims=[ClaimDraft(claim=top.excerpt, supporting_document_ids=[top.document_id])],
        )


class Investigator:
    """Python controls retrieval, retry bounds, and grounding; a policy supplies language reasoning."""
    def __init__(self, retriever: HybridRetriever, policy: ReasoningPolicy, max_retries: int = 2, evidence_limit: int = 3) -> None:
        if max_retries < 0 or evidence_limit < 1:
            raise ValueError("max_retries must be non-negative and evidence_limit must be at least 1")
        self.retriever, self.policy = retriever, policy
        self.max_retries, self.evidence_limit = max_retries, evidence_limit

    @staticmethod
    def _excerpt(document: dict[str, object], limit: int = 240) -> str:
        body = " ".join(str(document.get("body", "")).split())
        return body[:limit] + ("..." if len(body) > limit else "")

    def _evidence(self, results: list[HybridResult]) -> list[Evidence]:
        evidence: list[Evidence] = []
        for result in results[:self.evidence_limit]:
            metadata = result.document.get("metadata", {})
            subject = metadata.get("subject") if isinstance(metadata, dict) else None
            evidence.append(Evidence(
                document_id=result.document_id, source_path=str(result.document.get("source_path", "")),
                subject=str(subject) if subject else None, hybrid_score=result.score,
                lexical_score=result.lexical_score, semantic_score=result.semantic_score,
                retrieval_methods=list(result.retrieval_methods), excerpt=self._excerpt(result.document),
                verification_status=str((result.document.get("verification") or {}).get("status", "unverified")),
                verification_reason=str((result.document.get("verification") or {}).get("reason", "Corpus evidence is not independently verified.")),
            ))
        return evidence

    @staticmethod
    def _valid_ids(ids: list[str], evidence: list[Evidence]) -> list[str]:
        allowed = {item.document_id for item in evidence}
        return list(dict.fromkeys(item for item in ids if item in allowed))

    def _validate_claims(self, drafts: list[ClaimDraft], evidence: list[Evidence]) -> list[GroundedClaim]:
        """Reject uncited, unknown, or non-verbatim claims."""
        excerpts = {item.document_id: " ".join(item.excerpt.split()) for item in evidence}
        validated: list[GroundedClaim] = []
        for draft in drafts:
            ids, claim = self._valid_ids(draft.supporting_document_ids, evidence), " ".join(draft.claim.split())
            if ids and claim and any(claim in excerpts[item] for item in ids):
                source = next(item for item in evidence if item.document_id == ids[0])
                validated.append(GroundedClaim(claim=draft.claim, supporting_document_ids=ids,
                    verification_status=source.verification_status, verification_reason=source.verification_reason))
        return validated

    def investigate(self, question: str) -> InvestigationReport:
        state, query, final_claims = InvestigationState(question=question, max_retries=self.max_retries), question, []
        while True:
            evidence = self._evidence(self.retriever.search(query, self.evidence_limit))
            decision = self.policy.decide(question, query, evidence)
            theory_ids, claims = self._valid_ids(decision.theory_document_ids, evidence), self._validate_claims(decision.claims, evidence)
            theory = Theory(statement=decision.theory, supporting_document_ids=theory_ids)
            sufficient = decision.sufficient and bool(theory_ids) and bool(claims)
            reason = decision.sufficiency_reason if sufficient else (
                decision.sufficiency_reason + " Python grounding validation requires cited, verbatim claims and valid theory citations."
            )
            state.attempts.append(InvestigationAttempt(attempt_number=len(state.attempts) + 1, query=query,
                evidence=evidence, theory=theory, assessment=SufficiencyAssessment(sufficient=sufficient, reason=reason)))
            final_claims = claims if sufficient else []
            if sufficient:
                state.status = "sufficient"
                break
            if len(state.attempts) > self.max_retries or not decision.next_query or decision.next_query.strip() == query:
                state.status = "exhausted"
                break
            query = decision.next_query.strip()
        return InvestigationReport(state=state, final_theory=state.attempts[-1].theory, grounded_claims=final_claims)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded, LLM-reasoned investigation.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--evidence-limit", type=int, default=3)
    parser.add_argument("--lexical-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--llm-model", default=os.environ.get("OPENAI_MODEL", "gpt-5-mini"))
    parser.add_argument("--policy", choices=("openai", "rule"), default="openai")
    args = parser.parse_args()
    retriever = HybridRetriever(load_documents(Path(args.corpus)), args.lexical_weight, args.embedding_model)
    policy: ReasoningPolicy = OpenAIReasoningPolicy(args.llm_model) if args.policy == "openai" else RuleBasedReasoningPolicy()
    print(Investigator(retriever, policy, args.max_retries, args.evidence_limit).investigate(args.question).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
