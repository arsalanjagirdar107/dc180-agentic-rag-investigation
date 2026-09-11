"""Fresh adversarial retrieval and citation-validated fact checking."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import BaseModel

from investigation.investigator import InvestigationReport
from retrieval.hybrid import HybridRetriever
from retrieval.lexical import load_documents
from retrieval.semantic import DEFAULT_MODEL


class AdversarialSearch(BaseModel):
    purpose: Literal["contradiction", "timeline", "alternative"]
    query: str


class CheckEvidence(BaseModel):
    document_id: str
    subject: str | None
    excerpt: str
    hybrid_score: float
    retrieval_methods: list[str]


class FactCheckFinding(BaseModel):
    verdict: Literal["supported", "contradicted", "inconclusive"]
    explanation: str
    supporting_document_ids: list[str]
    contradicting_document_ids: list[str]
    alternative_document_ids: list[str]


class FactCheckDecision(BaseModel):
    verdict: Literal["supported", "contradicted", "inconclusive"]
    explanation: str
    supporting_document_ids: list[str]
    contradicting_document_ids: list[str]
    alternative_document_ids: list[str]


class FactCheckPolicy(Protocol):
    def decide(self, theory: str, evidence: list[CheckEvidence]) -> FactCheckDecision: ...


class OpenAIFactCheckPolicy:
    """LLM verdict policy constrained to freshly retrieved adversarial evidence."""
    def __init__(self, model: str = "gpt-5-mini") -> None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI Fact-Checker policy.")
        self.client, self.model = OpenAI(), model

    def decide(self, theory: str, evidence: list[CheckEvidence]) -> FactCheckDecision:
        response = self.client.responses.create(
            model=self.model,
            instructions=("Classify the hypothesis only from fresh evidence. Do not introduce facts or citations not "
                "present in the supplied evidence. Return supported, contradicted, or inconclusive with document IDs."),
            input=f"Hypothesis: {theory}\nFresh evidence: {[item.model_dump() for item in evidence]}",
            text={"format": {"type": "json_schema", "name": "fact_check_decision", "strict": True,
                "schema": FactCheckDecision.model_json_schema()}}, store=False,
        )
        return FactCheckDecision.model_validate_json(response.output_text)


class RuleBasedFactCheckPolicy:
    """Offline fallback retained for tests and no-key demos."""
    CONTRADICTION_CUES = ("cancel", "withdraw", "pending", "delay", "denied", "not execute", "reversed")
    ALTERNATIVE_CUES = ("counterparty", "revised", "different", "alternative")
    def decide(self, theory: str, evidence: list[CheckEvidence]) -> FactCheckDecision:
        contradicted = [item.document_id for item in evidence if any(cue in item.excerpt.lower() for cue in self.CONTRADICTION_CUES)]
        alternatives = [item.document_id for item in evidence if any(cue in item.excerpt.lower() for cue in self.ALTERNATIVE_CUES)]
        if contradicted:
            return FactCheckDecision(verdict="contradicted", explanation="Fresh evidence indicates a possible reversal.",
                supporting_document_ids=[], contradicting_document_ids=contradicted, alternative_document_ids=alternatives)
        return FactCheckDecision(verdict="inconclusive", explanation="Offline policy found no explicit reversal.",
            supporting_document_ids=[], contradicting_document_ids=[], alternative_document_ids=alternatives)


class FactCheckReport(BaseModel):
    investigated_theory: str
    searches: list[AdversarialSearch]
    retrieved_evidence: list[CheckEvidence]
    finding: FactCheckFinding


class FactChecker:
    """Independently challenge an Investigator report with new hybrid searches."""

    def __init__(self, retriever: HybridRetriever, evidence_limit: int = 3) -> None:
        self.retriever, self.evidence_limit = retriever, evidence_limit

    @staticmethod
    def _excerpt(document: dict[str, object], limit: int = 240) -> str:
        body = " ".join(str(document.get("body", "")).split())
        return body[:limit] + ("..." if len(body) > limit else "")

    def _searches(self, report: InvestigationReport) -> list[AdversarialSearch]:
        claims = " ".join(claim.claim for claim in report.grounded_claims)
        base = f"{report.state.question} {report.final_theory.statement} {claims}".strip()
        return [
            AdversarialSearch(purpose="contradiction", query=f"{base} cancelled withdrawn denied"),
            AdversarialSearch(purpose="timeline", query=f"{base} pending delayed revised"),
            AdversarialSearch(purpose="alternative", query=f"{base} counterparty alternative explanation"),
        ]

    def _retrieve(self, searches: list[AdversarialSearch]) -> list[CheckEvidence]:
        by_id: dict[str, CheckEvidence] = {}
        for search in searches:
            for result in self.retriever.search(search.query, self.evidence_limit):
                metadata = result.document.get("metadata", {})
                subject = metadata.get("subject") if isinstance(metadata, dict) else None
                item = CheckEvidence(document_id=result.document_id, subject=str(subject) if subject else None,
                    excerpt=self._excerpt(result.document), hybrid_score=result.score,
                    retrieval_methods=list(result.retrieval_methods))
                if item.document_id not in by_id or item.hybrid_score > by_id[item.document_id].hybrid_score:
                    by_id[item.document_id] = item
        return list(by_id.values())

    def _finding(self, evidence: list[CheckEvidence], policy: FactCheckPolicy, theory: str) -> FactCheckFinding:
        valid_ids = {item.document_id for item in evidence}
        decision = policy.decide(theory, evidence)
        keep = lambda ids: list(dict.fromkeys(item for item in ids if item in valid_ids))
        return FactCheckFinding(verdict=decision.verdict, explanation=decision.explanation,
            supporting_document_ids=keep(decision.supporting_document_ids),
            contradicting_document_ids=keep(decision.contradicting_document_ids),
            alternative_document_ids=keep(decision.alternative_document_ids))

    def check(self, report: InvestigationReport, policy: FactCheckPolicy | None = None) -> FactCheckReport:
        searches = self._searches(report)
        evidence = self._retrieve(searches)
        return FactCheckReport(investigated_theory=report.final_theory.statement, searches=searches,
            retrieved_evidence=evidence, finding=self._finding(evidence, policy or RuleBasedFactCheckPolicy(), report.final_theory.statement))


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarially fact-check an Investigator report.")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--investigation-file", required=True, type=Path)
    parser.add_argument("--evidence-limit", type=int, default=3)
    parser.add_argument("--lexical-weight", type=float, default=0.5)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    report = InvestigationReport.model_validate_json(args.investigation_file.read_text(encoding="utf-8"))
    retriever = HybridRetriever(load_documents(Path(args.corpus)), args.lexical_weight, args.embedding_model)
    print(FactChecker(retriever, args.evidence_limit).check(report).model_dump_json(indent=2))


if __name__ == "__main__":
    main()
