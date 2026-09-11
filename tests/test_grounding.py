import unittest

from investigation.investigator import ClaimDraft, Evidence, Investigator


class GroundingTest(unittest.TestCase):
    def test_misleading_status_is_preserved_on_grounded_claims(self) -> None:
        evidence = Evidence(document_id="doc-1", source_path="x", subject=None, hybrid_score=0.2,
            lexical_score=0.1, semantic_score=0.1, retrieval_methods=[], excerpt="Unverified statement.",
            verification_status="misleading", verification_reason="Conflicts with later evidence.")
        class NoRetriever: pass
        claims = Investigator(NoRetriever(), None)._validate_claims(
            [ClaimDraft(claim="Unverified statement.", supporting_document_ids=["doc-1"])], [evidence]
        )
        self.assertEqual(claims[0].verification_status, "misleading")
        self.assertEqual(claims[0].verification_reason, "Conflicts with later evidence.")
