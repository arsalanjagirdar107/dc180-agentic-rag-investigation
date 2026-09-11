import unittest
from pathlib import Path

from evidence_graph.builder import EvidenceGraph, stable_id
from retrieval.lexical import load_documents


class EvidenceGraphTest(unittest.TestCase):
    def test_sender_relationship_has_its_source_document(self) -> None:
        root = Path(__file__).resolve().parents[1]
        board = EvidenceGraph.from_documents(load_documents(root / "examples" / "phase5_sample.jsonl"))
        sender = stable_id("person", "trader@example.com")
        document = stable_id("document", "sample-001")

        edge = board.graph[sender][document]["sent"]
        self.assertEqual(edge["source_document_ids"], ["sample-001"])
        self.assertIn(document, board.document_subgraph("sample-001").nodes)


if __name__ == "__main__":
    unittest.main()
