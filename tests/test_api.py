from pathlib import Path
import unittest

from fastapi.testclient import TestClient

from api.main import create_app


ROOT = Path(__file__).resolve().parents[1]
client = TestClient(create_app(ROOT / "examples" / "phase5_sample.jsonl"))


class ApiTest(unittest.TestCase):
    def test_health_and_search(self) -> None:
        self.assertEqual(client.get("/health").json()["status"], "ok")
        response = client.post("/search_evidence", json={"query": "contract approval", "top_k": 1})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["document_id"], "sample-001")


    def test_investigate_fact_check_and_graph(self) -> None:
        isolated = TestClient(create_app(ROOT / "examples" / "phase5_sample.jsonl"))
        investigation = isolated.post("/investigate", json={
            "question": "attorney clearance before transaction", "policy": "rule", "evidence_limit": 1
        })
        self.assertEqual(investigation.status_code, 200)
        self.assertEqual(investigation.json()["state"]["status"], "sufficient")
        fact_check = isolated.post("/fact_check", json={"investigation": investigation.json(), "policy": "rule"})
        self.assertEqual(fact_check.status_code, 200)
        self.assertIn(fact_check.json()["finding"]["verdict"], {"supported", "contradicted", "inconclusive"})
        graph = isolated.get("/graph/document:sample-001")
        self.assertEqual(graph.status_code, 200)
        self.assertTrue(any(item["source_document_ids"] == ["sample-001"] for item in graph.json()["relationships"]))

    def test_ingest_search_and_verdict(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "maildir" / "person" / "inbox"
            raw.mkdir(parents=True)
            (raw / "1.txt").write_text("From: a@example.com\nTo: b@example.com\nSubject: Audit signal\n\nThe audit signal is ready.", encoding="utf-8")
            output = root / "processed.jsonl"
            ingest = client.post("/ingest", json={"input_dir": str(root / "maildir"), "output_file": str(output), "max_documents": 1})
            self.assertEqual(ingest.status_code, 200)
            self.assertEqual(client.get("/health").json()["documents"], 1)
            self.assertEqual(client.post("/search_evidence", json={"query": "audit signal"}).status_code, 200)
            verdict = client.post("/submit_verdict", json={"prediction": "ready"})
            self.assertEqual(verdict.json()["evaluation_status"], "not_evaluated")
