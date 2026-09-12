import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from api.main import create_app


ROOT = Path(__file__).resolve().parents[1]


class SemanticLifecycleTest(unittest.TestCase):
    def test_backend_defers_and_shares_one_semantic_model(self) -> None:
        created_models = 0

        class FakeTextEmbedding:
            def __init__(self, model_name: str) -> None:
                nonlocal created_models
                created_models += 1

            def embed(self, values: list[str]):
                return iter(np.ones((len(values), 2), dtype=np.float32))

        fake_module = types.SimpleNamespace(TextEmbedding=FakeTextEmbedding)
        with patch.dict(sys.modules, {"fastembed": fake_module}):
            app = create_app(ROOT / "examples" / "phase5_sample.jsonl")
            backend = app.state.backend

            self.assertFalse(backend.semantic.is_loaded)
            self.assertIs(backend.semantic, backend.hybrid.semantic)
            self.assertEqual(created_models, 0)

            self.assertEqual(TestClient(app).get("/health").status_code, 200)
            self.assertEqual(created_models, 0)

            semantic_results = backend.semantic.search("contract approval", 1)
            self.assertTrue(backend.semantic.is_loaded)
            self.assertEqual(created_models, 1)
            document_ids = {str(document["document_id"]) for document in backend.documents}
            self.assertIn(semantic_results[0].document["document_id"], document_ids)

            hybrid_results = backend.hybrid.search("approval", 1)
            self.assertEqual(created_models, 1)
            self.assertIn(hybrid_results[0].document_id, document_ids)
            self.assertEqual(
                hybrid_results[0].document["document_id"], hybrid_results[0].document_id
            )

            response = TestClient(app).post("/search_evidence", json={
                "query": "contract approval", "method": "hybrid", "top_k": 1
            })
            self.assertEqual(response.status_code, 200)
            self.assertTrue({"document_id", "score", "retrieval_methods", "metadata", "verification"}
                            <= response.json()["results"][0].keys())
            self.assertEqual(created_models, 1)


if __name__ == "__main__":
    unittest.main()
