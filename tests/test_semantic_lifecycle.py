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

        class FakeSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                nonlocal created_models
                created_models += 1

            def encode(self, values, **_: object) -> np.ndarray:
                count = len(values) if isinstance(values, list) else 1
                vectors = np.ones((count, 2), dtype=np.float32)
                return vectors if isinstance(values, list) else vectors[0]

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            app = create_app(ROOT / "examples" / "phase5_sample.jsonl")
            backend = app.state.backend

            self.assertFalse(backend.semantic.is_loaded)
            self.assertIs(backend.semantic, backend.hybrid.semantic)
            self.assertEqual(created_models, 0)

            self.assertEqual(TestClient(app).get("/health").status_code, 200)
            self.assertEqual(created_models, 0)

            backend.semantic.search("contract approval", 1)
            self.assertTrue(backend.semantic.is_loaded)
            self.assertEqual(created_models, 1)

            backend.hybrid.search("approval", 1)
            self.assertEqual(created_models, 1)


if __name__ == "__main__":
    unittest.main()
