import tempfile
import unittest
from pathlib import Path

import requests

from preprocessing.enron_loader import build_corpus
from retrieval.hybrid import HybridRetriever
from retrieval.lexical import TfidfRetriever, load_documents
from retrieval.semantic import SemanticRetriever


class RealEnronPipelineTest(unittest.TestCase):
    def test_public_enron_records_flow_through_pipeline(self) -> None:
        url = "https://datasets-server.huggingface.co/rows?dataset=TabMaven%2Fenron-mail-dataset-raw&config=default&split=train&offset=0&length=3"
        try:
            response = requests.get(url, timeout=30)
        except requests.RequestException as error:
            self.skipTest(f"Public Enron dataset is unavailable from this environment: {type(error).__name__}.")
        rows = response.json()["rows"]
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "maildir"
            for row in rows:
                item = row["row"]
                target = raw / item["file"].replace(".", "_")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(item["message"], encoding="utf-8")
            output = Path(temporary) / "enron.jsonl"
            self.assertEqual(build_corpus(raw, output, 3, None), 3)
            documents = load_documents(output)
            self.assertEqual(documents[0]["metadata"]["from"], "phillip.allen@enron.com")
            self.assertEqual(documents[0]["verification"]["status"], "unverified")
            self.assertTrue(TfidfRetriever(documents).search("forecast", 1))
            self.assertTrue(SemanticRetriever(documents).search("business forecast", 1))
            self.assertTrue(HybridRetriever(documents).search("forecast", 1))
