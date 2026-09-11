import unittest
from unittest.mock import patch
from ui.client import request_api

class FakeResponse:
    ok = True
    def json(self): return {"status": "ok"}

class UiClientTest(unittest.TestCase):
    @patch("ui.client.requests.request", return_value=FakeResponse())
    def test_client_calls_backend_over_http(self, request) -> None:
        self.assertEqual(request_api("http://api.test", "GET", "/health"), {"status": "ok"})
        request.assert_called_once_with("GET", "http://api.test/health", json=None, timeout=60)
