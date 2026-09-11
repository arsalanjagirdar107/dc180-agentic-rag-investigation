"""HTTP-only client used by the Streamlit frontend."""
from __future__ import annotations
from typing import Any
import requests

class BackendError(RuntimeError):
    pass

def request_api(base_url: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.request(method, f"{base_url.rstrip('/')}{path}", json=payload, timeout=60)
    except requests.RequestException as error:
        raise BackendError(f"Cannot reach the backend: {error}") from error
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise BackendError(f"Backend returned {response.status_code}: {detail}")
    return response.json()
