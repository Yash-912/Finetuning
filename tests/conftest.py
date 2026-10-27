"""Shared fixtures: a fake classifier and a mock upstream LLM, so no GPU or network is needed."""
from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

import middleware.app as app_module
from middleware.config import settings


class FakeClassifier:
    """Deterministic stand-in: 'ignore' -> injection (0.97), otherwise benign (0.99).

    A model path containing 'shadow' always predicts benign, so shadow/prod can disagree.
    """

    def __init__(self, model_path: str = "", max_length: int = 512, **_kwargs):
        self.model_path = model_path
        self.is_ready = True
        self.calls: list[str] = []

    def predict(self, text: str) -> tuple[int, float]:
        self.calls.append(text)
        if "shadow" in str(self.model_path):
            return 0, 0.90
        if "ignore" in text.lower():
            return 1, 0.97
        return 0, 0.99


class Upstream:
    """Records requests the proxy forwards and returns a canned completion."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.fail = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})


@pytest.fixture
def upstream() -> Upstream:
    return Upstream()


@pytest.fixture
def make_client(monkeypatch, tmp_path, upstream):
    """Factory so a test can choose settings (e.g. shadow mode) before startup."""
    monkeypatch.setattr(app_module, "InjectionClassifier", FakeClassifier)
    monkeypatch.setattr(settings, "review_queue_path", str(tmp_path / "queue.db"))
    monkeypatch.setattr(settings, "llm_endpoint", "http://upstream.test/v1")
    monkeypatch.setattr(settings, "llm_api_key", "test-key")
    monkeypatch.setattr(settings, "threshold", 0.85)
    monkeypatch.setattr(settings, "mode", "soft_flag")
    monkeypatch.setattr(settings, "shadow_enabled", False)
    monkeypatch.setattr(settings, "shadow_model_path", "")

    clients: list[TestClient] = []

    def _make(**overrides) -> TestClient:
        for key, value in overrides.items():
            monkeypatch.setattr(settings, key, value)
        client = TestClient(app_module.app)
        client.__enter__()
        client.app.state.httpx_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler))
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


def metric(name: str) -> float:
    return app_module._registry.get_sample_value(name) or 0.0
