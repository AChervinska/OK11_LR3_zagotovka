"""
Поглиблений рівень — крок 12: контрактні тести /predict/batch.
"""

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.advanced


def test_predict_batch_returns_valid_schema(client: TestClient) -> None:
    response = client.post(
        "/predict/batch",
        json={"texts": ["This course is great!", "This is a terrible experience."]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["batch_size"] == 2
    assert len(body["results"]) == 2
    assert body["results"][0]["label"] == "POSITIVE"
    assert body["results"][1]["label"] == "NEGATIVE"
    assert body["total_latency_ms"] >= 0.0
    assert body["throughput_per_sec"] >= 0.0


def test_predict_batch_preserves_order(client: TestClient) -> None:
    texts = ["good", "bad", "good", "bad", "good"]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    labels = [r["label"] for r in response.json()["results"]]
    assert labels == ["POSITIVE", "NEGATIVE", "POSITIVE", "NEGATIVE", "POSITIVE"]


def test_predict_batch_rejects_empty_list(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_predict_batch_rejects_over_schema_limit(client: TestClient) -> None:
    """BatchPredictRequest обмежує список 50 елементами на рівні схеми."""
    response = client.post("/predict/batch", json={"texts": ["x"] * 51})
    assert response.status_code == 422


def test_predict_batch_respects_configurable_max_size(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ліміт із Settings може бути СТРОГІШИМ за межу схеми (50)."""
    import app.main as main_module

    monkeypatch.setattr(main_module.settings, "max_batch_size", 2)
    response = client.post("/predict/batch", json={"texts": ["a", "b", "c"]})
    assert response.status_code == 422


def test_predict_batch_returns_503_when_model_not_ready(client_not_ready: TestClient) -> None:
    response = client_not_ready.post("/predict/batch", json={"texts": ["hello"]})
    assert response.status_code == 503
