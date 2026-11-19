"""
Контрактні тести сервісу інференсу (перенесено з ЛР1, без змін по суті).

Запуск: pytest -m "not advanced" -v
Ці тести НЕ залежать від бекенда (pytorch/onnx) — вони перевіряють лише
контракт API, тому мають проходити незалежно від того, який backend
обрано в Settings.
"""

from fastapi.testclient import TestClient


def test_health_ok_when_model_ready(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "distilbert" in body["model_name"].lower()
    assert body["backend"] in ("pytorch", "onnx", "onnx-int8")


def test_health_returns_503_when_model_not_ready(client_not_ready: TestClient) -> None:
    response = client_not_ready.get("/health")
    assert response.status_code == 503


def test_predict_returns_valid_schema(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "This course is fantastic!"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] in ("POSITIVE", "NEGATIVE")
    assert 0.0 <= body["score"] <= 1.0
    assert body["latency_ms"] >= 0.0


def test_predict_classifies_negative_text(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "This is a terrible experience."})
    assert response.status_code == 200
    assert response.json()["label"] == "NEGATIVE"


def test_predict_rejects_empty_text(client: TestClient) -> None:
    """min_length=1 у PredictRequest має відсікати порожній рядок ще на
    валідації Pydantic — до того, як запит дійде до classifier.predict()."""
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_returns_503_when_model_not_ready(client_not_ready: TestClient) -> None:
    response = client_not_ready.post("/predict", json={"text": "hello"})
    assert response.status_code == 503
