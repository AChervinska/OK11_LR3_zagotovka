"""
Фікстури для тестів.

Тести навмисно НЕ завантажують реальну модель Hugging Face — це
зробило б автоматичну перевірку залежною від мережі та повільною.
Замість цього класифікатор підміняється фейком (FakeClassifier),
який імітує контракт SentimentClassifier: load() / predict() / is_ready.

Реальну модель студент перевіряє вручну (крок 7 інструкції) та
тестом test_real_model.py, позначеним як повільний.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.main as main_module


class FakeClassifier:
    """
    Мінімальний двійник SentimentClassifier для швидких тестів.

    `always_ready=False` імітує ситуацію, коли load() відпрацював без
    винятку, але модель з якоїсь причини лишається непридатною до
    роботи (наприклад, не пройшла внутрішню перевірку) — це навмисно
    змушує /health і /predict перевіряти is_ready явно, а не вважати
    «load() виконався без помилки» достатньою умовою готовності.
    """

    def __init__(self, always_ready: bool = True) -> None:
        self._loaded = False
        self._always_ready = always_ready
        self.model_name = "fake/distilbert-stub"

    def load(self) -> None:
        self._loaded = True

    def predict(self, text: str) -> tuple[str, float]:
        if "bad" in text.lower() or "terrible" in text.lower():
            return "NEGATIVE", 0.98
        return "POSITIVE", 0.99

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.predict(t) for t in texts]

    @property
    def is_ready(self) -> bool:
        return self._loaded and self._always_ready


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient із підміненим (готовим) класифікатором."""
    monkeypatch.setattr(main_module, "classifier", FakeClassifier(always_ready=True))
    with TestClient(main_module.app) as c:
        yield c


@pytest.fixture
def client_not_ready(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """TestClient із класифікатором, що імітує НЕ готову модель."""
    monkeypatch.setattr(main_module, "classifier", FakeClassifier(always_ready=False))
    with TestClient(main_module.app) as c:
        yield c
