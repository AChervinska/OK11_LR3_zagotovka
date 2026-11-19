"""
Крок 4: перевірка, що ONNXSentimentClassifier.load() коректно
використовує ORTModelForSequenceClassification і AutoTokenizer з
ПРАВИЛЬНИМ model_dir, і що успадкований predict() і далі працює.

Реальні optimum/onnxruntime та завантаження файлів з диска тут НЕ
задіяні — обидві залежності підмінені фейками, щоб тест виконувався
миттєво й без реальної експортованої моделі. Перевірка на СПРАВЖНІЙ
експортованій моделі виконується вручну (крок 3 інструкції).
"""

import sys
import types

import pytest

from app.model import ONNXSentimentClassifier


class _FakeORTModel:
    """Двійник ORTModelForSequenceClassification.from_pretrained()."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "_FakeORTModel":
        return cls(model_dir)


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, model_dir: str) -> "_FakeTokenizer":
        return cls()


class _FakePipeline:
    """Двійник transformers.pipeline(...), що імітує sentiment-analysis."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, texts):
        single = isinstance(texts, str)
        items = [texts] if single else texts
        results = [
            {"label": "NEGATIVE", "score": 0.97} if "bad" in t.lower() else {"label": "POSITIVE", "score": 0.98}
            for t in items
        ]
        return results[0:1] if single else results


@pytest.fixture
def fake_onnx_backend(monkeypatch: pytest.MonkeyPatch):
    """Підміняє optimum.onnxruntime, transformers.AutoTokenizer і pipeline."""
    fake_optimum = types.ModuleType("optimum")
    fake_ort = types.ModuleType("optimum.onnxruntime")
    fake_ort.ORTModelForSequenceClassification = _FakeORTModel
    monkeypatch.setitem(sys.modules, "optimum", fake_optimum)
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", fake_ort)

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", _FakeTokenizer.from_pretrained)

    calls = []

    def fake_pipeline(task, model=None, tokenizer=None):
        calls.append({"task": task, "model": model, "tokenizer": tokenizer})
        return _FakePipeline(model, tokenizer)

    monkeypatch.setattr("app.model.pipeline", fake_pipeline)
    return calls


def test_load_uses_correct_model_dir(fake_onnx_backend) -> None:
    clf = ONNXSentimentClassifier("some/model-name", "models/onnx-fp32")
    assert not clf.is_ready

    clf.load()

    assert clf.is_ready
    assert isinstance(clf._pipeline.model, _FakeORTModel)
    assert clf._pipeline.model.model_dir == "models/onnx-fp32"
    assert fake_onnx_backend[0]["task"] == "sentiment-analysis"


def test_predict_and_predict_batch_inherited_unchanged(fake_onnx_backend) -> None:
    """predict()/predict_batch() успадковані від SentimentClassifier —
    тест демонструє, що їх не довелось переписувати для ONNX-бекенда."""
    clf = ONNXSentimentClassifier("some/model-name", "models/onnx-fp32")
    clf.load()

    label, score = clf.predict("This is bad")
    assert label == "NEGATIVE"

    pairs = clf.predict_batch(["great!", "so bad"])
    assert [p[0] for p in pairs] == ["POSITIVE", "NEGATIVE"]
