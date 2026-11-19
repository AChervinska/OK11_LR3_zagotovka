"""
Крок 5: перевірка, що create_classifier() обирає правильний клас і
правильну теку моделі залежно від settings.backend.

Ці тести НЕ завантажують жодної реальної моделі — вони перевіряють
лише логіку вибору, тому виконуються миттєво і без мережі.
"""

import pytest

import app.main as main_module
from app.config import Backend
from app.model import ONNXSentimentClassifier, SentimentClassifier


def test_factory_returns_plain_pytorch_classifier_for_pytorch_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main_module.settings, "backend", Backend.PYTORCH)
    clf = main_module.create_classifier()
    assert isinstance(clf, SentimentClassifier)
    assert not isinstance(clf, ONNXSentimentClassifier)
    assert clf.model_name == main_module.settings.model_name


def test_factory_returns_onnx_classifier_with_fp32_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.settings, "backend", Backend.ONNX)
    clf = main_module.create_classifier()
    assert isinstance(clf, ONNXSentimentClassifier)
    assert clf.model_dir == main_module.settings.onnx_model_dir


def test_factory_returns_onnx_classifier_with_int8_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.settings, "backend", Backend.ONNX_INT8)
    clf = main_module.create_classifier()
    assert isinstance(clf, ONNXSentimentClassifier)
    assert clf.model_dir == main_module.settings.onnx_int8_model_dir
