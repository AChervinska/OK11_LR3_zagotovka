"""
Тест на РЕАЛЬНІЙ моделі з Hugging Face Hub — крок 7 інструкції.

Цей тест навмисно не входить у звичайний прогін pytest: він тягне
~260 МБ ваг при першому запуску і вимагає інтернету. Запускається
окремо, коли студент хоче перевірити, що app/model.py справді працює
з бібліотекою transformers, а не лише проходить фейкові тести:

    RUN_SLOW=1 pytest tests/test_real_model.py -v
"""

import os

import pytest

from app.model import SentimentClassifier

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SLOW") != "1",
    reason="Повільний тест з реальним завантаженням моделі. Запускати явно: RUN_SLOW=1 pytest tests/test_real_model.py",
)


def test_real_model_loads_and_predicts() -> None:
    clf = SentimentClassifier("distilbert-base-uncased-finetuned-sst-2-english")
    assert not clf.is_ready

    clf.load()
    assert clf.is_ready

    label, score = clf.predict("I absolutely love this course!")
    assert label == "POSITIVE"
    assert score > 0.9

    label, score = clf.predict("This is the worst lecture I have ever attended.")
    assert label == "NEGATIVE"
    assert score > 0.9
