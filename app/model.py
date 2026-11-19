"""
Обгортки над моделлю класифікації тональності.

SentimentClassifier — базовий (PyTorch) клас, перенесений з ЛР1 без змін.
ONNXSentimentClassifier — новий у ЛР2 клас, що замінює лише load():
predict(), predict_batch() та is_ready успадковуються без єдиної зміни.
Це навмисна демонстрація того, навіщо контракт (спільний інтерфейс)
було зафіксовано в ЛР1: заміна виконавця моделі — це заміна ОДНОГО
методу, а не переписування бізнес-логіки.
"""

from __future__ import annotations

from transformers import pipeline


class SentimentClassifier:
    """PyTorch-версія — еталонний бекенд, з яким порівнюються інші."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None

    def load(self) -> None:
        self._pipeline = pipeline("sentiment-analysis", model=self.model_name)

    def predict(self, text: str) -> tuple[str, float]:
        result = self._pipeline(text)[0]
        return result["label"], float(result["score"])

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        results = self._pipeline(texts)
        return [(r["label"], float(r["score"])) for r in results]

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None


class ONNXSentimentClassifier(SentimentClassifier):
    """
    ONNX-версія того самого класифікатора.

    model_dir — тека з експортованою моделлю (результат кроку 2:
    optimum-cli export onnx, або кроку 9 — квантизована версія).
    Та сама тека годиться і для FP32-, і для INT8-варіанта — різниця
    лише в тому, звідки саме зчитано model_dir (крок 5 проти кроку 10).
    """

    def __init__(self, model_name: str, model_dir: str) -> None:
        super().__init__(model_name)
        self.model_dir = model_dir

    def load(self) -> None:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        model = ORTModelForSequenceClassification.from_pretrained(self.model_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self._pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
