"""
Pydantic-схеми контракту сервісу інференсу.

Цей файл видається студенту вже готовим — контракт API є відправною
точкою роботи, а не тим, що студент вигадує сам. Змінювати poля тут
не потрібно; завдання ЛР1 — реалізувати логіку, яка ці схеми наповнює.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Вхідні дані для запиту класифікації тональності тексту."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Текст природною мовою для класифікації тональності.",
        examples=["This course is absolutely fantastic!"],
    )


class PredictResponse(BaseModel):
    """Відповідь сервісу інференсу."""

    label: str = Field(..., description="Мітка класу: 'POSITIVE' або 'NEGATIVE'.")
    score: float = Field(..., ge=0.0, le=1.0, description="Впевненість моделі у мітці, [0, 1].")
    latency_ms: float = Field(..., ge=0.0, description="Час виконання інференсу в мілісекундах.")


class HealthResponse(BaseModel):
    """Відповідь healthcheck-ендпоінта."""

    status: str = Field(..., description="'ok', якщо модель завантажена і готова приймати запити.")
    model_name: str = Field(..., description="Ідентифікатор моделі на Hugging Face Hub.")
    backend: str = Field(..., description="Активний виконавець моделі: pytorch, onnx або onnx-int8 (ЛР2).")


# ---------------------------------------------------------------------------
# Поглиблений рівень: контракт пакетного (batch) інференсу.
# Ці моделі так само видаються готовими — проєктування batch-контракту вже
# виконано, завдання поглибленого рівня полягає в реалізації логіки, що
# заповнює ці схеми (app/main.py, app/model.py).
# ---------------------------------------------------------------------------


class BatchPredictRequest(BaseModel):
    """Вхідні дані для пакетної класифікації кількох текстів за один виклик."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Список текстів для пакетного інференсу (від 1 до 50 елементів).",
    )


class PredictionItem(BaseModel):
    """Результат класифікації одного тексту всередині пакетної відповіді."""

    label: str
    score: float = Field(..., ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    """Відповідь пакетного ендпоінта /predict/batch."""

    results: list[PredictionItem] = Field(..., description="Результати в тому ж порядку, що й вхідні тексти.")
    batch_size: int = Field(..., ge=1, description="Кількість оброблених текстів.")
    total_latency_ms: float = Field(..., ge=0.0, description="Сумарний час виконання пакетного інференсу.")
    throughput_per_sec: float = Field(..., ge=0.0, description="Текстів за секунду: batch_size / (total_latency_ms/1000).")
