"""
Сервіс інференсу на FastAPI — ЛР2 з дисципліни
«Інтеграція та супровід систем штучного інтелекту».

ЕТАЛОННЕ РІШЕННЯ — для викладача.

Порівняйте цей файл з ЛР1: lifespan, health, predict, predict_batch і
middleware НЕ ЗМІНИЛИСЬ ЖОДНИМ РЯДКОМ. Єдина зміна — функція
create_classifier(), яка обирає бекенд за налаштуванням. Це і є
practical payoff Dependency Injection з ЛР1 — заміна виконавця моделі
не торкається жодного ендпоінта.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from app.config import Backend, Settings
from app.model import ONNXSentimentClassifier, SentimentClassifier
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictionItem,
    PredictRequest,
    PredictResponse,
)

settings = Settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("ok11.lr2")


def create_classifier() -> SentimentClassifier:
    """Обирає реалізацію класифікатора за settings.backend (ЛР2)."""
    if settings.backend == Backend.PYTORCH:
        return SentimentClassifier(settings.model_name)
    if settings.backend == Backend.ONNX:
        return ONNXSentimentClassifier(settings.model_name, settings.onnx_model_dir)
    if settings.backend == Backend.ONNX_INT8:
        return ONNXSentimentClassifier(settings.model_name, settings.onnx_int8_model_dir)
    raise ValueError(f"Невідомий backend: {settings.backend}")


# Ліниве створення — див. коментар у заготовці app/main.py: це дозволяє
# тестам підміняти classifier через monkeypatch ДО того, як його
# створить get_classifier(), не чіпаючи реальні бекенди.
classifier: SentimentClassifier | None = None


def get_classifier() -> SentimentClassifier:
    global classifier
    if classifier is None:
        classifier = create_classifier()
    return classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_classifier().load()
    yield


app = FastAPI(
    title="Сервіс класифікації тональності тексту",
    description="ЛР2 · ОК11 «Інтеграція та супровід систем штучного інтелекту»",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health(clf: SentimentClassifier = Depends(get_classifier)) -> HealthResponse:
    if not clf.is_ready:
        raise HTTPException(status_code=503, detail="Модель ще не завантажена")
    return HealthResponse(status="ok", model_name=clf.model_name, backend=settings.backend.value)


@app.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest, clf: SentimentClassifier = Depends(get_classifier)
) -> PredictResponse:
    if not clf.is_ready:
        raise HTTPException(status_code=503, detail="Модель ще не завантажена")

    start = time.perf_counter()
    label, score = await run_in_threadpool(clf.predict, request.text)
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(label=label, score=score, latency_ms=latency_ms)


@app.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    request: BatchPredictRequest,
    clf: SentimentClassifier = Depends(get_classifier),
) -> BatchPredictResponse:
    if not clf.is_ready:
        raise HTTPException(status_code=503, detail="Модель ще не завантажена")

    if len(request.texts) > settings.max_batch_size:
        raise HTTPException(
            status_code=422,
            detail=f"Розмір пакета {len(request.texts)} перевищує ліміт {settings.max_batch_size}",
        )

    start = time.perf_counter()
    pairs = await run_in_threadpool(clf.predict_batch, request.texts)
    total_latency_ms = (time.perf_counter() - start) * 1000

    results = [PredictionItem(label=label, score=score) for label, score in pairs]
    throughput = len(results) / (total_latency_ms / 1000) if total_latency_ms > 0 else 0.0

    return BatchPredictResponse(
        results=results,
        batch_size=len(results),
        total_latency_ms=total_latency_ms,
        throughput_per_sec=throughput,
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id, request.method, request.url.path, response.status_code, duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response
