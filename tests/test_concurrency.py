"""
Поглиблений рівень — крок 13: перевірка, що синхронний CPU-bound виклик
моделі не блокує event loop.

Використовує httpx.AsyncClient з ASGITransport — конкурентні запити йдуть
напряму до ASGI-застосунку в тому самому процесі, без реального TCP-сокета,
тому тест швидкий і детермінований (жодного реального мережевого стеку).

Важливо: час завершення "швидкого" запиту (/health) вимірюється
ВСЕРЕДИНІ його власної корутини, а не після asyncio.gather() — інакше
вимірюється час найповільнішої задачі, а не тієї, що нас цікавить.
"""

import asyncio
import time

import httpx
import pytest

import app.main as main_module

pytestmark = pytest.mark.advanced

SLEEP_SECONDS = 0.4
STARTUP_DELAY = 0.05  # щоб slow-запит гарантовано стартував першим


class SlowFakeClassifier:
    """Класифікатор, чий predict() імітує важкий синхронний інференс."""

    def __init__(self) -> None:
        self.model_name = "fake/slow-stub"

    def load(self) -> None:
        pass

    def predict(self, text: str) -> tuple[str, float]:
        time.sleep(SLEEP_SECONDS)  # синхронний, блокуючий виклик — навмисно
        return "POSITIVE", 0.99

    def predict_batch(self, texts: list[str]) -> list[tuple[str, float]]:
        return [self.predict(t) for t in texts]

    @property
    def is_ready(self) -> bool:
        return True


async def _run_scenario() -> float:
    """Повертає час (у секундах від старту), за який завершився /health,
    поки паралельно виконується повільний /predict."""

    async with main_module.app.router.lifespan_context(main_module.app):
        transport = httpx.ASGITransport(app=main_module.app)
        start = time.perf_counter()

        async def slow_predict() -> None:
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.post("/predict", json={"text": "heavy request"})
                assert response.status_code == 200

        async def fast_health() -> float:
            await asyncio.sleep(STARTUP_DELAY)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                response = await ac.get("/health")
                assert response.status_code == 200
            return time.perf_counter() - start

        _, fast_elapsed = await asyncio.gather(slow_predict(), fast_health())
        return fast_elapsed


def test_slow_predict_does_not_block_health(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module, "classifier", SlowFakeClassifier())

    fast_elapsed = asyncio.run(_run_scenario())

    assert fast_elapsed < SLEEP_SECONDS * 0.5, (
        f"/health завершився за {fast_elapsed:.3f} с, поки /predict виконувався "
        f"{SLEEP_SECONDS} с — event loop, схоже, заблокований синхронним "
        "викликом моделі. Перевірте, чи виклик clf.predict() у /predict "
        "обгорнутий у run_in_threadpool (крок 13 інструкції)."
    )
