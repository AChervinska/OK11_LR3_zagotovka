#!/usr/bin/env bash
# Наскрізна перевірка ЛР3 — аналог pytest для інфраструктурного коду.
# Запускати з кореня репозиторію після виконання export/quantize з ЛР2
# (тобто коли models/onnx-fp32 і models/onnx-int8 уже існують).
#
# Використання:
#   bash verify.sh            # базовий рівень (backend з compose.yaml)
#   bash verify.sh advanced   # додатково перевіряє /predict/batch і healthcheck

set -euo pipefail

PORT=8080
FAILED=0

pass() { echo "OK: $1"; }
fail() { echo "FAIL: $1"; FAILED=1; }

echo "== Збірка образів =="
docker compose build

echo "== Запуск сервісів =="
docker compose up -d

echo "== Очікування healthy-стану api (до 60с) =="
for i in $(seq 1 30); do
    status=$(docker compose ps api --format '{{.Health}}' 2>/dev/null || echo "")
    if [ "$status" = "healthy" ]; then
        pass "api перейшов у стан healthy за ${i}0 с"
        break
    fi
    sleep 2
done

echo "== Перевірка /health через nginx =="
if curl -sf "http://127.0.0.1:${PORT}/health" | grep -q '"status":"ok"'; then
    pass "/health повертає status=ok"
else
    fail "/health не повернув очікувану відповідь"
fi

echo "== Перевірка /predict через nginx =="
response=$(curl -sf -X POST "http://127.0.0.1:${PORT}/predict" \
    -H "Content-Type: application/json" \
    -d '{"text": "This lab was really useful"}')
if echo "$response" | grep -q "POSITIVE"; then
    pass "/predict коректно класифікував тестовий текст"
else
    fail "/predict повернув неочікуваний результат: $response"
fi

if [ "${1:-}" = "advanced" ]; then
    echo "== [Поглиблений рівень] Перевірка /predict/batch =="
    batch_response=$(curl -sf -X POST "http://127.0.0.1:${PORT}/predict/batch" \
        -H "Content-Type: application/json" \
        -d '{"texts": ["great!", "terrible"]}')
    if echo "$batch_response" | grep -q "batch_size"; then
        pass "/predict/batch відповідає коректно через nginx"
    else
        fail "/predict/batch не відповів як очікувалось"
    fi

    echo "== [Поглиблений рівень] Розмір образу api =="
    docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}" | grep -i "$(basename "$(pwd)")" || \
        docker compose images api
fi

echo "== Логи api (останні 15 рядків) =="
docker compose logs --tail=15 api

echo "== Зупинка =="
docker compose down

if [ "$FAILED" -eq 0 ]; then
    echo ""
    echo "УСІ ПЕРЕВІРКИ ПРОЙДЕНО"
    exit 0
else
    echo ""
    echo "Є ПОМИЛКИ — див. FAIL вище"
    exit 1
fi
