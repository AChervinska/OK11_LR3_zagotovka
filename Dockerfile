
# syntax=docker/dockerfile:1

# Stage 1: builder
FROM python:3.12-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

ARG TORCH_VARIANT=cuda

RUN if [ "$TORCH_VARIANT" = "cpu" ]; then \
        pip install --no-cache-dir torch==2.5.1 \
          --index-url https://download.pytorch.org/whl/cpu; \
    else \
        pip install --no-cache-dir torch==2.5.1; \
    fi

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Stage 2: runtime
FROM python:3.12-slim

RUN groupadd --system app && \
    useradd --system --gid app --create-home app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

COPY app/ ./app/

RUN chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
    --start-period=120s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]