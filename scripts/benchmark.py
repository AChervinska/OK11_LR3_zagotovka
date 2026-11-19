"""
Порівняння бекендів моделі: розмір, латентність (медіана/p95), точність.

Запуск (крок 6, базовий рівень — 2 бекенди):
    python scripts/benchmark.py pytorch onnx

Запуск (крок 11, поглиблений рівень — 3 бекенди):
    python scripts/benchmark.py pytorch onnx onnx-int8

Для відтворюваних замірів рекомендовано (крок 11):
    OMP_NUM_THREADS=1 python scripts/benchmark.py ...

ЕТАЛОННЕ РІШЕННЯ — для викладача.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.model import ONNXSentimentClassifier, SentimentClassifier

settings = Settings()

# Невеликий розмічений набір для перевірки точності — ілюстративний,
# не заміна повноцінного тестового спліту SST-2.
TEST_EXAMPLES: list[tuple[str, str]] = [
    ("This movie was absolutely wonderful, I loved every minute of it.", "POSITIVE"),
    ("Waste of time, I want my money back.", "NEGATIVE"),
    ("Best purchase I've made all year.", "POSITIVE"),
    ("Terrible service, will not come back.", "NEGATIVE"),
    ("The plot was engaging and the acting superb.", "POSITIVE"),
    ("Broken on arrival and support never replied.", "NEGATIVE"),
    ("A delightful surprise from start to finish.", "POSITIVE"),
    ("Completely disappointed with the quality.", "NEGATIVE"),
    ("Highly recommend this to everyone I know.", "POSITIVE"),
    ("The worst experience I've had with any product.", "NEGATIVE"),
    ("Solid performance and great value for money.", "POSITIVE"),
    ("It stopped working after just two days.", "NEGATIVE"),
]

WARMUP_TEXTS = ["This is a warm-up request."] * 3


def build_classifier(backend: str) -> SentimentClassifier:
    if backend == "pytorch":
        return SentimentClassifier(settings.model_name)
    if backend == "onnx":
        return ONNXSentimentClassifier(settings.model_name, settings.onnx_model_dir)
    if backend == "onnx-int8":
        return ONNXSentimentClassifier(settings.model_name, settings.onnx_int8_model_dir)
    raise ValueError(f"Невідомий бекенд: {backend}")


def measure_latency(classifier: SentimentClassifier, texts: list[str]) -> dict:
    for warmup_text in WARMUP_TEXTS:
        classifier.predict(warmup_text)

    latencies_ms = []
    for text in texts:
        start = time.perf_counter()
        classifier.predict(text)
        latencies_ms.append((time.perf_counter() - start) * 1000)

    latencies_ms.sort()
    median_ms = statistics.median(latencies_ms)
    p95_ms = latencies_ms[max(int(len(latencies_ms) * 0.95) - 1, 0)]
    return {"median_ms": median_ms, "p95_ms": p95_ms, "n": len(texts)}


def measure_accuracy(classifier: SentimentClassifier, examples: list[tuple[str, str]]) -> float:
    correct = 0
    for text, expected_label in examples:
        label, _score = classifier.predict(text)
        if label == expected_label:
            correct += 1
    return correct / len(examples)


def measure_dir_size_mb(path: str) -> float:
    p = Path(path)
    if not p.exists():
        return float("nan")
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backends", nargs="+", choices=["pytorch", "onnx", "onnx-int8"])
    args = parser.parse_args()

    rows = []
    for backend in args.backends:
        clf = build_classifier(backend)
        clf.load()

        latency = measure_latency(clf, [ex[0] for ex in TEST_EXAMPLES])
        accuracy = measure_accuracy(clf, TEST_EXAMPLES)
        size_mb = (
            None
            if backend == "pytorch"
            else measure_dir_size_mb(
                settings.onnx_model_dir if backend == "onnx" else settings.onnx_int8_model_dir
            )
        )
        rows.append({"backend": backend, "size_mb": size_mb, "accuracy": accuracy, **latency})

    header = f"{'backend':<12}{'size_MB':>12}{'median_ms':>12}{'p95_ms':>10}{'accuracy':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        size_str = f"{r['size_mb']:.1f}" if r["size_mb"] is not None else "див. du -sh"
        print(
            f"{r['backend']:<12}{size_str:>12}{r['median_ms']:>12.1f}"
            f"{r['p95_ms']:>10.1f}{r['accuracy']:>10.1%}"
        )


if __name__ == "__main__":
    main()
