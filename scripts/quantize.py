"""
Динамічна INT8-квантизація ONNX-моделі (поглиблений рівень, кроки 9-10).

Запуск (після експорту в ONNX, крок 2):
    python scripts/quantize.py

ЕТАЛОННЕ РІШЕННЯ — для викладача.
"""

import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

from app.config import Settings

settings = Settings()

# Винесено в змінну, щоб тести могли підмінити шлях (tmp_path) без
# реального доступу до /proc.
CPUINFO_PATH = "/proc/cpuinfo"


def pick_quantization_config():
    """Обирає конфігурацію квантизації залежно від архітектури CPU."""
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return AutoQuantizationConfig.arm64(is_static=False)

    try:
        with open(CPUINFO_PATH) as f:
            flags = f.read()
        if "avx512_vnni" in flags:
            return AutoQuantizationConfig.avx512_vnni(is_static=False)
    except FileNotFoundError:
        pass  # не Linux (напр. macOS) — падаємо на безпечний варіант нижче

    return AutoQuantizationConfig.avx2(is_static=False)


def main() -> None:
    qconfig = pick_quantization_config()
    quantizer = ORTQuantizer.from_pretrained(settings.onnx_model_dir)
    quantizer.quantize(save_dir=settings.onnx_int8_model_dir, quantization_config=qconfig)
    print(f"Квантизовану модель збережено в {settings.onnx_int8_model_dir}")


if __name__ == "__main__":
    main()
