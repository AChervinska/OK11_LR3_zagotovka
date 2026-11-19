"""
Конфігурація застосунку через змінні середовища.

ЛР2 додає одне нове поле — backend: який виконавець моделі
використовувати. Це і є місце, де окупається Dependency Injection
з ЛР1 — щоб перемкнути сервіс на ONNX чи квантизовану версію,
достатньо змінити ОДНУ змінну середовища, не займаючи жодного
рядка в app/main.py чи app/model.py.

Документація pydantic-settings:
https://docs.pydantic.dev/latest/concepts/pydantic_settings/
"""

from enum import Enum

from pydantic_settings import BaseSettings, SettingsConfigDict


class Backend(str, Enum):
    """Три виконавці моделі, між якими перемикається сервіс у ЛР2."""

    PYTORCH = "pytorch"        # базовий рівень — з ЛР1, без змін
    ONNX = "onnx"               # базовий рівень — крок 4
    ONNX_INT8 = "onnx-int8"     # поглиблений рівень — крок 9


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OK11_", env_file=".env", extra="ignore")

    model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"
    max_batch_size: int = 50
    log_level: str = "INFO"

    # --- нове в ЛР2 ---
    backend: Backend = Backend.PYTORCH
    onnx_model_dir: str = "models/onnx-fp32"
    onnx_int8_model_dir: str = "models/onnx-int8"
