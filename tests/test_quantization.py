"""
Крок 9 (поглиблений рівень): тести pick_quantization_config() —
логіка вибору конфігурації квантизації залежно від архітектури CPU.

AutoQuantizationConfig підмінюється фейком: тест перевіряє ЛОГІКУ
вибору (яку гілку коду виконано), а не реальну квантизацію моделі.
"""

import pytest

pytestmark = pytest.mark.advanced

import scripts.quantize as quantize_module


class _FakeAutoQuantizationConfig:
    calls: list[str] = []

    @classmethod
    def arm64(cls, is_static: bool = False) -> str:
        cls.calls.append("arm64")
        return "arm64-config"

    @classmethod
    def avx512_vnni(cls, is_static: bool = False) -> str:
        cls.calls.append("avx512_vnni")
        return "avx512_vnni-config"

    @classmethod
    def avx2(cls, is_static: bool = False) -> str:
        cls.calls.append("avx2")
        return "avx2-config"


@pytest.fixture(autouse=True)
def _reset_and_patch(monkeypatch: pytest.MonkeyPatch):
    _FakeAutoQuantizationConfig.calls.clear()
    monkeypatch.setattr(quantize_module, "AutoQuantizationConfig", _FakeAutoQuantizationConfig)
    yield


def test_picks_arm64_on_apple_silicon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(quantize_module.platform, "machine", lambda: "arm64")
    quantize_module.pick_quantization_config()
    assert _FakeAutoQuantizationConfig.calls == ["arm64"]


def test_picks_avx512_vnni_when_supported(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(quantize_module.platform, "machine", lambda: "x86_64")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags\t\t: fpu vme de pse avx512_vnni avx2 sse4_2")
    monkeypatch.setattr(quantize_module, "CPUINFO_PATH", str(cpuinfo))

    quantize_module.pick_quantization_config()
    assert _FakeAutoQuantizationConfig.calls == ["avx512_vnni"]


def test_falls_back_to_avx2_without_vnni_flag(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(quantize_module.platform, "machine", lambda: "x86_64")
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("flags\t\t: fpu vme de pse avx2 sse4_2")
    monkeypatch.setattr(quantize_module, "CPUINFO_PATH", str(cpuinfo))

    quantize_module.pick_quantization_config()
    assert _FakeAutoQuantizationConfig.calls == ["avx2"]


def test_falls_back_to_avx2_when_cpuinfo_missing(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Наприклад, macOS — де /proc/cpuinfo не існує."""
    monkeypatch.setattr(quantize_module.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(quantize_module, "CPUINFO_PATH", str(tmp_path / "does-not-exist"))

    quantize_module.pick_quantization_config()
    assert _FakeAutoQuantizationConfig.calls == ["avx2"]
