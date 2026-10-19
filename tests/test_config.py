"""Test configuration loading and validation."""

from __future__ import annotations

import pytest

from cosmion.config import PROJECT_ROOT, ConfigError, load_settings

BASE_ENV = {
    "LIVEKIT_URL": "wss://example.livekit.cloud",
    "LIVEKIT_API_KEY": "test-key",
    "LIVEKIT_API_SECRET": "test-secret",
}


def test_defaults_are_local_first() -> None:
    settings = load_settings(BASE_ENV)

    assert settings.ollama_base_url == "http://localhost:11434/v1"
    assert settings.ollama_model == "llama3.2"
    assert settings.whisper_model == "small"
    assert settings.whisper_device == "cpu"
    assert settings.whisper_compute_type == "int8"
    assert settings.whisper_language == "en"
    assert settings.piper_model_path == (PROJECT_ROOT / "models/en_GB-alan-medium.onnx")
    assert settings.piper_speed == 0.95
    assert settings.piper_volume == 1.0
    assert settings.log_level == "INFO"


def test_missing_required_variable_is_actionable() -> None:
    with pytest.raises(ConfigError, match="LIVEKIT_URL"):
        load_settings({})


def test_invalid_livekit_url_is_rejected() -> None:
    with pytest.raises(ConfigError, match="ws://"):
        load_settings({**BASE_ENV, "LIVEKIT_URL": "https://example.com"})


def test_invalid_speed_is_rejected() -> None:
    with pytest.raises(ConfigError, match="PIPER_SPEED"):
        load_settings({**BASE_ENV, "PIPER_SPEED": "0"})


def test_invalid_volume_is_rejected() -> None:
    with pytest.raises(ConfigError, match="PIPER_VOLUME"):
        load_settings({**BASE_ENV, "PIPER_VOLUME": "1.5"})


def test_invalid_temperature_is_rejected() -> None:
    with pytest.raises(ConfigError, match="OLLAMA_TEMPERATURE"):
        load_settings({**BASE_ENV, "OLLAMA_TEMPERATURE": "3"})


def test_empty_language_means_auto_detect() -> None:
    settings = load_settings({**BASE_ENV, "WHISPER_LANGUAGE": ""})
    assert settings.whisper_language is None


def test_relative_voice_path_resolves_against_project_root() -> None:
    settings = load_settings({**BASE_ENV, "PIPER_MODEL_PATH": "voices/custom.onnx"})
    assert settings.piper_model_path == (PROJECT_ROOT / "voices/custom.onnx").resolve()
    assert settings.piper_config_path.name == "custom.onnx.json"


def test_voice_file_validation_reports_download_hint(tmp_path) -> None:
    settings = load_settings(
        {**BASE_ENV, "PIPER_MODEL_PATH": str(tmp_path / "missing.onnx")}
    )
    with pytest.raises(ConfigError, match="download_voice.py"):
        settings.validate_voice_files()
