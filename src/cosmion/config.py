"""Typed configuration loaded from environment variables and ``.env.local``."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env.local"
DEFAULT_PIPER_MODEL = "models/en_GB-alan-medium.onnx"

VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def _required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key, "").strip()
    if not value:
        raise ConfigError(
            f"{key} is missing. Copy .env.example to .env.local and fill in the value."
        )
    return value


def _text(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default).strip()
    return value or default


def _number(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number (got {raw!r}).") from exc


def _boolean(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be true or false (got {raw!r}).")


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Validated runtime settings for the COSMION worker."""

    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str
    ollama_base_url: str
    ollama_model: str
    ollama_temperature: float
    whisper_model: str
    whisper_device: str
    whisper_compute_type: str
    whisper_language: str | None
    piper_model_path: Path
    piper_speed: float
    piper_volume: float
    piper_noise_scale: float
    piper_noise_w: float
    piper_use_cuda: bool
    log_level: str

    def __post_init__(self) -> None:
        if not self.livekit_url.startswith(("ws://", "wss://")):
            raise ConfigError("LIVEKIT_URL must start with ws:// or wss://.")
        if not 0.0 <= self.ollama_temperature <= 2.0:
            raise ConfigError("OLLAMA_TEMPERATURE must be between 0.0 and 2.0.")
        if self.piper_speed <= 0.0:
            raise ConfigError("PIPER_SPEED must be greater than 0.0.")
        if not 0.0 < self.piper_volume <= 1.0:
            raise ConfigError("PIPER_VOLUME must be between 0.0 (exclusive) and 1.0.")
        if not 0.0 <= self.piper_noise_scale <= 1.0:
            raise ConfigError("PIPER_NOISE_SCALE must be between 0.0 and 1.0.")
        if not 0.0 <= self.piper_noise_w <= 1.0:
            raise ConfigError("PIPER_NOISE_W must be between 0.0 and 1.0.")
        if self.log_level not in VALID_LOG_LEVELS:
            allowed = ", ".join(sorted(VALID_LOG_LEVELS))
            raise ConfigError(f"LOG_LEVEL must be one of: {allowed}.")

    @property
    def piper_config_path(self) -> Path:
        """Piper stores voice metadata next to the model as ``<model>.json``."""
        return Path(f"{self.piper_model_path}.json")

    def validate_voice_files(self) -> None:
        """Fail early with an actionable message if the voice is not installed."""
        missing = [p for p in (self.piper_model_path, self.piper_config_path) if not p.is_file()]
        if missing:
            joined = ", ".join(str(path) for path in missing)
            raise ConfigError(
                f"Piper voice files not found: {joined}. "
                "Run `python scripts/download_voice.py` to download the default voice."
            )


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Load settings from ``.env.local`` (when present) and environment variables.

    Passing ``env`` skips ``.env.local`` loading, which keeps the function easy to test.
    """
    if env is None:
        if DEFAULT_ENV_FILE.is_file():
            load_dotenv(dotenv_path=DEFAULT_ENV_FILE, override=False)
        env = os.environ

    return Settings(
        livekit_url=_required(env, "LIVEKIT_URL"),
        livekit_api_key=_required(env, "LIVEKIT_API_KEY"),
        livekit_api_secret=_required(env, "LIVEKIT_API_SECRET"),
        ollama_base_url=_text(env, "OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        ollama_model=_text(env, "OLLAMA_MODEL", "llama3.2"),
        ollama_temperature=_number(env, "OLLAMA_TEMPERATURE", 0.5),
        whisper_model=_text(env, "WHISPER_MODEL", "small"),
        whisper_device=_text(env, "WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_text(env, "WHISPER_COMPUTE_TYPE", "int8"),
        whisper_language=(env.get("WHISPER_LANGUAGE", "en").strip() or None),
        piper_model_path=_resolve_path(
            _text(env, "PIPER_MODEL_PATH", DEFAULT_PIPER_MODEL)
        ),
        piper_speed=_number(env, "PIPER_SPEED", 0.95),
        piper_volume=_number(env, "PIPER_VOLUME", 1.0),
        piper_noise_scale=_number(env, "PIPER_NOISE_SCALE", 0.667),
        piper_noise_w=_number(env, "PIPER_NOISE_W", 0.8),
        piper_use_cuda=_boolean(env, "PIPER_USE_CUDA", False),
        log_level=_text(env, "LOG_LEVEL", "INFO").upper(),
    )
