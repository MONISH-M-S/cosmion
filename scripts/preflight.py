#!/usr/bin/env python
"""Check that COSMION's local prerequisites are ready before starting the worker."""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmion.config import DEFAULT_ENV_FILE, ConfigError, load_settings  # noqa: E402

failures = 0


def ok(label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[ OK ] {label}{suffix}")


def warn(label: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[WARN] {label}{suffix}")


def fail(label: str, detail: str = "") -> None:
    global failures
    failures += 1
    suffix = f" - {detail}" if detail else ""
    print(f"[FAIL] {label}{suffix}")


def check_python() -> None:
    ok("Python version", sys.version.split()[0])


def check_dependency(module: str, label: str) -> None:
    if importlib.util.find_spec(module):
        ok(label)
    else:
        fail(label, "not installed; run `pip install -e .`")


def check_env_file() -> None:
    if DEFAULT_ENV_FILE.is_file():
        ok(".env.local", "found")
    else:
        warn(".env.local", "not found; copy .env.example and fill it in")


def check_settings():
    try:
        settings = load_settings()
    except ConfigError as exc:
        fail("Configuration", str(exc))
        return None
    ok("Configuration", f"Ollama model {settings.ollama_model}")
    return settings


def check_voice(settings) -> None:
    try:
        settings.validate_voice_files()
    except ConfigError as exc:
        fail("Piper voice", str(exc))
        return
    size_mb = settings.piper_model_path.stat().st_size / (1024 * 1024)
    ok("Piper voice", f"{settings.piper_model_path.name} ({size_mb:.1f} MB)")


def check_ollama(settings) -> None:
    url = f"{settings.ollama_base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            payload = json.load(response)
    except Exception as exc:
        fail("Ollama", f"unreachable at {settings.ollama_base_url}: {exc}")
        return

    installed = {
        str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)
    }
    requested = settings.ollama_model
    if requested in installed or any(
        model.startswith(f"{requested}:") for model in installed
    ):
        ok("Ollama", f"reachable; model {requested!r} available")
    else:
        fail(
            "Ollama model",
            f"{requested!r} not installed; run `ollama pull {requested}`",
        )


def main() -> int:
    print("COSMION preflight\n")
    check_python()
    check_dependency("livekit.agents", "livekit-agents")
    check_dependency("faster_whisper", "faster-whisper")
    check_dependency("piper", "piper-tts")
    check_dependency("soundfile", "soundfile")
    check_env_file()

    settings = check_settings()
    if settings is not None:
        check_voice(settings)
        check_ollama(settings)

    print()
    if failures:
        print(f"{failures} check(s) failed. Fix the items above and run this again.")
        return 1
    print("All checks passed. Start COSMION with: python -m cosmion dev")
    return 0


if __name__ == "__main__":
    sys.exit(main())
