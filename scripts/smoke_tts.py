#!/usr/bin/env python
"""Render a sentence with the configured Piper voice to a WAV file.

This verifies the entire local TTS path without needing LiveKit, Ollama, or a
microphone:

    python scripts/smoke_tts.py "Good evening, sir."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cosmion.config import ConfigError, load_settings  # noqa: E402
from cosmion.tts import PiperTTS, _downmix_chunk  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "text",
        nargs="?",
        default="Good evening, sir. All systems are online and operating normally.",
        help="Sentence to synthesize.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "cosmion-smoke.wav",
        help="Output WAV path (default: artifacts/cosmion-smoke.wav).",
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
        settings.validate_voice_files()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    plugin = PiperTTS(
        settings.piper_model_path,
        speed=settings.piper_speed,
        volume=settings.piper_volume,
    )
    chunks = list(
        plugin.voice.synthesize(args.text, syn_config=plugin._synthesis_config())
    )
    if not chunks:
        print("Piper returned no audio.", file=sys.stderr)
        return 1

    audio = np.concatenate(
        [np.frombuffer(_downmix_chunk(chunk), dtype=np.int16) for chunk in chunks]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, audio, plugin.sample_rate, subtype="PCM_16")

    duration = len(audio) / plugin.sample_rate
    print(f'Wrote "{args.text}"')
    print(f"  file:        {args.out}")
    print(f"  sample rate: {plugin.sample_rate} Hz")
    print(f"  duration:    {duration:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
