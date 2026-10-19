#!/usr/bin/env python
"""Download a Piper voice model (``.onnx`` + ``.onnx.json``) from Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEST = PROJECT_ROOT / "models"
BASE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# Single-speaker British voices. Multi-speaker voices are intentionally omitted
# because COSMION does not currently select a speaker id.
VOICES: dict[str, str] = {
    "en_GB-alan-medium": "Calm British male; the default COSMION voice.",
    "en_GB-northern_english_male-medium": "Northern English male; warmer, less RP.",
}


def _voice_url(voice: str, suffix: str) -> str:
    try:
        language, name, quality = voice.split("-")
    except ValueError as exc:
        raise SystemExit(
            f"Invalid voice name {voice!r}; expected <language>-<name>-<quality>."
        ) from exc
    family = language.split("_")[0]
    return f"{BASE_URL}/{family}/{language}/{name}/{quality}/{voice}{suffix}"


def _download(url: str, destination: Path, *, force: bool) -> None:
    if destination.is_file() and not force:
        print(f"  exists  {destination.name} (use --force to re-download)")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    def progress(blocks: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(blocks * block_size, total_size)
        percent = downloaded / total_size * 100
        print(f"\r  fetching {destination.name}  {percent:5.1f}%", end="", flush=True)

    print(f"Downloading {url}")
    try:
        urllib.request.urlretrieve(url, partial, reporthook=progress)
        partial.replace(destination)
        print()
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"Download failed for {url}: {exc}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--voice",
        default="en_GB-alan-medium",
        choices=sorted(VOICES),
        help="Piper voice identifier (default: en_GB-alan-medium).",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Destination directory (default: ./models).",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--list", action="store_true", help="List curated voices and exit.")
    args = parser.parse_args()

    if args.list:
        print("Curated COSMION voices:\n")
        for name, description in VOICES.items():
            print(f"  {name:<40} {description}")
        return 0

    destination = args.dest.resolve()
    model_path = destination / f"{args.voice}.onnx"
    config_path = destination / f"{args.voice}.onnx.json"

    print(f"Voice: {args.voice}")
    print(f"Target: {destination}\n")

    _download(_voice_url(args.voice, ".onnx"), model_path, force=args.force)
    _download(_voice_url(args.voice, ".onnx.json"), config_path, force=args.force)

    try:
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        sample_rate = metadata["audio"]["sample_rate"]
    except (OSError, KeyError, ValueError) as exc:
        raise SystemExit(f"Downloaded voice metadata is invalid: {exc}") from exc

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"\nReady: {model_path.name} ({size_mb:.1f} MB, {sample_rate} Hz)")
    print("\nPoint PIPER_MODEL_PATH at this file in .env.local if you changed the default.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
