"""Local text-to-speech powered by Piper."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
from livekit.agents import tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from piper import PiperVoice, SynthesisConfig

logger = logging.getLogger(__name__)

_VOICE_CACHE: dict[tuple[str, bool], PiperVoice] = {}
_VOICE_CACHE_LOCK = threading.Lock()
_SYNTHESIS_LOCK = threading.Lock()


def _downmix_chunk(chunk: Any) -> bytes:
    """Return raw PCM bytes, downmixing multi-channel Piper output to mono."""
    data: bytes = chunk.audio_int16_bytes
    channels = int(getattr(chunk, "sample_channels", 1) or 1)
    if channels <= 1:
        return data

    audio = np.frombuffer(data, dtype=np.int16)
    if audio.size % channels != 0:
        return data
    mono = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return mono.tobytes()


def _resolve_sample_rate(voice: PiperVoice, model_path: Path) -> int:
    """Read the sample rate from the loaded voice, falling back to its sidecar."""
    configured = getattr(getattr(voice, "config", None), "sample_rate", None)
    if configured:
        return int(configured)

    sidecar = Path(f"{model_path}.json")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return int(payload["audio"]["sample_rate"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Could not determine the Piper sample rate from {sidecar}. "
            "The voice sidecar JSON is missing or malformed."
        ) from exc


def preload_piper_voice(model_path: str | Path, *, use_cuda: bool = False) -> PiperVoice:
    """Load and cache a Piper voice once per process.

    Loading the ONNX session takes several seconds. Doing it in LiveKit's
    ``prewarm`` hook keeps that work off the conversation event loop.
    """
    path = Path(model_path).expanduser().resolve()
    key = (str(path), use_cuda)
    with _VOICE_CACHE_LOCK:
        cached = _VOICE_CACHE.get(key)
        if cached is not None:
            return cached

        logger.info("Loading Piper voice %s (use_cuda=%s)", path.name, use_cuda)
        voice = PiperVoice.load(str(path), use_cuda=use_cuda)
        _VOICE_CACHE[key] = voice
        logger.info("Piper voice loaded")
        return voice


class PiperTTS(tts.TTS):
    """A LiveKit TTS plugin that speaks through a local Piper voice.

    ``speed`` is a multiplier: 1.0 is the voice's natural pace, values below 1.0
    are slower and values above 1.0 are faster. COSMION defaults to 0.95 for a
    calm, deliberate JARVIS-style delivery.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        speed: float = 0.95,
        volume: float = 1.0,
        noise_scale: float = 0.667,
        noise_w: float = 0.8,
        use_cuda: bool = False,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Piper voice model not found: {path}")

        self._model_path = path
        self._voice = preload_piper_voice(path, use_cuda=use_cuda)
        self._sample_rate = _resolve_sample_rate(self._voice, path)
        self._speed = speed
        self._volume = volume
        self._noise_scale = noise_scale
        self._noise_w = noise_w

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._sample_rate,
            num_channels=1,
        )

    @property
    def voice(self) -> PiperVoice:
        """The loaded Piper voice, exposed for smoke tests and tooling."""
        return self._voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options=DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _PiperChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
        )

    def _synthesis_config(self) -> SynthesisConfig:
        # Piper's length_scale is the inverse of speed: larger values are slower.
        return SynthesisConfig(
            volume=self._volume,
            length_scale=1.0 / self._speed,
            noise_scale=self._noise_scale,
            noise_w_scale=self._noise_w,
            normalize_audio=True,
        )


class _PiperChunkedStream(tts.ChunkedStream):
    """Synthesizes Piper audio on a worker thread and streams chunks to LiveKit."""

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=self._tts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()
        config = self._tts._synthesis_config()
        text = self.input_text

        def produce() -> None:
            try:
                # A prewarmed voice can be shared by sequential jobs in the
                # same process; serialize synthesis so it is never used twice
                # at once.
                with _SYNTHESIS_LOCK:
                    for chunk in self._tts.voice.synthesize(text, syn_config=config):
                        loop.call_soon_threadsafe(queue.put_nowait, _downmix_chunk(chunk))
            except BaseException as exc:  # propagate synthesis failures
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                output_emitter.push(item)
        except Exception:
            logger.exception("Piper synthesis failed")
            raise
        finally:
            await producer
