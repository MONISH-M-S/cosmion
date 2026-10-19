"""Local speech-to-text powered by faster-whisper."""

from __future__ import annotations

import asyncio
import io
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from livekit import rtc
from livekit.agents import APIConnectionError, APIConnectOptions, stt
from livekit.agents.utils import AudioBuffer

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[tuple[str, str, str, str | None], WhisperModel] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_TRANSCRIBE_LOCK = threading.Lock()

WhisperModelName = Literal[
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
]


@dataclass(slots=True)
class WhisperOptions:
    """Configuration for :class:`WhisperSTT`."""

    model: str
    language: str | None
    device: str
    compute_type: str
    model_cache_directory: str | None = None
    beam_size: int = 1


def _resample_to_whisper_rate(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample mono float32 audio to the 16 kHz rate Whisper expects."""
    target_rate = 16000
    if sample_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32, copy=False)

    target_size = max(1, int(round(audio.size * target_rate / sample_rate)))
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1.0, target_size)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def decode_wav(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes into mono float32 PCM at Whisper's 16 kHz rate.

    ``AudioFrame.to_wav_bytes()`` returns a complete WAV file, including its
    RIFF header. Treating those bytes as raw PCM feeds the header into Whisper
    as audio, so we decode the file properly with soundfile instead.

    LiveKit may deliver 24 kHz or 48 kHz frames. Whisper always interprets a
    float32 array as 16 kHz, so resampling here prevents faster, pitched-up
    speech and the accuracy loss that comes with it.
    """
    if not wav_bytes:
        raise ValueError("Received an empty WAV payload from LiveKit.")

    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    if sample_rate != 16000:
        logger.debug("Resampling STT audio from %d Hz to 16000 Hz", sample_rate)
    return _resample_to_whisper_rate(audio.mean(axis=1), sample_rate)


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
    *,
    language: str | None,
    beam_size: int = 1,
) -> str:
    """Run a single transcription and return the joined text."""
    # A model can be shared by warm job threads; CTranslate2 inference is
    # serialized so concurrent sessions cannot race inside one model.
    with _TRANSCRIBE_LOCK:
        segments, _info = model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            best_of=1,
            # Short, command-style utterances are safer without cross-utterance
            # context, which can otherwise cause repetition and hallucination.
            condition_on_previous_text=False,
            # Silero VAD has already segmented the turn; double filtering is wasteful.
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


def preload_whisper_model(
    model: str,
    device: str = "cpu",
    compute_type: str = "int8",
    model_cache_directory: str | None = None,
    *,
    local_files_only: bool = False,
) -> WhisperModel:
    """Load a Whisper model once per process.

    LiveKit's worker prewarms each job process before the event loop starts.
    ``prewarm`` calls this with ``local_files_only=True`` so cached models are
    ready before the first greeting. If the model is missing, the caller loads
    it in a worker thread instead, keeping downloads off the event loop.
    """
    key = (str(model), device, compute_type, model_cache_directory)
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached

        cache_dir = model_cache_directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        logger.info(
            "Loading Whisper model %r (device=%s, compute_type=%s)",
            model,
            device,
            compute_type,
        )
        loaded = WhisperModel(
            model_size_or_path=str(model),
            device=device,
            compute_type=compute_type,
            download_root=cache_dir,
            local_files_only=local_files_only,
        )
        _MODEL_CACHE[key] = loaded
        logger.info("Whisper model loaded")
        return loaded


class WhisperSTT(stt.STT):
    """A non-streaming LiveKit STT plugin backed by a local Whisper model."""

    def __init__(
        self,
        model: WhisperModelName | str = "small",
        language: str | None = "en",
        device: str = "cpu",
        compute_type: str = "int8",
        model_cache_directory: str | None = None,
        warmup_audio: str | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._opts = WhisperOptions(
            model=str(model),
            language=language,
            device=device,
            compute_type=compute_type,
            model_cache_directory=model_cache_directory,
        )
        self._lock = threading.Lock()
        self._model = self._load_model()

        if warmup_audio:
            self._warmup(warmup_audio)

    def _load_model(self) -> WhisperModel:
        return preload_whisper_model(
            self._opts.model,
            device=self._opts.device,
            compute_type=self._opts.compute_type,
            model_cache_directory=self._opts.model_cache_directory,
        )

    def _warmup(self, path: str) -> None:
        try:
            start = time.perf_counter()
            audio, _sample_rate = sf.read(path, dtype="float32", always_2d=True)
            text = self._transcribe_audio(audio.mean(axis=1), self._opts.language)
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("Whisper warmup finished in %.0f ms: %r", elapsed_ms, text)
        except Exception:
            # Warmup is an optimization; never prevent the agent from starting.
            logger.warning("Whisper warmup failed", exc_info=True)

    def _transcribe_audio(self, audio: np.ndarray, language: str | None) -> str:
        return transcribe(
            self._model,
            audio,
            language=language,
            beam_size=self._opts.beam_size,
        )

    def _recognize_sync(self, buffer: AudioBuffer, language: str | None) -> str:
        wav_bytes = rtc.combine_audio_frames(buffer).to_wav_bytes()
        audio = decode_wav(wav_bytes)
        if audio.size == 0:
            return ""
        return self._transcribe_audio(audio, language or self._opts.language)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        try:
            # Whisper is CPU-bound; keep it off the event loop.
            text = await asyncio.to_thread(self._recognize_sync, buffer, language)
        except Exception as exc:
            logger.error("Whisper transcription failed", exc_info=True)
            raise APIConnectionError(f"Whisper transcription failed: {exc}") from exc

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(
                    text=text,
                    language=language or self._opts.language or "en",
                )
            ],
        )

    def update_options(
        self,
        *,
        model: WhisperModelName | str | None = None,
        language: str | None = None,
        model_cache_directory: str | None = None,
    ) -> None:
        """Update options, reloading the model only when necessary."""
        reinitialize = False
        with self._lock:
            if model is not None:
                self._opts.model = str(model)
                reinitialize = True
            if model_cache_directory is not None:
                self._opts.model_cache_directory = model_cache_directory
                reinitialize = True
            if language is not None:
                self._opts.language = language
            if reinitialize:
                self._model = self._load_model()
