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


def decode_wav(wav_bytes: bytes) -> np.ndarray:
    """Decode WAV bytes into mono float32 PCM.

    ``AudioFrame.to_wav_bytes()`` returns a complete WAV file, including its
    RIFF header. Treating those bytes as raw PCM feeds the header into Whisper
    as audio, so we decode the file properly with soundfile instead.
    """
    if not wav_bytes:
        raise ValueError("Received an empty WAV payload from LiveKit.")

    audio, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
    if sample_rate != 16000:
        logger.warning("Expected 16 kHz audio from LiveKit, received %d Hz", sample_rate)
    return audio.mean(axis=1)


def transcribe(
    model: WhisperModel,
    audio: np.ndarray,
    *,
    language: str | None,
    beam_size: int = 1,
) -> str:
    """Run a single transcription and return the joined text."""
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
        cache_dir = self._opts.model_cache_directory
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        logger.info(
            "Loading Whisper model %r (device=%s, compute_type=%s)",
            self._opts.model,
            self._opts.device,
            self._opts.compute_type,
        )
        model = WhisperModel(
            model_size_or_path=self._opts.model,
            device=self._opts.device,
            compute_type=self._opts.compute_type,
            download_root=cache_dir,
        )
        logger.info("Whisper model loaded")
        return model

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
