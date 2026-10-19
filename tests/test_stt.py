"""Test WAV decoding and Whisper transcription plumbing."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

from cosmion.stt import decode_wav, transcribe


def _wav_bytes(seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    samples = int(seconds * sample_rate)
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    tone = 0.1 * np.sin(2 * np.pi * 440 * timeline)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_decode_wav_returns_exactly_the_audio_samples() -> None:
    decoded = decode_wav(_wav_bytes())
    assert decoded.dtype == np.float32
    # The old implementation fed the 44-byte RIFF header to Whisper as samples.
    assert decoded.shape == (16000,)
    assert np.max(np.abs(decoded)) < 0.2


def test_decode_wav_resamples_to_16khz_for_whisper() -> None:
    decoded = decode_wav(_wav_bytes(seconds=1.0, sample_rate=24000))
    assert decoded.dtype == np.float32
    assert decoded.shape == (16000,)


def test_decode_wav_downmixes_stereo() -> None:
    left = np.full(1600, 0.5, dtype=np.float32)
    right = np.full(1600, -0.5, dtype=np.float32)
    stereo = np.stack([left, right], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, stereo, 16000, format="WAV", subtype="FLOAT")

    decoded = decode_wav(buffer.getvalue())
    assert decoded.ndim == 1
    assert np.allclose(decoded, 0.0, atol=1e-6)


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter([_FakeSegment(" Hello, "), _FakeSegment("sir. ")]), None


def test_transcribe_joins_segments_and_disables_history() -> None:
    model = _FakeModel()
    audio = np.zeros(1600, dtype=np.float32)

    text = transcribe(model, audio, language="en", beam_size=1)

    assert text == "Hello, sir."
    assert model.kwargs is not None
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["language"] == "en"


def test_transcribe_supports_language_auto_detection() -> None:
    model = _FakeModel()
    transcribe(model, np.zeros(1600, dtype=np.float32), language=None)
    assert model.kwargs is not None
    assert model.kwargs["language"] is None
