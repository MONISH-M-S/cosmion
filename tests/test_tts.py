"""Test Piper TTS helpers and the chunked-stream adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cosmion import tts as cosmion_tts


class _FakeChunk:
    def __init__(self, audio: np.ndarray, channels: int = 1) -> None:
        self.audio_int16_bytes = audio.astype(np.int16).tobytes()
        self.sample_channels = channels


class _FakeVoice:
    def __init__(self, chunks: list[_FakeChunk], sample_rate: int = 22050) -> None:
        self._chunks = chunks
        self.config = SimpleNamespace(sample_rate=sample_rate)

    def synthesize(self, text: str, syn_config=None):
        return iter(self._chunks)


def test_downmix_chunk_averages_stereo_channels() -> None:
    stereo = np.array([[100, 300], [200, 400]], dtype=np.int16)
    chunk = _FakeChunk(stereo.reshape(-1), channels=2)

    mono = np.frombuffer(cosmion_tts._downmix_chunk(chunk), dtype=np.int16)
    assert mono.tolist() == [200, 300]


def test_downmix_chunk_leaves_mono_untouched() -> None:
    mono = np.array([1, 2, 3], dtype=np.int16)
    chunk = _FakeChunk(mono)
    assert cosmion_tts._downmix_chunk(chunk) == mono.tobytes()


def test_resolve_sample_rate_prefers_voice_config(tmp_path: Path) -> None:
    voice = _FakeVoice([], sample_rate=16000)
    assert cosmion_tts._resolve_sample_rate(voice, tmp_path / "voice.onnx") == 16000


def test_resolve_sample_rate_falls_back_to_sidecar(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not-a-real-model")
    Path(f"{model}.json").write_text(
        json.dumps({"audio": {"sample_rate": 24000}}), encoding="utf-8"
    )
    voice = SimpleNamespace(config=SimpleNamespace(sample_rate=None))

    assert cosmion_tts._resolve_sample_rate(voice, model) == 24000


def test_piper_tts_loads_voice_and_uses_its_sample_rate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")
    fake = _FakeVoice([_FakeChunk(np.zeros(8))], sample_rate=22050)
    monkeypatch.setattr(
        cosmion_tts.PiperVoice, "load", staticmethod(lambda *args, **kwargs: fake)
    )

    plugin = cosmion_tts.PiperTTS(model, speed=0.95)

    assert plugin.sample_rate == 22050
    assert plugin.voice is fake


@pytest.mark.asyncio
async def test_chunked_stream_pushes_every_chunk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")
    chunks = [
        _FakeChunk(np.array([1, 2], dtype=np.int16)),
        _FakeChunk(np.array([3, 4], dtype=np.int16)),
    ]
    monkeypatch.setattr(
        cosmion_tts.PiperVoice,
        "load",
        staticmethod(lambda *args, **kwargs: _FakeVoice(chunks, sample_rate=22050)),
    )
    plugin = cosmion_tts.PiperTTS(model)

    frame = await plugin.synthesize("Hello, sir.").collect()

    expected = b"".join(chunk.audio_int16_bytes for chunk in chunks)
    data = bytes(frame.data)
    assert frame.sample_rate == 22050
    assert frame.num_channels == 1
    assert data[: len(expected)] == expected
    # The emitter pads the final frame with silence to its 200 ms frame size.
    assert not any(data[len(expected) :])


@pytest.mark.asyncio
async def test_chunked_stream_propagates_synthesis_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake")

    class BrokenVoice(_FakeVoice):
        def synthesize(self, text: str, syn_config=None):
            raise RuntimeError("piper exploded")

    monkeypatch.setattr(
        cosmion_tts.PiperVoice,
        "load",
        staticmethod(lambda *args, **kwargs: BrokenVoice([])),
    )
    plugin = cosmion_tts.PiperTTS(model)

    with pytest.raises(RuntimeError, match="piper exploded"):
        await plugin.synthesize("Hello, sir.").collect()
