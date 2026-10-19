"""End-to-end Piper smoke test using the real voice model (opt-in)."""

from __future__ import annotations

import numpy as np
import pytest

from cosmion.config import PROJECT_ROOT
from cosmion.tts import PiperTTS

MODEL_PATH = PROJECT_ROOT / "models" / "en_GB-alan-medium.onnx"


@pytest.mark.slow
@pytest.mark.skipif(not MODEL_PATH.is_file(), reason="Piper voice model not downloaded")
def test_real_voice_produces_audio() -> None:
    plugin = PiperTTS(MODEL_PATH, speed=0.95)
    chunks = list(
        plugin.voice.synthesize(
            "Good evening, sir. All systems are online.", syn_config=plugin._synthesis_config()
        )
    )

    assert chunks
    audio = np.concatenate(
        [np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16) for chunk in chunks]
    )
    assert audio.size > 0
    assert 0 < plugin.sample_rate <= 48000
