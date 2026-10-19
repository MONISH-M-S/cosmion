"""Manual end-to-end check against a real LiveKit room.

This script joins a temporary room as a synthetic participant, speaks a
question with the local Piper voice, records COSMION's spoken reply, and
transcribes the recording with Whisper so the result can be checked from the
terminal.

The agent worker must already be running in another terminal::

    python -m cosmion dev

Then run::

    python scripts/live_e2e.py

The check is intentionally not part of the automated test suite because it
needs LiveKit credentials, network access, Ollama, and a running worker.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from livekit import api, rtc
from piper import PiperVoice, SynthesisConfig

from cosmion.config import PROJECT_ROOT, Settings, load_settings

CAPTURE_RATE = 48_000
AGENT_JOIN_TIMEOUT = 25.0
GREETING_SETTLE_SECONDS = 8.0
REPLY_TIMEOUT = 30.0


def _mono_pcm16(audio: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return audio
    usable = audio.size - (audio.size % channels)
    return audio[:usable].reshape(-1, channels).mean(axis=1).astype(np.int16)


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.int16, copy=False)
    duration = audio.size / source_rate
    target_size = max(1, int(round(duration * target_rate)))
    source_positions = np.arange(audio.size, dtype=np.float64)
    target_positions = np.linspace(0.0, audio.size - 1.0, target_size)
    return np.interp(target_positions, source_positions, audio).astype(np.int16)


def synthesize_question(settings: Settings, text: str) -> np.ndarray:
    """Render the test question with COSMION's own Piper voice."""
    voice = PiperVoice.load(str(settings.piper_model_path))
    config = SynthesisConfig(
        volume=settings.piper_volume,
        length_scale=1.0 / settings.piper_speed,
        noise_scale=settings.piper_noise_scale,
        noise_w_scale=settings.piper_noise_w,
        normalize_audio=True,
    )
    chunks: list[np.ndarray] = []
    for chunk in voice.synthesize(text, syn_config=config):
        samples = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)
        chunks.append(_mono_pcm16(samples, int(getattr(chunk, "sample_channels", 1) or 1)))
    if not chunks:
        raise RuntimeError("Piper produced no audio for the test question.")

    audio = np.concatenate(chunks)
    source_rate = int(voice.config.sample_rate)
    return _resample(audio, source_rate, CAPTURE_RATE)


def _padded(audio: np.ndarray, seconds: float) -> np.ndarray:
    silence = np.zeros(int(CAPTURE_RATE * seconds), dtype=np.int16)
    return np.concatenate([silence, audio, silence])


async def _publish_question(room: rtc.Room, audio: np.ndarray) -> None:
    source = rtc.AudioSource(CAPTURE_RATE, 1)
    track = rtc.LocalAudioTrack.create_audio_track("question", source)
    options = rtc.TrackPublishOptions()
    options.source = rtc.TrackSource.SOURCE_MICROPHONE
    await room.local_participant.publish_track(track, options)

    await asyncio.sleep(0.4)
    frame_samples = CAPTURE_RATE // 100  # 10 ms
    for offset in range(0, audio.size, frame_samples):
        chunk = audio[offset : offset + frame_samples]
        if chunk.size < frame_samples:
            chunk = np.pad(chunk, (0, frame_samples - chunk.size))
        frame = rtc.AudioFrame(chunk.tobytes(), CAPTURE_RATE, 1, chunk.size)
        await source.capture_frame(frame)
        await asyncio.sleep(chunk.size / CAPTURE_RATE)

    # Keep the track alive briefly so the final phoneme is delivered.
    await asyncio.sleep(0.5)


async def _wait_for_agent(room: rtc.Room, identity: str) -> bool:
    deadline = time.monotonic() + AGENT_JOIN_TIMEOUT
    while time.monotonic() < deadline:
        if any(pid != identity for pid in room.remote_participants):
            return True
        await asyncio.sleep(0.2)
    return False


async def run(settings: Settings, question: str, expected: str, output: Path) -> bool:
    room_name = f"cosmion-e2e-{int(time.time())}"
    identity = "cosmion-e2e-tester"
    token = (
        api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(identity)
        .with_name("COSMION E2E Tester")
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

    api_url = settings.livekit_url.replace("wss://", "https://", 1).replace(
        "ws://", "http://", 1
    )
    api_client = api.LiveKitAPI(
        api_url, settings.livekit_api_key, settings.livekit_api_secret
    )
    room = rtc.Room()
    captured: list[bytes] = []
    recorders: list[asyncio.Task[None]] = []

    def start_recorder(track: rtc.Track) -> None:
        async def record() -> None:
            stream = rtc.AudioStream(track, sample_rate=CAPTURE_RATE, num_channels=1)
            try:
                async for event in stream:
                    captured.append(event.frame.data.tobytes())
            finally:
                await stream.aclose()

        recorders.append(asyncio.create_task(record()))

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        _publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if participant.identity != identity and track.kind == rtc.TrackKind.KIND_AUDIO:
            print(f"Recording agent audio from {participant.identity}")
            start_recorder(track)

    try:
        await api_client.room.create_room(
            api.CreateRoomRequest(name=room_name, empty_timeout=120)
        )
        print(f"Joining {room_name}")
        await room.connect(settings.livekit_url, token)
        print("Waiting for COSMION to join...")
        if not await _wait_for_agent(room, identity):
            print("ERROR: COSMION did not join the room.", file=sys.stderr)
            return False

        print("COSMION joined; waiting for the greeting to finish...")
        await asyncio.sleep(GREETING_SETTLE_SECONDS)

        question_audio = _padded(synthesize_question(settings, question), 0.5)
        print(f"Asking: {question!r}")
        await _publish_question(room, question_audio)

        print("Waiting for the spoken reply...")
        await asyncio.sleep(REPLY_TIMEOUT)
    finally:
        for task in recorders:
            task.cancel()
        if recorders:
            await asyncio.gather(*recorders, return_exceptions=True)
        await room.disconnect()
        try:
            await api_client.room.delete_room(api.DeleteRoomRequest(room=room_name))
        except Exception as exc:  # cleanup best effort
            print(f"Room cleanup warning: {exc}", file=sys.stderr)
        await api_client.aclose()

    if not captured:
        print("ERROR: no agent audio was captured.", file=sys.stderr)
        return False

    pcm = np.frombuffer(b"".join(captured), dtype=np.int16)
    peak = int(np.abs(pcm).max()) if pcm.size else 0
    if peak < 100:
        print("ERROR: the captured agent audio is silent.", file=sys.stderr)
        return False

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, pcm, CAPTURE_RATE, subtype="PCM_16")
    duration = pcm.size / CAPTURE_RATE
    print(f"Saved {duration:.1f}s of agent audio to {output} (peak={peak})")

    print("Transcribing the captured reply...")
    model = WhisperModel(
        settings.whisper_model,
        device=settings.whisper_device,
        compute_type=settings.whisper_compute_type,
    )
    # Pass decoded float32 audio rather than a filename: this avoids a PyAV
    # version incompatibility in faster-whisper's file decoder and matches the
    # decode-with-soundfile path used by the live STT plugin.
    recorded, sample_rate = sf.read(output, dtype="float32", always_2d=True)
    mono = recorded.mean(axis=1)
    if sample_rate != 16_000:
        target_size = max(1, int(round(mono.size * 16_000 / sample_rate)))
        mono = np.interp(
            np.linspace(0.0, mono.size - 1.0, target_size),
            np.arange(mono.size, dtype=np.float64),
            mono,
        ).astype(np.float32)
    segments, _info = model.transcribe(
        mono, language=settings.whisper_language, beam_size=5, vad_filter=True
    )
    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    print(f"Transcript: {transcript!r}")

    if expected.lower() not in transcript.lower():
        print(f"ERROR: the reply did not contain {expected!r}.", file=sys.stderr)
        return False
    print("PASS: COSMION heard the question and answered correctly.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="COSMION, what is the capital of France?",
        help="Question to speak into the room.",
    )
    parser.add_argument(
        "--expect",
        default="paris",
        help="Word or phrase the transcribed reply must contain.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "live-e2e.wav",
        help="Where to save the captured agent audio.",
    )
    args = parser.parse_args()

    settings = load_settings()
    settings.validate_voice_files()
    ok = asyncio.run(run(settings, args.question, args.expect, args.output))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
