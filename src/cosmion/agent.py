"""COSMION's LiveKit worker entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    SessionUsageUpdatedEvent,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.room_io import RoomOptions
from livekit.plugins import openai, silero

from .config import ConfigError, Settings, load_settings
from .stt import WhisperSTT, preload_whisper_model
from .tts import PiperTTS, preload_piper_voice

logger = logging.getLogger("cosmion")

COSMION_INSTRUCTIONS = """You are COSMION, a personal AI assistant modeled after JARVIS from Iron Man.
Address the user as "sir" and speak in a calm, formal, composed British tone.
Keep responses concise and conversational. Everything you say is spoken aloud, so avoid
markdown, bullet lists, code blocks, emoji, and long explanations.
Lead with a short acknowledgment when it fits ("Of course, sir." or "Right away, sir."),
but do not force it into every reply.
If you do not know something, say so plainly and offer the most useful next step."""


class Cosmion(Agent):
    """The agent persona: JARVIS-inspired, concise, and polite."""

    def __init__(self) -> None:
        super().__init__(instructions=COSMION_INSTRUCTIONS)

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Greet the user briefly as COSMION coming online.",
            allow_interruptions=True,
        )


async def _build_session(settings: Settings, vad: silero.VAD) -> AgentSession:
    # Creating the plugins can touch ONNX Runtime or the Hugging Face cache.
    # Build them off the event loop so a cold or slow model never freezes
    # audio handling. Both calls are cache hits when prewarm succeeded.
    stt_plugin, tts_plugin = await asyncio.gather(
        asyncio.to_thread(
            WhisperSTT,
            model=settings.whisper_model,
            language=settings.whisper_language,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        ),
        asyncio.to_thread(
            PiperTTS,
            settings.piper_model_path,
            speed=settings.piper_speed,
            volume=settings.piper_volume,
            noise_scale=settings.piper_noise_scale,
            noise_w=settings.piper_noise_w,
            use_cuda=settings.piper_use_cuda,
        ),
    )

    return AgentSession(
        vad=vad,
        stt=stt_plugin,
        llm=openai.LLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            api_key="ollama",  # required by the client, ignored by Ollama
            temperature=settings.ollama_temperature,
        ),
        tts=tts_plugin,
        turn_handling={
            "endpointing": {"min_delay": 0.5, "max_delay": 5.0},
        },
    )


def _check_ollama(settings: Settings) -> None:
    """Warn early when Ollama or the configured model is unavailable."""
    url = f"{settings.ollama_base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.load(response)
    except Exception as exc:
        logger.warning(
            "Ollama is not reachable at %s: %s", settings.ollama_base_url, exc
        )
        logger.warning(
            "Start Ollama and pull the model with: ollama pull %s", settings.ollama_model
        )
        return

    installed = {
        str(item.get("id", "")) for item in payload.get("data", []) if isinstance(item, dict)
    }
    requested = settings.ollama_model
    if requested not in installed and not any(
        model.startswith(f"{requested}:") for model in installed
    ):
        logger.warning(
            "Ollama model %r is not installed. Run: ollama pull %s", requested, requested
        )


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()

    # Load cached models before the event loop starts so the first greeting is
    # not delayed. Whisper is only preloaded from the local cache: a first-run
    # download can take minutes and would exceed JobProcess's init timeout.
    # When it is not cached, entrypoint loads it off-loop instead.
    settings = load_settings()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="cosmion-prewarm") as pool:
        whisper = pool.submit(
            preload_whisper_model,
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            local_files_only=True,
        )
        piper = pool.submit(
            preload_piper_voice,
            settings.piper_model_path,
            use_cuda=settings.piper_use_cuda,
        )

        try:
            whisper.result()
        except Exception as exc:
            logger.info(
                "Whisper model is not cached yet; it will download on the first "
                "conversation (%s).",
                exc,
            )
        piper.result()


async def entrypoint(ctx: JobContext) -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings.validate_voice_files()

    # A 2-second check keeps startup fast while still surfacing common mistakes.
    await asyncio.to_thread(_check_ollama, settings)

    logger.info("Connecting to room %s", ctx.room.name)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    try:
        participant = await ctx.wait_for_participant()
    except RuntimeError as exc:
        # The user closed the page or denied microphone access before joining.
        logger.info("Participant never joined; ending the job (%s)", exc)
        return
    logger.info("COSMION online for participant %s", participant.identity)

    latest_usage: list[object] = []

    def on_usage_updated(event: SessionUsageUpdatedEvent) -> None:
        # This event fires on every metric update; keep the latest snapshot and
        # log it once when the session ends instead of repeating it each turn.
        latest_usage[:] = event.usage.model_usage

    session = await _build_session(settings, ctx.proc.userdata["vad"])
    session.on("session_usage_updated", on_usage_updated)

    async def _on_shutdown() -> None:
        if latest_usage:
            logger.info("Session usage: %s", latest_usage)
        logger.info("COSMION shutting down")

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        room=ctx.room,
        agent=Cosmion(),
        room_options=RoomOptions(),
    )


def main() -> None:
    try:
        load_settings()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            # Loading two local ONNX models can take longer than the 10s
            # default on cold disks; give the first job room to initialize.
            initialize_process_timeout=30.0,
        )
    )


if __name__ == "__main__":
    main()
