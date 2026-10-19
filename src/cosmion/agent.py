"""COSMION's LiveKit worker entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import urllib.request

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
)
from livekit.plugins import openai, silero

from .config import ConfigError, Settings, load_settings
from .stt import WhisperSTT
from .tts import PiperTTS

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


def _build_session(settings: Settings, vad: silero.VAD) -> AgentSession:
    return AgentSession(
        vad=vad,
        stt=WhisperSTT(
            model=settings.whisper_model,
            language=settings.whisper_language,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
        ),
        llm=openai.LLM(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            api_key="ollama",  # required by the client, ignored by Ollama
            temperature=settings.ollama_temperature,
        ),
        tts=PiperTTS(
            settings.piper_model_path,
            speed=settings.piper_speed,
            volume=settings.piper_volume,
        ),
        min_endpointing_delay=0.5,
        max_endpointing_delay=5.0,
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
    participant = await ctx.wait_for_participant()
    logger.info("COSMION online for participant %s", participant.identity)

    usage_collector = metrics.UsageCollector()

    def on_metrics_collected(agent_metrics: metrics.AgentMetrics) -> None:
        metrics.log_metrics(agent_metrics)
        usage_collector.collect(agent_metrics)

    session = _build_session(settings, ctx.proc.userdata["vad"])
    session.on("metrics_collected", on_metrics_collected)

    async def _on_shutdown() -> None:
        logger.info("COSMION shutting down")

    ctx.add_shutdown_callback(_on_shutdown)

    await session.start(
        room=ctx.room,
        agent=Cosmion(),
        room_input_options=RoomInputOptions(),
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
        )
    )


if __name__ == "__main__":
    main()
