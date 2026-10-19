# COSMION

**A JARVIS-style voice assistant that hears, thinks, and speaks on your machine.**

COSMION is a local-first voice agent built on
[LiveKit Agents](https://github.com/livekit/agents). It listens with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper), reasons with a local
[Ollama](https://ollama.com/) model, and answers through a British neural
[Piper](https://github.com/OHF-Voice/piper1-gpl) voice. LiveKit Cloud is used only
for WebRTC signaling and encrypted media relay — no audio, transcript, or prompt is
sent to a cloud AI service.

```
Microphone ──WebRTC──▶ LiveKit room ──▶ COSMION worker (your machine)
                                          │
                                          ├─ Silero VAD        (local)
                                          ├─ faster-whisper    (local)
                                          ├─ Ollama LLM        (local)
                                          └─ Piper TTS         (local)
                                                     │
Speaker ◀──WebRTC── LiveKit room ◀───────────────────┘
```

## Features

- **Fully local AI pipeline** — STT, LLM, TTS, and VAD all run on your hardware.
- **JARVIS persona** — addresses you as "sir", speaks calmly, and keeps replies short.
- **British neural voice** — `en_GB-alan-medium` by default, tuned slightly slower.
- **Typed configuration** — one `.env.local`, validated at startup with clear errors.
- **Preflight + smoke tests** — verify Ollama, the voice model, and audio output before a call.
- **No secrets in git** — `.env.local`, voice models, and caches are ignored.

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10–3.13 (3.13 recommended) |
| [Ollama](https://ollama.com/download) | Local LLM runtime; default model `llama3.2` (~2 GB) |
| [LiveKit Cloud](https://cloud.livekit.io/) | Free tier is enough, or self-host LiveKit |
| Disk | ~1 GB for dependencies, ~60 MB for the voice model |
| Hardware | CPU is fine; a CUDA GPU speeds up Whisper |

## Quickstart (Windows / PowerShell)

```powershell
cd C:\C.O.S.M.I.O.N\cosmion_livekit

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"

Copy-Item .env.example .env.local
notepad .env.local          # paste your LiveKit URL, key, and secret

python scripts/download_voice.py
ollama pull llama3.2

python scripts/preflight.py
python scripts/smoke_tts.py
python -m cosmion dev
```

Then open the [LiveKit Agents Playground](https://agents-playground.livekit.io/),
enter the same LiveKit URL/key/secret, and connect. Allow microphone access and say
"Hello, COSMION."

### Quickstart (macOS / Linux)

```bash
cd cosmion_livekit
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"

cp .env.example .env.local
$EDITOR .env.local

python scripts/download_voice.py
ollama pull llama3.2

python scripts/preflight.py
python scripts/smoke_tts.py
python -m cosmion dev
```

## Configuration

Copy `.env.example` to `.env.local` and adjust as needed. `.env.local` is ignored by git.

| Variable | Default | Purpose |
|---|---|---|
| `LIVEKIT_URL` | — | LiveKit WebSocket URL (`wss://…`) |
| `LIVEKIT_API_KEY` | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | — | LiveKit API secret |
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama's OpenAI-compatible endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model name passed to Ollama |
| `OLLAMA_TEMPERATURE` | `0.5` | Response creativity (0.0–2.0) |
| `WHISPER_MODEL` | `small` | `tiny`, `base`, `small`, `medium`, `large-v3`, … |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` on CPU, `float16` on CUDA |
| `WHISPER_LANGUAGE` | `en` | Leave empty to auto-detect |
| `PIPER_MODEL_PATH` | `models/en_GB-alan-medium.onnx` | Voice model path |
| `PIPER_SPEED` | `0.95` | 1.0 = natural, lower = slower |
| `PIPER_VOLUME` | `1.0` | 0.0–1.0 |
| `PIPER_NOISE_SCALE` | `0.667` | Voice variation; lower is steadier |
| `PIPER_NOISE_W` | `0.8` | Phoneme-duration variation |
| `PIPER_USE_CUDA` | `false` | GPU inference (`onnxruntime-gpu` required) |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |

## The JARVIS voice

The default is **`en_GB-alan-medium`**: a calm British male voice. `PIPER_SPEED=0.95`
gives it a slightly measured, deliberate cadence.

Ways to tune the character:

```dotenv
# Slower and more deliberate (try 0.90–0.95)
PIPER_SPEED=0.93

# A little quieter
PIPER_VOLUME=0.9
```

To try another single-speaker British voice:

```powershell
python scripts/download_voice.py --list
python scripts/download_voice.py --voice en_GB-northern_english_male-medium
# then set PIPER_MODEL_PATH=models/en_GB-northern_english_male-medium.onnx
```

Piper is a compact neural TTS, not a voice-cloning system. It will not reproduce
Paul Bettany exactly. If you want a closer match later, the `tts.TTS` interface makes
it straightforward to swap in another local engine or a cloud voice plugin.

## Commands

| Command | What it does |
|---|---|
| `python scripts/preflight.py` | Checks Python, packages, env vars, voice files, and Ollama |
| `python scripts/download_voice.py` | Downloads the default Piper voice |
| `python scripts/smoke_tts.py "Hello, sir."` | Renders speech to `artifacts/cosmion-smoke.wav` |
| `python -m cosmion dev` | Starts the worker with auto-reload |
| `python -m cosmion start` | Starts the worker for production |
| `pytest -m "not slow"` | Runs unit tests |
| `pytest -m slow` | Runs the real-voice end-to-end test |
| `ruff check .` | Lints the project |

`make` targets are provided for Unix users (`make setup`, `make test`, `make lint`).

## Project layout

```
cosmion_livekit/
├── src/cosmion/
│   ├── agent.py        # LiveKit worker, persona, session wiring
│   ├── config.py       # Typed env configuration
│   ├── stt.py          # faster-whisper plugin
│   ├── tts.py          # Piper plugin
│   └── __main__.py     # `python -m cosmion`
├── scripts/
│   ├── download_voice.py
│   ├── preflight.py
│   └── smoke_tts.py
├── tests/
├── models/             # Voice files (gitignored)
└── .env.local          # Secrets (gitignored)
```

## Security

- **Never commit `.env.local`.** It contains your LiveKit API secret. The included
  `.gitignore` excludes it, along with voice models and caches.
- If a secret has ever been shared or committed, **rotate it** in the LiveKit
  dashboard before using this project.
- LiveKit Cloud relays encrypted WebRTC media. All AI inference stays local. For a
  fully local network path, self-host LiveKit as well.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Configuration error: LIVEKIT_URL is missing` | Copy `.env.example` to `.env.local` and fill it in |
| `Piper voice files not found` | Run `python scripts/download_voice.py` |
| `Ollama is not reachable` | Start Ollama, then run `ollama pull llama3.2` |
| `Ollama model 'llama3.2' is not installed` | Run `ollama pull llama3.2` |
| LiveKit connects but no audio | Check microphone permissions in the playground/browser |
| Whisper is slow | Use `WHISPER_MODEL=base` or a CUDA GPU |
| `No module named 'cosmion'` | Activate the venv and run `pip install -e .` |
| Python 3.14 install errors | Use Python 3.13; some native wheels lag behind |

## Credits

- [LiveKit Agents](https://github.com/livekit/agents) — real-time agent framework
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — Whisper inference
- [Piper](https://github.com/OHF-Voice/piper1-gpl) — neural TTS
- [Ollama](https://ollama.com/) — local LLM runtime
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- STT plugin originally inspired by [taresh18/livekit-whisper](https://github.com/taresh18/livekit-whisper)
- Piper plugin originally inspired by [nay-cat/LiveKit-PiperTTS-Plugin](https://github.com/nay-cat/LiveKit-PiperTTS-Plugin)

## License

MIT — see [LICENSE](LICENSE).
