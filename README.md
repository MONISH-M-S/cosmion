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
- **JARVIS persona** — calm, concise, occasionally addresses you as "sir", and names
  **Monish M S** as its creator.
- **British neural voice** — `en_GB-alan-medium` by default, tuned slightly slower.
- **Typed configuration** — one `.env.local`, validated at startup with clear errors.
- **Preflight + smoke tests** — verify Ollama, the voice model, and audio output before a call.
- **Warm first turn** — cached models load during prewarm; cold loads run in worker threads.
- **One-click launcher** — `Start COSMION.bat` starts the worker and opens a local
  page with a built-in **Start Conversation** button.
- **No secrets in git** — `.env.local`, voice models, and caches are ignored.

## Requirements

| Requirement | Notes |
|---|---|
| Python | 3.10–3.13 (3.13 recommended) |
| [Ollama](https://ollama.com/download) | Local LLM runtime; default model `llama3.2` (~2 GB) |
| [LiveKit Cloud](https://cloud.livekit.io/) | Free tier is enough, or self-host LiveKit |
| Disk | ~1 GB dependencies, ~60 MB voice model, ~2 GB Ollama model, ~500 MB Whisper model |
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
```

That is the whole setup. From then on, start COSMION by double-clicking
**`Start COSMION.bat`** (or a desktop shortcut to it). The launcher starts the
worker and opens **http://127.0.0.1:8765/** in your browser. Click
**Start Conversation**, allow microphone access once, and talk.

If you prefer the generic LiveKit playground, start the worker with
`python -m cosmion dev` and open the
[LiveKit Agents Playground](https://agents-playground.livekit.io/) instead. Paste
the same LiveKit URL/key/secret and connect.

On first run, faster-whisper downloads the `small` model (~500 MB) once. It is
cached in `%USERPROFILE%\.cache\huggingface` on Windows or `~/.cache/huggingface`
elsewhere. Piper is preloaded during the worker's prewarm phase, and Whisper is
prewarmed on later runs once cached. On the very first run, the download and ONNX
initialization happen in a worker thread instead of blocking the conversation loop.

> **Using someone else's copy?** The repository ships no credentials. Create your own
> free [LiveKit Cloud](https://cloud.livekit.io/) project, copy its WebSocket URL,
> API key, and API secret into `.env.local`, then follow the same steps. Speech
> recognition, the language model, and the voice all run on your own machine, so
> there are no per-minute AI API bills.

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
python scripts/one_click.py
```

On macOS and Linux the launcher opens the same local page at
`http://127.0.0.1:8765/`. To run only the worker instead, use
`python -m cosmion dev` and connect through the LiveKit playground.

## One-click launcher

`Start COSMION.bat` (Windows) and `python scripts/one_click.py` (any platform) do
three things:

1. Start the COSMION worker as a child process, logging to
   `artifacts/one-click-worker.log`.
2. Serve a small local web app on **127.0.0.1 only**. The page asks the launcher
   for a short-lived room token; your LiveKit API secret never reaches the browser.
3. Open the page in your default browser.

The page has one primary action: **Start Conversation**. It connects to a fresh
room, enables your microphone, and plays COSMION's reply. Allow microphone access the
first time; after that it is a single click. Use **Mute**, **End**, or
**Quit COSMION** to control the session.

Each conversation uses a new random room, so you can start again without restarting
the launcher. Starting the launcher a second time just reopens the existing page
instead of spawning a duplicate worker. On Windows, the worker is tied to a job
object, so it is cleaned up even if the launcher window is closed abruptly.

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
| `python scripts/live_e2e.py` | Speaks a test question into a real room, records the reply, and transcribes it |
| `python scripts/one_click.py` | Starts the worker and the local web UI, then opens the browser |
| `Start COSMION.bat` | Windows double-click launcher for `one_click.py` |
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
│   ├── live_e2e.py
│   ├── one_click.py       # Local launcher + token server
│   ├── preflight.py
│   └── smoke_tts.py
├── web/                   # One-click UI (served from 127.0.0.1)
│   ├── app.js
│   ├── cosmion.ico
│   ├── index.html
│   ├── style.css
│   └── vendor/            # Pinned LiveKit JS client (Apache-2.0)
├── tests/
├── Start COSMION.bat      # Windows double-click launcher
├── models/                # Voice files (gitignored)
└── .env.local             # Secrets (gitignored)
```

## Security

- **Never commit `.env.local`.** It contains your LiveKit API secret. The included
  `.gitignore` excludes it, along with voice models and caches.
- If a secret has ever been shared or committed, **rotate it** in the LiveKit
  dashboard before using this project.
- LiveKit Cloud relays encrypted WebRTC media. All AI inference stays local. For a
  fully local network path, self-host LiveKit as well.
- The one-click launcher binds to `127.0.0.1` only and mints room tokens in-process.
  Your LiveKit API secret is never sent to the browser; only a short-lived,
  room-scoped token is. The token endpoint requires a same-origin request header so
  other local pages cannot silently mint tokens.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Configuration error: LIVEKIT_URL is missing` | Copy `.env.example` to `.env.local` and fill it in |
| `Piper voice files not found` | Run `python scripts/download_voice.py` |
| `Ollama is not reachable` | Start Ollama, then run `ollama pull llama3.2` |
| `Ollama model 'llama3.2' is not installed` | Run `ollama pull llama3.2` |
| LiveKit connects but no audio | Check microphone permissions in the playground/browser |
| Local page says "Waiting for COSMION" | The worker is still starting or crashed; check `artifacts/one-click-worker.log` |
| "Port 8765 is already in use" | Another app owns the port; pass `--port 8766` to `one_click.py` |
| One-click page does not open | Open `http://127.0.0.1:8765/` manually or run `one_click.py --no-browser` |
| `live_e2e.py` waits forever for COSMION | Start `python -m cosmion dev` in another terminal first |
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
- [LiveKit JavaScript client](https://github.com/livekit/client-sdk-js) — vendored in
  `web/vendor/` under Apache-2.0 (see `web/vendor/LIVEKIT_LICENSE`)

## License

MIT — see [LICENSE](LICENSE).
