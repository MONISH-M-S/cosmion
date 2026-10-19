"""One-click local launcher and browser UI for COSMION.

Double-click ``Start COSMION.bat`` (Windows) or run this script to start both
the LiveKit worker and a small local web app. The app generates a short-lived
room token in-process, so the browser never needs your LiveKit API secret.

The server binds to ``127.0.0.1`` only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import webbrowser
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from livekit import api  # noqa: E402

from cosmion.config import ConfigError, Settings, load_settings  # noqa: E402

HOST = "127.0.0.1"
DEFAULT_PORT = 8765
WEB_ROOT = PROJECT_ROOT / "web"
ARTIFACTS = PROJECT_ROOT / "artifacts"
WORKER_LOG = ARTIFACTS / "one-click-worker.log"
READY_MARKER = "registered worker"
READY_TIMEOUT = 45.0
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
KEY_EVENTS = (
    "registered worker",
    "COSMION online",
    "received job request",
    "ERROR",
    "Traceback",
    "shutting down",
)


class _NoopJob:
    """Keep the worker cleanup API identical on non-Windows platforms."""

    def assign(self, process: subprocess.Popen[str]) -> None:
        return None


if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
    _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9

    class _IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class _JobBasicLimitInformation(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _JobExtendedLimitInformation(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _JobBasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    class _WindowsJob:
        """Terminate the worker when this launcher process exits.

        The launcher owns a job object configured with
        ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Closing the console window or
        killing the launcher therefore cannot leave an orphaned worker behind.
        """

        def __init__(self) -> None:
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
            self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
            self._kernel32.SetInformationJobObject.argtypes = [
                wintypes.HANDLE,
                ctypes.c_int,
                ctypes.c_void_p,
                wintypes.DWORD,
            ]
            self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
            self._kernel32.AssignProcessToJobObject.argtypes = [
                wintypes.HANDLE,
                wintypes.HANDLE,
            ]
            self._kernel32.CloseHandle.restype = wintypes.BOOL
            self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

            self._handle = self._kernel32.CreateJobObjectW(None, None)
            if not self._handle:
                return

            info = _JobExtendedLimitInformation()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            configured = self._kernel32.SetInformationJobObject(
                self._handle,
                _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            if not configured:
                self._kernel32.CloseHandle(self._handle)
                self._handle = None

        def assign(self, process: subprocess.Popen[str]) -> None:
            if not self._handle:
                return
            try:
                handle = int(process._handle)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                return
            self._kernel32.AssignProcessToJobObject(self._handle, handle)

    _WorkerJob = _WindowsJob
else:
    _WorkerJob = _NoopJob

_WORKER_JOB: _WorkerJob | None = None


class CosmionHTTPServer(ThreadingHTTPServer):
    """HTTP server that carries the loaded settings for token generation."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        settings: Settings,
    ) -> None:
        super().__init__(server_address, handler)
        self.settings = settings


class CosmionRequestHandler(BaseHTTPRequestHandler):
    """Serve the local UI and mint room tokens on demand."""

    server: CosmionHTTPServer
    server_version = "CosmionLocal/1.0"

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        path = urllib.parse.urlsplit(self.path).path
        if path == "/":
            self._serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif path == "/app.js":
            self._serve_file(WEB_ROOT / "app.js", "text/javascript; charset=utf-8")
        elif path == "/style.css":
            self._serve_file(WEB_ROOT / "style.css", "text/css; charset=utf-8")
        elif path == "/vendor/livekit-client.umd.js":
            self._serve_file(
                WEB_ROOT / "vendor" / "livekit-client.umd.js",
                "text/javascript; charset=utf-8",
            )
        elif path == "/cosmion.ico":
            self._serve_file(WEB_ROOT / "cosmion.ico", "image/x-icon")
        elif path == "/health":
            self._send_json({"status": "ok", "app": "cosmion"})
        elif path == "/token":
            if self.headers.get("X-Cosmion") != "1":
                self.send_error(403, "Missing COSMION request header")
                return
            self._send_token()
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        path = urllib.parse.urlsplit(self.path).path
        if path != "/shutdown":
            self.send_error(404)
            return
        if self.headers.get("X-Cosmion") != "1":
            self.send_error(403, "Missing COSMION request header")
            return

        self._send_json({"status": "stopping"})
        threading.Thread(target=self.server.shutdown, daemon=True).start()

    def log_message(self, format: str, *args: object) -> None:
        """Keep the launcher console focused on worker events."""
        return

    def _serve_file(self, path: Path, content_type: str) -> None:
        try:
            payload = path.read_bytes()
        except OSError:
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "img-src 'self' data:; media-src 'self' blob:; "
            "connect-src 'self' wss: https:;",
        )
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_token(self) -> None:
        settings = self.server.settings
        room = f"cosmion-{uuid.uuid4().hex[:10]}"
        identity = f"cosmion-user-{uuid.uuid4().hex[:6]}"
        token = (
            api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(identity)
            .with_name("COSMION User")
            .with_ttl(timedelta(hours=1))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
            .to_jwt()
        )
        self._send_json(
            {
                "url": settings.livekit_url,
                "token": token,
                "room": room,
            }
        )


def _worker_command() -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "cosmion",
        "dev",
        "--no-reload",
        "--log-level",
        "info",
    ]


def _start_worker() -> tuple[subprocess.Popen[str], threading.Event]:
    global _WORKER_JOB

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    log_file = WORKER_LOG.open("w", encoding="utf-8", errors="replace")
    worker = subprocess.Popen(
        _worker_command(),
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    ready = threading.Event()

    if _WORKER_JOB is None:
        _WORKER_JOB = _WorkerJob()
    _WORKER_JOB.assign(worker)

    def pump() -> None:
        try:
            assert worker.stdout is not None
            for raw in worker.stdout:
                log_file.write(raw)
                log_file.flush()
                line = ANSI_RE.sub("", raw.rstrip())
                if READY_MARKER in line:
                    ready.set()
                if any(marker in line for marker in KEY_EVENTS):
                    print(f"  {line}", flush=True)
        finally:
            log_file.close()
            ready.set()

    threading.Thread(target=pump, daemon=True, name="cosmion-worker-log").start()
    return worker, ready


def _stop_worker(worker: subprocess.Popen[str]) -> None:
    if worker.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(worker.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        return
    worker.terminate()
    try:
        worker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        worker.kill()


def _existing_instance(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=1.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return payload.get("app") == "cosmion"


def _open_browser(url: str, disabled: bool) -> None:
    if disabled:
        print(f"  Open this address in your browser: {url}")
        return
    try:
        webbrowser.open(url)
    except Exception:
        print(f"  Could not open a browser automatically. Open: {url}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the local app without opening a browser window.",
    )
    args = parser.parse_args()

    try:
        settings = load_settings()
        settings.validate_voice_files()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    print()
    print("  COSMION — one-click launcher")
    print("  -----------------------------")
    print()

    try:
        server = CosmionHTTPServer((HOST, args.port), CosmionRequestHandler, settings)
    except OSError:
        if _existing_instance(args.port):
            url = f"http://{HOST}:{args.port}/"
            print(f"  COSMION is already running. Opening {url}")
            _open_browser(url, args.no_browser)
            return 0
        print(
            f"  Port {args.port} is already in use. Close the other app or pass --port.",
            file=sys.stderr,
        )
        return 1

    server_thread = threading.Thread(
        target=server.serve_forever,
        name="cosmion-http",
        daemon=True,
    )
    server_thread.start()

    worker, ready = _start_worker()
    print("  Starting the local worker...")
    if not ready.wait(READY_TIMEOUT):
        print(
            "  The worker is taking longer than expected. Opening the page anyway; "
            f"see {WORKER_LOG}",
            file=sys.stderr,
        )
    elif worker.poll() is not None:
        print(
            f"  The worker exited during startup. See {WORKER_LOG}",
            file=sys.stderr,
        )
        server.shutdown()
        server.server_close()
        return 1
    else:
        print("  COSMION worker is ready.")

    url = f"http://{HOST}:{args.port}/"
    print(f"  Opening {url}")
    print("  Use Quit COSMION in the page, or press Ctrl+C here, to stop.")
    print()
    _open_browser(url, args.no_browser)

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  Stopping COSMION...")
    finally:
        server.shutdown()
        server.server_close()
        _stop_worker(worker)

    print("  COSMION stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
