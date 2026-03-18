"""Server lifecycle management — launchd on macOS, systemd on Linux."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx

from ormah.config import settings

LAUNCHD_LABEL = "com.ormah.server"
PLIST_DIR = Path.home() / "Library" / "LaunchAgents"
PLIST_PATH = PLIST_DIR / f"{LAUNCHD_LABEL}.plist"
LOG_DIR = Path.home() / ".local" / "share" / "ormah" / "logs"

SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
SYSTEMD_UNIT = SYSTEMD_DIR / "ormah.service"

SYSTEMD_TEMPLATE = """\
[Unit]
Description=Ormah memory server
After=network.target

[Service]
ExecStart={wrapper_path}
Environment="PATH={bin_dir}:/usr/local/bin:/usr/bin:/bin"
StandardOutput=append:{log_dir}/ormah.out.log
StandardError=append:{log_dir}/ormah.err.log
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""

PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{wrapper_path}</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>{bin_dir}:/usr/local/bin:/usr/bin:/bin</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>{log_dir}/ormah.out.log</string>
  <key>StandardErrorPath</key><string>{log_dir}/ormah.err.log</string>
</dict>
</plist>
"""


def get_ormah_bin_path() -> str:
    """Find the absolute path to the ormah binary."""
    path = shutil.which("ormah")
    if path:
        return path
    # Fallback: the current Python interpreter's bin directory
    bin_dir = Path(sys.executable).parent
    candidate = bin_dir / "ormah"
    if candidate.exists():
        return str(candidate)
    return "ormah"


def is_server_running() -> bool:
    """Check if the ormah server is reachable via health endpoint."""
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"http://localhost:{settings.port}/admin/health")
            return r.status_code == 200
    except Exception:
        return False


def install_launchd_agent(ormah_bin: str, wrapper_path: str | None = None) -> None:
    """Install and load a launchd agent for auto-starting the server on macOS."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PLIST_DIR.mkdir(parents=True, exist_ok=True)

    bin_dir = str(Path(ormah_bin).parent)
    effective_wrapper = wrapper_path or ormah_bin
    plist_content = PLIST_TEMPLATE.format(
        label=LAUNCHD_LABEL,
        wrapper_path=effective_wrapper,
        bin_dir=bin_dir,
        log_dir=LOG_DIR,
    )

    # Unload existing agent if present (ignore errors)
    if PLIST_PATH.exists():
        subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)],
            capture_output=True,
        )

    PLIST_PATH.write_text(plist_content)
    subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=True)
    print(f"Installed launchd agent: {PLIST_PATH}")


def uninstall_launchd_agent() -> None:
    """Unload and remove the launchd agent."""
    if not PLIST_PATH.exists():
        print("No launchd agent installed.")
        return

    subprocess.run(
        ["launchctl", "unload", str(PLIST_PATH)],
        capture_output=True,
    )
    PLIST_PATH.unlink(missing_ok=True)
    print("Removed launchd agent.")


def install_systemd_service(ormah_bin: str, wrapper_path: str | None = None) -> None:
    """Install and enable a user-space systemd service for auto-starting the server."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)

    bin_dir = str(Path(ormah_bin).parent)
    effective_wrapper = wrapper_path or ormah_bin
    unit_content = SYSTEMD_TEMPLATE.format(
        wrapper_path=effective_wrapper,
        bin_dir=bin_dir,
        log_dir=LOG_DIR,
    )

    # Stop existing service if present (ignore errors)
    if SYSTEMD_UNIT.exists():
        subprocess.run(
            ["systemctl", "--user", "stop", "ormah.service"],
            capture_output=True,
        )

    SYSTEMD_UNIT.write_text(unit_content)
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["systemctl", "--user", "enable", "--now", "ormah.service"],
        check=True,
    )


def uninstall_systemd_service() -> None:
    """Disable and remove the systemd user service."""
    if not SYSTEMD_UNIT.exists():
        print("No systemd service installed.")
        return

    subprocess.run(
        ["systemctl", "--user", "disable", "--now", "ormah.service"],
        capture_output=True,
    )
    SYSTEMD_UNIT.unlink(missing_ok=True)
    subprocess.run(
        ["systemctl", "--user", "daemon-reload"],
        capture_output=True,
    )
    print("Removed systemd service.")


def _start_server_background(wrapper_path: str) -> None:
    """Start the server as a background process (fallback when no init system)."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_log = open(LOG_DIR / "ormah.out.log", "a")
    err_log = open(LOG_DIR / "ormah.err.log", "a")
    subprocess.Popen(
        [wrapper_path],
        stdout=out_log,
        stderr=err_log,
        start_new_session=True,
    )


def install_autostart(ormah_bin: str, wrapper_path: str | None = None) -> None:
    """Install auto-start using the platform-appropriate mechanism."""
    system = platform.system()
    if system == "Darwin":
        install_launchd_agent(ormah_bin, wrapper_path=wrapper_path)
    elif system == "Linux":
        if shutil.which("systemctl"):
            install_systemd_service(ormah_bin, wrapper_path=wrapper_path)
        else:
            # No systemd (e.g. Docker container) — start directly
            if wrapper_path:
                _start_server_background(wrapper_path)
            else:
                _start_server_background(ormah_bin)
    else:
        print(
            f"Auto-start not supported on {system}. "
            "Run `ormah server start` manually."
        )


def uninstall_autostart() -> None:
    """Remove auto-start using the platform-appropriate mechanism."""
    system = platform.system()
    if system == "Darwin":
        uninstall_launchd_agent()
    elif system == "Linux":
        uninstall_systemd_service()
    else:
        print(f"Auto-start not supported on {system}.")


def is_first_run() -> bool:
    """Check if the fastembed model cache exists — if not, first download needed."""
    cache_dir = Path(tempfile.gettempdir()) / "fastembed_cache"
    if not cache_dir.exists():
        return True
    # Check if any model directories exist inside the cache
    try:
        return not any(cache_dir.iterdir())
    except OSError:
        return True


# Phase markers: log substring -> human-friendly label
_PHASE_MAP: list[tuple[str, str]] = [
    ("Starting ormah server", "Starting server..."),
    ("Initializing memory engine", "Initializing memory engine..."),
    ("Initial index rebuild", "Building search index..."),
    ("Loading embedding model", "Downloading embedding model (~420 MB)..."),
    ("Embedding model ready", "Embedding model loaded"),
    ("Memory engine ready", "Memory engine ready"),
    ("Background scheduler", "Starting background jobs..."),
]


def _tail_server_log(
    callback: callable,
    stop_event: threading.Event,
) -> None:
    """Tail ormah.err.log for phase markers, calling callback on each new phase.

    Runs on a background thread. Polls until the log file appears, then
    reads new lines, matching against known phase markers.
    """
    log_path = LOG_DIR / "ormah.err.log"

    # Wait for the log file to appear
    while not stop_event.is_set():
        if log_path.exists():
            break
        stop_event.wait(0.3)

    if stop_event.is_set():
        return

    try:
        with open(log_path, "r") as f:
            # Seek to end — ignore old log lines
            f.seek(0, 2)
            while not stop_event.is_set():
                line = f.readline()
                if not line:
                    stop_event.wait(0.2)
                    continue
                for marker, label in _PHASE_MAP:
                    if marker in line:
                        callback(label)
                        break
    except OSError:
        pass


def wait_for_server(
    timeout: float = 10.0,
    show_progress: bool = False,
) -> bool:
    """Poll the health endpoint until server is up or timeout is reached.

    When *show_progress* is True, shows an animated spinner with phase
    updates from the server log (for interactive CLI use).
    """
    if not show_progress:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if is_server_running():
                return True
            time.sleep(0.5)
        return False

    # --- Interactive mode with spinner + log tailing ---
    from ormah.console import Spinner

    first_run = is_first_run()
    if first_run:
        effective_timeout = max(timeout, 600.0)
        initial_msg = "Starting server (first run — downloading embedding model)..."
    else:
        effective_timeout = max(timeout, 60.0)
        initial_msg = "Starting server..."

    stop_event = threading.Event()

    with Spinner(initial_msg) as sp:
        # Start log tailer thread
        tail_thread = threading.Thread(
            target=_tail_server_log,
            args=(sp.update, stop_event),
            daemon=True,
        )
        tail_thread.start()

        try:
            deadline = time.monotonic() + effective_timeout
            while time.monotonic() < deadline:
                if is_server_running():
                    stop_event.set()
                    sp.succeed("Server is running")
                    return True
                time.sleep(1.0)

            stop_event.set()
            sp.fail("Server did not start in time")
            return False
        except KeyboardInterrupt:
            stop_event.set()
            sp.fail("Interrupted")
            return False
