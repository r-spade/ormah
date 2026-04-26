"""Server lifecycle management — launchd on macOS, systemd on Linux."""

from __future__ import annotations

from dataclasses import dataclass
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

from ormah.config import settings
from ormah.embeddings.cache import get_fastembed_cache_dir

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
    PLIST_DIR.mkdir(parents=True, exist_ok=True)

    bin_dir = str(Path(ormah_bin).parent)
    effective_wrapper = wrapper_path or ormah_bin
    plist_content = PLIST_TEMPLATE.format(
        label=LAUNCHD_LABEL,
        wrapper_path=effective_wrapper,
        bin_dir=bin_dir,
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
    SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)

    bin_dir = str(Path(ormah_bin).parent)
    effective_wrapper = wrapper_path or ormah_bin
    unit_content = SYSTEMD_TEMPLATE.format(
        wrapper_path=effective_wrapper,
        bin_dir=bin_dir,
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
    subprocess.Popen(
        [wrapper_path],
        start_new_session=True,
    )


def _called_process_error_output(exc: subprocess.CalledProcessError) -> str:
    """Return a concise subprocess error message for user-facing fallback output."""
    output = exc.stderr or exc.stdout or ""
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    return " ".join(str(output).strip().split())


def install_autostart(ormah_bin: str, wrapper_path: str | None = None) -> None:
    """Install auto-start using the platform-appropriate mechanism."""
    system = platform.system()
    effective_wrapper = wrapper_path or ormah_bin
    if system == "Darwin":
        install_launchd_agent(ormah_bin, wrapper_path=effective_wrapper)
    elif system == "Linux":
        if shutil.which("systemctl"):
            try:
                install_systemd_service(ormah_bin, wrapper_path=effective_wrapper)
                return
            except subprocess.CalledProcessError as exc:
                SYSTEMD_UNIT.unlink(missing_ok=True)
                print("User systemd is unavailable; starting server in background instead.")
                details = _called_process_error_output(exc)
                if details:
                    print(f"systemctl error: {details}")
                _start_server_background(effective_wrapper)
        else:
            # No systemd (e.g. Docker container) — start directly
            _start_server_background(effective_wrapper)
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


def _wait_for_pid_exit(pid: int, timeout: float = 5.0) -> bool:
    """Poll until pid exits or timeout elapses. Returns True if the process exited."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        time.sleep(0.1)
    return False


@dataclass
class _StopServerResult:
    found: bool = False
    stopped: bool = False
    failed: bool = False


def _is_ormah_server_start_command(command: str) -> bool:
    """Return True only for direct Ormah server-start process commands."""
    try:
        args = shlex.split(command)
    except ValueError:
        args = command.split()

    if len(args) >= 3 and Path(args[0]).name == "ormah" and args[1:3] == ["server", "start"]:
        return True

    if len(args) >= 4:
        launcher = Path(args[0]).name
        script = Path(args[1]).name
        if launcher.startswith("python") and script == "ormah":
            return args[2:4] == ["server", "start"]

    return False


def _find_manual_server_pids() -> list[int]:
    """Find manually-started Ormah server PIDs via token-level ps matching."""
    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,uid=,command="],
            capture_output=True, text=True, timeout=5,
        )
    except Exception:
        return []

    current_pid = os.getpid()
    current_uid = os.getuid()
    pids: list[int] = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid_text, uid_text, command = parts
        try:
            pid = int(pid_text)
            uid = int(uid_text)
        except ValueError:
            continue
        if pid == current_pid or uid != current_uid:
            continue
        if _is_ormah_server_start_command(command):
            pids.append(pid)
    return pids


def _stop_running_server() -> _StopServerResult:
    """Stop the Ormah server and return a structured result for failure propagation."""
    res = _StopServerResult()
    system = platform.system()

    if system == "Linux" and shutil.which("systemctl"):
        active = subprocess.run(
            ["systemctl", "--user", "is-active", "ormah.service"],
            capture_output=True, text=True,
        )
        if active.stdout.strip() == "active":
            res.found = True
            stop = subprocess.run(
                ["systemctl", "--user", "stop", "ormah.service"],
                capture_output=True,
            )
            if stop.returncode == 0:
                print("Stopped Ormah server (systemd).")
                res.stopped = True
            else:
                print("Failed to stop Ormah server via systemd.")
                res.failed = True
    elif system == "Darwin" and PLIST_PATH.exists():
        res.found = True
        stop = subprocess.run(
            ["launchctl", "unload", str(PLIST_PATH)],
            capture_output=True,
        )
        if stop.returncode == 0:
            print("Stopped Ormah server (launchd).")
            res.stopped = True
        else:
            print("Failed to stop Ormah server via launchd.")
            res.failed = True

    killed = 0
    for pid in _find_manual_server_pids():
        res.found = True
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            res.failed = True
            continue
        if _wait_for_pid_exit(pid):
            killed += 1
        else:
            print(f"Warning: process {pid} did not exit after SIGTERM.")
            res.failed = True

    if killed:
        noun = "process" if killed == 1 else "processes"
        print(f"Stopped Ormah server ({killed} {noun}).")
        res.stopped = True

    if not res.found:
        print("No running Ormah server found.")

    return res


def stop_running_server() -> bool:
    """Stop the running Ormah server. Returns True if at least one process was stopped."""
    return _stop_running_server().stopped


def is_first_run() -> bool:
    """Check if the fastembed model cache exists — if not, first download needed."""
    cache_dir = get_fastembed_cache_dir()
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
    ("Loading embedding model", "Loading embedding model..."),
    ("Embedding model ready", "Embedding model loaded"),
    ("Loading whisper reranker", "Loading whisper reranker..."),
    ("Whisper reranker ready", "Whisper reranker loaded"),
    ("Re-indexing embeddings", "Re-embedding memories..."),
    ("Memory engine ready", "Memory engine ready"),
    ("Background scheduler", "Starting background jobs..."),
]

_REEMBED_RE = re.compile(r"Re-embedding memories: (\d+)/(\d+)")


def _tail_server_log(
    callback: callable,
    stop_event: threading.Event,
    phase_map: list[tuple[str, str]] | None = None,
) -> None:
    """Tail the server log file for phase markers, calling callback on each new phase.

    Runs on a background thread. Polls until the log file appears, then
    reads new lines, matching against known phase markers.
    """
    log_path = LOG_DIR / "ormah.log"
    effective_phase_map = phase_map if phase_map is not None else _PHASE_MAP

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
                m = _REEMBED_RE.search(line)
                if m:
                    done, total = int(m.group(1)), int(m.group(2))
                    pct = int(done / total * 100)
                    callback(f"Re-embedding memories: {done}/{total} ({pct}%)")
                    continue
                for marker, label in effective_phase_map:
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
        phase_map = [
            (marker, "Downloading embedding model (~420 MB)..." if marker == "Loading embedding model" else label)
            for marker, label in _PHASE_MAP
        ]
    else:
        effective_timeout = max(timeout, 300.0)
        initial_msg = "Starting server..."
        phase_map = _PHASE_MAP

    stop_event = threading.Event()

    with Spinner(initial_msg) as sp:
        # Start log tailer thread
        tail_thread = threading.Thread(
            target=_tail_server_log,
            args=(sp.update, stop_event, phase_map),
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
