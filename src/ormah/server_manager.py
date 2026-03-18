"""Server lifecycle management — launchd on macOS, systemd on Linux."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
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


def install_autostart(ormah_bin: str, wrapper_path: str | None = None) -> None:
    """Install auto-start using the platform-appropriate mechanism."""
    system = platform.system()
    if system == "Darwin":
        install_launchd_agent(ormah_bin, wrapper_path=wrapper_path)
    elif system == "Linux":
        install_systemd_service(ormah_bin, wrapper_path=wrapper_path)
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


def wait_for_server(timeout: float = 10.0) -> bool:
    """Poll the health endpoint until server is up or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_server_running():
            return True
        time.sleep(0.5)
    return False
