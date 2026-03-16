"""Hatch build hook — auto-builds the UI if npm is available."""

import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        ui_dir = Path("ui")
        ui_dist = Path("src/ormah/ui_dist")

        # Already built — skip
        if (ui_dist / "index.html").exists():
            return

        # No UI source — nothing to build
        if not (ui_dir / "package.json").exists():
            return

        # npm not available — skip silently
        npm = shutil.which("npm")
        if not npm:
            self.app.display_warning(
                "npm not found — skipping UI build. "
                "The server will work but the graph UI won't be available."
            )
            return

        self.app.display_info("Building UI...")
        subprocess.run([npm, "install"], cwd="ui", check=True, capture_output=True)
        subprocess.run([npm, "run", "build"], cwd="ui", check=True, capture_output=True)
        self.app.display_info("UI built successfully.")
