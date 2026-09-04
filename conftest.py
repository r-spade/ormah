"""Pytest bootstrap — runs before tests/conftest.py, and before anything
imports ormah.

`ormah.config` builds its `Settings()` singleton at module import time, and
`Settings` reads `~/.config/ormah/.env` plus every `ORMAH_*` variable in the
environment. On a developer machine that runs ormah, those describe the live
install, not the test tree: a provider this branch does not accept aborts
collection outright, and a non-default embedding model silently changes what
the cache tests look for.

Point HOME at an empty directory and drop the ORMAH_* variables here, before
that import happens. This also keeps `Settings.memory_dir` — which defaults
under `Path.home()` — away from the developer's real store.
"""

from __future__ import annotations

import os
import tempfile

for _key in [k for k in os.environ if k.startswith("ORMAH_")]:
    del os.environ[_key]

os.environ["HOME"] = tempfile.mkdtemp(prefix="ormah-test-home-")
