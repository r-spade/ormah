import os
import tempfile

# ormah.config resolves ~/.config/ormah/.env at import and builds Settings() there, so import it
# under an empty HOME: the developer's ORMAH_* values must never reach the suite. This package
# init runs before conftest and in every spawned child process that imports a test module.
_HOME_AT_START = os.environ.get("HOME")
os.environ["HOME"] = tempfile.mkdtemp(prefix="ormah-test-home-")
try:
    import ormah.config  # noqa: F401
finally:
    if _HOME_AT_START is None:
        del os.environ["HOME"]
    else:
        os.environ["HOME"] = _HOME_AT_START
