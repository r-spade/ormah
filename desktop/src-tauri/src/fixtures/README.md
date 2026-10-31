Captured real uv 0.10.9 stderr on Linux, 2026-10-31, in isolated HOME/XDG/uv directories:

- `uv-pypi-down.txt`: pinned install through dead HTTP(S) proxy at127.0.0.1:9.
- `uv-python-down.txt`: pinned install with empty managed-Python directory,
  `UV_PYTHON_PREFERENCE=only-managed`, dead `UV_PYTHON_INSTALL_MIRROR` at127.0.0.1:9.
- `uv-no-solution.txt`: normal public PyPI; main's0.17.0 pin was unpublished.
- `uv-*-default-retries.txt`: same network conditions with uv's default HTTP
  retries, including its own 'Request failed after3retries' wording.

Command: `uv tool install --reinstall --force --python 3.11 ormah==0.17.0`.
The first network fixtures use UV_HTTP_RETRIES=0 for fast, deterministic probes;
all network probes use UV_HTTP_TIMEOUT=10. Default-retries fixtures omit the
retry override. No fixture is from a GUI/Finder launch. Full command, environment
allowlist, exit status, timing, and successful local-wheel upgrade/sentinel
verification are retained in the implementation job's evidence directory.
