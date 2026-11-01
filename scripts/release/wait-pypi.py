#!/usr/bin/env python3
"""Fail closed while allowing five minutes for PyPI/CDN propagation (#180)."""

import argparse
import json
import time
import urllib.request


def wait_for_version(version, timeout=300, interval=10):
    deadline = time.monotonic() + timeout
    while True:
        try:
            request = urllib.request.Request(
                f"https://pypi.org/pypi/ormah/json?release_check={time.time_ns()}",
                headers={"Cache-Control": "no-cache"},
            )
            with urllib.request.urlopen(request, timeout=15) as response:
                info = json.load(response)
            observed = info["info"]["version"]
            available = any(
                f["packagetype"] == "bdist_wheel" and not f.get("yanked", False)
                for f in info.get("releases", {}).get(version, [])
            )
            print(f"PyPI latest={observed}, expected={version}, wheel={available}", flush=True)
            if observed == version and available:
                return
        except (OSError, ValueError, KeyError) as exc:
            print(f"PyPI check: {exc}", flush=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise SystemExit(f"PyPI did not confirm {version} within {timeout}s")
        time.sleep(min(interval, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version")
    args = parser.parse_args()
    wait_for_version(args.version)
