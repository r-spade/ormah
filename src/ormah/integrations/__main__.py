"""Explicit integration setup without starting services or changing memory stores."""
import argparse
import json
from importlib import import_module
from pathlib import Path

from .registry import descriptors


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ormah agents")
    parser.add_argument("action", choices=["list", "connect", "disconnect", "status"])
    parser.add_argument("host", nargs="?")
    parser.add_argument("--project", type=Path, help="Configure this project's workspace")
    args = parser.parse_args(argv)
    available = {d.id: d for d in descriptors()}
    if args.action == "list":
        print(json.dumps(list(available)))
        return
    if args.host not in available:
        parser.error("Unknown host; use 'ormah agents list'")
    module = import_module(f"ormah.integrations.hosts.{args.host}")
    project = args.project.resolve() if args.project else None
    if project is not None and not project.is_dir():
        parser.error("--project must be an existing workspace directory")
    try:
        if args.action == "status":
            print(json.dumps(module.status(project), indent=2))
        else:
            getattr(module, args.action)(project)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
