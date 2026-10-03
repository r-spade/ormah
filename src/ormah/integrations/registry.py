"""Discover built-in integrations shipped in this distribution."""
from importlib import import_module
from pkgutil import iter_modules

from . import hosts


def descriptors():
    return [
        import_module(f"{hosts.__name__}.{module.name}").descriptor()
        for module in sorted(iter_modules(hosts.__path__), key=lambda m: m.name)
        if not module.name.startswith("_")
    ]
