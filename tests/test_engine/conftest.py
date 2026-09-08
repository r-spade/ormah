"""Fixtures shared by the engine tests."""

from __future__ import annotations

import pytest

from ormah.engine import prompt_classifier


@pytest.fixture(autouse=True)
def pin_temporal_locales(monkeypatch):
    """Pin the enabled temporal packs so no test reads the operator's ``.env``.

    The module-level temporal functions build their parser from a fresh
    ``Settings()``, so without this pin the PT-BR assertions would silently
    depend on whatever ``ORMAH_TEMPORAL_LOCALES`` the machine running the
    suite happens to have. The cache clear is part of the pin, not a sweep:
    ``_default_parser`` is an ``lru_cache`` keyed on nothing, so a parser built
    before this fixture ran would survive the ``setenv`` and the pin would mean
    nothing. Cleaning up after a test that *changes* the setting is that test's
    own job — see ``TestTheSettingGatesThePublicFunctions`` in
    ``test_temporal_public_functions.py``.
    """
    monkeypatch.setenv("ORMAH_TEMPORAL_LOCALES", "en,pt-BR")
    prompt_classifier._default_parser.cache_clear()
    yield
