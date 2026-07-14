"""Regression tests for the shared pytest safety fixtures."""

from pathlib import Path

from ormah.embeddings.cache import get_fastembed_cache_dir


def test_fastembed_cache_isolated_from_real_ormah_install(isolate_fastembed_cache):
    cache_dir = get_fastembed_cache_dir()
    real_ormah_data_dir = Path.home() / ".local" / "share" / "ormah"

    assert cache_dir == isolate_fastembed_cache
    assert not cache_dir.is_relative_to(real_ormah_data_dir)

    cleanup_probe = cache_dir / ".cleanup-probe.incomplete"
    cleanup_probe.write_text("partial download", encoding="utf-8")
    cleanup_probe.unlink()

    assert not cleanup_probe.exists()
