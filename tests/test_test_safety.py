"""Regression tests for the shared pytest safety fixtures."""

from pathlib import Path

from ormah import config
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


def test_settings_ignore_the_real_home_env_file():
    real_env_file = Path.home() / ".config" / "ormah" / ".env"

    assert str(real_env_file) not in config._EXISTING_ENV_FILES


def test_settings_ignore_a_checkout_env_file(tmp_path, monkeypatch):
    # config also reads a relative ".env", so a developer's checkout file would reach the suite.
    (tmp_path / ".env").write_text("ORMAH_CONSOLIDATION_MAX_CLUSTERS_PER_RUN=3\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    default = config.Settings.model_fields["consolidation_max_clusters_per_run"].default

    assert config.Settings().consolidation_max_clusters_per_run == default
