# Task 01: Deletion config fields

**Files:**
- Modify: `src/ormah/config.py`
- Test: `tests/test_config.py`

Add the 8 deletion knobs (all `ORMAH_`-prefixed env vars) plus validators mirroring the
existing decay/threshold validators. Defaults make the feature OFF and conservative.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_deletion_defaults_are_off_and_conservative():
    from ormah.config import Settings
    s = Settings()
    assert s.deletion_enabled is False
    assert s.forgetting_interval_hours == 24
    assert s.deletion_min_archival_days == 90
    assert s.deletion_retrievability_floor == 0.05
    assert s.deletion_max_degree == 2
    assert s.deletion_strong_edge_weight == 0.7
    assert s.deletion_retention_days == 30
    assert s.archival_soft_cap == 0


def test_deletion_enabled_from_env(monkeypatch):
    from ormah.config import Settings
    monkeypatch.setenv("ORMAH_DELETION_ENABLED", "true")
    monkeypatch.setenv("ORMAH_DELETION_RETENTION_DAYS", "7")
    s = Settings()
    assert s.deletion_enabled is True
    assert s.deletion_retention_days == 7


def test_deletion_retrievability_floor_must_be_unit_range():
    import pytest
    from ormah.config import Settings
    with pytest.raises(ValueError):
        Settings(deletion_retrievability_floor=1.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config.py -k deletion -v`
Expected: FAIL with `AttributeError`/validation errors (fields not defined).

- [ ] **Step 3: Add the fields**

In `src/ormah/config.py`, add this block right after the existing
`decay_importance_threshold: float = 0.5` line (around line 143):

```python
    # Bounded forgetting (#28). Master switch OFF by default — deletion is
    # irreversible, so it must be armed explicitly via ORMAH_DELETION_ENABLED.
    deletion_enabled: bool = False
    forgetting_interval_hours: int = 24
    deletion_min_archival_days: int = 90       # graveyard age before eligible
    deletion_retrievability_floor: float = 0.05  # FSRS R must be below this
    deletion_max_degree: int = 2               # only weakly-connected leaves
    deletion_strong_edge_weight: float = 0.7   # any edge >= this protects the node
    deletion_retention_days: int = 30          # soft-delete reversibility window
    archival_soft_cap: int = 0                 # 0 = disabled; >0 = evict worst-first to cap
```

- [ ] **Step 4: Add validators**

Add to the validator section (near the other `@field_validator`s, e.g. after the
`_decay_threshold_range` validator around line 393):

```python
    @field_validator("forgetting_interval_hours", "deletion_min_archival_days", "deletion_retention_days")
    @classmethod
    def _deletion_days_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"must be >= 1, got {v}")
        return v

    @field_validator("deletion_retrievability_floor", "deletion_strong_edge_weight")
    @classmethod
    def _deletion_unit_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"must be in [0, 1], got {v}")
        return v

    @field_validator("deletion_max_degree", "archival_soft_cap")
    @classmethod
    def _deletion_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"must be >= 0, got {v}")
        return v
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_config.py -k deletion -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Lint + commit**

```bash
.venv/bin/ruff check src/ormah/config.py tests/test_config.py
git add src/ormah/config.py tests/test_config.py
git commit -m "feat(config): bounded forgetting settings (#28)"
```
