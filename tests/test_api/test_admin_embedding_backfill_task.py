"""embedding_backfill must be a registered admin task in the sleep-cycle (#32)."""
from __future__ import annotations

from ormah.api import routes_admin


def test_embedding_backfill_in_task_registry():
    assert "embedding_backfill" in routes_admin._TASK_RUNNERS
    module, func = routes_admin._TASK_RUNNERS["embedding_backfill"]
    assert module == "ormah.background.embedding_backfill"
    assert func == "run_embedding_backfill"


def test_embedding_backfill_has_description():
    assert "embedding_backfill" in routes_admin._TASK_DESCRIPTIONS


def test_embedding_backfill_in_sleep_cycle_order():
    order = routes_admin._SLEEP_CYCLE_ORDER
    assert "embedding_backfill" in order
    # runs after the index is updated
    assert order.index("embedding_backfill") > order.index("index_updater")


def test_run_all_tasks_degraded_returns_503_when_a_task_raises(monkeypatch):
    """C1/I1: a failed task yields status=degraded AND HTTP 503 (not 200)."""
    import importlib
    import json
    from unittest.mock import MagicMock

    from fastapi.responses import JSONResponse

    import ormah.background.embedding_backfill as ebf
    from ormah.api import routes_admin

    def _raise(engine):
        raise RuntimeError("encoder down")

    monkeypatch.setattr(ebf, "run_embedding_backfill", _raise)

    # Stub every other runner so real background code doesn't run with a mock engine.
    for task_id, (module_path, func_name) in routes_admin._TASK_RUNNERS.items():
        if task_id != "embedding_backfill":
            mod = importlib.import_module(module_path)
            monkeypatch.setattr(mod, func_name, lambda e: None)

    mock_request = MagicMock()
    mock_request.app.state.engine = MagicMock()

    result = routes_admin.run_all_tasks(mock_request)

    assert isinstance(result, JSONResponse)
    assert result.status_code == 503
    body = json.loads(bytes(result.body))
    assert body["status"] == "degraded"
    assert body["results"]["embedding_backfill"].startswith("error:")


def test_run_all_tasks_completed_returns_dict_when_all_ok(monkeypatch):
    """Happy path stays a plain dict (HTTP 200) with status=completed."""
    import importlib
    from unittest.mock import MagicMock

    from ormah.api import routes_admin

    for task_id, (module_path, func_name) in routes_admin._TASK_RUNNERS.items():
        mod = importlib.import_module(module_path)
        monkeypatch.setattr(mod, func_name, lambda e: None)

    mock_request = MagicMock()
    engine = MagicMock()
    engine.builder.incremental_update.return_value = (0, 0)
    mock_request.app.state.engine = engine

    result = routes_admin.run_all_tasks(mock_request)

    assert isinstance(result, dict)
    assert result["status"] == "completed"
    assert all(v == "ok" for v in result["results"].values())
