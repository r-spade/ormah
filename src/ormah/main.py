"""FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ormah.api.middleware import AgentMiddleware
from ormah.api.local_auth import load_or_create_local_admin_token
from ormah.api.routes_account import router as account_router
from ormah.api.routes_admin import router as admin_router
from ormah.api.routes_agent import router as agent_router
from ormah.api.routes_ingest import router as ingest_router
from ormah.api.routes_protection import router as protection_router
from ormah.api.routes_stats import router as stats_router
from ormah.api.routes_ui import router as ui_router
from ormah.background.maintenance_manager import MaintenanceManager
from ormah.config import settings
from ormah.engine.memory_engine import MemoryEngine
from ormah.logging_setup import setup_logging
from ormah.server_manager import LOG_DIR

setup_logging(
    log_format=settings.log_format,
    level=getattr(logging, settings.log_level),
    log_file=LOG_DIR / "ormah.log",
)
logger = logging.getLogger(__name__)

_RESERVED_API_PREFIXES = {"agent", "admin", "ingest", "stats", "ui"}
_LOCAL_CORS_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"


def _is_reserved_api_path(full_path: str) -> bool:
    return full_path.split("/", 1)[0] in _RESERVED_API_PREFIXES


try:
    APP_VERSION = pkg_version("ormah")
except PackageNotFoundError:
    APP_VERSION = "0.0.0"


def _initialize_local_admin(app: FastAPI) -> None:
    """Enable sensitive local routes without making them a core-server dependency."""
    try:
        app.state.local_admin_token = load_or_create_local_admin_token()
    except (OSError, RuntimeError):
        app.state.local_admin_token = None
        logger.warning(
            "Local account and billing routes are disabled because their capability "
            "could not be secured."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ormah server on port %d...", settings.port)
    _initialize_local_admin(app)
    logger.info("Initializing memory engine...")
    engine = MemoryEngine(settings)
    engine.startup()
    app.state.engine = engine
    from ormah.cloud.operations import (
        ProtectionOperationCoordinator,
        resume_interrupted_enable,
    )

    app.state.protection_operations = ProtectionOperationCoordinator()
    resume_interrupted_enable(engine, app.state.protection_operations)
    logger.info("Memory engine ready.")

    # Start background scheduler if available
    try:
        from ormah.background.scheduler import start_scheduler

        logger.info("Starting background scheduler...")
        scheduler, tracker = start_scheduler(engine)
        app.state.scheduler = scheduler
        app.state.job_tracker = tracker
        app.state.maintenance_manager = MaintenanceManager(engine, tracker=tracker)
        logger.info("Background scheduler ready.")
    except Exception as e:
        logger.warning("Background scheduler not started: %s", e)

    if not hasattr(app.state, "job_tracker"):
        # The manual admin routes use the tracker for single-flight exclusion. It was only
        # created inside the scheduler's try block, so a failed scheduler startup left the
        # guard with nothing to claim against and it silently degraded to a no-op — two
        # concurrent HTTP triggers could then run the same edge-writing job at once (#117).
        # No scheduler means no scheduled job to collide with, but concurrent requests
        # still collide, so the tracker must always exist.
        from ormah.background.job_tracker import JobTracker

        app.state.job_tracker = JobTracker()

    if not hasattr(app.state, "maintenance_manager"):
        app.state.maintenance_manager = MaintenanceManager(engine)

    # Start hippocampus file watchers
    try:
        from ormah.background.hippocampus import start_hippocampus, stop_hippocampus

        observers = start_hippocampus(engine)
        app.state.hippocampus_observers = observers
    except Exception as e:
        logger.warning("Hippocampus watchers not started: %s", e)

    # Start session watcher for agent transcripts
    try:
        from ormah.background.session_watcher import start_session_watcher, stop_session_watcher

        session_watches = start_session_watcher(engine)
        app.state.session_watcher_observers = session_watches
        if hasattr(app.state, "scheduler"):
            from ormah.background.scheduler import register_session_reconcile_job
            register_session_reconcile_job(
                app.state.scheduler, app.state.job_tracker, session_watches,
                engine.settings.session_watcher_reconcile_interval_minutes,
            )
    except Exception as e:
        logger.warning("Session watcher not started: %s", e)

    yield

    # Unschedule the reconcile job before stopping the watchers, to shrink the window where
    # a tick recreates an Observer that nothing then stops. remove_job() only cancels future
    # triggers, not an already-running tick, so a single in-flight tick can still recreate one
    # Observer; that leaked daemon thread dies with the process (same tradeoff as the engine
    # connection below). Fully closing it would require shutting the scheduler down before the
    # watchers, which the bind-sensitive shutdown order avoids.
    if hasattr(app.state, "scheduler"):
        try:
            app.state.scheduler.remove_job("session_reconcile")
        except Exception:
            pass

    # Shutdown — stop session watcher first
    if hasattr(app.state, "session_watcher_observers"):
        stop_session_watcher(app.state.session_watcher_observers)

    # Shutdown — stop hippocampus watchers
    if hasattr(app.state, "hippocampus_observers"):
        stop_hippocampus(app.state.hippocampus_observers)

    # Shutdown — wait for running jobs to finish
    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown(wait=True)
    if hasattr(app.state, "protection_operations"):
        app.state.protection_operations.shutdown(wait=True)
    engine.shutdown()
    logger.info("Ormah stopped")


app = FastAPI(
    title="Ormah",
    description="Local-first, LLM-agnostic memory system for AI agents",
    version=APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=_LOCAL_CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AgentMiddleware)

app.include_router(agent_router)
app.include_router(admin_router)
app.include_router(account_router)
app.include_router(protection_router)
app.include_router(stats_router)
app.include_router(ui_router)
app.include_router(ingest_router)

# Serve the built frontend bundled inside the package
_ui_dist = Path(__file__).resolve().parent / "ui_dist"
if _ui_dist.is_dir():
    app.mount("/assets", StaticFiles(directory=_ui_dist / "assets"), name="static")

    _ui_dist_resolved = _ui_dist.resolve()

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the SPA index.html for all non-API routes."""
        if _is_reserved_api_path(full_path):
            raise HTTPException(status_code=404, detail="Not found")
        file = (_ui_dist / full_path).resolve()
        try:
            file.relative_to(_ui_dist_resolved)
        except ValueError:
            return FileResponse(_ui_dist / "index.html")
        if file.is_file():
            return FileResponse(file)
        return FileResponse(_ui_dist / "index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ormah.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
