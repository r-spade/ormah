"""File system watcher for memory node changes."""

from __future__ import annotations

import logging
from pathlib import Path

from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class NodeFileHandler(FileSystemEventHandler):
    """Watches memory/nodes/ for file changes and triggers re-indexing."""

    def __init__(self, on_change: callable) -> None:
        self.on_change = on_change

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            logger.debug("File modified: %s", event.src_path)
            self.on_change(Path(event.src_path))

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            logger.debug("File created: %s", event.src_path)
            self.on_change(Path(event.src_path))

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            logger.debug("File deleted: %s", event.src_path)
            self.on_change(None)  # trigger full incremental update


def start_watcher(nodes_dir: Path, on_change: callable) -> Observer:
    """Start watching the nodes directory for changes."""
    observer = Observer()
    handler = NodeFileHandler(on_change)
    observer.schedule(handler, str(nodes_dir), recursive=False)
    observer.start()
    logger.info("File watcher started on %s", nodes_dir)
    return observer
