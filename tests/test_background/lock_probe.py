"""Count real L_mem acquisitions, ignoring RLock re-entries.

L_mem is an RLock and every engine mutator is decorated with it, so a job that
holds it and calls engine.remember() re-enters. Only a per-thread 0 -> 1
transition is a hold a foreground writer would have waited on.
"""

from __future__ import annotations

import threading


class LockProbe:
    """Drop-in wrapper around the engine's L_mem RLock."""

    def __init__(self, real_lock) -> None:
        self._real = real_lock
        self._local = threading.local()
        self._counter_lock = threading.Lock()
        self.acquisitions = 0

    @property
    def _depth(self) -> int:
        return getattr(self._local, "depth", 0)

    @_depth.setter
    def _depth(self, value: int) -> None:
        self._local.depth = value

    @property
    def held(self) -> bool:
        """Is the calling thread currently inside L_mem?"""
        return self._depth > 0

    def __enter__(self):
        if self._depth == 0:
            with self._counter_lock:
                self.acquisitions += 1
        self._depth += 1
        return self._real.__enter__()

    def __exit__(self, *args):
        self._depth -= 1
        return self._real.__exit__(*args)

    # FileStore and the engine only ever use `with`, but keep the RLock surface
    # intact so an unexpected direct call does not silently bypass the probe.
    def acquire(self, *args, **kwargs):
        acquired = self._real.acquire(*args, **kwargs)
        if acquired:
            if self._depth == 0:
                with self._counter_lock:
                    self.acquisitions += 1
            self._depth += 1
        return acquired

    def release(self):
        self._depth -= 1
        return self._real.release()


def install_probe(engine) -> LockProbe:
    """Swap a LockProbe into both references to L_mem and return it."""
    probe = LockProbe(engine._memory_operation_lock)
    engine._memory_operation_lock = probe
    engine.file_store._operation_lock = probe
    return probe
