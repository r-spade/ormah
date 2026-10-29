"""Process-wide scheduling for the shared local ONNX models."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from functools import wraps
import logging
from threading import local
from time import perf_counter
from typing import Callable, Iterator, Literal, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
InferenceOrigin = Literal["general", "recall"]
_origin: ContextVar[InferenceOrigin] = ContextVar("inference_origin", default="general")
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ormah-inference")
_recall_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ormah-recall")
_worker_state = local()
logger = logging.getLogger(__name__)

# Bound activation memory even when every input reaches the model's token limit.
# Keep text intact: a character cutoff discards query context well before the
# tokenizer's existing model-specific token limit.
MAX_BATCH_SIZE = 8


@contextmanager
def inference_request(origin: InferenceOrigin) -> Iterator[None]:
    """Scope a synchronous request's origin, also usable as a decorator.

    AnyIO copies this context into request threads; local_inference explicitly
    copies it into executor threads. Always reset it, including after errors.
    Generic search/embedding helpers inherit the origin, never choose a lane.
    """
    token = _origin.set(origin)
    try:
        yield
    finally:
        _origin.reset(token)


def local_inference(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """Run local model loading/inference on the request's reusable worker.

    One general worker and one reserved for deliberate recall bound aggregate
    concurrency to two. Recalls can still queue behind recalls and contend for
    CPU or cold model initialization. Model caches need construction locks.
    Nested calls run inline on the current worker, even if the origin changes,
    so workers never wait on each other. Consume lazy iterators before returning.
    """
    @wraps(fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        if getattr(_worker_state, "active", False):
            return fn(*args, **kwargs)

        origin = _origin.get()
        worker = _recall_worker if origin == "recall" else _worker
        context = copy_context()
        traced = logger.isEnabledFor(logging.DEBUG)
        enqueued = perf_counter() if traced else 0.0

        def run() -> _T:
            _worker_state.active = True
            started = perf_counter() if traced else 0.0
            timing = {
                "operation": fn.__qualname__, "origin": origin,
                "enqueued": enqueued, "started": started,
            }
            try:
                if traced:
                    logger.debug("local inference start", extra={"inference": timing})
                return fn(*args, **kwargs)
            finally:
                _worker_state.active = False
                if traced:
                    ended = perf_counter()
                    logger.debug("local inference end", extra={"inference": {
                        **timing, "ended": ended,
                        "queue_seconds": started - enqueued,
                        "execution_seconds": ended - started,
                    }})

        return worker.submit(context.run, run).result()

    return wrapped
