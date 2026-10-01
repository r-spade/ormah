"""Process-wide resource limits for the shared local ONNX models."""

from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from threading import local
from typing import Callable, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ormah-inference")
_worker_state = local()

# Bound activation memory even when every input reaches the model's token limit.
# Keep text intact: a character cutoff discards query context well before the
# tokenizer's existing model-specific token limit.
MAX_BATCH_SIZE = 8


def local_inference(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    """Run local model loading/inference on one reusable thread.

    A lock alone bounds simultaneous activations but still lets each request
    thread retain native allocator buffers. One worker bounds both effects and
    serializes the model caches. Nested calls run inline (e.g. lazy model load).
    Decorated functions must consume FastEmbed's lazy iterators before returning.
    """
    @wraps(fn)
    def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        if getattr(_worker_state, "active", False):
            return fn(*args, **kwargs)

        def run() -> _T:
            _worker_state.active = True
            try:
                return fn(*args, **kwargs)
            finally:
                _worker_state.active = False

        return _worker.submit(run).result()

    return wrapped
