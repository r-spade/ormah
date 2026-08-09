"""In-process exclusion for jobs that mutate the live memory graph."""

from functools import wraps


def serialized_memory_job(job):
    """Keep a background mutation from overlapping a full graph restore."""

    @wraps(job)
    def locked(engine, *args, **kwargs):
        with engine.memory_operation():
            return job(engine, *args, **kwargs)

    return locked
