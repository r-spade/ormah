"""Request middleware for agent_id extraction and logging."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class AgentMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract agent_id from header or query param
        agent_id = request.headers.get("x-agent-id") or request.query_params.get("agent_id")
        request.state.agent_id = agent_id

        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start

        logger.info(
            "%s %s [agent=%s] %dms",
            request.method,
            request.url.path,
            agent_id or "anon",
            int(elapsed * 1000),
        )
        return response
