"""
api/middleware/correlation.py
------------------------------
FastAPI middleware that injects a correlation ID into every request.

The correlation ID is:
  1. Read from the X-Correlation-ID header if the caller sends one.
  2. Generated as a new UUID if not present.
  3. Set on the structured logger so every log line for this request
     automatically includes it.
  4. Echoed back in the X-Correlation-ID response header.

This is the standard pattern at companies like Stripe, Uber, and GitHub for
distributed request tracing without a full OpenTelemetry deployment.
"""
from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from evaluation_platform.logging_.structured import set_correlation_id

CORRELATION_ID_HEADER = "X-Correlation-ID"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Any) -> Response:
        cid = request.headers.get(CORRELATION_ID_HEADER) or str(uuid.uuid4())
        set_correlation_id(cid)
        response: Response = await call_next(request)
        response.headers[CORRELATION_ID_HEADER] = cid
        return response


# avoid import issues
from typing import Any  # noqa: E402
