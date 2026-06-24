"""Per-client sliding-window rate limiter for the Shef web application."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

# ── Configuration ───────────────────────────────────────────────────────────

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 5

# ── Internal state ──────────────────────────────────────────────────────────

_request_times_by_client: dict[str, deque[float]] = defaultdict(deque)


# ── Public helpers ──────────────────────────────────────────────────────────


def request_client_key(request: Request) -> str:
    """Derive a stable client identifier from the request.

    Prefers the ``X-Forwarded-For`` header when the direct client is
    localhost (typical behind a reverse proxy).
    """
    client_host = (
        request.client.host
        if request.client and request.client.host
        else "unknown-client"
    )
    forwarded_for = request.headers.get("x-forwarded-for", "")
    forwarded_host = forwarded_for.split(",", 1)[0].strip()
    if client_host in {"127.0.0.1", "::1", "localhost"} and forwarded_host:
        return forwarded_host
    return client_host


def enforce_rate_limit(request: Request) -> None:
    """Enforce a sliding-window rate limit per client.

    Raises ``HTTPException`` (429) when the client exceeds
    ``RATE_LIMIT_MAX_REQUESTS`` requests within
    ``RATE_LIMIT_WINDOW_SECONDS``.
    """
    now = time.monotonic()
    client_key = request_client_key(request)
    request_times = _request_times_by_client[client_key]

    while request_times and now - request_times[0] >= RATE_LIMIT_WINDOW_SECONDS:
        request_times.popleft()

    if len(request_times) >= RATE_LIMIT_MAX_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Too many chat requests. Try again in a minute.",
        )

    request_times.append(now)


def reset_rate_limit_state() -> None:
    """Clear all tracked request times.  Used by tests."""
    _request_times_by_client.clear()
