"""
In-memory rate limiter for WebSocket connections and messages.

Tracks:
  - Connection rate per IP (max N connections per minute)
  - Concurrent active sessions per IP
  - Message throughput per session (safety valve)

No external dependencies (no Redis) — suitable for single-instance demo/staging.
For multi-instance production, replace with Redis-backed implementation.
"""

import time
import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from loguru import logger


@dataclass
class _IPState:
    """Tracks rate-limit state for a single IP address."""

    connection_timestamps: list[float] = field(default_factory=list)
    active_sessions: set[str] = field(default_factory=set)


@dataclass
class _SessionState:
    """Tracks message-rate state for a single session."""

    message_timestamps: list[float] = field(default_factory=list)


class WebSocketRateLimiter:
    """
    IP-based rate limiter for WebSocket connections and messages.

    Usage:
        limiter = WebSocketRateLimiter()

        # Before accepting a connection:
        if not limiter.check_connection(ip, session_id):
            reject(...)

        # Inside message loop:
        if not limiter.check_message(session_id):
            warn/throttle(...)

        # On disconnect:
        limiter.release_session(ip, session_id)
    """

    def __init__(
        self,
        max_connections_per_minute: int = 5,
        max_concurrent_per_ip: int = 3,
        max_messages_per_second: int = 200,
    ):
        self.max_conn_per_min = max_connections_per_minute
        self.max_concurrent = max_concurrent_per_ip
        self.max_msg_per_sec = max_messages_per_second

        self._ip_states: dict[str, _IPState] = defaultdict(_IPState)
        self._session_states: dict[str, _SessionState] = defaultdict(_SessionState)
        self._lock = asyncio.Lock()

    def _prune_timestamps(self, timestamps: list[float], window_secs: float) -> list[float]:
        """Remove timestamps older than the window."""
        cutoff = time.monotonic() - window_secs
        return [t for t in timestamps if t > cutoff]

    async def check_connection(self, ip: str, session_id: str) -> bool:
        """
        Check if a new WebSocket connection from this IP is allowed.

        Returns True if allowed, False if rate-limited.
        """
        async with self._lock:
            state = self._ip_states[ip]
            now = time.monotonic()

            # Prune old connection timestamps (1-minute window)
            state.connection_timestamps = self._prune_timestamps(
                state.connection_timestamps, 60.0
            )

            # Check connection rate
            if len(state.connection_timestamps) >= self.max_conn_per_min:
                logger.warning(
                    "Rate limit: too many connections | ip={} count={}/min",
                    ip,
                    len(state.connection_timestamps),
                )
                return False

            # Check concurrent sessions
            if len(state.active_sessions) >= self.max_concurrent:
                logger.warning(
                    "Rate limit: too many concurrent sessions | ip={} active={}",
                    ip,
                    len(state.active_sessions),
                )
                return False

            # Allow — record the connection
            state.connection_timestamps.append(now)
            state.active_sessions.add(session_id)
            return True

    def check_message(self, session_id: str) -> bool:
        """
        Check if a message from this session is allowed.
        Called synchronously in the hot path (audio frame processing).

        Returns True if allowed, False if throttled.
        """
        state = self._session_states[session_id]
        now = time.monotonic()

        # Prune old message timestamps (1-second window)
        state.message_timestamps = self._prune_timestamps(
            state.message_timestamps, 1.0
        )

        if len(state.message_timestamps) >= self.max_msg_per_sec:
            # Don't log every throttled message — too noisy
            return False

        state.message_timestamps.append(now)
        return True

    async def release_session(self, ip: str, session_id: str) -> None:
        """Release a session when the WebSocket disconnects."""
        async with self._lock:
            state = self._ip_states.get(ip)
            if state:
                state.active_sessions.discard(session_id)

            # Clean up session message state
            self._session_states.pop(session_id, None)

    @property
    def stats(self) -> dict:
        """Return current limiter stats for monitoring."""
        total_active = sum(
            len(s.active_sessions) for s in self._ip_states.values()
        )
        return {
            "tracked_ips": len(self._ip_states),
            "total_active_sessions": total_active,
        }
