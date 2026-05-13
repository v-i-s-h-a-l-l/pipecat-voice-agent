"""
WebSocket security middleware.

Provides:
  - Origin header validation (prevent Cross-Site WebSocket Hijacking)
  - Message size limits (prevent memory exhaustion)
  - Session timeout enforcement (prevent zombie connections)
  - Optional JWT token validation (query param based)

No external dependencies beyond the standard library + loguru.
"""

import os
import time
import hmac
import hashlib
import base64
import json
from urllib.parse import urlparse

from fastapi import WebSocket
from loguru import logger


class WebSocketSecurity:
    """
    Security middleware for WebSocket connections.

    Usage:
        security = WebSocketSecurity(allowed_origins=["http://localhost:3000"])

        # Before accepting a WebSocket:
        if not security.validate_origin(websocket):
            await websocket.close(code=4403)
            return

        # In message loop:
        if not security.validate_message_size(data):
            warn/close(...)

        # Check session timeout:
        if security.is_session_expired(start_time):
            close(...)
    """

    def __init__(
        self,
        allowed_origins: list[str] | None = None,
        max_message_bytes: int = 10240,         # 10KB — enough for 5120 PCM16 samples
        session_timeout_secs: int = 600,         # 10 minutes
        require_auth: bool = False,
        jwt_secret: str | None = None,
    ):
        # Parse allowed origins — if ["*"] or empty, allow all
        self.allow_all_origins = (
            not allowed_origins
            or allowed_origins == ["*"]
            or "*" in allowed_origins
        )
        self.allowed_origins = set(allowed_origins or [])
        self.max_message_bytes = max_message_bytes
        self.session_timeout_secs = session_timeout_secs
        self.require_auth = require_auth
        self.jwt_secret = jwt_secret or os.getenv("JWT_SECRET", "")

    def validate_origin(self, websocket: WebSocket) -> bool:
        """
        Validate the Origin header against allowed origins.
        Returns True if the origin is allowed, False otherwise.
        """
        if self.allow_all_origins:
            return True

        origin = websocket.headers.get("origin", "")
        if not origin:
            # No origin header — reject in strict mode
            logger.warning("Security: missing Origin header | client={}", websocket.client)
            return False

        # Normalize: strip trailing slashes for comparison
        origin_normalized = origin.rstrip("/")
        for allowed in self.allowed_origins:
            if origin_normalized == allowed.rstrip("/"):
                return True

        logger.warning(
            "Security: rejected origin | origin={} allowed={} client={}",
            origin,
            self.allowed_origins,
            websocket.client,
        )
        return False

    def validate_message_size(self, data: bytes | str) -> bool:
        """
        Check if a message exceeds the size limit.
        Returns True if the message is within limits.
        """
        size = len(data) if isinstance(data, (bytes, bytearray)) else len(data.encode("utf-8"))
        if size > self.max_message_bytes:
            logger.warning("Security: message too large | size={} limit={}", size, self.max_message_bytes)
            return False
        return True

    def is_session_expired(self, session_start: float) -> bool:
        """
        Check if a session has exceeded the timeout.
        session_start should be time.monotonic() from when the session started.
        """
        elapsed = time.monotonic() - session_start
        return elapsed > self.session_timeout_secs

    def validate_token(self, websocket: WebSocket) -> bool:
        """
        Validate a JWT-like token from query params.
        This is a simple HMAC-based validation — not full JWT.

        Token format: base64(payload).base64(signature)
        Payload: {"exp": unix_timestamp, "sub": "user_id"}

        For production, replace with proper JWT (PyJWT) validation.
        """
        if not self.require_auth:
            return True

        token = websocket.query_params.get("token", "")
        if not token:
            logger.warning("Security: missing auth token | client={}", websocket.client)
            return False

        try:
            parts = token.split(".")
            if len(parts) != 2:
                return False

            payload_b64, sig_b64 = parts
            payload_bytes = base64.urlsafe_b64decode(payload_b64 + "==")
            expected_sig = hmac.new(
                self.jwt_secret.encode(),
                payload_bytes,
                hashlib.sha256,
            ).digest()
            actual_sig = base64.urlsafe_b64decode(sig_b64 + "==")

            if not hmac.compare_digest(expected_sig, actual_sig):
                logger.warning("Security: invalid token signature | client={}", websocket.client)
                return False

            payload = json.loads(payload_bytes)
            exp = payload.get("exp", 0)
            if exp and time.time() > exp:
                logger.warning("Security: token expired | client={}", websocket.client)
                return False

            return True

        except Exception as e:
            logger.warning("Security: token validation failed | error={} client={}", e, websocket.client)
            return False
