"""
RawAudioSerializer — bridges raw PCM-16 binary from the browser
to Pipecat's InputAudioRawFrame, and serializes outgoing audio
frames back to raw bytes and RTVI events as JSON text.

Why this exists
---------------
Pipecat's FastAPIWebsocketTransport silently drops **all** incoming
WebSocket messages when no serializer is configured (see
FastAPIWebsocketInputTransport._receive_messages).  Likewise the
output transport will not send anything without a serializer.
This custom serializer keeps the client side simple (raw PCM-16 +
JSON text) while satisfying the transport contract.
"""

import json
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

SAMPLE_RATE = 16_000
NUM_CHANNELS = 1


class RawAudioSerializer(FrameSerializer):
    """
    Incoming (browser → server):
      bytes  → InputAudioRawFrame (raw PCM-16 LE mono @ 16 kHz)
      str    → ignored (no client→server control messages needed)

    Outgoing (server → browser):
      OutputAudioRawFrame              → raw bytes  (PCM-16 LE)
      OutputTransportMessageFrame      → JSON text  (RTVI events)
      OutputTransportMessageUrgentFrame→ JSON text  (RTVI urgent events)
      everything else                  → None  (silently skipped)
    """

    def __init__(self):
        # Disable the RTVI-message filter so RTVI events reach the browser.
        params = FrameSerializer.InputParams(ignore_rtvi_messages=False)
        super().__init__(params=params)

    # ── Outgoing frames (server → browser) ───────────────────────────────────

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio  # raw PCM-16 bytes

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            try:
                return json.dumps(frame.message)
            except Exception as exc:
                logger.warning(f"[RawAudioSerializer] serialize JSON failed: {exc}")
                return None

        return None  # drop all other frame types

    # ── Incoming frames (browser → server) ───────────────────────────────────

    async def deserialize(self, data: str | bytes) -> Frame | None:
        if isinstance(data, (bytes, bytearray)):
            return InputAudioRawFrame(
                audio=bytes(data),
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )

        # Text messages from the browser are not used yet.
        return None
