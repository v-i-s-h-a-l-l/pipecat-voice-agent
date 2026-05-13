import json

from pipecat.frames.frames import (
    InputAudioRawFrame,
    InputTransportMessageFrame,
    OutputAudioRawFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

_TARGET_BYTES = 1024  # 512 samples × 2 bytes = 1024 bytes for Silero at 16 kHz


class RawPCMSerializer(FrameSerializer):
    """Per-connection PCM re-chunker: buffers incoming bytes until _TARGET_BYTES."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._accumulator = bytearray()
        self._count = 0

    async def serialize(self, frame) -> bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: bytes | str):
        if isinstance(data, bytes):
            self._accumulator.extend(data)
            if len(self._accumulator) < _TARGET_BYTES:
                return None  # not enough yet, wait for more
            chunk = bytes(self._accumulator[:_TARGET_BYTES])
            del self._accumulator[:_TARGET_BYTES]
            self._count += 1
            if self._count == 1 or self._count % 50 == 0:
                print(
                    f"[RawPCMSerializer] emitting chunk bytes={len(chunk)}, count={self._count}",
                    flush=True,
                )
            return InputAudioRawFrame(
                audio=chunk,
                sample_rate=16000,
                num_channels=1,
            )
        if isinstance(data, str):
            try:
                msg = json.loads(data)
                return InputTransportMessageFrame(message=msg)
            except Exception:
                return None
        return None
