import json
from pipecat.frames.frames import (
    InputAudioRawFrame,
    OutputAudioRawFrame,
    InputTransportMessageFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer

_count = 0
_accumulator = bytearray()
_TARGET_BYTES = 1024  # 512 samples × 2 bytes = 1024 bytes for Silero at 16kHz


class RawPCMSerializer(FrameSerializer):
    async def serialize(self, frame) -> bytes | None:
        if isinstance(frame, OutputAudioRawFrame):
            return frame.audio
        return None

    async def deserialize(self, data: bytes | str):
        global _count, _accumulator
        if isinstance(data, bytes):
            _accumulator.extend(data)
            if len(_accumulator) < _TARGET_BYTES:
                return None  # not enough yet, wait for more
            chunk = bytes(_accumulator[:_TARGET_BYTES])
            _accumulator = _accumulator[_TARGET_BYTES:]
            _count += 1
            if _count == 1 or _count % 50 == 0:
                print(
                    f"[RawPCMSerializer] emitting chunk bytes={len(chunk)}, count={_count}",
                    flush=True,
                )
            return InputAudioRawFrame(
                audio=chunk,
                sample_rate=16000,
                num_channels=1,
            )
        elif isinstance(data, str):
            try:
                msg = json.loads(data)
                return InputTransportMessageFrame(message=msg)
            except Exception:
                return None
        return None
