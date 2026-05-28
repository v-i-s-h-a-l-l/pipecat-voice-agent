"""
ContextSanitizerProcessor
─────────────────────────
Merges or replaces consecutive same-role messages in the LLM context before
every LLM call.
"""

from loguru import logger

from pipecat.frames.frames import Frame, LLMMessagesAppendFrame, LLMRunFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class ContextSanitizerProcessor(FrameProcessor):
    """
    Sanitizes the LLM context before each LLM call:
    - Consecutive USER messages: keep only the latest
    - Consecutive ASSISTANT messages: replace if truncated, merge if complete
    - Trims history to max_history_messages (non-system)
    """

    TRUNCATION_LENGTH_THRESHOLD = 60
    MAX_HISTORY_MESSAGES = 16

    def __init__(self, context: LLMContext, **kwargs):
        super().__init__(**kwargs)
        self._context = context

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (LLMMessagesAppendFrame, LLMRunFrame)):
            self._prepare_context()

        await self.push_frame(frame, direction)

    def _prepare_context(self):
        """Sanitize and trim before the LLM reads context."""
        self._sanitize()
        self._trim_history()

    @classmethod
    def drop_truncated_last_assistant(cls, context: LLMContext) -> bool:
        """Remove the last assistant message if it looks cut off by barge-in."""
        messages = context.messages
        if len(messages) < 2:
            return False

        last = messages[-1]
        if last.get("role") != "assistant":
            return False

        content = (last.get("content") or "").strip()
        if not cls._is_truncated_assistant(content):
            return False

        messages.pop()
        logger.info(
            "[ContextSanitizer] Removed truncated assistant: '{}'",
            content[:60],
        )
        return True

    @classmethod
    def _is_truncated_assistant(cls, content: str) -> bool:
        content = content.strip()
        return len(content) < cls.TRUNCATION_LENGTH_THRESHOLD and not content.endswith(
            (".", "?", "!", ",", "—")
        )

    def _sanitize(self):
        """Sanitize self._context.messages in-place before LLM sees it."""
        messages = self._context.messages
        if not messages:
            return

        merged: list[dict] = [dict(messages[0])]

        for msg in messages[1:]:
            role = msg.get("role", "")
            prev_role = merged[-1].get("role", "")

            if role == prev_role and role == "user":
                prev_content = (merged[-1].get("content") or "").strip()
                new_content = (msg.get("content") or "").strip()
                # Keep whichever message is longer/more complete —
                # Sarvam STT sometimes delivers a fuller partial first,
                # then a shorter correction; always taking the latest
                # would lose context.
                if len(new_content) >= len(prev_content):
                    merged[-1] = dict(msg)
                    logger.info(
                        "[ContextSanitizer] Replaced user message '{}' → '{}'",
                        prev_content[:60],
                        new_content[:60],
                    )
                else:
                    logger.info(
                        "[ContextSanitizer] Kept longer user message '{}' (discarded '{}')",
                        prev_content[:60],
                        new_content[:60],
                    )

            elif role == prev_role and role == "assistant":
                prev_content = (merged[-1].get("content") or "").strip()
                new_content = (msg.get("content") or "").strip()

                if self._is_truncated_assistant(prev_content):
                    merged[-1] = dict(msg)
                    logger.info(
                        "[ContextSanitizer] Replaced truncated assistant '{}' → '{}'",
                        prev_content[:60],
                        new_content[:60],
                    )
                else:
                    merged[-1] = dict(merged[-1])
                    merged[-1]["content"] = (prev_content + " " + new_content).strip()
                    logger.info(
                        "[ContextSanitizer] Merged consecutive assistant messages → '{}'",
                        merged[-1]["content"][:80],
                    )

            else:
                merged.append(dict(msg))

        self._context.messages[:] = merged

    def _trim_history(self):
        """Keep system prompt plus the most recent non-system messages."""
        messages = self._context.messages
        if len(messages) < 1:
            return

        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        if len(other_msgs) <= self.MAX_HISTORY_MESSAGES:
            return

        trimmed = other_msgs[-self.MAX_HISTORY_MESSAGES :]
        before = len(other_msgs)
        self._context.messages[:] = system_msgs + trimmed
        logger.info(
            "[ContextSanitizer] Trimmed history {} → {} messages",
            before,
            len(trimmed),
        )
