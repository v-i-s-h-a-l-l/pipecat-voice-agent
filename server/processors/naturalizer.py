import re
import random
from collections import deque

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMFullResponseEndFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)


STARTERS = [
    # Neutral / conversational
    "Yeah, ",
    "Oh, ",
    "Right, ",
    "So, ",
    "Hmm, ",
    "Okay, ",
    "Sure, ",
    "Got it — ",
    # Warm
    "Ah, ",
    "Actually, ",
    # Hinglish
    "Haan, ",
    "Bilkul, ",
    "Acha, ",
    "Dekho, ",
    "Toh, ",
    # Empty weighted higher for realism
    "",
    "",
    "",
    "",
    "",
    "",
]

ROBOTIC_SUBS = [
    (r"(?i)^(As an AI[,.]?\s*)", ""),
    (r"(?i)^(Certainly!\s*I'?d be happy to help\.?\s*)", ""),
    (r"(?i)^(Of course!\s*I'?d be happy to assist\.?\s*)", ""),
    (r"(?i)\bAs a language model\b", ""),
    (r"(?i)\bI don't have personal opinions\b", "From what I can tell"),
    (r"(?i)\bI'm just an AI\b", ""),
    (r"(?i)\bI cannot provide\b", "I can't"),
]

TTS_PUNCTUATION_SUBS = [
    # remove ALL dots/periods — replace with comma so TTS pauses naturally
    (r"\.{2,}", ", "),  # ellipsis / multiple dots → pause
    (r"\.", ", "),  # single dot → comma pause
    # markdown cleanup
    (r"\*+", ""),
    (r"#+\s?", ""),
    (r"_+", ""),
    # better TTS cadence
    (r"\s-\s", " — "),
    # repeated punctuation cleanup
    (r",\s*,+", ", "),
    (r"\s{2,}", " "),
]

# Patterns that match spelled-out letter sequences — stripped out entirely
SPELLING_PATTERNS = [
    # V I S H A L  (space-separated single letters)
    re.compile(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b"),
    # V-I-S-H-A-L  (hyphen-separated)
    re.compile(r"\b(?:[A-Za-z][\-\s]){3,}[A-Za-z]\b"),
    # A. B. C. D.  (dot-separated initials)
    re.compile(r"\b(?:[A-Za-z]\.\s*){3,}"),
    # Mixed uppercase sequences like  A B C D
    re.compile(r"(?:\b[A-Z]\b[\s\-\.,]*){4,}"),
]


class ResponseNaturalizerProcessor(FrameProcessor):
    def __init__(
        self,
        add_starters: bool = True,
        min_chunk_length: int = 12,
        starter_cooldown: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._add_starters = add_starters
        self._first_chunk = True

        # avoid awkward tiny chunk starters
        self._min_chunk_length = min_chunk_length

        # avoid repetitive discourse markers
        self._recent_starters = deque(maxlen=starter_cooldown)

    def _strip_spellings(self, text: str) -> str:
        """Remove spelled-out letter sequences from text instead of dropping the whole chunk."""
        for pattern in SPELLING_PATTERNS:
            text = pattern.sub("", text)
        # clean up any double spaces or leading/trailing mess left behind
        text = re.sub(r"\s{2,}", " ", text).strip()
        # remove orphaned punctuation like ", ," or "— ,"
        text = re.sub(r"^[\s,\-—]+", "", text)
        text = re.sub(r"[\s,\-—]+$", "", text)
        return text

    def _clean(self, text: str) -> str:
        if not text:
            return ""

        text = text.strip()

        # remove robotic phrases
        for pattern, repl in ROBOTIC_SUBS:
            text = re.sub(pattern, repl, text)

        # strip spelled-out letter sequences (don't drop whole chunk)
        text = self._strip_spellings(text)

        if not text:
            return ""

        # TTS cleanup
        for pattern, repl in TTS_PUNCTUATION_SUBS:
            text = re.sub(pattern, repl, text)

        # normalize spaces
        text = re.sub(r"\s{2,}", " ", text)

        return text.strip()

    def _pick_starter(self) -> str:
        available = [s for s in STARTERS if s not in self._recent_starters]

        if not available:
            available = STARTERS

        starter = random.choice(available)

        if starter:
            self._recent_starters.append(starter)

        return starter

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame):
            text = self._clean(frame.text)

            if not text:
                return

            # add starters only for meaningful chunks
            if (
                self._add_starters
                and self._first_chunk
                and len(text) >= self._min_chunk_length
            ):
                starter = self._pick_starter()
                text = starter + text
                self._first_chunk = False

            await self.push_frame(
                TextFrame(text=text),
                direction,
            )

        elif isinstance(frame, LLMFullResponseEndFrame):
            self._first_chunk = True
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)
