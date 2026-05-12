from loguru import logger

# -- Smart Turn
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_start import TranscriptionUserTurnStartStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

# -- VAD
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# -- Pipeline
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask

# -- Context aggregators
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
    LLMAssistantAggregatorParams,
)

# -- RTVI
from pipecat.processors.frameworks.rtvi.processor import RTVIProcessor
from pipecat.processors.frameworks.rtvi.observer import RTVIObserver

# -- Services
from pipecat.services.cerebras.llm import CerebrasLLMService
from pipecat.services.sarvam.stt import SarvamSTTService
from pipecat.services.sarvam.tts import SarvamTTSService

# -- Transport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)

# -- Language
from pipecat.transcriptions.language import Language

# -- Config and custom processors
from config import CEREBRAS_API_KEY, SARVAM_API_KEY, LLM_MODEL, SAMPLE_RATE
from serializers.raw_pcm import RawPCMSerializer
from processors.pivot_detector import PivotDetectorProcessor
from processors.naturalizer import ResponseNaturalizerProcessor

SYSTEM_PROMPT = """You are a warm, efficient, and natural-sounding voice support assistant.

Speak like a calm, capable human support agent -- clear, conversational, and helpful.

Behavior:
- Keep responses short and easy to understand
- Default to one short spoken sentence unless more detail is required
- Ask only ONE question at a time
- React naturally, but stay professional and composed
- Match the user's tone and language style naturally
- Support English, Hindi, and Hinglish naturally
- If the user changes topic, adapt smoothly and continue naturally
- If interrupted, stop naturally and continue from the new context
- Briefly acknowledge frustrations or confusion when appropriate

Accuracy:
- NEVER guess facts, policies, balances, actions, or outcomes
- If information is missing or unclear, ask a short clarifying question
- If you are uncertain, say so briefly and guide the user forward
- Avoid overexplaining unless the user asks for details

Voice Style:
- Use contractions naturally (I'm, you're, that's, let's)
- Speak like audio, not written text
- Prefer natural spoken phrasing over perfectly written grammar
- Keep replies concise to reduce response latency in conversation
- Avoid long explanations and unnecessary filler
- Avoid repetitive phrases like "Certainly" or "I'd be happy to help"
- NEVER say "As an AI" or "I'm just a language model"
- NEVER use markdown, bullet points, emojis, or formal formatting
- NEVER sound robotic, scripted, or overly corporate
- NEVER spell words, names, codes, or sentences letter-by-letter unless the user explicitly asks
- NEVER read punctuation, symbols, markdown, URLs, or formatting aloud
- NEVER say "dot" while speaking naturally unless the user explicitly asks for an email, URL, or spelling


Conversation Style:
- Focus on helping the user quickly and naturally
- Keep the conversation flowing smoothly in realtime
- Prioritize clarity over perfect grammar
- Sound confident, calm, and human
- Avoid unnecessary apologies or excessive politeness

Your goal is to help users quickly, clearly, and naturally in realtime voice conversation.
"""


async def create_pipeline(websocket, language: str = "hi-IN"):
    logger.info(f"Creating pipeline for language: {language}")

    # -- Transport
    # RTVIProcessor() has NO transport= argument so it does NOT disable audio on start.
    # SileroVADAnalyzer drives voice activity detection locally in the transport.
    # vad_signals=False in STT so Sarvam doesn't try to handle VAD itself.
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            audio_in_stream_on_start=True,
            audio_in_passthrough=True,
            serializer=RawPCMSerializer(),
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.2,
                    min_volume=0.2,
                )
            ),
        ),
    )
    logger.info("Transport created")

    # -- Smart Turn
    smart_turn_stop = TurnAnalyzerUserTurnStopStrategy(
        turn_analyzer=LocalSmartTurnAnalyzerV3()
    )
    transcription_turn_start = TranscriptionUserTurnStartStrategy(use_interim=True)
    logger.info("Turn strategies created")

    # -- STT
    # vad_signals=False: VAD is handled by Silero in the transport, not by Sarvam
    stt = SarvamSTTService(
        api_key=SARVAM_API_KEY,
        mode="transcribe",
        sample_rate=SAMPLE_RATE,
        settings=SarvamSTTService.Settings(
            model="saaras:v3",
            language=Language.HI_IN if language == "hi-IN" else Language.EN_IN,
            vad_signals=True,
            high_vad_sensitivity=True,
        ),
    )
    logger.info("STT service created")

    # -- LLM
    llm = CerebrasLLMService(
        api_key=CEREBRAS_API_KEY,
        settings=CerebrasLLMService.Settings(
            model=LLM_MODEL,
            temperature=0.6,
            max_completion_tokens=300,
        ),
    )
    logger.info("LLM service created")

    # -- TTS
    tts = SarvamTTSService(
        api_key=SARVAM_API_KEY,
        sample_rate=SAMPLE_RATE,
        settings=SarvamTTSService.Settings(
            model="bulbul:v3",
            voice="shubh",
            language=Language.HI_IN if language == "hi-IN" else Language.EN_IN,
            pace=1.0,
            pitch=0.0,
            enable_preprocessing=True,
            temperature=0.7,
        ),
    )
    logger.info("TTS service created")

    # -- Custom processors
    pivot_detector = PivotDetectorProcessor()
    naturalizer = ResponseNaturalizerProcessor(add_starters=True)
    logger.info("Custom processors created")

    # -- RTVI: no transport= so it does NOT gate audio behind client-ready handshake
    rtvi = RTVIProcessor()
    logger.info("RTVI processor created")

    # -- LLM context
    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}])

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[transcription_turn_start],
                stop=[smart_turn_stop],
            ),
        ),
        assistant_params=LLMAssistantAggregatorParams(),
    )
    logger.info("Context aggregators created")

    # -- Pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # raw audio from browser
            stt,  # audio -> TranscriptionFrame
            user_aggregator,  # turn detection + LLM context
            pivot_detector,  # topic-change detection
            llm,  # text -> streaming response
            naturalizer,  # clean robotic phrases
            tts,  # text -> audio chunks
            rtvi,  # RTVI events to browser
            assistant_aggregator,  # store reply in context
            transport.output(),  # stream audio to browser
        ]
    )
    logger.info("Pipeline assembled")

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=SAMPLE_RATE,
            audio_out_sample_rate=SAMPLE_RATE,
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )
    logger.info("Pipeline task created -- ready")

    return transport, task
