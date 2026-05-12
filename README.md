# Live Voice Agent

A realtime browser-to-server voice assistant built with:

- A lightweight browser client (`client/`) that captures microphone audio and plays bot audio.
- A FastAPI + Pipecat backend (`server/`) that runs STT -> LLM -> TTS pipeline over WebSocket.
- A UV-managed Python environment (`voice-agent/`) that contains the backend dependencies.

---

## 1) Prerequisites and Requirements

## OS / Runtime
- Windows (project is currently configured and tested with Windows-style commands).
- Python `>=3.12` (defined in `voice-agent/pyproject.toml`).
- Internet access for cloud AI services.

## Python package manager
- [`uv`](https://docs.astral.sh/uv/) (recommended and used by this repo).

## Required API keys
Create a `.env` file at project root (`c:\voice-agent\.env`) with:

```env
GROQ_API_KEY=your_groq_key
SARVAM_API_KEY=your_sarvam_key
CEREBRAS_API_KEY=your_cerebras_key
HOST=0.0.0.0
PORT=8805
```

Notes:
- `server/config.py` loads `.env` from the project root.
- The active backend pipeline currently uses **Cerebras** for LLM and **Sarvam** for STT/TTS.
- `GROQ_API_KEY` is still read in config for compatibility, but the current `server/pipeline.py` imports `CerebrasLLMService`.

---

## 2) Install Dependencies

From project root:

```powershell
cd c:\voice-agent
uv sync --project voice-agent
```

This installs all dependencies declared in `voice-agent/pyproject.toml`, including:
- `fastapi`, `uvicorn[standard]`
- `pipecat-ai[...]`
- `python-dotenv`
- `numpy`, `scipy`, `torch`

---

## 3) Project Folder Structure

```text
c:\voice-agent
├── .env                          # Local secrets and runtime config (create this)
├── .gitignore
├── README.md
├── client
│   ├── index.html               # UI (connect/disconnect, status, language selector)
│   ├── agent.js                 # Browser VoiceAgent (WebSocket + mic capture + playback)
│   └── audio-processor.js       # AudioWorklet processor, streams Float32 mic frames
├── server
│   ├── config.py                # Env loading + constants (keys, host/port, model/sample rate)
│   ├── main.py                  # FastAPI app + /health + /ws endpoint
│   ├── pipeline.py              # Pipecat pipeline assembly (transport/STT/LLM/TTS/processors)
│   ├── serializer.py            # Alternate serializer implementation (not currently used)
│   ├── processors
│   │   ├── naturalizer.py       # Post-LLM text cleanup for natural spoken output
│   │   └── pivot_detector.py    # Detect topic pivots, interrupt and re-steer response
│   └── serializers
│       └── raw_pcm.py           # Active serializer for raw PCM audio + JSON control messages
└── voice-agent
    ├── pyproject.toml           # Python project metadata and dependencies
    ├── uv.lock                  # Lockfile
    ├── .python-version
    ├── .gitignore
    └── main.py                  # Minimal placeholder script for package root
```

---

## 4) Architecture (Detailed)

## High-level data flow
1. Browser captures mic audio via `AudioWorklet` (`client/audio-processor.js`).
2. `client/agent.js` converts Float32 -> PCM16 and sends raw bytes over WebSocket.
3. FastAPI WebSocket endpoint (`server/main.py` at `/ws`) starts a Pipecat pipeline.
4. `RawPCMSerializer` deserializes incoming raw PCM to `InputAudioRawFrame`.
5. `SarvamSTTService` transcribes audio to text.
6. User-turn aggregation + stop strategy decide when to trigger LLM response.
7. `CerebrasLLMService` generates response text.
8. `ResponseNaturalizerProcessor` cleans and humanizes text for spoken output.
9. `SarvamTTSService` synthesizes audio.
10. Audio frames are serialized back to binary and played in browser.

## Server internals
- `server/main.py`
  - Initializes FastAPI.
  - Exposes:
    - `GET /health`
    - `WS /ws?lang=hi-IN|en-IN`
  - Per WS connection:
    - Calls `create_pipeline(...)`.
    - Runs pipeline task with `PipelineRunner`.

- `server/pipeline.py`
  - Transport: `FastAPIWebsocketTransport` with `RawPCMSerializer`.
  - Turn control:
    - `TranscriptionUserTurnStartStrategy`
    - `TurnAnalyzerUserTurnStopStrategy(LocalSmartTurnAnalyzerV3)`
  - AI stack:
    - STT: `SarvamSTTService`
    - LLM: `CerebrasLLMService`
    - TTS: `SarvamTTSService`
  - Custom processors:
    - `PivotDetectorProcessor`
    - `ResponseNaturalizerProcessor`
  - Pipeline sequence:
    - `transport.input() -> stt -> user_aggregator -> pivot_detector -> llm -> naturalizer -> tts -> rtvi -> assistant_aggregator -> transport.output()`

## Client internals
- `client/index.html`
  - Renders UI and connects to `ws://localhost:8805/ws`.
  - Shows states: disconnected, listening, thinking, speaking, error.

- `client/agent.js`
  - Handles:
    - mic permission
    - WebSocket lifecycle
    - worklet wiring
    - PCM conversion
    - binary audio playback queue
    - simple mic health check
  - Decodes RTVI-style JSON control messages and drives UI hooks.

---

## 5) How to Run (Exactly as requested)

Use **two terminals**.

## Terminal 1 (Client)
From `client` directory:

```powershell
cd c:\voice-agent\client
python -m http.server 3000
```

Open:
- [http://localhost:3000](http://localhost:3000)

## Terminal 2 (Server)
From `voice-agent` directory (so relative app-dir resolves correctly):

```powershell
cd c:\voice-agent\voice-agent
uv run uvicorn main:app --host 0.0.0.0 --port 8805 --app-dir ..\server
```

Why this works:
- `--app-dir ..\server` points Uvicorn module loading to the `server` folder (where `main.py` lives).
- `uv run` ensures dependencies from `voice-agent/pyproject.toml` are used.

---

## 6) Quick Verification Checklist

1. Server terminal should show startup complete and WebSocket connection logs.
2. In browser, click **Connect**.
3. Status should move to listening.
4. Speak 2-3 seconds.
5. Server should receive frames and process STT/LLM/TTS path.
6. Browser should receive binary audio and play response.

---

## 7) Common Issues and Notes

- If startup works but no response:
  - Verify `.env` keys are valid (`SARVAM_API_KEY`, `CEREBRAS_API_KEY`).
  - Ensure browser microphone permission is granted.
  - Ensure selected language matches speech (`hi-IN` / `en-IN`).
  - Confirm server is really on port `8805` and client connects to `ws://localhost:8805/ws`.

- If dependency errors occur:
  - Re-run:
    - `uv sync --project voice-agent`
  - Then run server again via `uv run ...`, not global Python.

---

## 8) Development Notes

- `server/serializer.py` exists as an alternative serializer reference, but active pipeline uses `server/serializers/raw_pcm.py`.
- `voice-agent/main.py` is currently a placeholder and not part of runtime voice flow.

