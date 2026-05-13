# Live Voice Agent

A realtime browser-to-server voice assistant built with:

- A lightweight browser client (`client/`) that captures microphone audio and plays bot audio.
- A FastAPI + Pipecat backend (`server/`) that runs STT -> LLM -> TTS pipeline over WebSocket.
- A UV-managed Python environment (`voice-agent/`) that contains the backend dependencies.
- Production security middleware (`server/middleware/`) for rate limiting and origin validation.

## 🌐 Live Instance

| Service | URL |
|:--------|:----|
| **Backend API** | `https://ministros-voice-agent.bravebay-2f859642.centralindia.azurecontainerapps.io` |
| **Health Check** | [/health](https://ministros-voice-agent.bravebay-2f859642.centralindia.azurecontainerapps.io/health) |
| **Ready Check** | [/ready](https://ministros-voice-agent.bravebay-2f859642.centralindia.azurecontainerapps.io/ready) |
| **WebSocket** | `wss://ministros-voice-agent.bravebay-2f859642.centralindia.azurecontainerapps.io/ws` |
| **Frontend** | `https://ministros-frontend.bravebay-2f859642.centralindia.azurecontainerapps.io` |

> **Note:** The live instance is hosted on Azure Container Apps (Central India region) for low-latency access to Sarvam AI services.

---

## 1) Prerequisites and Requirements

### OS / Runtime
- Windows, macOS, or Linux.
- Python `>=3.12` (defined in `voice-agent/pyproject.toml`).
- Internet access for cloud AI services.

### Python package manager
- [`uv`](https://docs.astral.sh/uv/) (recommended and used by this repo).

### Required API keys
Create a `.env` file at project root with:

```env
GROQ_API_KEY=your_groq_key
SARVAM_API_KEY=your_sarvam_key
CEREBRAS_API_KEY=your_cerebras_key
HOST=0.0.0.0
PORT=8805
```

See `.env.example` for the full list of configurable variables including security settings.

Notes:
- `server/config.py` loads `.env` from the project root.
- The active backend pipeline uses **Cerebras** for LLM and **Sarvam** for STT/TTS.
- `GROQ_API_KEY` is still read in config for compatibility, but the current `server/pipeline.py` imports `CerebrasLLMService`.

---

## 2) Install Dependencies

From project root:

```powershell
cd pipecat-voice-agent
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
pipecat-voice-agent/
├── .env                          # Local secrets and runtime config (create this)
├── .env.example                  # Template with all configurable variables
├── .gitignore
├── Dockerfile                    # Production container build
├── README.md
├── client/
│   ├── index.html               # UI (connect/disconnect, status, language selector)
│   ├── agent.js                 # Browser VoiceAgent (WebSocket + mic capture + playback)
│   └── audio-processor.js       # AudioWorklet processor, streams Float32 mic frames
├── server/
│   ├── config.py                # Env loading + constants (keys, host/port, model/sample rate)
│   ├── main.py                  # FastAPI app + /health + /ready + /ws endpoint
│   ├── pipeline.py              # Pipecat pipeline assembly (transport/STT/LLM/TTS/processors)
│   ├── serializer.py            # Alternate serializer implementation (not currently used)
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── rate_limiter.py      # IP-based connection + message rate limiting
│   │   └── security.py          # Origin validation, message size limits, session timeouts
│   ├── processors/
│   │   ├── naturalizer.py       # Post-LLM text cleanup for natural spoken output
│   │   └── pivot_detector.py    # Detect topic pivots, interrupt and re-steer response
│   └── serializers/
│       └── raw_pcm.py           # Active serializer for raw PCM audio + JSON control messages
└── voice-agent/
    ├── pyproject.toml           # Python project metadata and dependencies
    ├── uv.lock                  # Lockfile
    ├── .python-version
    └── main.py                  # Minimal placeholder script for package root
```

---

## 4) Architecture (Detailed)

### High-level data flow
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

### Latency targets
- **Voice-to-voice:** ~635ms (achieved via tuned VAD/LLM/TTS parameters)
- **VAD endpointing:** 150ms `stop_secs`
- **LLM TTFB:** <100ms (Cerebras 8B)
- **TTS pace:** 1.15x for faster synthesis

### Server internals
- `server/main.py`
  - Initializes FastAPI.
  - Exposes:
    - `GET /health` — returns status + rate limiter stats
    - `GET /ready` — readiness probe
    - `WS /ws?lang=hi-IN|en-IN` — voice agent WebSocket
  - Per WS connection:
    - 3 security gates: origin validation, auth token, rate limiting
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

### Security middleware (`server/middleware/`)
- **Rate Limiter** — IP-based connection and message throughput limiting (in-memory)
- **Security** — Origin header validation, max message size (64KB), session timeouts
- Configurable via environment variables (see `.env.example`)

### Client internals
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

## 5) How to Run

### Option A: Local Development (two terminals)

#### Terminal 1 (Client)
From `client` directory:

```powershell
cd pipecat-voice-agent/client
python -m http.server 3000
```

Open: [http://localhost:3000](http://localhost:3000)

#### Terminal 2 (Server)
From `voice-agent` directory (so relative app-dir resolves correctly):

```powershell
cd pipecat-voice-agent/voice-agent
uv run uvicorn main:app --host 0.0.0.0 --port 8805 --app-dir ../server
```

Why this works:
- `--app-dir ../server` points Uvicorn module loading to the `server` folder (where `main.py` lives).
- `uv run` ensures dependencies from `voice-agent/pyproject.toml` are used.

### Option B: Docker

```bash
# Build
docker build -t ministros-voice-agent .

# Run
docker run -p 8805:8805 --env-file .env ministros-voice-agent
```

### Option C: Connect to Live Instance

Point any WebSocket client to:

```
wss://ministros-voice-agent.bravebay-2f859642.centralindia.azurecontainerapps.io/ws?lang=hi-IN
```

Supported languages: `hi-IN` (Hindi), `en-IN` (English India)

---

## 6) Quick Verification Checklist

1. Server terminal should show startup complete and WebSocket connection logs.
2. In browser, click **Connect**.
3. Status should move to listening.
4. Speak 2-3 seconds.
5. Server should receive frames and process STT/LLM/TTS path.
6. Browser should receive binary audio and play response.

Health check:
```bash
curl https://ministros-voice-agent.bravebay-2f859642.centralindia.azurecontainerapps.io/health
# {"status":"ok","service":"ministros-voice-agent","version":"1.1.0",...}
```

---

## 7) Environment Variables Reference

| Variable | Required | Default | Description |
|:---------|:---------|:--------|:------------|
| `SARVAM_API_KEY` | Yes | — | Sarvam AI API key (STT/TTS) |
| `CEREBRAS_API_KEY` | Yes | — | Cerebras inference API key (LLM) |
| `GROQ_API_KEY` | Yes | — | Groq API key (fallback LLM) |
| `HOST` | No | `0.0.0.0` | Server bind host |
| `PORT` | No | `8805` | Server bind port |
| `CORS_ORIGINS` | No | `*` | Comma-separated CORS origins |
| `ALLOWED_WS_ORIGINS` | No | `*` | WebSocket origin validation |
| `MAX_CONCURRENT_SESSIONS` | No | `10` | Max simultaneous WebSocket sessions |
| `SESSION_TIMEOUT_SECS` | No | `600` | Session timeout in seconds |
| `REQUIRE_AUTH` | No | `false` | Enable HMAC token auth |
| `JWT_SECRET` | No | — | Secret for token verification |

---

## 8) Common Issues and Notes

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

## 9) Development Notes

- `server/serializer.py` exists as an alternative serializer reference, but active pipeline uses `server/serializers/raw_pcm.py`.
- `voice-agent/main.py` is currently a placeholder and not part of runtime voice flow.
- Security middleware is in-memory (no Redis dependency) — suitable for single-instance deployment.

## 10) Deployment

The backend is deployed on **Azure Container Apps** (Central India region).

```bash
# Build and push image
az acr build --registry ministrosacr --image ministros-voice-agent:latest --file Dockerfile .

# Update container app
az containerapp update --name ministros-voice-agent --resource-group ministros-voice-rg --image "ministrosacr.azurecr.io/ministros-voice-agent:latest"
```

See `azure-deploy.sh` for the full deployment script.
