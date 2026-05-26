"""
query_pipeline/query_api.py
─────────────────────────────
WHAT:  FastAPI server exposing the query pipeline as an HTTP REST API
       and a WebSocket endpoint for real-time streaming.

ENDPOINTS:
  POST /query          → blocking (returns full response)
  POST /query/stream   → SSE streaming (real-time token delivery)
  WS   /ws/{session_id} → WebSocket for bi-directional chat
  GET  /health         → health check

WHY FASTAPI:
  - Async-first → handles concurrent restaurant customers
  - Built-in streaming support (StreamingResponse)
  - Auto-generates OpenAPI docs at /docs
  - WebSocket built-in — no extra packages

USAGE:
  pip install fastapi uvicorn
  uvicorn query_api:app --reload --port 8000

  # Then test:
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"message": "What veg dishes do you have?", "session_id": null}'
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.responses import StreamingResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from query_pipeline import query as run_query
from query_session  import get_session, add_turn, clear_expired_sessions


# ── Request/Response models ─────────────────────────────────────
if FASTAPI_AVAILABLE:
    class QueryRequest(BaseModel):
        message:    str
        session_id: str | None = None
        verbose:    bool = False

    class QueryResponse(BaseModel):
        session_id: str
        response:   str
        sources:    list
        intent:     dict
        latency_ms: dict


# ── App setup ───────────────────────────────────────────────────
if FASTAPI_AVAILABLE:
    app = FastAPI(
        title="Restaurant Grand Chennai — AI Query API",
        description="Hybrid RAG query pipeline powered by Qdrant + Cerebras",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # restrict in production
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # ── Health check ─────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "model": "cerebras/llama-3.3-70b"}


    # ── Blocking query endpoint ───────────────────────────────────
    @app.post("/query", response_model=QueryResponse)
    async def query_endpoint(req: QueryRequest):
        """
        Blocking query — returns full response after generation.
        Good for: simple integrations, testing, batch processing.
        """
        result = run_query(
            user_input = req.message,
            session_id = req.session_id,
            stream     = False,
            verbose    = req.verbose,
        )
        return QueryResponse(
            session_id = result["session_id"],
            response   = result["response"],
            sources    = result["sources"],
            intent     = result["intent"],
            latency_ms = result["latency_ms"],
        )


    # ── Streaming query endpoint (SSE) ────────────────────────────
    @app.post("/query/stream")
    async def query_stream_endpoint(req: QueryRequest):
        """
        Streaming query — returns Server-Sent Events (SSE) stream.
        Good for: web frontends, real-time chat UI.

        Frontend usage:
          const es = new EventSource('/query/stream');
          es.onmessage = (e) => {
            const data = JSON.parse(e.data);
            if (data.type === 'chunk') appendText(data.text);
            if (data.type === 'done')  showSources(data.sources);
          };
        """
        result = run_query(
            user_input = req.message,
            session_id = req.session_id,
            stream     = True,
            verbose    = False,
        )

        session_id = result["session_id"]
        generator  = result["response"]
        sources    = result["sources"]
        intent     = result["intent"]

        async def event_stream():
            # Stream metadata first
            yield f"data: {json.dumps({'type': 'meta', 'session_id': session_id, 'intent': intent})}\n\n"

            # Stream response chunks
            full_response = ""
            for chunk in generator:
                full_response += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"

            # Update session history after streaming
            session = get_session(session_id)
            if session:
                add_turn(session, req.message, full_response)

            # Final message with sources
            yield f"data: {json.dumps({'type': 'done', 'sources': sources})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control":               "no-cache",
                "X-Accel-Buffering":           "no",   # disable nginx buffering
                "Access-Control-Allow-Origin": "*",
            }
        )


    # ── WebSocket endpoint ────────────────────────────────────────
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """
        WebSocket for bi-directional streaming chat.
        Client sends: {"message": "..."}
        Server sends: {"type": "chunk"|"done"|"meta", ...}

        WHY WebSocket over SSE for chat:
        SSE is one-way (server → client).
        WebSocket is bi-directional — the client can send while
        the server is still streaming a previous response.
        Better for a real-time restaurant chat widget.
        """
        await websocket.accept()

        try:
            while True:
                # Receive customer message
                data = await websocket.receive_json()
                user_message = data.get("message", "").strip()
                if not user_message:
                    continue

                result = run_query(
                    user_input = user_message,
                    session_id = session_id,
                    stream     = True,
                    verbose    = False,
                )

                # Send metadata
                await websocket.send_json({
                    "type":       "meta",
                    "intent":     result["intent"],
                    "session_id": result["session_id"],
                })

                # Stream response
                full_response = ""
                for chunk in result["response"]:
                    full_response += chunk
                    await websocket.send_json({"type": "chunk", "text": chunk})

                # Update history
                session = get_session(session_id)
                if session:
                    add_turn(session, user_message, full_response)

                # Send completion with sources
                await websocket.send_json({
                    "type":    "done",
                    "sources": result["sources"],
                })

        except WebSocketDisconnect:
            pass   # client disconnected — normal


    # ── Session cleanup (run periodically) ───────────────────────
    @app.on_event("startup")
    async def startup():
        # Could also set up a background task for session cleanup
        print("🍽️  Restaurant Grand Chennai Query API started")
        print("   Docs: http://localhost:8000/docs")


# ── Direct run ──────────────────────────────────────────────────
if __name__ == "__main__":
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run:")
        print("  pip install fastapi uvicorn")
        sys.exit(1)

    import uvicorn
    uvicorn.run(
        "query_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
