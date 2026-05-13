import asyncio
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pipecat.pipeline.runner import PipelineRunner

from pipeline import create_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Voice Agent server starting up...")
    yield
    logger.info("Voice Agent server shutting down.")


app = FastAPI(title="Live Voice Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    session_id = str(uuid.uuid4())

    await websocket.accept()
    logger.info(
        "Client connected | session_id={} client={}",
        session_id,
        websocket.client,
    )

    # Accept ?lang=hi-IN or ?lang=en-IN
    language = websocket.query_params.get("lang", "hi-IN")

    try:
        transport, task = await create_pipeline(websocket, language=language)

        @transport.event_handler("on_client_connected")
        async def on_connected(t, ws):
            logger.info("Pipeline running | session_id={}", session_id)

        @transport.event_handler("on_client_disconnected")
        async def on_disconnected(t, ws):
            logger.info(
                "Client disconnected — stopping pipeline | session_id={}", session_id
            )
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected cleanly | session_id={}", session_id)
    except Exception as e:
        logger.error(
            "Pipeline error | session_id={} err={}",
            session_id,
            e,
            exc_info=True,
        )
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    from config import HOST, PORT

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=True,
        log_level="info",
    )
