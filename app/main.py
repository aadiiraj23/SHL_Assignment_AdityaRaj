import os
import time
import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.agent import SHLAgent
from app.catalog import get_catalog_store
from app.llm_client import get_llm_client
from app.models import ChatRequest, ChatResponse


def _configure_logging() -> None:
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
        "CRITICAL": 50,
    }
    level = level_map.get(level_name, 20)
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_logging()
logger = structlog.get_logger()

app = FastAPI(title="SHL Assessment Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(
            "request_failed",
            method=request.method,
            path=request.url.path,
            status_code=500,
            duration_ms=duration_ms,
            error=str(exc),
        )
        raise
    duration_ms = round((time.time() - start_time) * 1000, 2)
    logger.info(
        "request_completed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=duration_ms,
    )
    return response


@app.get("/health")
def health() -> dict:
    try:
        catalog = get_catalog_store()
        size = catalog.size()
        agent_ready = True
    except Exception:
        size = 0
        agent_ready = False
    return {
        "status": "ok",
        "catalog_size": size,
        "agent_ready": agent_ready,
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        catalog = get_catalog_store()
        llm = get_llm_client()
        agent = SHLAgent(catalog=catalog, llm=llm)
        response = await agent.run(request)
        return response
    except Exception as exc:
        logger.error("chat_request_failed", error=str(exc))
        error_response = ChatResponse(
            reply="I encountered an internal server error. Please try again.",
            recommendations=[],
            end_of_conversation=False,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=error_response.model_dump(),
        )