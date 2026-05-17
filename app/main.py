import os

# Force deep learning frameworks to operate in a low-overhead, single-core mode
# This completely prevents multi-threading spikes from crashing Render's 512MB RAM container
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import time
import threading
import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# ── Global singletons ─────────────────────────────────────────────────────────
_catalog = None
_llm = None
_agent = None
_ready = False


def _load_in_background():
    global _catalog, _llm, _agent, _ready
    import traceback

    try:
        from app.catalog import get_catalog_store
        logger.info("loading_catalog_start")
        _catalog = get_catalog_store()
        logger.info("catalog_loaded", size=_catalog.size())
    except Exception as e:
        logger.error("catalog_load_failed", error=str(e), trace=traceback.format_exc())
        _catalog = None

    try:
        from app.llm_client import get_llm_client
        logger.info("loading_llm_start")
        _llm = get_llm_client()
        logger.info("llm_loaded")
    except Exception as e:
        logger.error("llm_load_failed", error=str(e), trace=traceback.format_exc())
        _llm = None

    if _catalog and _llm:
        try:
            from app.agent import SHLAgent
            _agent = SHLAgent(catalog=_catalog, llm=_llm)
            logger.info("agent_ready", catalog_size=_catalog.size())
        except Exception as e:
            logger.error("agent_init_failed", error=str(e), trace=traceback.format_exc())
            _agent = None
    else:
        logger.error("agent_unavailable", catalog_ok=_catalog is not None, llm_ok=_llm is not None)
        _agent = None

    _ready = True
    logger.info("background_loading_complete", agent_ready=_agent is not None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background loading immediately — don't block port binding
    t = threading.Thread(target=_load_in_background, daemon=True)
    t.start()
    logger.info("background_loading_started")
    yield
    logger.info("shutdown")


app = FastAPI(
    title="SHL Assessment Recommender API",
    version="1.0.0",
    lifespan=lifespan
)

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


@app.api_route("/health", methods=["GET", "HEAD"])
async def health(request: Request):
    return {
        "status": "ok",
        "catalog_size": _catalog.size() if _catalog else 0,
        "agent_ready": _agent is not None,
        "loading": not _ready,
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if _agent is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "reply": "Service is still loading, please try again in a moment.",
                "recommendations": [],
                "end_of_conversation": False
            }
        )
    try:
        response = await _agent.run(request)
        return response
    except Exception as exc:
        logger.error("chat_request_failed", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "reply": "I encountered an internal error. Please try again.",
                "recommendations": [],
                "end_of_conversation": False
            }
        )