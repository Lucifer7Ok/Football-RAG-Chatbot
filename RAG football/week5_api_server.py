"""
====================================================================
WEEK 5: PRODUCTION-READY MULTI-MODAL RAG API SERVER  v2.2.0
====================================================================
Root fix: 503 on first /api/chat call
→ Engine chưa load xong khi request đầu tiên tới (race condition)
→ Giải pháp: startup_event + readiness flag + retry-after header
"""

import os
import sys
import gc
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

sys.stdout.reconfigure(encoding="utf-8")

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import torch
from week4_rag_engine import IFABHybridRAGEngine

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ifab_api")

# ============================================================
# CONFIG
# ============================================================
MAX_CONCURRENT_AI_TASKS = 2
AI_TIMEOUT_SECONDS      = 90

# ============================================================
# APP STATE
# ============================================================
class AppState:
    rag_engine:    Optional[IFABHybridRAGEngine] = None
    startup_error: Optional[str]  = None
    load_time:     float          = 0.0
    semaphore:     Optional[asyncio.Semaphore] = None

    # ── Readiness flag ──────────────────────────────────────
    # Event được set() SAU KHI engine load xong.
    # Tất cả request /api/chat phải await event này trước.
    ready_event: asyncio.Event = None

state = AppState()

# ============================================================
# LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Khởi tạo trong async context — bắt buộc
    state.semaphore   = asyncio.Semaphore(MAX_CONCURRENT_AI_TASKS)
    state.ready_event = asyncio.Event()   # ← chưa set, tức là "chưa ready"

    logger.info("⏳ [STARTUP] Loading RAG Engine (non-blocking)...")
    t0 = time.time()

    try:
        # ── Chạy blocking init trong thread riêng ──────────
        # Event loop vẫn nhận health-check requests trong lúc này
        state.rag_engine   = await asyncio.to_thread(IFABHybridRAGEngine)
        state.load_time    = round(time.time() - t0, 2)
        state.startup_error = None

        logger.info(f"✅ [STARTUP] Engine ready in {state.load_time}s")

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            logger.info(f"🚀 GPU : {p.name}")
            logger.info(f"🧠 VRAM: {round(p.total_memory / 1024**3, 2)} GB")
        else:
            logger.warning("⚠️  CUDA unavailable — running on CPU")

    except Exception as e:
        state.startup_error = str(e)
        state.rag_engine    = None
        logger.exception(f"❌ [STARTUP] Engine init FAILED: {e}")
        # Vẫn set event để request không bị treo mãi mãi
        # → sẽ nhận 503 với thông báo lỗi rõ ràng thay vì timeout

    finally:
        # Dù thành công hay thất bại → unblock tất cả request đang chờ
        state.ready_event.set()
        logger.info("🔓 [STARTUP] Readiness gate opened.")

    yield   # ── server running ──────────────────────────────

    # ── SHUTDOWN ────────────────────────────────────────────
    logger.info("🧹 [SHUTDOWN] Cleaning up...")
    state.ready_event.clear()   # Block request mới trong lúc shutdown

    try:
        del state.rag_engine
        state.rag_engine = None
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    logger.info("✅ [SHUTDOWN] Done.")


# ============================================================
# APP
# ============================================================
app = FastAPI(
    title="IFAB Football Laws RAG API",
    description="Production-ready Multi-modal RAG API",
    version="2.2.0",
    lifespan=lifespan,
    redirect_slashes=False,     # Fix 404 trailing-slash
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error — {request.method} {request.url}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ============================================================
# SCHEMAS
# ============================================================
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)


# ============================================================
# HELPER: chờ engine sẵn sàng (dùng trong mọi AI endpoint)
# ============================================================
READINESS_WAIT_SECONDS = 120   # Tối đa chờ bao lâu

async def require_engine_ready():
    """
    Await readiness event với timeout.
    - Nếu engine đang load  → client chờ (không bị 503 ngay)
    - Nếu engine load xong  → pass-through ngay lập tức
    - Nếu engine load thất bại → trả 503 kèm lý do
    - Nếu chờ quá lâu       → trả 503 timeout
    """
    try:
        await asyncio.wait_for(
            asyncio.shield(state.ready_event.wait()),
            timeout=READINESS_WAIT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Engine not ready after {READINESS_WAIT_SECONDS}s. Try again later.",
            headers={"Retry-After": "30"},
        )

    if state.rag_engine is None:
        detail = "RAG Engine failed to initialize."
        if state.startup_error:
            detail += f" Reason: {state.startup_error}"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            headers={"Retry-After": "60"},
        )


# ============================================================
# HEALTH CHECK  GET /
# ============================================================
@app.get("/", tags=["Health"], summary="Liveness + readiness status")
def health_check():
    gpu_enabled = torch.cuda.is_available()
    gpu_mem     = 0.0
    if gpu_enabled:
        gpu_mem = round(torch.cuda.memory_allocated() / 1024**3, 2)

    is_ready = (
        state.ready_event is not None
        and state.ready_event.is_set()
        and state.rag_engine is not None
    )

    return {
        "status":                    "ready" if is_ready else "starting",
        "rag_engine_loaded":         state.rag_engine is not None,
        "startup_error":             state.startup_error,
        "load_time_seconds":         state.load_time,
        "device":                    "GPU" if gpu_enabled else "CPU",
        "gpu_memory_allocated_gb":   gpu_mem,
        "max_concurrent_ai_tasks":   MAX_CONCURRENT_AI_TASKS,
    }


# ============================================================
# READINESS PROBE  GET /api/chat/ready
# ============================================================
@app.get(
    "/api/chat/ready",
    tags=["Health"],
    summary="Readiness probe — poll này trước khi gọi /api/chat",
)
async def readiness_probe():
    """
    Trả 200 khi engine đã load xong, 503 nếu chưa.
    Client / load-balancer dùng endpoint này để biết
    khi nào server thực sự sẵn sàng nhận AI request.
    """
    if state.ready_event is None or not state.ready_event.is_set():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine is still loading. Retry in a few seconds.",
            headers={"Retry-After": "5"},
        )

    if state.rag_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Engine failed: {state.startup_error}",
        )

    return {"ready": True, "load_time_seconds": state.load_time}


# ============================================================
# CHAT  POST /api/chat
# ============================================================
@app.post("/api/chat", tags=["AI"], summary="Ask a football law question")
async def chat_with_bot(request: ChatRequest):

    # ── Chờ engine sẵn sàng (fix race condition 503) ───────
    await require_engine_ready()

    clean_query = request.query.strip()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty.",
        )

    logger.info(f"📩 Query: {clean_query!r}")

    try:
        async with state.semaphore:
            t0 = time.time()

            response_data = await asyncio.wait_for(
                asyncio.to_thread(
                    state.rag_engine.generate_answer,
                    clean_query,
                ),
                timeout=AI_TIMEOUT_SECONDS,
            )

            logger.info(f"✅ Response in {round(time.time() - t0, 2)}s")

        # Cleanup sau khi release semaphore
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response_data

    except asyncio.TimeoutError:
        logger.error(f"⏰ Timeout ({AI_TIMEOUT_SECONDS}s): {clean_query!r}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"AI processing timed out after {AI_TIMEOUT_SECONDS}s.",
        )
    except Exception as e:
        logger.exception(f"❌ Inference error: {clean_query!r}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


# ============================================================
# ENTRYPOINT
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "week5_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # Bắt buộc False với GPU
        workers=1,      # Bắt buộc 1 với GPU
        log_level="info",
        access_log=True,
    )