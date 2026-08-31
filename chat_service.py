# -*- coding: utf-8 -*-
"""RAG 知识库问答 - FastAPI 服务。

接口：
- POST /api/chat          JSON 问答（返回 answer/references/sources/session_id/trace_id）
- POST /chat/stream       SSE 流式（阶段推送：改写→检索→生成）
- POST /api/ingest        上传文本入库（字段 text / source）
- GET  /api/knowledge     知识库统计（片段数/最近来源）
- GET  /health            健康检查

用法：python chat_service.py  →  http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import RagAgent
from config import LOG_LEVEL
from ingest import ingest_text
from retriever import retrieve

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] [trace=%(trace_id)s] %(message)s")

# ── trace_id 注入日志 ──
class TraceIdFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = getattr(record, "trace_id", "-")
        return True


for h in logging.root.handlers:
    h.addFilter(TraceIdFilter())
logger = logging.getLogger(__name__)

MAX_HISTORY = 10  # 多轮历史上限，防滥用


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    history: List[dict] = Field(default_factory=list)


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200000)
    source: str = Field(default="manual")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("RAG 知识库问答服务启动中...")
    yield
    logger.info("服务已关闭")


app = FastAPI(
    title="RAG 知识库问答 Agent",
    description="企业知识库检索增强问答：文档入库 → 向量检索 → LLM 带引用作答。"
                "支持 SSE 流式、多轮改写、无相关知识兜底，Mock 模式下无需 API Key。",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS：默认放开便于前端联调；生产可在环境变量中收紧（见 .env.example）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse({"error": "参数校验失败", "detail": exc.errors()}, status_code=422)


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception):
    trace_id = new_trace_id()
    logger.exception("[trace=%s] 未捕获异常: %s", trace_id, exc)
    return JSONResponse({"error": "服务内部错误", "trace_id": trace_id}, status_code=500)


_agent: Optional[RagAgent] = None


def get_agent() -> RagAgent:
    global _agent
    if _agent is None:
        _agent = RagAgent()
        logger.info("知识库片段数：%d", _agent.knowledge_size())
    return _agent


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


@app.get("/")
async def index():
    """Web 聊天界面。"""
    import os
    path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    from fastapi.responses import FileResponse
    return FileResponse(path)


@app.get("/health")
async def health():
    return {"status": "ok", "knowledge_chunks": get_agent().knowledge_size()}


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    trace_id = new_trace_id()
    session_id = req.session_id or f"sess-{uuid.uuid4().hex[:12]}"
    history = req.history[-MAX_HISTORY:]
    logger.info("[trace=%s] 收到问题 [session=%s]: %s", trace_id, session_id, req.message[:60])
    try:
        result = get_agent().answer(req.message, history=history)
        result.update({"session_id": session_id, "trace_id": trace_id})
        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("[trace=%s] 处理异常", trace_id)
        return JSONResponse({"error": str(e), "trace_id": trace_id}, status_code=500)


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式：阶段状态 + 最终回答。"""
    trace_id = new_trace_id()
    session_id = req.session_id or f"sess-{uuid.uuid4().hex[:12]}"
    logger.info("[trace=%s] SSE 收到问题 [session=%s]: %s", trace_id, session_id, req.message[:60])

    async def event_generator():
        agent = get_agent()
        history = req.history[-MAX_HISTORY:]
        yield _sse({"type": "status", "content": "正在分析问题..."})

        rewritten = req.message
        if history:
            yield _sse({"type": "status", "content": "正在结合上下文改写问题..."})
            try:
                from retriever import rewrite_query
                rewritten = rewrite_query(req.message, history,
                                          agent.llm if not agent.llm.mock else None)
            except Exception:  # noqa: BLE001
                pass

        yield _sse({"type": "status", "content": "正在检索知识库..."})
        hits = retrieve(rewritten, agent.emb, agent.store, agent.top_k)
        yield _sse({"type": "status", "content": f"检索到 {len(hits)} 个相关片段"})

        yield _sse({"type": "status", "content": "正在生成回答..."})
        result = agent.answer(req.message, history=history)
        result.update({"session_id": session_id, "trace_id": trace_id})
        yield _sse({"type": "response", "data": result})
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Trace-Id": trace_id},
    )


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.post("/api/ingest")
async def api_ingest(req: IngestRequest):
    try:
        n = ingest_text(req.text, source=req.source, emb=get_agent().emb, store=get_agent().store)
        return {"ok": True, "chunks": n, "total": get_agent().knowledge_size()}
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/api/knowledge")
async def knowledge():
    agent = get_agent()
    return {"chunks": agent.knowledge_size()}


if __name__ == "__main__":
    import uvicorn
    logger.info("启动 RAG 服务 http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
