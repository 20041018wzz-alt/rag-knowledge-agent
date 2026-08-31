# -*- coding: utf-8 -*-
"""FastAPI 接口测试（TestClient，全程离线 Mock）。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from chat_service import app, get_agent  # noqa: E402
from ingest import ingest_text  # noqa: E402
from vector_store import MemoryVectorStore  # noqa: E402

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_knowledge():
    """每个用例前隔离知识库（内存库，不污染真实 data/knowledge_index.json）。"""
    agent = get_agent()
    agent.store = MemoryVectorStore(persist_path=None)  # 测试隔离
    ingest_text("# 报销\n金额超过 2000 元的报销需部门负责人二次审批，财务审核周期为 3 个工作日。",
                source="m.md", emb=agent.emb, store=agent.store)
    yield


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_chat_hit_returns_structure():
    r = client.post("/api/chat", json={"message": "报销超过多少钱需要二次审批？", "history": []})
    assert r.status_code == 200
    body = r.json()
    assert body["has_knowledge"] is True
    assert body["answer"]
    assert body["session_id"] and body["trace_id"]
    assert isinstance(body["sources"], list)


def test_chat_out_of_knowledge_fallback():
    r = client.post("/api/chat", json={"message": "公司食堂几点开门？", "history": []})
    assert r.status_code == 200
    assert r.json()["has_knowledge"] is False
    assert "没有" in r.json()["answer"]


def test_chat_validation_error():
    r = client.post("/api/chat", json={"message": ""})  # 空消息
    assert r.status_code == 422


def test_chat_long_message_rejected():
    r = client.post("/api/chat", json={"message": "x" * 2001})
    assert r.status_code == 422


def test_ingest_endpoint():
    agent = get_agent()
    before = agent.knowledge_size()
    r = client.post("/api/ingest", json={"text": "# 新政策\n远程办公调整为每月 6 天。", "source": "policy.md"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["total"] == before + body["chunks"]


def test_ingest_empty_rejected():
    r = client.post("/api/ingest", json={"text": ""})
    assert r.status_code == 422


def test_knowledge_count():
    r = client.get("/api/knowledge")
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1


def test_sse_stream():
    r = client.post("/chat/stream", json={"message": "报销超过多少钱需要二次审批？"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    body = r.text
    assert "data: " in body
    assert "[DONE]" in body
    assert "response" in body
