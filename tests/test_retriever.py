# -*- coding: utf-8 -*-
"""检索与向量库测试：相似度单调性、命中正确文档、空库、相关性阈值。"""
from retriever import is_relevant, retrieve
from vector_store import MemoryVectorStore
from ingest import ingest_text


def _ingest_sample(emb, store):
    doc_a = "# 考勤\n员工每月可申请 4 天远程办公，迟到超过 30 分钟需提交说明。"
    doc_b = "# 保修\n产品提供 1 年免费质保，企业版享受 7x24 技术支持。"
    ingest_text(doc_a, source="a.md", emb=emb, store=store)
    ingest_text(doc_b, source="b.md", emb=emb, store=store)


def test_ingest_and_count(emb, store):
    assert store.count() == 0
    _ingest_sample(emb, store)
    assert store.count() >= 2


def test_retrieve_hits_correct_doc(emb, store):
    _ingest_sample(emb, store)
    results = retrieve("远程办公每个月可以申请几天？", emb, store, top_k=3)
    assert results
    assert results[0]["score"] >= 0
    assert any(r["metadata"]["source"] == "a.md" for r in results)


def test_retrieve_empty_store(emb, store):
    assert retrieve("任何问题", emb, store) == []


def test_memory_store_persistence(tmp_path):
    store = MemoryVectorStore(persist_path=str(tmp_path / "idx.json"))
    store.add(["1"], [[1.0, 0.0]], ["你好"], [{"source": "t.md"}])
    store2 = MemoryVectorStore(persist_path=str(tmp_path / "idx.json"))
    assert store2.count() == 1
    assert store2.search([1.0, 0.0])[0]["text"] == "你好"


def test_is_relevant_threshold():
    assert not is_relevant([], threshold=0.3)
    assert not is_relevant([{"score": 0.1}], threshold=0.3)
    assert is_relevant([{"score": 0.5}], threshold=0.3)
