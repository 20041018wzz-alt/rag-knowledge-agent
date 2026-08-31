# -*- coding: utf-8 -*-
"""Agent 测试：无知识兜底、引用解析、Mock 回答、多轮改写。"""
from ingest import ingest_text
from agent import RagAgent
from retriever import rewrite_query


def test_answer_no_knowledge(agent):
    result = agent.answer("公司食堂几点开门？")
    assert result["has_knowledge"] is False
    assert "没有" in result["answer"] or "无法" in result["answer"]


def test_answer_with_knowledge_returns_reply(agent, emb, store):
    ingest_text("# 报销\n超过 2000 元的报销需部门负责人二次审批。",
                source="m.md", emb=emb, store=store)
    result = agent.answer("报销超过多少钱需要二次审批？")
    assert result["has_knowledge"] is True
    assert result["answer"]  # Mock 模式也应有回复
    assert result["sources"]


def test_parse_references():
    results = [
        {"text": "片段A", "metadata": {"source": "a.md"}},
        {"text": "片段B", "metadata": {"source": "b.md"}},
    ]
    refs = RagAgent._parse_references("答案是A[1]，也是B[2][1]", results)
    indexes = {r["index"] for r in refs}
    assert indexes == {1, 2}
    assert refs[0]["source"] == "a.md"


def test_rewrite_query_no_llm_passthrough():
    assert rewrite_query("它的价格呢？", history=[{"role": "user", "content": "产品怎么部署？"}], llm=None) == "它的价格呢？"


def test_rewrite_query_with_mock_llm(llm):
    # Mock LLM 不真正改写，但接口不抛异常
    out = rewrite_query("它的价格呢？",
                        history=[{"role": "user", "content": "支持哪些部署方式？"}],
                        llm=llm)
    assert isinstance(out, str) and out


def test_answer_with_history(agent, emb, store):
    ingest_text("# 计费\n企业版每个坐席 199 元每月，基础版每个坐席 99 元每月，均含 10GB 知识库容量。",
                source="faq.md", emb=emb, store=store)
    result = agent.answer("企业版每个坐席每月多少钱？", history=[])
    assert result["has_knowledge"] is True
