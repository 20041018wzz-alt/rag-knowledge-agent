# -*- coding: utf-8 -*-
"""检索模块：多轮改写 → 向量检索 → 相关性判定。

面试点：
- Query Rewrite：多轮对话里"它的保修期呢"缺主语，直接检索必失败；
  先用 LLM 结合历史改写成完整问题再检索（无 LLM 时原样返回）。
- 相关性阈值：召回不等于相关，低于阈值判定"知识库无相关信息"，交给 Agent 兜底，
  避免 LLM 基于不相关片段硬答产生幻觉。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from config import RELEVANCE_THRESHOLD, TOP_K
from embeddings import EmbeddingBackend
from vector_store import VectorStore

logger = logging.getLogger(__name__)

REWRITE_PROMPT = """你负责把用户的多轮追问改写成独立、完整的检索问题。
规则：结合对话历史补全指代（它/这个/上一条等），只输出改写后的问题本身，不要解释。

对话历史：
{history}

当前问题：{question}
改写后的问题："""


def rewrite_query(question: str, history: Optional[List[Dict[str, str]]] = None,
                  llm: Any = None) -> str:
    """多轮追问改写；无 LLM 或单轮对话时原样返回。"""
    if llm is None or not history:
        return question

    history_text = "\n".join(
        f"{'用户' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')}"
        for m in history[-4:]
    )
    try:
        reply = llm.chat([
            {"role": "system", "content": REWRITE_PROMPT.format(
                history=history_text, question=question)},
        ])
        rewritten = (reply or "").strip().strip("\"'")
        return rewritten if rewritten else question
    except Exception as e:  # noqa: BLE001
        logger.warning("改写失败，使用原问题: %s", e)
        return question


def retrieve(query: str, emb: EmbeddingBackend, store: VectorStore,
             top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """检索 top_k 个相关片段。"""
    if store.count() == 0:
        return []
    vector = emb.embed([query])[0]
    results = store.search(vector, top_k=top_k)
    return results


def is_relevant(results: List[Dict[str, Any]], threshold: float = RELEVANCE_THRESHOLD) -> bool:
    """最高分低于阈值 → 视为无相关知识。"""
    if not results:
        return False
    return results[0]["score"] >= threshold
