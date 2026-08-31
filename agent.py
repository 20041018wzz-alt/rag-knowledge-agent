# -*- coding: utf-8 -*-
"""RAG 问答 Agent：检索 → 拼上下文 → LLM 带引用生成 → 引用解析。

面试点：
- 引用溯源：要求 LLM 用 [n] 标注依据，返回 references 供前端展示来源，防幻觉且可验证。
- 兜底：检索为空或低于相关性阈值 → 明确回复"知识库中没有相关信息"，绝不编造。
- Mock 模式：无 API Key 时 LLM 后端退化为 Mock（返回检索到的片段），全链路可离线演示、可测试。
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Dict, List, Optional

from config import (LLM_API_KEY, LLM_BASE_URL, LLM_MAX_RETRIES, LLM_MODEL,
                    LLM_TEMPERATURE, LLM_TIMEOUT)
from embeddings import EmbeddingBackend, create_embedding_backend
from retriever import is_relevant, retrieve, rewrite_query
from vector_store import VectorStore, create_vector_store

logger = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """你是一个严谨的企业知识库问答助手。规则：
1. 只能依据下面给出的【知识片段】回答，不得编造知识库中没有的内容。
2. 回答中引用知识片段时，在句末标注来源编号，如 [1][2]。
3. 如果知识片段不足以回答，明确说"知识库中没有相关信息"，不要猜测。
4. 用中文回答，简洁有条理。

【知识片段】
{context}

【对话历史】
{history}"""

NO_KNOWLEDGE_REPLY = "抱歉，知识库中没有与这个问题相关的信息，我无法回答。"
MOCK_REPLY = ("（Mock 模式）检索到以下相关片段，供联调/演示：\n{items}\n"
              "配置 LLM_API_KEY 后由模型生成正式回答。")


class LLMClient:
    """OpenAI 兼容 Chat Completions 客户端（含重试）；无 Key 时走 Mock。"""

    def __init__(self):
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL
        self.mock = not bool(self.api_key)

    def chat(self, messages: List[Dict[str, str]], temperature: float = LLM_TEMPERATURE) -> str:
        if self.mock:
            return self._mock_chat(messages)
        import httpx
        payload = {"model": self.model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"
        last_err: Optional[Exception] = None
        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            for attempt in range(LLM_MAX_RETRIES):
                try:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                    choices = data.get("choices") or []
                    if choices:
                        return choices[0].get("message", {}).get("content", "")
                    raise ValueError("API 响应缺少 choices")
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if attempt < LLM_MAX_RETRIES - 1:
                        time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM 调用失败: {last_err}")

    def _mock_chat(self, messages: List[Dict[str, str]]) -> str:
        """确定性 Mock：从消息里找【知识片段】并返回前两条，模拟"检索到了什么"。"""
        context = ""
        for m in messages:
            if m.get("role") == "system" and "【知识片段】" in m.get("content", ""):
                context = m["content"].split("【知识片段】")[-1]
                break
        items = [line.strip() for line in context.splitlines()
                 if line.strip().startswith("[")][:2]
        return MOCK_REPLY.format(items="\n".join(items) if items else "（无片段）")


class RagAgent:
    def __init__(self, emb: Optional[EmbeddingBackend] = None,
                 store: Optional[VectorStore] = None,
                 llm: Optional[LLMClient] = None,
                 top_k: int = 4):
        self.emb = emb or create_embedding_backend()
        self.store = store or create_vector_store()
        self.llm = llm or LLMClient()
        self.top_k = top_k

    def answer(self, question: str,
               history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """问答主流程，返回 {answer, references, sources, has_knowledge}。"""
        rewritten = rewrite_query(question, history, self.llm if not self.llm.mock else None)
        results = retrieve(rewritten, self.emb, self.store, self.top_k)

        if not is_relevant(results):
            return {
                "answer": NO_KNOWLEDGE_REPLY,
                "references": [],
                "sources": [],
                "has_knowledge": False,
            }

        context = "\n".join(
            f"[{i + 1}] {r['text']}" for i, r in enumerate(results)
        )
        history_text = "\n".join(
            f"{'用户' if m.get('role') == 'user' else 'AI'}: {m.get('content', '')}"
            for m in (history or [])[-4:]
        ) or "（无）"

        messages = [
            {"role": "system", "content": SYSTEM_TEMPLATE.format(
                context=context, history=history_text)},
            {"role": "user", "content": question},
        ]

        try:
            reply = self.llm.chat(messages)
        except Exception as e:  # noqa: BLE001
            logger.error("生成回答失败: %s", e)
            reply = "抱歉，生成回答时出现错误，请稍后重试。"

        references = self._parse_references(reply, results)
        sources = [{"source": r["metadata"].get("source", ""),
                    "heading": r["metadata"].get("heading", ""),
                    "score": r["score"]} for r in results]
        return {
            "answer": reply,
            "references": references,
            "sources": sources,
            "has_knowledge": True,
        }

    @staticmethod
    def _parse_references(reply: str, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """解析回答中的 [n] 引用，映射回片段文本。"""
        refs: List[Dict[str, str]] = []
        for num in sorted({int(x) for x in re.findall(r"\[(\d+)\]", reply)}):
            idx = num - 1
            if 0 <= idx < len(results):
                refs.append({
                    "index": num,
                    "text": results[idx]["text"],
                    "source": results[idx]["metadata"].get("source", ""),
                })
        return refs

    def knowledge_size(self) -> int:
        return self.store.count()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="RAG 知识库问答（命令行）")
    parser.add_argument("--ask", required=True, help="要问的问题")
    args = parser.parse_args()
    agent = RagAgent()
    result = agent.answer(args.ask)
    print(result["answer"])
    if result["references"]:
        print("\n--- 引用来源 ---")
        for ref in result["references"]:
            print(f"[{ref['index']}] ({ref['source']}) {ref['text'][:80]}")


if __name__ == "__main__":
    main()
