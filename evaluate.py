# -*- coding: utf-8 -*-
"""RAG 效果评测：基于 golden set 计算检索召回率与回答准确率。

用法：
    python evaluate.py                              # 使用默认 examples/golden_set.json
    python evaluate.py --golden path/to/golden.json
    python evaluate.py --top-k 6                    # 调整召回深度

golden set 格式（JSON 列表）：
    [{"question": "报销超过多少钱需要二次审批？",
      "expected_sources": ["company_manual.md"],     # 期望命中的来源（召回率）
      "expected_keyword": "2000"}]                    # 期望答案包含的关键词（准确率，需真实 LLM）

指标说明：
- 召回率（Recall@k）：检索结果是否包含期望来源（命中数 / 总条数）
- 准确率（Accuracy）：Mock 模式不计算；配置真实 LLM 后按答案是否含关键词统计
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List

from agent import RagAgent
from retriever import retrieve

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_GOLDEN = os.path.join(os.path.dirname(__file__), "examples", "golden_set.json")


def load_golden(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError("golden set 必须是非空 JSON 数组")
    return data


def evaluate(agent: RagAgent, golden: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    hit = 0
    total = len(golden)
    accuracy_hit, accuracy_total = 0, 0
    cases: List[Dict[str, Any]] = []

    for item in golden:
        question = item["question"]
        expected = set(item.get("expected_sources", []))
        results = retrieve(question, agent.emb, agent.store, top_k=top_k)
        hit_sources = {r["metadata"].get("source", "") for r in results}
        recall_ok = bool(expected & hit_sources) if expected else bool(results)
        hit += 1 if recall_ok else 0

        case = {
            "question": question,
            "expected_sources": sorted(expected),
            "hit_sources": sorted(hit_sources),
            "recall_ok": recall_ok,
        }

        keyword = item.get("expected_keyword")
        if keyword and not agent.llm.mock:
            answer = agent.answer(question).get("answer", "")
            ok = keyword in answer
            accuracy_hit += 1 if ok else 0
            accuracy_total += 1
            case["accuracy_ok"] = ok

        cases.append(case)

    return {
        "total": total,
        "recall@k": round(hit / total, 4) if total else 0.0,
        "hit": hit,
        "accuracy": round(accuracy_hit / accuracy_total, 4) if accuracy_total else None,
        "accuracy_hit": accuracy_hit,
        "accuracy_total": accuracy_total,
        "top_k": top_k,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG golden set 评测")
    parser.add_argument("--golden", default=DEFAULT_GOLDEN, help="golden set JSON 路径")
    parser.add_argument("--top-k", type=int, default=4, help="召回深度")
    args = parser.parse_args()

    agent = RagAgent()
    golden = load_golden(args.golden)
    print(f"加载 golden set：{len(golden)} 条；知识库片段数：{agent.knowledge_size()}")
    if agent.knowledge_size() == 0:
        print("⚠️  知识库为空，请先运行：python ingest.py --dir examples")
        sys.exit(1)

    result = evaluate(agent, golden, top_k=args.top_k)
    print(f"\n召回率 Recall@{args.top_k}: {result['hit']}/{result['total']} = {result['recall@k']:.2%}")
    if result["accuracy"] is not None:
        print(f"回答准确率: {result['accuracy_hit']}/{result['accuracy_total']} = {result['accuracy']:.2%}（真实 LLM 模式）")
    else:
        print("回答准确率: 未统计（Mock 模式；配置 LLM_API_KEY 后自动统计）")

    print("\n--- 逐条结果 ---")
    for c in result["cases"]:
        flag = "✓" if c["recall_ok"] else "✗"
        print(f"{flag} {c['question']}\n    期望: {c['expected_sources']} 命中: {c['hit_sources']}")


if __name__ == "__main__":
    main()
