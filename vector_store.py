# -*- coding: utf-8 -*-
"""向量库抽象：MemoryVectorStore（纯 Python 兜底）+ ChromaVectorStore（可选）。

设计（面试点）：
- 上层只依赖 VectorStore 接口；Chroma 装不上/没装 → 自动降级内存实现，
  与「多智能体客服系统」的 MySQL/Redis → 内存降级一脉相承。
- 相似度用余弦（向量已 L2 归一化，等价于点积），分数越高越相关。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from embeddings import EmbeddingBackend
from config import KNOWLEDGE_INDEX

logger = logging.getLogger(__name__)


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: List[str], vectors: List[List[float]],
            texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        ...

    @abstractmethod
    def search(self, vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        """返回 [{id, score, text, metadata}]，score 越高越相关。"""
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def clear(self) -> None:
        ...


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class MemoryVectorStore(VectorStore):
    """纯 Python 余弦相似度实现（默认后端，可持久化到 JSON，线程安全）。"""

    def __init__(self, persist_path: Optional[str] = KNOWLEDGE_INDEX):
        self.persist_path = persist_path
        self._items: List[Dict[str, Any]] = []  # [{id, vector, text, metadata}]
        self._lock = threading.RLock()  # FastAPI 多线程并发读写安全
        self._load()

    def add(self, ids, vectors, texts, metadatas) -> None:
        with self._lock:
            for i, vid in enumerate(ids):
                self._items.append({
                    "id": vid,
                    "vector": vectors[i],
                    "text": texts[i],
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })
            self._save()

    def search(self, vector, top_k=4):
        with self._lock:
            scored = [(_dot(vector, item["vector"]), item) for item in self._items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": item["id"],
                "score": round(score, 4),
                "text": item["text"],
                "metadata": item["metadata"],
            }
            for score, item in scored[:top_k]
        ]

    def count(self) -> int:
        with self._lock:
            return len(self._items)

    def clear(self) -> None:
        with self._lock:
            self._items = []
            self._save()

    def _save(self) -> None:
        if not self.persist_path:
            return
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(self._items, f, ensure_ascii=False)
        except Exception as e:  # noqa: BLE001
            logger.warning("向量库持久化失败: %s", e)

    def _load(self) -> None:
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                self._items = json.load(f)
        except Exception as e:  # noqa: BLE001
            logger.warning("向量库加载失败，以空库启动: %s", e)
            self._items = []


class ChromaVectorStore(VectorStore):
    """可选 Chroma 后端（本地文件、HNSW 索引，数据量大时更优）。"""

    def __init__(self, persist_dir: str = "data/chroma"):
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError("未安装 chromadb") from e
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="knowledge", metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, vectors, texts, metadatas) -> None:
        self._collection.add(
            ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas
        )

    def search(self, vector, top_k=4):
        res = self._collection.query(query_embeddings=[vector], n_results=top_k)
        results = []
        for i, doc_id in enumerate(res.get("ids", [[]])[0]):
            results.append({
                "id": doc_id,
                "score": round(float(res["distances"][0][i]), 4) if res.get("distances") else 0.0,
                "text": res["documents"][0][i] if res.get("documents") else "",
                "metadata": (res["metadatas"][0][i] or {}) if res.get("metadatas") else {},
            })
        # Chroma 返回距离（越小越近），统一转成"分数越高越相关"
        for r in results:
            r["score"] = round(1.0 - r["score"], 4) if r["score"] is not None else 0.0
        return results

    def count(self) -> int:
        return self._collection.count()

    def clear(self) -> None:
        self._collection.delete(where={})


def create_vector_store() -> VectorStore:
    """工厂：优先 Chroma，失败降级 Memory。"""
    try:
        store = ChromaVectorStore()
        logger.info("使用 Chroma 向量库")
        return store
    except Exception as e:  # noqa: BLE001
        logger.warning("Chroma 不可用，降级 MemoryVectorStore: %s", e)
        return MemoryVectorStore()
