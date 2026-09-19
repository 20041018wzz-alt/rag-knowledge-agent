# -*- coding: utf-8 -*-
"""向量库抽象：PgVector / Chroma / Memory 三实现 + 工厂（按优先级自动降级）。

选型依据（面试点）：
- **pgvector**：默认生产后端。知识片段与业务数据同库同事务，天然复用
  PostgreSQL 的备份 / 权限 / 连接池，百万级向量配 HNSW 索引可满足低延迟检索；
  团队不必为一个检索能力额外引入独立存储组件。
- **Chroma**：本地开发 / 单机 Demo 用（文件级持久化，零依赖启动）。
- **Memory**：纯 Python 兜底，保证「无任何外部依赖也能跑通全链路 + 离线测试」。
- 数据继续增长或要求极致检索吞吐时，接口不变可直接替换为 Milvus。
- 上层只依赖 VectorStore 接口；安装缺失 / 连接失败 → 自动降级下一级，
  与「多智能体客服系统」的 MySQL/Redis → 内存降级一脉相承。
- 相似度统一为「分数越高越相关」：Memory 用余弦（向量已 L2 归一化，等价点积），
  pgvector / Chroma 把距离换算为 `1 - 距离`。
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
from config import KNOWLEDGE_INDEX, PGVECTOR_DIM, PGVECTOR_TABLE, VECTOR_STORE_BACKEND

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


class PgVectorStore(VectorStore):
    """pgvector 后端（默认生产实现）。

    设计要点（面试常追问的工程细节）：
    - **维度守卫**：pgvector 的 `vector(n)` 维度建表时固定，而 embedding 后端可换
      （mock 256 / BGE-M3 1024）。因此把维度落在 meta 表里：首次入库自动定维，
      之后维度不一致直接报错，避免「查得出结果但相似度全错」的静默故障。
    - **连接池**：psycopg_pool 连接池，FastAPI 多线程并发安全，不每请求建连。
    - **索引两段式**：先建表入库，再按实际维度 `CREATE INDEX ... USING hnsw
      (embedding vector_cosine_ops)`（HNSW 需 pgvector ≥ 0.5.0）；索引建不上
      只告警不阻断，退化为顺序扫描，功能仍可用。
    - **幂等建表**：`CREATE TABLE IF NOT EXISTS` + 自愈补建 meta 表 / 索引，
      容器与数据库可任意顺序启动，不需要人工执行迁移。
    - **检索 SQL**：`ORDER BY embedding <=> %s::vector LIMIT k`，`<=>` 为余弦距离，
      换算成分数 `1 - 距离` 与其余后端口径一致。
    """

    def __init__(self, dsn: str, table: str = PGVECTOR_TABLE,
                 dim: int = PGVECTOR_DIM, pool_min: int = 1, pool_max: int = 4) -> None:
        if not dsn:
            raise ValueError("DATABASE_URL 未设置")
        try:
            import psycopg
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool
        except ImportError as e:
            raise RuntimeError("未安装 psycopg[binary] / psycopg_pool") from e

        self._psycopg = psycopg
        self._table = self._safe_ident(table)
        self._configured_dim = int(dim or 0)
        self._lock = threading.RLock()
        self._pool = ConnectionPool(
            conninfo=dsn, min_size=pool_min, max_size=pool_max,
            kwargs={"row_factory": dict_row}, open=False,
            timeout=10.0, name="pgvector-store",
        )
        self._pool.open(wait=True, timeout=10.0)
        self._init_schema()

    # ── 内部：SQL 与幂等初始化 ────────────────────────────────────────────
    @staticmethod
    def _safe_ident(name: str) -> str:
        """表名白名单校验（表名无法参数化，只能拼进 SQL，必须防注入）。"""
        cleaned = "".join(c for c in (name or "") if c.isalnum() or c == "_")
        if not cleaned or cleaned[0].isdigit():
            raise ValueError(f"非法表名: {name!r}")
        return cleaned

    def _init_schema(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._table} ("
                    "  id        TEXT PRIMARY KEY,"
                    "  embedding vector,"
                    "  text      TEXT NOT NULL,"
                    "  metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,"
                    "  provenance TEXT NOT NULL,"
                    "  created_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                    ")"
                )
                # 自愈：早期版本建的表没有 provenance 列
                cur.execute(
                    f"ALTER TABLE {self._table} "
                    "ADD COLUMN IF NOT EXISTS provenance TEXT NOT NULL DEFAULT ''"
                )
                cur.execute(
                    f"CREATE TABLE IF NOT EXISTS {self._table}_meta ("
                    "  key   TEXT PRIMARY KEY,"
                    "  value TEXT NOT NULL"
                    ")"
                )
                dim = self._configured_dim
                if dim:
                    cur.execute(
                        f"ALTER TABLE {self._table} "
                        f"ALTER COLUMN embedding TYPE vector({dim})"
                    )
                    self._create_index(cur, dim)
            conn.commit()

    def _create_index(self, cur, dim: int) -> None:
        """HNSW 余弦索引；pgvector < 0.5 不支持时仅告警。"""
        try:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {self._table}_embedding_hnsw "
                f"ON {self._table} USING hnsw (embedding vector_cosine_ops) "
                f"WHERE embedding IS NOT NULL"
            )
            logger.info("pgvector HNSW 索引就绪（dim=%d）", dim)
        except Exception as e:  # noqa: BLE001
            logger.warning("HNSW 索引创建失败（pgvector 可能 < 0.5.0），退化为顺序扫描: %s", e)

    def _current_dim(self, cur) -> int:
        if self._configured_dim:
            return self._configured_dim
        cur.execute(f"SELECT value FROM {self._table}_meta WHERE key = 'dim'")
        row = cur.fetchone()
        if not row:
            return 0
        value = row["value"] if isinstance(row, dict) else row[0]
        # 值为 NULL / 空串时视为「尚未定维」，交由本次入库决定
        return int(value) if value not in (None, "") else 0

    # ── VectorStore 接口 ────────────────────────────────────────────────
    def add(self, ids: List[str], vectors: List[List[float]],
            texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        if not ids:
            return
        dim = len(vectors[0]) if vectors else 0
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                current = self._current_dim(cur)
                if current and dim and current != dim:
                    raise ValueError(
                        f"向量维度不一致：库中已固定 dim={current}，本次 dim={dim}。"
                        f"换 embedding 后端后请先 clear() 重新入库"
                        f"（或设置 VECTOR_STORE_BACKEND=memory 做隔离实验）。"
                    )
                if not current and dim:
                    cur.execute(
                        f"ALTER TABLE {self._table} "
                        f"ALTER COLUMN embedding TYPE vector({dim})"
                    )
                    cur.execute(
                        f"INSERT INTO {self._table}_meta (key, value) VALUES ('dim', %s) "
                        f"ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                        (str(dim),),
                    )
                    self._configured_dim = dim
                    self._create_index(cur, dim)

                rows = [
                    (vid, _as_list(vectors[i]), texts[i],
                     json.dumps(metadatas[i] if i < len(metadatas) else {},
                                ensure_ascii=False),
                     (metadatas[i] or {}).get("source", "") if i < len(metadatas) else "")
                    for i, vid in enumerate(ids)
                ]
                cur.executemany(
                    f"INSERT INTO {self._table} (id, embedding, text, metadata, provenance) "
                    f"VALUES (%s, %s::vector, %s, %s::jsonb, %s) "
                    f"ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                    f"  text = EXCLUDED.text, metadata = EXCLUDED.metadata, "
                    f"  provenance = EXCLUDED.provenance",
                    rows,
                )
            conn.commit()

    def search(self, vector: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, text, metadata, embedding <=> %s::vector AS distance "
                    f"FROM {self._table} WHERE embedding IS NOT NULL "
                    f"ORDER BY distance LIMIT %s",
                    (_as_list(vector), int(top_k)),
                )
                rows = cur.fetchall()
        results = []
        for row in rows:
            distance = row["distance"]
            results.append({
                "id": row["id"],
                "score": round(1.0 - float(distance), 4) if distance is not None else 0.0,
                "text": row["text"],
                "metadata": row["metadata"] or {},
            })
        return results

    def count(self) -> int:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT count(*) AS n FROM {self._table}")
                row = cur.fetchone()
        return int(row["n"] if isinstance(row, dict) else row[0])

    def clear(self) -> None:
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._table}")
            conn.commit()

    def close(self) -> None:
        try:
            self._pool.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("关闭 pgvector 连接池失败: %s", e)


def _as_list(vec: Any) -> List[float]:
    """兼容 numpy.ndarray / list / tuple → list[float]（psycopg 的 vector 适配需要）。"""
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    return [float(x) for x in vec]


def create_vector_store() -> VectorStore:
    """工厂：按优先级 pgvector → Chroma → Memory 自动降级。

    可用 VECTOR_STORE_BACKEND 强制指定（pgvector / chroma / memory），
    指定后端不可用时仍打印告警并继续降级，保证服务永远能起来。
    """
    forced = (VECTOR_STORE_BACKEND or "auto").lower()
    if forced == "memory":
        logger.info("向量库后端：memory（按 VECTOR_STORE_BACKEND 强制指定）")
        return MemoryVectorStore()

    candidates = {"pgvector": _try_pgvector, "chroma": _try_chroma}
    attempts = list(candidates.items()) if forced == "auto" else \
        ([(forced, candidates[forced])] if forced in candidates else [])

    for name, factory in attempts:
        try:
            store = factory()
            logger.info("向量库后端：%s", name)
            return store
        except Exception as e:  # noqa: BLE001
            logger.warning("%s 向量库不可用，降级下一级: %s", name, e)

    logger.info("向量库后端：memory（纯 Python 兜底）")
    return MemoryVectorStore()


def _try_pgvector() -> VectorStore:
    import config
    if not config.DATABASE_URL:
        raise RuntimeError("未配置 DATABASE_URL")
    return PgVectorStore(config.DATABASE_URL)


def _try_chroma() -> VectorStore:
    return ChromaVectorStore()
