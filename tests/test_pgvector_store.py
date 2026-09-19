# -*- coding: utf-8 -*-
"""pgvector 后端单元测试（离线）。

用「假连接池」替换 psycopg 的真实连接：断言的是 PgVectorStore 发出的 SQL 与
结果映射逻辑，不需要启动 PostgreSQL。真实连通性由 docker compose 冒烟验证，
这样 CI 仍然全离线、跨三平台稳定。

覆盖点：
- 幂等建表 / 扩展 / meta 表 SQL
- 首次入库自动定维 + 写 meta；维度不一致必须显式报错（防静默故障）
- 检索 SQL 用余弦距离 `<=>` 且分数换算为 1 - 距离
- numpy 数组入参归一化为 list[float]
- 工厂降级链：pgvector 不可用 → 下一级
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vector_store  # noqa: E402
from vector_store import MemoryVectorStore, PgVectorStore, create_vector_store  # noqa: E402

pgvector_driver = pytest.importorskip(
    "psycopg", reason="未安装 psycopg，跳过 pgvector 后端单测"
)


# ── 假连接池：记录所有 SQL，并按注入的脚本返回结果 ──────────────────────
class FakeCursor:
    def __init__(self, conn):
        self._conn = conn
        self._rows = []

    def execute(self, sql, params=None):
        self._conn.executed.append((sql, params))
        self._rows = self._conn.script.pop(0) if self._conn.script else []

    def executemany(self, sql, rows):
        self._conn.executed.append((sql, list(rows)))
        self._rows = []

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeConn:
    def __init__(self):
        self.executed = []
        self.script = []
        self.commits = 0

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.commits += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakePool:
    def __init__(self, conn):
        self._conn = conn
        self.opened = False
        self.closed = False

    def open(self, wait=False, timeout=None):
        self.opened = True

    def connection(self):
        return self._conn

    def close(self):
        self.closed = True


@pytest.fixture()
def store_and_conn(monkeypatch):
    """构造一个用假连接池的 PgVectorStore。"""
    conn = FakeConn()
    pool = FakePool(conn)
    monkeypatch.setattr(vector_store, "PgVectorStore", PgVectorStore)

    def fake_init(self, dsn, table="knowledge_chunks", dim=0, pool_min=1, pool_max=4):
        self._table = self._safe_ident(table)
        self._configured_dim = int(dim or 0)
        import threading
        self._lock = threading.RLock()
        self._pool = pool

    monkeypatch.setattr(PgVectorStore, "__init__", fake_init)
    store = PgVectorStore("postgresql://x/y", table="knowledge_chunks")
    store._init_schema()
    return store, conn, pool


def _sql(conn):
    return "\n".join(s for s, _ in conn.executed)


# ── 建表 / 初始化 ──────────────────────────────────────────────────────
def test_init_creates_extension_and_tables(store_and_conn):
    _, conn, _ = store_and_conn
    sql = _sql(conn)
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE IF NOT EXISTS knowledge_chunks" in sql
    assert "CREATE TABLE IF NOT EXISTS knowledge_chunks_meta" in sql
    # 自愈补列
    assert "ADD COLUMN IF NOT EXISTS provenance" in sql


def test_init_creates_hnsw_index_when_dim_configured(monkeypatch):
    conn = FakeConn()
    pool = FakePool(conn)

    def fake_init(self, dsn, table="knowledge_chunks", dim=1024, pool_min=1, pool_max=4):
        self._table = table
        self._configured_dim = 1024
        import threading
        self._lock = threading.RLock()
        self._pool = pool

    monkeypatch.setattr(PgVectorStore, "__init__", fake_init)
    PgVectorStore("postgresql://x/y")._init_schema()
    sql = _sql(conn)
    assert "USING hnsw (embedding vector_cosine_ops)" in sql
    assert "vector(1024)" in sql


def test_hnsw_failure_does_not_break_init(store_and_conn, monkeypatch):
    """pgvector < 0.5 不支持 HNSW 时，只告警不阻断初始化。"""
    store, conn, _ = store_and_conn
    conn.executed.clear()

    def boom(sql, params=None):
        conn.executed.append((sql, params))
        if "hnsw" in sql:
            raise RuntimeError("type \"hnsw\" does not exist")
        return FakeCursor(conn)

    monkeypatch.setattr(conn, "cursor", lambda: type("C", (), {
        "execute": lambda self, sql, params=None: boom(sql, params),
        "fetchone": lambda self: None,
        "fetchall": lambda self: [],
        "__enter__": lambda self: self,
        "__exit__": lambda self, *e: False,
    })())

    store._create_index(None, 256)  # 不应抛出


# ── 维度守卫 ───────────────────────────────────────────────────────────
def test_add_sets_dimension_on_first_ingest(store_and_conn):
    store, conn, _ = store_and_conn
    conn.script = [[{"value": None}]]  # _current_dim 查 meta → 空
    conn.executed.clear()
    store.add(["a"], [[1.0, 0.0, 0.0]], ["你好"], [{"source": "t.md"}])

    sql = _sql(conn)
    assert "ALTER COLUMN embedding TYPE vector(3)" in sql
    assert "INSERT INTO knowledge_chunks_meta" in sql
    assert store._configured_dim == 3


def test_add_rejects_dimension_mismatch(store_and_conn):
    """换 embedding 后端导致维度变了 → 必须显式报错，不能静默写坏索引。"""
    store, conn, _ = store_and_conn
    conn.script = [[{"value": "256"}]]  # 库中已固定 256 维
    with pytest.raises(ValueError) as ei:
        store.add(["a"], [[1.0, 0.0]], ["你好"], [{"source": "t.md"}])
    assert "维度不一致" in str(ei.value)
    assert "256" in str(ei.value)


def test_add_uses_configured_dim_without_querying_meta(store_and_conn, monkeypatch):
    store, conn, _ = store_and_conn
    store._configured_dim = 4
    conn.executed.clear()
    store.add(["a"], [[1.0, 0.0, 0.0, 0.0]], ["x"], [{}])
    assert "SELECT value FROM knowledge_chunks_meta" not in _sql(conn)
    assert store._configured_dim == 4  # 配置维度优先，不被覆盖


def test_add_upserts_with_on_conflict(store_and_conn):
    store, conn, _ = store_and_conn
    store._configured_dim = 2
    conn.executed.clear()
    store.add(["a"], [[1.0, 0.0]], ["x"], [{"source": "s.md"}])
    sql, rows = conn.executed[-1]
    assert "ON CONFLICT (id) DO UPDATE" in sql
    # 入参 tuple 结构：(id, vector, text, metadata_json, provenance)
    assert rows[0][0] == "a"
    assert rows[0][1] == [1.0, 0.0]
    assert rows[0][4] == "s.md"


def test_add_empty_is_noop(store_and_conn):
    store, conn, _ = store_and_conn
    conn.executed.clear()
    store.add([], [], [], [])
    assert conn.executed == []


# ── 检索 ───────────────────────────────────────────────────────────────
def test_search_uses_cosine_operator_and_maps_score(store_and_conn):
    store, conn, _ = store_and_conn
    conn.script = [[
        {"id": "1", "text": "甲", "metadata": {"source": "a.md"}, "distance": 0.2},
        {"id": "2", "text": "乙", "metadata": {}, "distance": 0.75},
    ]]
    results = store.search([1.0, 0.0], top_k=2)

    sql, params = conn.executed[-1]
    assert "embedding <=> %s::vector" in sql
    assert "ORDER BY distance" in sql
    assert params[1] == 2

    # 距离 → 分数（越高越相关），与 Memory / Chroma 后端口径一致
    assert results[0]["score"] == 0.8
    assert results[1]["score"] == 0.25
    assert results[0]["id"] == "1"
    assert results[0]["metadata"] == {"source": "a.md"}
    assert results[1]["metadata"] == {}


def test_search_handles_null_distance(store_and_conn):
    store, conn, _ = store_and_conn
    conn.script = [[{"id": "1", "text": "甲", "metadata": {}, "distance": None}]]
    assert store.search([1.0], top_k=1)[0]["score"] == 0.0


def test_count_and_clear(store_and_conn):
    store, conn, _ = store_and_conn
    conn.script = [[{"n": 7}]]
    assert store.count() == 7

    conn.executed.clear()
    store.clear()
    assert "DELETE FROM knowledge_chunks" in _sql(conn)
    assert conn.commits >= 1


# ── 工厂降级链 ─────────────────────────────────────────────────────────
def test_factory_falls_back_to_memory_without_database_url(monkeypatch):
    import config
    monkeypatch.setattr(vector_store, "VECTOR_STORE_BACKEND", "auto")
    monkeypatch.setattr(config, "DATABASE_URL", "")
    monkeypatch.setattr(vector_store, "_try_chroma",
                        lambda: (_ for _ in ()).throw(RuntimeError("未安装 chromadb")))
    assert isinstance(create_vector_store(), MemoryVectorStore)


def test_factory_uses_pgvector_when_available(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(vector_store, "VECTOR_STORE_BACKEND", "auto")
    monkeypatch.setattr(vector_store, "_try_pgvector", lambda: sentinel)
    assert create_vector_store() is sentinel


def test_factory_pgvector_failure_falls_through_to_chroma(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(vector_store, "VECTOR_STORE_BACKEND", "auto")
    monkeypatch.setattr(vector_store, "_try_pgvector",
                        lambda: (_ for _ in ()).throw(RuntimeError("连接失败")))
    monkeypatch.setattr(vector_store, "_try_chroma", lambda: sentinel)
    assert create_vector_store() is sentinel


def test_factory_forced_memory(monkeypatch):
    monkeypatch.setattr(vector_store, "VECTOR_STORE_BACKEND", "memory")
    assert isinstance(create_vector_store(), MemoryVectorStore)


def test_factory_forced_backend_unavailable_still_starts(monkeypatch):
    """强制指定 pgvector 但连不上 → 仍降级 Memory，保证服务能起来。"""
    monkeypatch.setattr(vector_store, "VECTOR_STORE_BACKEND", "pgvector")
    monkeypatch.setattr(vector_store, "_try_pgvector",
                        lambda: (_ for _ in ()).throw(RuntimeError("连不上")))
    assert isinstance(create_vector_store(), MemoryVectorStore)


# ── 入参归一化 / 表名防注入 ────────────────────────────────────────────
def test_as_list_accepts_numpy_like():
    np = pytest.importorskip("numpy", reason="未安装 numpy，跳过 ndarray 归一化用例")

    class FakeArray:
        def tolist(self):
            return [1, 2, 3]

    assert vector_store._as_list(FakeArray()) == [1.0, 2.0, 3.0]
    assert vector_store._as_list((1, 2)) == [1.0, 2.0]
    assert all(isinstance(x, float) for x in vector_store._as_list([1, 2]))


def test_safe_ident_blocks_injection():
    assert PgVectorStore._safe_ident("knowledge_chunks") == "knowledge_chunks"
    # 表名无法参数化，必须白名单过滤：注入串被剥成安全标识符后再校验
    assert PgVectorStore._safe_ident("chunks; DROP TABLE users--") == "chunksDROPTABLEusers"
    with pytest.raises(ValueError):
        PgVectorStore._safe_ident("")
    with pytest.raises(ValueError):
        PgVectorStore._safe_ident("; --")
    with pytest.raises(ValueError):
        PgVectorStore._safe_ident("1bad")


def test_pgvector_requires_dsn():
    with pytest.raises(ValueError):
        PgVectorStore("")
