# -*- coding: utf-8 -*-
"""RAG 知识库问答 Agent - 配置（环境变量优先，全部可离线默认）。"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── LLM（对话生成）──
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # 留空 = Mock 模式
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))

# ── Embedding（三选一：mock / api / local）──
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "mock")
# api 后端参数（OpenAI 兼容 /embeddings，如 SiliconFlow 的 BGE）
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.siliconflow.cn/v1/embeddings")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_API_MODEL = os.getenv("EMBEDDING_API_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))  # mock 向量维度

# ── 切分参数 ──
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))      # 每块目标字符数
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50")) # 相邻块重叠，保持上下文连贯
MIN_CHUNK_LEN = int(os.getenv("MIN_CHUNK_LEN", "20")) # 过短片段丢弃

# ── 检索 ──
TOP_K = int(os.getenv("TOP_K", "4"))            # 召回片段数
# 相关性阈值：低于则判"无相关知识"。该默认值按 mock embedding 校准；
# 切换 api/local 后端后建议上调（如 BGE 余弦相似度相关片段通常 > 0.5）。
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.25"))

# ── 存储 ──
DATA_DIR = os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
KNOWLEDGE_INDEX = os.path.join(DATA_DIR, "knowledge_index.json")  # Memory 向量库持久化文件

# ── 向量库（选型：pgvector 生产 / Chroma 本地 / Memory 兜底）──
# auto：按 pgvector → Chroma → Memory 优先级自动降级；也可强制单个后端。
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "auto")
# pgvector 连接串；留空则跳过 pgvector 直接降级（保证离线可跑）
DATABASE_URL = os.getenv("DATABASE_URL", "")
PGVECTOR_TABLE = os.getenv("PGVECTOR_TABLE", "knowledge_chunks")
# 建表维度：留 0 = 由首次入库的向量维度自动决定（推荐）。
# 显式指定（如 1024 对应 BGE-M3）可让 HNSW 索引在启动时就建好。
PGVECTOR_DIM = int(os.getenv("PGVECTOR_DIM", "0"))

# ── 日志 ──
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
