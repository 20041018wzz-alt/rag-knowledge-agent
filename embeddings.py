# -*- coding: utf-8 -*-
"""Embedding 抽象：api / local / mock 三实现 + 工厂。

设计（面试点）：
- 上层只依赖 EmbeddingBackend 接口，换后端不改业务代码（依赖倒置）。
- mock 后端：字符 bigram 哈希 → 归一化向量，离线演示/测试可确定；
  语义效果一般，仅作降级，生产用 api / local。
"""
from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

from config import (EMBEDDING_API_KEY, EMBEDDING_API_MODEL, EMBEDDING_API_URL,
                    EMBEDDING_BACKEND, EMBEDDING_DIM, LLM_TIMEOUT)

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    dim: int = 0

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """把一批文本转成向量。"""


def l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class MockEmbeddingBackend(EmbeddingBackend):
    """字符 unigram + bigram 哈希向量（离线演示/测试，确定性）。

    相比纯 bigram，加入字级特征可缓解"短查询 vs 长片段"的稀疏问题；
    语义效果有限，仅作降级，生产用 api / local。
    """

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            grams = _char_features(text)
            for g in grams:
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            vectors.append(l2_normalize(vec))
        return vectors


class ApiEmbeddingBackend(EmbeddingBackend):
    """OpenAI 兼容 /embeddings 接口（SiliconFlow 等提供 BGE）。"""

    def __init__(self, url: str = EMBEDDING_API_URL, api_key: str = EMBEDDING_API_KEY,
                 model: str = EMBEDDING_API_MODEL, timeout: int = LLM_TIMEOUT):
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("EMBEDDING_API_KEY 未设置")
        payload = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self.url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        # OpenAI 兼容返回: {"data": [{"embedding": [...]}, ...]}
        data_list = sorted(data["data"], key=lambda x: x.get("index", 0))
        vectors = [item["embedding"] for item in data_list]
        self.dim = len(vectors[0]) if vectors else 0
        return [l2_normalize(v) for v in vectors]


class LocalEmbeddingBackend(EmbeddingBackend):
    """本地 sentence-transformers BGE（中文效果好，需 torch；懒加载）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("未安装 sentence-transformers") from e
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        vectors = model.encode(texts, normalize_embeddings=True).tolist()
        return [list(map(float, v)) for v in vectors]


def create_embedding_backend() -> EmbeddingBackend:
    """工厂：按配置创建后端，失败时降级 mock。"""
    try:
        if EMBEDDING_BACKEND == "api":
            return ApiEmbeddingBackend()
        if EMBEDDING_BACKEND == "local":
            return LocalEmbeddingBackend()
    except Exception as e:  # noqa: BLE001
        logger.warning("Embedding 后端初始化失败，降级 mock: %s", e)
    return MockEmbeddingBackend()


def _char_features(text: str) -> List[str]:
    """字级特征：单字 + 相邻两字组合（去掉空白）。"""
    text = text.lower()
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return []
    grams = list(chars)
    grams += [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    return grams
