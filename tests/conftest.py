# -*- coding: utf-8 -*-
"""pytest 公共夹具：Mock embedding + Memory 向量库，全程离线。"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from embeddings import MockEmbeddingBackend  # noqa: E402
from vector_store import MemoryVectorStore  # noqa: E402
from agent import LLMClient, RagAgent  # noqa: E402


@pytest.fixture()
def emb():
    return MockEmbeddingBackend(dim=64)


@pytest.fixture()
def store(tmp_path):
    return MemoryVectorStore(persist_path=None)


@pytest.fixture()
def llm():
    """Mock LLM：确定性返回，不依赖 API Key。"""
    client = LLMClient()
    client.mock = True
    return client


@pytest.fixture()
def agent(emb, store, llm):
    return RagAgent(emb=emb, store=store, llm=llm, top_k=3)
