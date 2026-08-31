# -*- coding: utf-8 -*-
"""文档入库管线：解析 → 切分 → 向量化 → 写入向量库。

用法：
    python ingest.py --dir examples        # 批量入库目录下 md/txt
    python ingest.py --file examples/a.md  # 单个文件
    python ingest.py --clear               # 清空知识库
"""
from __future__ import annotations

import argparse
import logging
import os
import uuid
from typing import List

from chunker import chunk_document
from embeddings import create_embedding_backend
from vector_store import create_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".md", ".txt", ".markdown"}


def read_document(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in SUPPORTED_EXT:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if ext == ".pdf":
        try:
            import pdfplumber
        except ImportError as e:
            raise RuntimeError("解析 PDF 需要安装 pdfplumber") from e
        with pdfplumber.open(path) as pdf:
            return "\n\n".join(page.extract_text() or "" for page in pdf.pages)
    raise ValueError(f"不支持的格式: {ext}")


def ingest_text(text: str, source: str, emb, store) -> int:
    """单篇文本入库，返回入库片段数。"""
    chunks = chunk_document(text, source=source)
    if not chunks:
        logger.warning("文档 %s 切分后为空，跳过", source)
        return 0

    texts = [c["text"] for c in chunks]
    vectors = emb.embed(texts)
    ids = [f"{uuid.uuid4().hex[:12]}" for _ in chunks]
    metadatas = [{"source": c["source"], "heading": c["heading"], "index": c["index"]}
                 for c in chunks]
    store.add(ids, vectors, texts, metadatas)
    logger.info("已入库 %s：%d 个片段", source, len(chunks))
    return len(chunks)


def ingest_file(path: str, emb, store) -> int:
    text = read_document(path)
    return ingest_text(text, source=os.path.basename(path), emb=emb, store=store)


def ingest_directory(dir_path: str, emb, store) -> int:
    total = 0
    files = [f for f in sorted(os.listdir(dir_path))
             if os.path.splitext(f)[1].lower() in SUPPORTED_EXT]
    for fname in files:
        total += ingest_file(os.path.join(dir_path, fname), emb, store)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 知识库入库")
    parser.add_argument("--dir", help="批量入库目录")
    parser.add_argument("--file", help="入库单个文件")
    parser.add_argument("--clear", action="store_true", help="清空知识库")
    args = parser.parse_args()

    emb = create_embedding_backend()
    store = create_vector_store()

    if args.clear:
        store.clear()
        logger.info("知识库已清空")
        return

    if args.file:
        ingest_file(args.file, emb, store)
    elif args.dir:
        ingest_directory(args.dir, emb, store)
    else:
        parser.print_help()
        return

    logger.info("当前知识库片段数：%d", store.count())


if __name__ == "__main__":
    main()
