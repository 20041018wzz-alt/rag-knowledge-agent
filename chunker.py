# -*- coding: utf-8 -*-
"""文档切分模块。

为什么切分重要（面试点）：
- 向量检索的粒度由切分决定：块太大 → 语义混杂、召回不准；块太小 → 上下文不完整。
- 本实现做三级策略：按标题（Markdown #）保留章节语义 → 按段落聚合到目标长度 → 超长段落用重叠窗口切，避免切断句子上下文。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_LEN


def split_by_headings(text: str) -> List[Dict[str, str]]:
    """按 Markdown 标题把文档切成一节节（保留标题作为 heading 元数据）。"""
    pattern = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return [{"heading": "", "body": text.strip()}]

    sections: List[Dict[str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections.append({"heading": heading, "body": body})
    return [s for s in sections if s["body"]]


def split_by_paragraphs(body: str, chunk_size: int = CHUNK_SIZE,
                        overlap: int = CHUNK_OVERLAP) -> List[str]:
    """按段落聚合到目标长度；超长段落用固定窗口 + 重叠切。"""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            # 超长段落：固定窗口 + 重叠
            step = chunk_size - overlap
            for i in range(0, len(para), step):
                chunk = para[i:i + chunk_size]
                if len(chunk) >= MIN_CHUNK_LEN:
                    chunks.append(chunk)
                if i + chunk_size >= len(para):
                    break
            continue

        if len(current) + len(para) + 1 <= chunk_size:
            current = f"{current}\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            # 保留上一块尾部 overlap 字符，维持跨块上下文
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n{para}".strip() if tail else para

    if current and len(current) >= MIN_CHUNK_LEN:
        chunks.append(current)
    return chunks


def chunk_document(text: str, source: str = "") -> List[Dict[str, Any]]:
    """文档 → 切分片段列表。

    返回: [{text, source, heading, index}]
    """
    chunks: List[Dict[str, Any]] = []
    idx = 0
    for section in split_by_headings(text):
        heading = section["heading"]
        for piece in split_by_paragraphs(section["body"]):
            if len(piece) < MIN_CHUNK_LEN:
                continue
            chunks.append({
                "text": piece,
                "source": source,
                "heading": heading,
                "index": idx,
            })
            idx += 1
    return chunks
