# -*- coding: utf-8 -*-
"""切分模块测试：标题感知、重叠窗口、空输入。"""
from chunker import chunk_document, split_by_paragraphs, split_by_headings


def test_split_by_headings():
    text = "# 标题一\n内容A\n\n# 标题二\n内容B\n"
    sections = split_by_headings(text)
    assert len(sections) == 2
    assert sections[0]["heading"] == "标题一"
    assert "内容A" in sections[0]["body"]


def test_chunk_document_preserves_source_and_heading():
    text = "# 考勤\n弹性工作制，核心时间 10-16 点。\n\n远程办公每月 4 天。"
    chunks = chunk_document(text, source="a.md")
    assert chunks
    assert all(c["source"] == "a.md" for c in chunks)
    assert all(c["heading"] == "考勤" for c in chunks)
    assert chunks[0]["index"] == 0


def test_overlap_keeps_context():
    # 超长单段：固定窗口 + 重叠，两块之间应共享部分字符
    long_para = "字" * 300
    chunks = split_by_paragraphs(long_para, chunk_size=100, overlap=20)
    assert len(chunks) >= 3
    assert chunks[0][-20:] == chunks[1][:20]


def test_short_chunks_dropped():
    chunks = chunk_document("# 标题\n短", source="x.md")
    assert len(chunks) == 0  # 低于 MIN_CHUNK_LEN


def test_empty_document():
    assert chunk_document("", source="e.md") == []
