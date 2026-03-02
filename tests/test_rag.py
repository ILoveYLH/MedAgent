"""
RAG 知识库测试

测试知识库构建和检索功能。
使用临时目录避免污染正式数据库。
"""

import os
import tempfile
from pathlib import Path

import pytest

from app.rag.knowledge_base import (
    _split_text,
    build_knowledge_base,
    get_collection_info,
    query_knowledge,
)


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def temp_docs_dir(tmp_path):
    """创建包含示例医学文档的临时目录"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # 创建示例文档
    (docs_dir / "lung_cancer_test.md").write_text(
        "# 肺癌\n\n肺癌是最常见的恶性肿瘤之一。CT影像表现为肺部实性结节或肿块，边缘可见毛刺征和分叶征。"
        "早期肺癌直径通常小于3厘米。需要与肺结核球和炎性假瘤进行鉴别诊断。",
        encoding="utf-8",
    )
    (docs_dir / "pneumonia_test.md").write_text(
        "# 肺炎\n\n肺炎是肺部感染性疾病。CT表现为肺叶或肺段的实变影，可见空气支气管征。"
        "磨玻璃影提示轻度肺炎。细菌性肺炎多呈大叶性实变，病毒性肺炎多为双侧弥漫磨玻璃影。",
        encoding="utf-8",
    )

    return str(docs_dir)


@pytest.fixture
def temp_chroma_dir(tmp_path, monkeypatch):
    """创建临时 ChromaDB 目录，并修改配置"""
    chroma_dir = tmp_path / "chroma_test"
    chroma_dir.mkdir()

    # 修改配置中的 chroma_db_path 和 collection_name
    from app.config import settings
    monkeypatch.setattr(settings, "chroma_db_path", str(chroma_dir))
    monkeypatch.setattr(settings, "chroma_collection_name", "test_medical_knowledge")

    return str(chroma_dir)


# ─────────────────────────────────────────
# 文本分块测试
# ─────────────────────────────────────────

class TestSplitText:
    """测试文本分块函数"""

    def test_short_text_no_split(self):
        """短文本不应被分割"""
        text = "这是一段短文本。"
        chunks = _split_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self):
        """长文本应被分割为多个块"""
        text = "A" * 1000
        chunks = _split_text(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        """每个文本块的长度不应超过 chunk_size"""
        text = "X" * 2000
        chunk_size = 300
        chunks = _split_text(text, chunk_size=chunk_size, overlap=30)
        for chunk in chunks:
            assert len(chunk) <= chunk_size, f"文本块长度超过限制: {len(chunk)}"

    def test_empty_text(self):
        """空文本应返回空列表"""
        chunks = _split_text("", chunk_size=500)
        assert chunks == []


# ─────────────────────────────────────────
# 知识库构建测试
# ─────────────────────────────────────────

class TestBuildKnowledgeBase:
    """测试知识库构建函数"""

    def test_build_returns_count(self, temp_docs_dir, temp_chroma_dir):
        """构建函数应返回存入的文本块数量"""
        count = build_knowledge_base(docs_dir=temp_docs_dir)
        assert isinstance(count, int)
        assert count > 0, "应至少存入一个文本块"

    def test_build_with_empty_dir(self, tmp_path, temp_chroma_dir):
        """空目录应返回 0"""
        empty_dir = str(tmp_path / "empty_docs")
        os.makedirs(empty_dir, exist_ok=True)
        count = build_knowledge_base(docs_dir=empty_dir)
        assert count == 0

    def test_build_with_nonexistent_dir(self, temp_chroma_dir):
        """不存在的目录应返回 0 而不是抛出异常"""
        count = build_knowledge_base(docs_dir="/nonexistent/path")
        assert count == 0

    def test_rebuild_overwrites_existing(self, temp_docs_dir, temp_chroma_dir):
        """重建应覆盖已有数据"""
        count1 = build_knowledge_base(docs_dir=temp_docs_dir)
        count2 = build_knowledge_base(docs_dir=temp_docs_dir)
        # 两次构建结果应相同
        assert count1 == count2


# ─────────────────────────────────────────
# 知识库检索测试
# ─────────────────────────────────────────

class TestQueryKnowledge:
    """测试知识库检索函数"""

    def test_query_returns_list(self, temp_docs_dir, temp_chroma_dir):
        """检索应返回列表"""
        build_knowledge_base(docs_dir=temp_docs_dir)
        results = query_knowledge("肺癌", top_k=3)
        assert isinstance(results, list)

    def test_query_top_k_limit(self, temp_docs_dir, temp_chroma_dir):
        """检索结果数量不应超过 top_k"""
        build_knowledge_base(docs_dir=temp_docs_dir)
        results = query_knowledge("肺部疾病", top_k=2)
        assert len(results) <= 2

    def test_query_results_are_strings(self, temp_docs_dir, temp_chroma_dir):
        """检索结果中的每个元素应为字符串"""
        build_knowledge_base(docs_dir=temp_docs_dir)
        results = query_knowledge("肺炎CT表现", top_k=3)
        for r in results:
            assert isinstance(r, str)

    def test_query_empty_kb_returns_list(self, temp_chroma_dir):
        """空知识库检索应返回空列表而不是抛出异常"""
        results = query_knowledge("测试查询", top_k=3)
        assert isinstance(results, list)


# ─────────────────────────────────────────
# 知识库信息测试
# ─────────────────────────────────────────

class TestGetCollectionInfo:
    """测试知识库信息获取函数"""

    def test_returns_dict(self, temp_chroma_dir):
        """应返回字典"""
        info = get_collection_info()
        assert isinstance(info, dict)

    def test_required_keys(self, temp_chroma_dir):
        """返回字典应包含 name, count, status 字段"""
        info = get_collection_info()
        assert "name" in info
        assert "count" in info
        assert "status" in info

    def test_count_after_build(self, temp_docs_dir, temp_chroma_dir):
        """构建后 count 应大于 0"""
        build_knowledge_base(docs_dir=temp_docs_dir)
        info = get_collection_info()
        assert info["count"] > 0
        assert info["status"] == "ready"
