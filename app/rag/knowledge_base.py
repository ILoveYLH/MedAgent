"""
医学知识库模块

使用 ChromaDB 作为向量数据库，实现医学知识文档的存储和检索。
支持从 docs/ 目录批量加载文档，进行文本分块，向量化后存入 ChromaDB。
"""

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb

from app.config import settings
from app.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)

# 文档分块配置
CHUNK_SIZE = 500        # 每个文本块的最大字符数
CHUNK_OVERLAP = 50      # 相邻文本块的重叠字符数
CHROMA_BATCH_SIZE = 100 # ChromaDB 批量写入大小


def _split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    将长文本分割为多个重叠的文本块

    参数:
        text: 待分割的文本
        chunk_size: 每个文本块的最大字符数
        overlap: 相邻文本块的重叠字符数

    返回:
        文本块列表
    """
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap

    return chunks


def _load_documents(docs_dir: str) -> list[dict]:
    """
    从指定目录加载所有 .md 和 .txt 文档

    参数:
        docs_dir: 文档目录路径

    返回:
        文档列表，每个文档包含 content 和 metadata
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.warning("文档目录不存在: %s", docs_dir)
        return []

    documents = []
    supported_extensions = {".md", ".txt"}

    for file_path in docs_path.iterdir():
        if file_path.suffix.lower() not in supported_extensions:
            continue
        try:
            content = file_path.read_text(encoding="utf-8")
            documents.append({
                "content": content,
                "metadata": {
                    "source": file_path.name,
                    "file_path": str(file_path),
                },
            })
            logger.info("已加载文档: %s (%d 字)", file_path.name, len(content))
        except Exception as exc:
            logger.error("加载文档失败 %s: %s", file_path, exc)

    return documents


def build_knowledge_base(docs_dir: Optional[str] = None) -> int:
    """
    从文档目录构建医学知识库

    将目录中的 .md 和 .txt 文件加载、分块、向量化后存入 ChromaDB。

    参数:
        docs_dir: 文档目录路径，默认使用配置中的 DOCS_DIR

    返回:
        成功存入 ChromaDB 的文本块数量
    """
    docs_dir = docs_dir or settings.docs_dir
    logger.info("开始构建知识库，文档目录: %s", docs_dir)

    # 加载文档
    documents = _load_documents(docs_dir)
    if not documents:
        logger.warning("未找到任何文档，知识库为空")
        return 0

    # 分块
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_idx = 0

    for doc in documents:
        chunks = _split_text(doc["content"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                **doc["metadata"],
                "chunk_index": chunk_idx,
            })
            all_ids.append(f"chunk_{chunk_idx}")
            chunk_idx += 1

    logger.info("文档分块完成，共 %d 个文本块", len(all_chunks))

    # 存入 ChromaDB
    client = chromadb.PersistentClient(path=settings.chroma_db_path)
    embedding_fn = get_embedding_function()

    # 删除已有集合（重建时覆盖）
    try:
        client.delete_collection(settings.chroma_collection_name)
        logger.info("已删除旧知识库集合")
    except Exception:
        pass

    collection = client.create_collection(
        name=settings.chroma_collection_name,
        embedding_function=embedding_fn,
        metadata={"description": "医学CT诊断知识库"},
    )

    # 批量添加（ChromaDB 建议每批不超过 5000 条）
    for i in range(0, len(all_chunks), CHROMA_BATCH_SIZE):
        batch_chunks = all_chunks[i:i + CHROMA_BATCH_SIZE]
        batch_metadatas = all_metadatas[i:i + CHROMA_BATCH_SIZE]
        batch_ids = all_ids[i:i + CHROMA_BATCH_SIZE]
        collection.add(
            documents=batch_chunks,
            metadatas=batch_metadatas,
            ids=batch_ids,
        )

    logger.info("知识库构建完成，共存入 %d 个文本块", len(all_chunks))
    return len(all_chunks)


def query_knowledge(query: str, top_k: Optional[int] = None) -> list[str]:
    """
    从知识库检索与查询最相关的知识片段

    参数:
        query: 查询文本
        top_k: 返回的最大知识片段数量，默认使用配置中的 RAG_TOP_K

    返回:
        相关知识片段列表（字符串列表）
    """
    top_k = top_k or settings.rag_top_k
    logger.info("知识库检索: %s (top_k=%d)", query, top_k)

    try:
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        embedding_fn = get_embedding_function()

        # 获取集合（若不存在则先构建）
        try:
            collection = client.get_collection(
                name=settings.chroma_collection_name,
                embedding_function=embedding_fn,
            )
        except Exception:
            logger.warning("知识库集合不存在，尝试自动构建...")
            build_knowledge_base()
            collection = client.get_collection(
                name=settings.chroma_collection_name,
                embedding_function=embedding_fn,
            )

        # 执行相似度检索
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        documents = results.get("documents", [[]])[0]
        logger.info("检索完成，返回 %d 条知识片段", len(documents))
        return documents

    except Exception as exc:
        logger.error("知识库检索失败: %s", exc)
        return []


def get_collection_info() -> dict:
    """
    获取知识库集合的基本信息

    返回:
        包含集合名称和文档数量的字典
    """
    try:
        client = chromadb.PersistentClient(path=settings.chroma_db_path)
        collection = client.get_collection(name=settings.chroma_collection_name)
        return {
            "name": settings.chroma_collection_name,
            "count": collection.count(),
            "status": "ready",
        }
    except Exception as exc:
        return {
            "name": settings.chroma_collection_name,
            "count": 0,
            "status": f"not_ready: {exc}",
        }
