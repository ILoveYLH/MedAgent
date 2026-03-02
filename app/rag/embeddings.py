"""
文档向量化模块

使用 Google Embedding API 将文本转换为向量表示，用于 RAG 检索。
"""

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)


# 随机向量维度（与 Google Embedding 模型输出维度一致）
EMBEDDING_DIMENSION = 768


class GoogleEmbeddingFunction:
    """
    ChromaDB 兼容的 Google Embedding 函数

    使用 google-generativeai SDK 调用 Google Embedding API，
    将文本列表转换为向量列表。
    """

    def __init__(self, model_name: Optional[str] = None):
        """
        初始化 Google Embedding 函数

        参数:
            model_name: Embedding 模型名称，默认使用配置中的 EMBEDDING_MODEL
        """
        self.model_name = model_name or settings.embedding_model
        self._client = None

    def _get_client(self):
        """懒加载 Google Generative AI 客户端"""
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            self._client = genai
        return self._client

    def __call__(self, input: list[str]) -> list[list[float]]:
        """
        将文本列表转换为向量列表（ChromaDB EmbeddingFunction 接口）

        参数:
            input: 需要向量化的文本列表

        返回:
            对应的向量列表
        """
        if not settings.google_api_key:
            logger.warning("未配置 GOOGLE_API_KEY，使用随机向量（仅用于测试）")
            import random
            return [[random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)] for _ in input]

        try:
            client = self._get_client()
            embeddings = []
            for text in input:
                result = client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document",
                )
                embeddings.append(result["embedding"])
            return embeddings
        except Exception as exc:
            logger.error("Google Embedding 调用失败: %s，使用随机向量", exc)
            import random
            return [[random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)] for _ in input]

    def embed_query(self, text: str) -> list[float]:
        """
        将单个查询文本转换为向量（用于检索时的查询向量化）

        参数:
            text: 查询文本

        返回:
            查询向量
        """
        if not settings.google_api_key:
            import random
            return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]

        try:
            client = self._get_client()
            result = client.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_query",
            )
            return result["embedding"]
        except Exception as exc:
            logger.error("查询向量化失败: %s，使用随机向量", exc)
            import random
            return [random.uniform(-1, 1) for _ in range(EMBEDDING_DIMENSION)]


# 全局 Embedding 函数实例（懒加载）
_embedding_function = None


def get_embedding_function() -> GoogleEmbeddingFunction:
    """获取全局 Embedding 函数实例"""
    global _embedding_function
    if _embedding_function is None:
        _embedding_function = GoogleEmbeddingFunction()
    return _embedding_function
