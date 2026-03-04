"""
文档向量化模块

支持多 LLM 提供商：
  - Google Embedding API (google-generativeai)
  - 通义千问 Embedding API (DashScope OpenAI 兼容接口)

通过 .env 中的 LLM_PROVIDER 自动切换。
"""

import logging
from typing import Optional

from app.config import settings

logger = logging.getLogger(__name__)

# 随机向量维度（Google = 768, Qwen text-embedding-v3 = 1024）
EMBEDDING_DIMENSION_GOOGLE = 768
EMBEDDING_DIMENSION_QWEN = 1024


def _get_embedding_dimension() -> int:
    """根据当前提供商返回 embedding 维度"""
    if settings.is_qwen:
        return EMBEDDING_DIMENSION_QWEN
    return EMBEDDING_DIMENSION_GOOGLE


def _random_vectors(count: int) -> list[list[float]]:
    """生成随机向量（API 不可用时的降级方案）"""
    import random
    dim = _get_embedding_dimension()
    return [[random.uniform(-1, 1) for _ in range(dim)] for _ in range(count)]


def _random_vector() -> list[float]:
    """生成单个随机向量"""
    import random
    dim = _get_embedding_dimension()
    return [random.uniform(-1, 1) for _ in range(dim)]


class GoogleEmbeddingFunction:
    """ChromaDB 兼容的 Google Embedding 函数"""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.embedding_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            self._client = genai
        return self._client

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not settings.google_api_key:
            logger.warning("未配置 GOOGLE_API_KEY，使用随机向量（仅用于测试）")
            return _random_vectors(len(input))

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
            return _random_vectors(len(input))

    def embed_query(self, text: str) -> list[float]:
        if not settings.google_api_key:
            return _random_vector()
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
            return _random_vector()


class QwenEmbeddingFunction:
    """
    ChromaDB 兼容的通义千问 Embedding 函数

    使用 DashScope 的 OpenAI 兼容接口 (/v1/embeddings)。
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.qwen_embedding_model
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=settings.qwen_api_key,
                base_url=settings.qwen_base_url,
            )
        return self._client

    def __call__(self, input: list[str]) -> list[list[float]]:
        if not settings.qwen_api_key:
            logger.warning("未配置 QWEN_API_KEY，使用随机向量（仅用于测试）")
            return _random_vectors(len(input))

        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model_name,
                input=input,
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            logger.error("Qwen Embedding 调用失败: %s，使用随机向量", exc)
            return _random_vectors(len(input))

    def embed_query(self, text: str) -> list[float]:
        if not settings.qwen_api_key:
            return _random_vector()
        try:
            client = self._get_client()
            response = client.embeddings.create(
                model=self.model_name,
                input=[text],
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.error("查询向量化失败: %s，使用随机向量", exc)
            return _random_vector()


# ─────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────

_embedding_function = None


def get_embedding_function():
    """
    获取全局 Embedding 函数实例（根据 LLM_PROVIDER 自动选择）
    """
    global _embedding_function
    if _embedding_function is None:
        provider = settings.llm_provider.lower()
        logger.info("初始化 Embedding 提供商: %s, 模型: %s", provider, settings.active_embedding_model)

        if provider == "qwen":
            _embedding_function = QwenEmbeddingFunction()
        else:
            _embedding_function = GoogleEmbeddingFunction()

    return _embedding_function
