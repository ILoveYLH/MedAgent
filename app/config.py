"""
全局配置管理模块

使用 pydantic-settings 从 .env 文件读取环境变量配置。
支持多 LLM 提供商切换（Google Gemini / 通义千问）。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用全局配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ─── LLM 提供商切换 ───────────────────────
    llm_provider: str = Field(
        default="google",
        description="LLM 提供商: google / qwen",
    )

    # ─── Google API 配置 ──────────────────────
    google_api_key: str = Field(default="", description="Google Gemini API密钥")
    model_name: str = Field(default="gemini-2.0-flash", description="Gemini模型名称")
    embedding_model: str = Field(
        default="models/embedding-001", description="Google Embedding模型名称"
    )

    # ─── 通义千问 (Qwen) API 配置 ─────────────
    qwen_api_key: str = Field(default="", description="通义千问 DashScope API密钥")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="通义千问 API Base URL (OpenAI 兼容)",
    )
    qwen_model_name: str = Field(
        default="qwen-plus", description="通义千问对话模型名称"
    )
    qwen_embedding_model: str = Field(
        default="text-embedding-v3", description="通义千问 Embedding 模型名称"
    )

    # ─── ChromaDB 配置 ────────────────────────
    chroma_db_path: str = Field(default="./data/chroma_db", description="ChromaDB存储路径")
    chroma_collection_name: str = Field(
        default="medical_knowledge", description="ChromaDB集合名称"
    )

    # ─── 服务器配置 ───────────────────────────
    server_host: str = Field(default="0.0.0.0", description="服务器主机地址")
    server_port: int = Field(default=8000, description="服务器端口")

    # ─── Gradio 前端配置 ──────────────────────
    gradio_host: str = Field(default="0.0.0.0", description="Gradio服务器主机")
    gradio_port: int = Field(default=7860, description="Gradio服务器端口")

    # ─── RAG 配置 ─────────────────────────────
    rag_top_k: int = Field(default=3, description="RAG检索返回的最大文档数")
    docs_dir: str = Field(default="./app/rag/docs", description="医学知识文档目录")

    # ─── Telegram Bot 配置 ────────────────────
    telegram_bot_token: str = Field(default="", description="Telegram Bot Token")

    @property
    def is_google(self) -> bool:
        return self.llm_provider.lower() == "google"

    @property
    def is_qwen(self) -> bool:
        return self.llm_provider.lower() == "qwen"

    @property
    def active_api_key(self) -> str:
        """当前激活的 API Key"""
        return self.google_api_key if self.is_google else self.qwen_api_key

    @property
    def active_model_name(self) -> str:
        """当前激活的对话模型名称"""
        return self.model_name if self.is_google else self.qwen_model_name

    @property
    def active_embedding_model(self) -> str:
        """当前激活的 Embedding 模型名称"""
        return self.embedding_model if self.is_google else self.qwen_embedding_model


# 全局配置实例
settings = Settings()
