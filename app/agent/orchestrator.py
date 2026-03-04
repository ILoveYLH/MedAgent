"""
主Agent调度器

使用 LangGraph 构建医学CT诊断Agent。
支持多 LLM 提供商 (Google Gemini / 通义千问)，通过 .env 中的 LLM_PROVIDER 切换。

Phase 1: 单体 ReACT Agent (create_react_agent)
Phase 2: + MemorySaver 多轮对话记忆
Phase 3: 多专家 Supervisor 架构
"""

import logging
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from app.agent.prompts import REACT_SYSTEM_PROMPT
from app.config import settings
from app.skills.ct_classifier import analyze_ct_tool
from app.skills.lesion_detector import detect_lesion_tool
from app.skills.medical_knowledge_tool import query_medical_knowledge_tool

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 工具列表
# ─────────────────────────────────────────

TOOLS = [analyze_ct_tool, detect_lesion_tool, query_medical_knowledge_tool]


# ─────────────────────────────────────────
# LLM 工厂（根据 LLM_PROVIDER 自动切换）
# ─────────────────────────────────────────

_llm = None


def _get_llm() -> BaseChatModel:
    """
    获取全局 LLM 实例（根据 .env 中的 LLM_PROVIDER 自动选择）

    - google → ChatGoogleGenerativeAI (Gemini)
    - qwen   → ChatOpenAI (DashScope OpenAI 兼容接口)
    """
    global _llm
    if _llm is not None:
        return _llm

    provider = settings.llm_provider.lower()
    logger.info("初始化 LLM 提供商: %s, 模型: %s", provider, settings.active_model_name)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm = ChatGoogleGenerativeAI(
            model=settings.model_name,
            google_api_key=settings.google_api_key,
            temperature=0.3,
        )
    elif provider == "qwen":
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=settings.qwen_model_name,
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
            temperature=0.3,
        )
    else:
        raise ValueError(
            f"不支持的 LLM 提供商: {provider}。"
            f"请在 .env 中设置 LLM_PROVIDER=google 或 LLM_PROVIDER=qwen"
        )

    return _llm


# ─────────────────────────────────────────
# Phase 1: 单体 ReACT Agent
# ─────────────────────────────────────────

_react_agent = None


def _get_react_agent():
    """获取全局 ReACT Agent 实例（无 Memory）"""
    global _react_agent
    if _react_agent is None:
        llm = _get_llm()
        _react_agent = create_react_agent(
            model=llm,
            tools=TOOLS,
            prompt=REACT_SYSTEM_PROMPT,
        )
    return _react_agent


def run_agent(image_path: str, user_request: str) -> dict[str, Any]:
    """
    运行 ReACT Agent 进行 CT 诊断（Phase 1，无记忆）

    参数:
        image_path: CT图片路径
        user_request: 用户文本请求

    返回:
        包含诊断报告的字典
    """
    agent = _get_react_agent()

    # 构建用户消息
    message = f"{user_request}\n\nCT图片路径: {image_path}"

    logger.info("ReACT Agent 开始处理: %s", user_request[:50])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]}
    )

    # 提取最终回复
    final_message = result["messages"][-1]
    report = final_message.content if hasattr(final_message, "content") else str(final_message)

    logger.info("ReACT Agent 处理完成")

    return {
        "report": report,
        "classification": {},
        "detection": {},
        "knowledge": [],
        "intents": ["react_agent"],
    }


# ─────────────────────────────────────────
# Phase 2: 带记忆的 ReACT Agent
# ─────────────────────────────────────────

_memory_agent = None
_memory_saver = None


def _get_memory_agent():
    """获取带 MemorySaver 的 ReACT Agent 实例"""
    global _memory_agent, _memory_saver
    if _memory_agent is None:
        from langgraph.checkpoint.memory import MemorySaver
        _memory_saver = MemorySaver()
        llm = _get_llm()
        _memory_agent = create_react_agent(
            model=llm,
            tools=TOOLS,
            prompt=REACT_SYSTEM_PROMPT,
            checkpointer=_memory_saver,
        )
    return _memory_agent


def run_agent_with_memory(
    message: str,
    thread_id: str,
    image_path: Optional[str] = None,
) -> str:
    """
    运行带多轮记忆的 ReACT Agent（Phase 2）

    参数:
        message: 用户消息文本
        thread_id: 会话线程ID（用于多轮记忆，如 Telegram User ID）
        image_path: CT图片路径（可选）

    返回:
        Agent 最终回复文本
    """
    agent = _get_memory_agent()

    # 构建用户消息
    user_content = message
    if image_path:
        user_content = f"{message}\n\nCT图片路径: {image_path}"

    logger.info("Memory Agent [thread=%s] 处理: %s", thread_id, message[:50])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # 提取最终回复
    final_message = result["messages"][-1]
    reply = final_message.content if hasattr(final_message, "content") else str(final_message)

    logger.info("Memory Agent [thread=%s] 处理完成", thread_id)
    return reply


# ─────────────────────────────────────────
# Phase 3: 多专家 Supervisor 架构
# ─────────────────────────────────────────

_supervisor_agent = None
_supervisor_memory = None


def _get_supervisor_agent():
    """获取多专家 Supervisor Agent 实例"""
    global _supervisor_agent, _supervisor_memory
    if _supervisor_agent is None:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph_supervisor import create_supervisor

        from app.agent.prompts import (
            CLINICIAN_PROMPT,
            RADIOLOGIST_PROMPT,
            REVIEWER_PROMPT,
            SUPERVISOR_PROMPT,
        )

        _supervisor_memory = MemorySaver()
        llm = _get_llm()

        # Agent A: 放射科医生 — 视觉分析专家
        radiologist = create_react_agent(
            model=llm,
            tools=[analyze_ct_tool, detect_lesion_tool],
            prompt=RADIOLOGIST_PROMPT,
            name="radiologist",
        )

        # Agent B: 临床医生 — RAG 知识查阅
        clinician = create_react_agent(
            model=llm,
            tools=[query_medical_knowledge_tool],
            prompt=CLINICIAN_PROMPT,
            name="clinician",
        )

        # Agent C: 主任审核员 — Reflection 交叉验证（无工具）
        reviewer = create_react_agent(
            model=llm,
            tools=[],
            prompt=REVIEWER_PROMPT,
            name="reviewer",
        )

        # Supervisor 编排
        _supervisor_agent = create_supervisor(
            agents=[radiologist, clinician, reviewer],
            model=llm,
            prompt=SUPERVISOR_PROMPT,
            checkpointer=_supervisor_memory,
        ).compile()

    return _supervisor_agent


def run_supervisor_agent(
    message: str,
    thread_id: str,
    image_path: Optional[str] = None,
) -> str:
    """
    运行多专家 Supervisor Agent（Phase 3）

    参数:
        message: 用户消息文本
        thread_id: 会话线程ID
        image_path: CT图片路径（可选）

    返回:
        最终诊断报告文本
    """
    agent = _get_supervisor_agent()

    user_content = message
    if image_path:
        user_content = f"{message}\n\nCT图片路径: {image_path}"

    logger.info("Supervisor Agent [thread=%s] 开始会诊: %s", thread_id, message[:50])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_content}]},
        config={"configurable": {"thread_id": thread_id}},
    )

    # 提取最终回复
    final_message = result["messages"][-1]
    reply = final_message.content if hasattr(final_message, "content") else str(final_message)

    logger.info("Supervisor Agent [thread=%s] 会诊完成", thread_id)
    return reply
