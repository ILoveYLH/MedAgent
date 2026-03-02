"""
主Agent调度器

使用 LangGraph 构建有状态的医学CT诊断Agent工作流。
工作流步骤：
  1. 接收用户输入（CT图片路径 + 文本请求）
  2. 意图识别
  3. CT疾病分类
  4. 病灶检测
  5. RAG知识库检索
  6. 诊断报告生成
  7. 返回综合结果
"""

import logging
from datetime import datetime
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agent.prompts import (
    INTENT_RECOGNITION_PROMPT,
    KNOWLEDGE_QUERY_PROMPT,
    REPORT_GENERATION_PROMPT,
)
from app.config import settings
from app.skills.ct_classifier import classify_ct
from app.skills.lesion_detector import detect_lesions

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────
# 工作流状态定义
# ─────────────────────────────────────────
class AgentState(TypedDict):
    """Agent工作流状态，贯穿整个处理流程"""

    # 输入
    image_path: str          # CT图片路径
    user_request: str        # 用户文本请求

    # 中间状态
    intents: list[str]       # 识别到的意图列表
    classification_result: dict[str, Any]   # CT分类结果
    detection_result: dict[str, Any]        # 病灶检测结果
    rag_knowledge: list[str]                # RAG检索结果

    # 输出
    report: str              # 最终诊断报告
    error: str               # 错误信息（如有）


# ─────────────────────────────────────────
# 工作流节点函数
# ─────────────────────────────────────────

DEFAULT_INTENTS = ["classification", "detection", "report"]


def recognize_intent(state: AgentState) -> AgentState:
    """
    节点1：意图识别
    判断用户需要分类、检测、报告还是知识查询
    """
    try:
        # 尝试使用 Gemini 进行意图识别
        if settings.google_api_key:
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            model = genai.GenerativeModel(settings.model_name)
            prompt = INTENT_RECOGNITION_PROMPT.format(user_request=state["user_request"])
            response = model.generate_content(prompt)
            raw = response.text.strip().lower()
            intents = [i.strip() for i in raw.split(",") if i.strip()]
        else:
            # 无API Key时默认执行全流程
            intents = list(DEFAULT_INTENTS)

        # 确保至少有一个意图
        if not intents:
            intents = list(DEFAULT_INTENTS)

        logger.info("识别到的用户意图: %s", intents)
        return {**state, "intents": intents}
    except Exception as exc:
        logger.warning("意图识别失败，使用默认全流程: %s", exc)
        return {**state, "intents": list(DEFAULT_INTENTS)}


def run_classification(state: AgentState) -> AgentState:
    """
    节点2：CT疾病分类
    调用 ct_classifier skill 获取疾病分类结果
    """
    try:
        result = classify_ct(state["image_path"])
        logger.info("CT分类完成: %s", result.get("disease_type"))
        return {**state, "classification_result": result}
    except Exception as exc:
        logger.error("CT分类失败: %s", exc)
        return {**state, "classification_result": {"error": str(exc)}}


def run_detection(state: AgentState) -> AgentState:
    """
    节点3：病灶检测
    调用 lesion_detector skill 获取病灶检测结果
    """
    try:
        result = detect_lesions(state["image_path"])
        logger.info("病灶检测完成，发现 %d 处病灶", len(result.get("lesions", [])))
        return {**state, "detection_result": result}
    except Exception as exc:
        logger.error("病灶检测失败: %s", exc)
        return {**state, "detection_result": {"error": str(exc)}}


def query_knowledge(state: AgentState) -> AgentState:
    """
    节点4：RAG知识库检索
    根据分类和检测结果检索相关医学知识
    """
    try:
        from app.rag.knowledge_base import query_knowledge as kb_query

        cls = state.get("classification_result", {})
        det = state.get("detection_result", {})
        disease_type = cls.get("disease_type", "未知疾病")
        lesion_count = len(det.get("lesions", []))
        lesion_info = f"发现 {lesion_count} 处病灶" if lesion_count else "未发现明显病灶"

        # 构建查询语句
        if settings.google_api_key:
            import google.generativeai as genai
            genai.configure(api_key=settings.google_api_key)
            model = genai.GenerativeModel(settings.model_name)
            prompt = KNOWLEDGE_QUERY_PROMPT.format(
                disease_type=disease_type,
                lesion_info=lesion_info,
            )
            query = model.generate_content(prompt).text.strip()
        else:
            query = disease_type

        knowledge_chunks = kb_query(query, top_k=settings.rag_top_k)
        logger.info("RAG检索完成，获取 %d 条知识", len(knowledge_chunks))
        return {**state, "rag_knowledge": knowledge_chunks}
    except Exception as exc:
        logger.warning("RAG检索失败，跳过: %s", exc)
        return {**state, "rag_knowledge": []}


def generate_report(state: AgentState) -> AgentState:
    """
    节点5：诊断报告生成
    整合所有分析结果，调用 Gemini 生成结构化诊断报告
    """
    try:
        from app.report.generator import generate_diagnosis_report

        report = generate_diagnosis_report(
            user_request=state["user_request"],
            classification_result=state.get("classification_result", {}),
            detection_result=state.get("detection_result", {}),
            rag_knowledge=state.get("rag_knowledge", []),
        )
        logger.info("诊断报告生成完成")
        return {**state, "report": report}
    except Exception as exc:
        logger.error("报告生成失败: %s", exc)
        # 降级：返回简单摘要
        cls = state.get("classification_result", {})
        det = state.get("detection_result", {})
        fallback = (
            f"## CT诊断摘要\n\n"
            f"**疾病分类**: {cls.get('disease_type', '未知')} "
            f"（置信度: {cls.get('confidence', 0):.1%}）\n\n"
            f"**病灶数量**: {len(det.get('lesions', []))} 处\n\n"
            f"*报告生成失败，请检查API配置。错误: {exc}*"
        )
        return {**state, "report": fallback}


# ─────────────────────────────────────────
# 构建 LangGraph 工作流
# ─────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    """构建并返回编译好的 LangGraph Agent工作流"""

    workflow = StateGraph(AgentState)

    # 注册节点
    workflow.add_node("recognize_intent", recognize_intent)
    workflow.add_node("run_classification", run_classification)
    workflow.add_node("run_detection", run_detection)
    workflow.add_node("query_knowledge", query_knowledge)
    workflow.add_node("generate_report", generate_report)

    # 设置入口节点
    workflow.set_entry_point("recognize_intent")

    # 定义边（顺序执行）
    workflow.add_edge("recognize_intent", "run_classification")
    workflow.add_edge("run_classification", "run_detection")
    workflow.add_edge("run_detection", "query_knowledge")
    workflow.add_edge("query_knowledge", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()


# 全局编译好的工作流实例（懒加载）
_agent_graph = None


def get_agent_graph():
    """获取全局Agent工作流实例（懒加载）"""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(image_path: str, user_request: str) -> dict[str, Any]:
    """
    运行完整的CT诊断Agent工作流

    参数:
        image_path: CT图片路径
        user_request: 用户文本请求

    返回:
        包含分类结果、检测结果、RAG知识和诊断报告的字典
    """
    graph = get_agent_graph()

    # 初始化状态
    initial_state: AgentState = {
        "image_path": image_path,
        "user_request": user_request,
        "intents": [],
        "classification_result": {},
        "detection_result": {},
        "rag_knowledge": [],
        "report": "",
        "error": "",
    }

    # 执行工作流
    final_state = graph.invoke(initial_state)

    return {
        "classification": final_state.get("classification_result", {}),
        "detection": final_state.get("detection_result", {}),
        "knowledge": final_state.get("rag_knowledge", []),
        "report": final_state.get("report", ""),
        "intents": final_state.get("intents", []),
    }
