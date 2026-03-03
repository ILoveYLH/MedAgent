"""
医学知识库检索工具

将 RAG 检索能力封装为 LangChain Tool，供 ReACT Agent 调用。
"""

import logging

from langchain_core.tools import tool

from app.config import settings

logger = logging.getLogger(__name__)


@tool
def query_medical_knowledge_tool(query: str) -> str:
    """医学知识库查询工具。

    当你需要查阅某种疾病的临床指南、治疗方案、鉴别诊断要点、
    或者某个影像学征象的临床意义时，就调用这个工具。
    比如你想知道"磨玻璃结节的随访建议"或者"肺腺癌的TNM分期标准"，
    把你的问题传进来，它会从医学知识库中检索最相关的参考资料给你。

    参数:
        query: 你想查询的医学问题，用自然语言描述即可，例如"肺结节恶性概率评估标准"
    """
    try:
        from app.rag.knowledge_base import query_knowledge

        results = query_knowledge(query, top_k=settings.rag_top_k)
        if not results:
            return "知识库中暂未找到与该查询相关的医学资料。建议尝试换一个关键词或更具体的描述。"

        # 格式化返回
        formatted = []
        for i, chunk in enumerate(results, 1):
            formatted.append(f"【参考资料 {i}】\n{chunk}")
        return "\n\n".join(formatted)

    except Exception as exc:
        logger.error("医学知识库检索失败: %s", exc)
        return f"知识库检索出错: {exc}"
