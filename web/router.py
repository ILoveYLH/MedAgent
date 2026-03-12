"""
多模态路由器 (Router)

接收前端输入（文本 + 附件），通过 LLM 进行意图识别，
决定需要激活哪些 Agent 进行并行处理。

Agent 列表（3个）：
  - GeneralAgent: 通用对话
  - ImagingAgent: 影像分析（上传图片时）
  - AnalysisAgent: LLM 驱动的综合医学分析（症状/血液/基因等文本数据）

路由策略：
    1. 优先使用 LLM（Qwen API）进行智能意图识别
    2. LLM 不可用时降级为关键词匹配规则引擎
    3. 有图片附件 → 必激活 ImagingAgent
    4. 有医学数据文本 → 激活 AnalysisAgent
    5. 纯闲聊/问候 → GeneralAgent
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Agent 名称常量 ──
GENERAL = "GeneralAgent"
IMAGING = "ImagingAgent"
ANALYSIS = "AnalysisAgent"

ALL_AGENTS = [GENERAL, IMAGING, ANALYSIS]
SPECIALIST_AGENTS = [IMAGING, ANALYSIS]

# ── LLM 路由 Prompt ──
ROUTER_SYSTEM_PROMPT = """你是一个医疗意图识别系统。根据用户输入，判断应该调用哪些 Agent。

可选的 Agent 列表：
- GeneralAgent: 通用对话，用于问候、闲聊、感谢、一般性问题（如"你好"、"谢谢"、"你是谁"、"能做什么"）
- ImagingAgent: 影像分析，当涉及 CT、MRI、X光、影像、结节、图片、图像等内容时调用
- AnalysisAgent: 综合医学分析，当用户提供了任何医疗数据时调用，包括：症状描述、血液检验数据、基因检测结果、肿瘤标志物、病史等

判断规则：
1. 如果用户只是打招呼、闲聊、问通用问题，只返回 GeneralAgent
2. 如果用户描述了症状、提供了医学数据（血液/基因/病史等），返回 AnalysisAgent
3. 如果涉及影像（CT/MRI/图像），返回 ImagingAgent（可与 AnalysisAgent 并用）
4. 如果同时有影像需求和数据分析需求，同时返回 ImagingAgent 和 AnalysisAgent

请只返回一个 JSON 数组，包含需要调用的 Agent 名称，不要返回其他任何内容。
示例：["GeneralAgent"]
示例：["ImagingAgent", "AnalysisAgent"]
示例：["AnalysisAgent"]"""

# ── 关键词路由规则（降级方案）──
KEYWORD_RULES: dict[str, list[str]] = {
    GENERAL: [
        "你好", "hello", "hi", "嗨", "谢谢", "感谢", "再见", "拜拜",
        "你是谁", "能做什么", "介绍", "帮助", "help",
    ],
    IMAGING: [
        "CT", "影像", "拍片", "X光", "MRI", "核磁", "超声", "B超",
        "结节", "肿块", "阴影", "病灶", "图片", "图像",
        "DICOM", "dcm",
    ],
    ANALYSIS: [
        # 症状
        "症状", "主诉", "病史", "诊断", "分析", "检查", "评估", "就诊",
        "咳嗽", "干咳", "胸闷", "胸痛", "呼吸", "发热",
        # 血液指标
        "血液", "血常规", "血检", "化验", "指标", "肿瘤标志物",
        "白细胞", "红细胞", "血红蛋白", "CEA", "NSE", "CYFRA",
        "CRP", "生化",
        # 基因
        "基因", "突变", "靶向", "EGFR", "ALK", "ROS1", "KRAS",
        "PD-L1", "免疫治疗", "NGS", "测序", "遗传",
        # 综合
        "综合", "全面", "整体",
    ],
}

# ── 文件类型路由规则 ──
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx", ".csv", ".xlsx"}


def route_input(
    text: str,
    attachments: Optional[list[str]] = None,
) -> list[str]:
    """
    根据用户输入文本和附件类型，智能决定需要激活的 Agent 列表

    参数:
        text: 用户输入的文本消息
        attachments: 上传的文件路径列表

    返回:
        list[str]: 需要激活的 Agent 名称列表（去重后）
    """
    # ─ 1. 尝试 LLM 路由 ─
    llm_result = _route_with_llm(text)
    if llm_result is not None:
        activated = set(llm_result)
    else:
        # ─ 2. 降级为关键词匹配 ─
        logger.info("LLM 路由不可用，降级为关键词匹配")
        activated = _route_with_keywords(text)

    # ─ 3. 文件类型补充路由 ─
    if attachments:
        file_agents = _route_by_file_type(attachments)
        if file_agents:
            activated.discard(GENERAL)
            activated.update(file_agents)

    # ─ 4. 一致性保障：有专业 Agent 则移除 GeneralAgent ─
    has_specialist = any(a in SPECIALIST_AGENTS for a in activated)
    if has_specialist:
        activated.discard(GENERAL)

    # ─ 5. 兜底：什么都没匹配到 → GeneralAgent ─
    if not activated:
        activated.add(GENERAL)

    result = sorted(activated)
    logger.info("路由结果: '%s' → %s", text[:50], result)
    return result


def _route_with_llm(text: str) -> Optional[list[str]]:
    """使用 LLM (Qwen) 进行意图识别"""
    qwen_key = os.getenv("QWEN_API_KEY", "")
    if not qwen_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=qwen_key,
            base_url=os.getenv(
                "QWEN_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )

        response = client.chat.completions.create(
            model=os.getenv("QWEN_MODEL_NAME", "qwen-plus"),
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            temperature=0.1,
            max_tokens=100,
        )

        content = response.choices[0].message.content.strip()
        logger.info("LLM 路由原始返回: %s", content)

        if "```" in content:
            import re
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                content = match.group(0)

        agents = json.loads(content)
        if not isinstance(agents, list):
            return None

        valid_agents = [a for a in agents if a in ALL_AGENTS]
        if not valid_agents:
            return None

        logger.info("LLM 路由成功: %s → %s", text[:30], valid_agents)
        return valid_agents

    except Exception as exc:
        logger.warning("LLM 路由调用失败: %s", exc)
        return None


def _route_with_keywords(text: str) -> set[str]:
    """关键词匹配路由（降级方案）"""
    activated: set[str] = set()
    text_lower = text.lower()

    for agent_name, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                activated.add(agent_name)
                break

    return activated


def _route_by_file_type(attachments: list[str]) -> set[str]:
    """根据文件类型判断需要激活的 Agent"""
    agents: set[str] = set()
    for file_path in attachments:
        ext = Path(file_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            agents.add(IMAGING)
        if ext in DOCUMENT_EXTENSIONS:
            agents.add(ANALYSIS)
    return agents
