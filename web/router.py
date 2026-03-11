"""
多模态路由器 (Router)

接收前端输入（文本 + 附件），通过 LLM 进行意图识别，
决定需要激活哪些 Agent 进行并行处理。

路由策略：
    1. 优先使用 LLM（Qwen API）进行智能意图识别
    2. LLM 不可用时降级为关键词匹配规则引擎
    3. 支持识别"通用对话"意图，避免误触专业 Agent

未来升级：
    - 支持更复杂的条件路由（如：基于患者历史数据决定是否激活基因检测）
    - 支持动态 Agent 注册和发现机制
    - 多轮对话上下文路由
"""

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── Agent 名称常量 ──
GENERAL = "GeneralAgent"
CLINICAL = "ClinicalAgent"
IMAGING = "ImagingAgent"
BLOOD = "BloodAgent"
GENETICS = "GeneticsAgent"

ALL_AGENTS = [GENERAL, CLINICAL, IMAGING, BLOOD, GENETICS]
SPECIALIST_AGENTS = [CLINICAL, IMAGING, BLOOD, GENETICS]

# ── LLM 路由 Prompt ──
ROUTER_SYSTEM_PROMPT = """你是一个医疗意图识别系统。根据用户输入，判断应该调用哪些 Agent。

可选的 Agent 列表：
- GeneralAgent: 通用对话，用于问候、闲聊、感谢、一般性问题（如"你好"、"谢谢"、"你是谁"、"能做什么"）
- ClinicalAgent: 临床诊断分析，当用户描述症状、病史、就诊需求时调用
- ImagingAgent: 影像分析，当涉及 CT、MRI、X光、影像、结节、图片等内容时调用
- BloodAgent: 血液检验分析，当涉及血液、血常规、化验、肿瘤标志物、CEA 等指标时调用
- GeneticsAgent: 基因检测分析，当涉及基因、突变、靶向、EGFR、ALK、PD-L1 等内容时调用

判断规则：
1. 如果用户只是打招呼、闲聊、问通用问题，只返回 GeneralAgent
2. 如果用户描述了具体医学需求，返回相关的专业 Agent（可以多个），不要返回 GeneralAgent
3. 如果涉及多个领域，返回多个 Agent
4. 如果有专业 Agent 被选中，始终同时包含 ClinicalAgent（它负责综合分析）

请只返回一个 JSON 数组，包含需要调用的 Agent 名称，不要返回其他任何内容。
示例：["GeneralAgent"]
示例：["ClinicalAgent", "ImagingAgent"]
示例：["ClinicalAgent", "BloodAgent", "GeneticsAgent"]"""

# ── 关键词路由规则（降级方案）──
KEYWORD_RULES: dict[str, list[str]] = {
    GENERAL: [
        "你好", "hello", "hi", "嗨", "谢谢", "感谢", "再见", "拜拜",
        "你是谁", "能做什么", "介绍", "帮助", "help",
    ],
    CLINICAL: [
        "症状", "主诉", "病史", "诊断", "分析", "检查", "评估", "就诊",
        "咳嗽", "干咳", "胸闷", "胸痛", "呼吸", "发热",
        "综合", "全面", "整体",
    ],
    IMAGING: [
        "CT", "影像", "拍片", "X光", "MRI", "核磁", "超声", "B超",
        "结节", "肿块", "阴影", "病灶", "图片", "图像",
        "DICOM", "dcm",
    ],
    BLOOD: [
        "血液", "血常规", "血检", "化验", "指标", "肿瘤标志物",
        "白细胞", "红细胞", "血红蛋白", "CEA", "NSE", "CYFRA",
        "CRP", "生化",
    ],
    GENETICS: [
        "基因", "突变", "靶向", "EGFR", "ALK", "ROS1", "KRAS",
        "PD-L1", "免疫治疗", "NGS", "测序", "遗传",
    ],
}

# ── 文件类型路由规则 ──
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm"}
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx", ".csv", ".xlsx"}


def route_input(
    text: str,
) -> list[str]:
    """
    根据用户输入文本，智能决定需要激活的 Agent 列表

    优先使用 LLM 进行意图识别，失败时降级为关键词匹配。

    参数:
        text: 用户输入的文本消息

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

    # ─ 3. 一致性保障：有专业 Agent 则必加 Clinical ─
    has_specialist = any(a in SPECIALIST_AGENTS for a in activated)
    if has_specialist:
        activated.add(CLINICAL)
        activated.discard(GENERAL)  # 有专业需求就不走通用对话

    # ─ 4. 兜底：什么都没匹配到 → GeneralAgent ─
    if not activated:
        activated.add(GENERAL)

    result = sorted(activated)
    logger.info("路由结果: '%s' → %s", text[:50], result)
    return result


def _route_with_llm(text: str) -> Optional[list[str]]:
    """
    使用 LLM (Qwen) 进行意图识别

    参数:
        text: 用户输入

    返回:
        list[str] | None: 成功则返回 Agent 列表，失败返回 None
    """
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
            temperature=0.1,  # 低温度确保稳定输出
            max_tokens=100,
        )

        content = response.choices[0].message.content.strip()
        logger.info("LLM 路由原始返回: %s", content)

        # 提取 JSON 数组
        # 处理可能的 markdown 代码块包裹
        if "```" in content:
            import re
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                content = match.group(0)

        agents = json.loads(content)
        if not isinstance(agents, list):
            logger.warning("LLM 路由返回格式异常: %s", content)
            return None

        # 验证返回的 Agent 名称合法
        valid_agents = [a for a in agents if a in ALL_AGENTS]
        if not valid_agents:
            logger.warning("LLM 路由返回无有效 Agent: %s", agents)
            return None

        logger.info("LLM 路由成功: %s → %s", text[:30], valid_agents)
        return valid_agents

    except Exception as exc:
        logger.warning("LLM 路由调用失败: %s", exc)
        return None


def _route_with_keywords(text: str) -> set[str]:
    """
    关键词匹配路由（降级方案）

    参数:
        text: 用户输入

    返回:
        set[str]: 匹配到的 Agent 集合
    """
    activated: set[str] = set()
    text_lower = text.lower()

    for agent_name, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                activated.add(agent_name)
                break

    return activated


def _route_by_file_type(attachments: list[str]) -> set[str]:
    """
    根据文件类型判断需要激活的 Agent

    参数:
        attachments: 文件路径列表

    返回:
        set[str]: 文件类型对应的 Agent 集合
    """
    agents: set[str] = set()
    for file_path in attachments:
        ext = Path(file_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            agents.add(IMAGING)
        if ext in DOCUMENT_EXTENSIONS:
            agents.add(BLOOD)
            agents.add(GENETICS)
    return agents
