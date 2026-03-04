"""
多模态路由器 (Router)

接收前端输入（文本 + 附件），通过关键词匹配和文件类型判断，
决定需要激活哪些 Agent 进行并行处理。

MVP 阶段：基于关键词和文件扩展名的规则引擎。

未来升级：
    - route_with_llm(): 使用 LLM 进行精确的意图识别和多标签分类
    - 支持更复杂的条件路由（如：基于患者历史数据决定是否激活基因检测）
    - 支持动态 Agent 注册和发现机制
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Agent 名称常量 ──
CLINICAL = "ClinicalAgent"
IMAGING = "ImagingAgent"
BLOOD = "BloodAgent"
GENETICS = "GeneticsAgent"

ALL_AGENTS = [CLINICAL, IMAGING, BLOOD, GENETICS]

# ── 关键词路由规则 ──
KEYWORD_RULES: dict[str, list[str]] = {
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
    attachments: Optional[list[str]] = None,
) -> list[str]:
    """
    根据用户输入文本和附件类型，决定需要激活的 Agent 列表

    参数:
        text: 用户输入的文本消息
        attachments: 上传的文件路径列表

    返回:
        list[str]: 需要激活的 Agent 名称列表（去重后）
    """
    activated: set[str] = set()
    text_lower = text.lower()

    # ─ 1. 关键词匹配 ─
    for agent_name, keywords in KEYWORD_RULES.items():
        for kw in keywords:
            if kw.lower() in text_lower:
                activated.add(agent_name)
                break

    # ─ 2. 文件类型判断 ─
    if attachments:
        for file_path in attachments:
            ext = Path(file_path).suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                activated.add(IMAGING)
            if ext in DOCUMENT_EXTENSIONS:
                # PDF/TXT 可能是检验报告或基因报告
                activated.add(BLOOD)
                activated.add(GENETICS)

    # ─ 3. 默认策略：若无法匹配，则激活临床 + 影像 ─
    if not activated:
        activated.add(CLINICAL)
        activated.add(IMAGING)
        logger.info("未匹配到特定关键词，默认激活 Clinical + Imaging")

    # ─ 4. 通用策略：临床 Agent 始终参与 ─
    activated.add(CLINICAL)

    result = sorted(activated)
    logger.info("路由结果: %s → %s", text[:50], result)
    return result


def route_with_llm(text: str, attachments: Optional[list[str]] = None) -> list[str]:
    """
    使用 LLM 进行意图识别和 Agent 路由（未来升级）

    # TODO: 接入真实 LLM 意图识别
    # 实现步骤：
    #   1. 构建多标签分类 Prompt
    #   2. 调用 LLM API 获取意图标签
    #   3. 将意图标签映射到 Agent 列表
    #   4. 结合文件类型信息补充路由

    参数:
        text: 用户输入
        attachments: 附件列表

    返回:
        list[str]: Agent 名称列表
    """
    raise NotImplementedError("LLM 路由尚未实现，当前使用关键词匹配")
