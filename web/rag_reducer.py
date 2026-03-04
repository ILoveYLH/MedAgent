"""
Mock 医疗知识库 (RAG Reducer)

收集各 Agent 返回的 JSON，提取异常关键词，
从本地 Mock 知识字典中查找对应的医学解释和诊疗指南。

MVP 阶段：不使用真实 Vector DB，纯字典匹配。

未来升级：
    - retrieve_medical_guidelines() 接入 Chroma/FAISS + 真实医学文档
    - 集成医学知识图谱（如 UMLS、SNOMED CT）
    - 支持多语言医学文献检索
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Mock 医学知识库 ──
# 关键词 → (医学解释, 诊疗指南)
MOCK_KNOWLEDGE_BASE: dict[str, dict[str, str]] = {
    "结节": {
        "explanation": "肺结节是肺内直径≤30mm的类圆形病灶。根据密度可分为实性结节、部分实性结节（混合磨玻璃结节）和纯磨玻璃结节（GGO）。",
        "guideline": "根据《肺结节诊治中国专家共识(2024)》：8mm以上实性结节或部分实性结节建议3个月内随访薄层CT，必要时行PET-CT或穿刺活检。Fleischner Society指南推荐对高风险人群（吸烟史≥30包年）进行低剂量CT筛查。",
    },
    "磨玻璃": {
        "explanation": "肺磨玻璃影（GGO）指CT上肺内密度轻度增高的云雾状淡薄影。纯GGO恶变率约为18%，含实性成分的混合GGO恶变率可达63%。",
        "guideline": "纯GGO＜6mm：建议年度随访。纯GGO 6-8mm：6个月复查后年度随访。混合GGO或实性成分增加：建议3个月内复查，必要时穿刺活检。",
    },
    "EGFR": {
        "explanation": "表皮生长因子受体（EGFR）突变是非小细胞肺癌最常见的驱动基因突变，在亚洲人群中发生率约40-55%。常见突变类型包括外显子19缺失（19del）和外显子21点突变（L858R）。",
        "guideline": "NCCN/CSCO指南推荐：EGFR敏感突变阳性肺癌一线推荐三代TKI奥希替尼（Osimertinib），OS获益显著。对于19del亚型，奥希替尼较一代TKI中位OS延长超6个月。",
    },
    "白细胞": {
        "explanation": "白细胞计数（WBC）正常范围3.5-9.5×10⁹/L。升高提示感染、炎症或血液系统疾病，降低提示骨髓抑制或免疫缺陷。",
        "guideline": "WBC轻度升高（<15）且CRP正常，可暂观察。WBC>15或CRP明显升高建议抗感染治疗并追查病因。",
    },
    "CYFRA21-1": {
        "explanation": "细胞角蛋白19片段（CYFRA21-1）正常值<3.3 ng/mL，是非小细胞肺癌（尤其是鳞癌）较为特异的肿瘤标志物。",
        "guideline": "CYFRA21-1升高需结合影像学和病理学综合判断。轻度升高（3.3-10）可见于肺炎、肺结核等良性疾病，但持续升高或动态增高应警惕恶性肿瘤。",
    },
    "CEA": {
        "explanation": "癌胚抗原（CEA）正常值<5.0 ng/mL，在肺腺癌中阳性率约为50-70%。CEA在肝脏代谢，肝功能异常时也可升高。",
        "guideline": "CEA>5.0 建议进一步检查。CEA>10.0 高度提示恶性肿瘤。CEA持续缓慢上升或动态监测倍增时间<2个月，提示恶性可能性大。吸烟者CEA可轻度升高（一般<10）。",
    },
    "肺癌": {
        "explanation": "肺癌是全球癌症死亡首位原因。非小细胞肺癌（NSCLC）占85%，小细胞肺癌（SCLC）占15%。早期NSCLC（I-IIA期）5年生存率可达60-90%。",
        "guideline": "I期NSCLC首选手术治疗（肺叶切除+系统淋巴结清扫）。IB-IIIA期术后辅助化疗±靶向治疗。晚期NSCLC根据驱动基因状态选择靶向治疗或免疫+化疗。",
    },
    "PD-L1": {
        "explanation": "PD-L1是免疫检查点蛋白，肿瘤细胞PD-L1高表达（TPS≥50%）提示对免疫治疗应答率较高（约45%）。",
        "guideline": "PD-L1 TPS≥50%且无驱动基因突变：一线推荐帕博利珠单抗单药。PD-L1 1-49%：推荐免疫+化疗联合方案。注意：EGFR/ALK阳性患者免疫治疗获益有限，应优先靶向治疗。",
    },
}


def retrieve_medical_guidelines(
    query: str,
    agent_results: Optional[dict[str, Any]] = None,
) -> list[dict[str, str]]:
    """
    根据查询关键词和 Agent 分析结果，检索相关医学知识

    MVP 阶段：从 Mock 知识字典中匹配。
    # TODO: 接入 Chroma/FAISS 向量数据库 + 真实医学文档
    # TODO: 接入医学知识图谱 (UMLS, SNOMED CT)

    参数:
        query: 查询文本（用户输入 + Agent 发现的关键词）
        agent_results: 各 Agent 的分析结果（用于提取补充关键词）

    返回:
        list[dict]: 匹配到的知识条目列表，每条含 keyword, explanation, guideline
    """
    matched: list[dict[str, str]] = []
    search_text = query.lower()

    # 从 Agent 结果中提取额外关键词
    if agent_results:
        for agent_output in agent_results.values():
            if hasattr(agent_output, "findings"):
                for finding in agent_output.findings:
                    search_text += " " + finding.lower()
            if hasattr(agent_output, "abnormal_metrics"):
                for metric in agent_output.abnormal_metrics:
                    search_text += " " + metric.name.lower()

    # 关键词匹配
    for keyword, knowledge in MOCK_KNOWLEDGE_BASE.items():
        if keyword.lower() in search_text:
            matched.append({
                "keyword": keyword,
                "explanation": knowledge["explanation"],
                "guideline": knowledge["guideline"],
            })

    if not matched:
        # 默认返回通用知识
        matched.append({
            "keyword": "通用",
            "explanation": "建议结合临床症状、实验室检查和影像学资料综合分析。",
            "guideline": "如有异常发现，建议尽早就诊于专科门诊，获取专业医师的诊治意见。",
        })

    logger.info("RAG 检索完成，匹配到 %d 条知识", len(matched))
    return matched


def format_guidelines_for_prompt(guidelines: list[dict[str, str]]) -> str:
    """
    将知识检索结果格式化为 LLM Prompt 可用的文本段

    参数:
        guidelines: retrieve_medical_guidelines() 的返回值

    返回:
        str: 格式化后的知识参考文本
    """
    if not guidelines:
        return "暂无相关医学知识参考。"

    sections = []
    for i, g in enumerate(guidelines, 1):
        sections.append(
            f"### 参考 {i}：{g['keyword']}\n"
            f"**医学解释**：{g['explanation']}\n"
            f"**诊疗指南**：{g['guideline']}"
        )
    return "\n\n".join(sections)
