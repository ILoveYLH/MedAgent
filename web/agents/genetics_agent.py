"""
基因检测分析 Agent (Mock)

分析患者基因突变检测结果，辅助靶向治疗方案制定。
当前为 Mock 实现，返回预设的基因检测数据。

未来升级：
    - 接入 NGS (二代测序) 数据解析管线
    - 集成 ClinVar、OncoKB 等基因突变数据库
    - 支持免疫组化 (IHC) 结果联合分析
    - 集成 TMB (肿瘤突变负荷) 和 MSI (微卫星不稳定) 评估
"""

import logging
import time
from typing import Any, Optional

from web.agents.base import AbnormalMetric, AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)


class GeneticsAgent:
    """
    基因检测分析 Agent

    职责：
        - 分析常见驱动基因突变（EGFR, ALK, ROS1, KRAS 等）
        - 评估靶向治疗适应症
        - 提供免疫治疗相关生物标志物信息

    参数:
        simulate_delay: 模拟处理延迟时间（秒）
    """

    AGENT_NAME = "GeneticsAgent"
    DISPLAY_NAME = "🧬 基因检测"

    def __init__(self, simulate_delay: float = 3.5) -> None:
        self.simulate_delay = simulate_delay

    def run(
        self,
        text: str = "",
        attachments: Optional[list[str]] = None,
        patient_profile: Optional[dict[str, Any]] = None,
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行基因检测分析

        参数:
            text: 用户描述
            attachments: 基因检测报告文件路径
            patient_profile: 患者档案
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的基因检测结果
        """
        logger.info("[%s] 开始基因检测分析...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()
        time.sleep(self.simulate_delay)

        # ── Mock 分析结果 ──
        findings = [
            "EGFR 基因外显子19缺失突变（19del）阳性 — 对应一代/三代 TKI 靶向药",
            "ALK 融合基因阴性",
            "ROS1 重排阴性",
            "KRAS G12C 突变阴性",
            "PD-L1 表达：TPS 约 30%，具有一定的免疫治疗应答潜力",
        ]

        abnormal_metrics = [
            AbnormalMetric(
                name="EGFR 19del",
                value="阳性（突变丰度 15.2%）",
                reference_range="阴性",
                severity="severe",
                description="EGFR 外显子19缺失突变，一线推荐奥希替尼（Osimertinib）",
            ),
            AbnormalMetric(
                name="PD-L1 TPS",
                value="30%",
                reference_range="<1% 为阴性",
                severity="moderate",
                description="中等表达，可考虑免疫联合治疗方案",
            ),
        ]

        elapsed = time.time() - start
        logger.info("[%s] 基因检测分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

        output = BaseAgentOutput(
            agent_name=self.AGENT_NAME,
            agent_display_name=self.DISPLAY_NAME,
            status=AgentStatus.SUCCESS,
            findings=findings,
            abnormal_metrics=abnormal_metrics,
            confidence=0.92,
            processing_time=round(elapsed, 2),
            raw_data={
                "driver_mutations": {
                    "EGFR": {
                        "status": "positive",
                        "mutation": "Exon 19 deletion",
                        "allele_frequency": 0.152,
                    },
                    "ALK": {"status": "negative"},
                    "ROS1": {"status": "negative"},
                    "KRAS_G12C": {"status": "negative"},
                    "BRAF_V600E": {"status": "negative"},
                    "MET_ex14": {"status": "negative"},
                },
                "immunotherapy_markers": {
                    "PD-L1_TPS": 30,
                    "TMB": {"value": 6.5, "unit": "mut/Mb", "level": "low"},
                    "MSI": "MSS",
                },
                "recommended_targets": [
                    "奥希替尼 (Osimertinib) — EGFR TKI 三代",
                    "吉非替尼 (Gefitinib) — EGFR TKI 一代（备选）",
                ],
            },
        )

        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.SUCCESS)

        return output
