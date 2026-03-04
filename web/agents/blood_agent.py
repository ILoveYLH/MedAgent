"""
血液检验分析 Agent (Mock)

分析患者血液检验结果，识别异常指标并给出临床提示。
当前为 Mock 实现，返回预设的异常血液指标数据。

未来升级：
    - 接入医院 LIS (检验信息系统) 数据接口
    - 集成自动异常指标解读引擎
    - 支持趋势分析（多次检验结果对比）
"""

import logging
import time
from typing import Any, Optional

from web.agents.base import AbnormalMetric, AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)


class BloodAgent:
    """
    血液检验分析 Agent

    职责：
        - 解析血常规、肿瘤标志物、生化指标
        - 标记异常值并给出临床参考意义
        - 辅助鉴别诊断

    参数:
        simulate_delay: 模拟处理延迟时间（秒）
    """

    AGENT_NAME = "BloodAgent"
    DISPLAY_NAME = "🩸 血液分析"

    def __init__(self, simulate_delay: float = 2.0) -> None:
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
        执行血液检验分析

        参数:
            text: 用户描述或补充信息
            attachments: 上传的检验报告文件路径列表
            patient_profile: 患者档案
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的血液分析结果
        """
        logger.info("[%s] 开始血液检验分析...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()
        time.sleep(self.simulate_delay)

        # ── Mock 分析结果 ──
        findings = [
            "血常规各项指标（WBC、RBC、HGB、PLT）均在正常范围内",
            "CRP 2.1 mg/L，感染指标正常，排除急性感染可能",
            "肿瘤标志物 CYFRA21-1 升高至 3.8 ng/mL（正常 <3.3），提示角蛋白代谢异常",
            "CEA 4.8 ng/mL 接近临界值（正常 <5.0），建议动态监测",
            "NSE 16.2 ng/mL 处于正常上限（正常 <16.3），暂无特殊意义",
        ]

        abnormal_metrics = [
            AbnormalMetric(
                name="CYFRA21-1",
                value="3.8 ng/mL",
                reference_range="0 - 3.3 ng/mL",
                severity="moderate",
                description="细胞角蛋白19片段，升高常见于非小细胞肺癌（鳞癌敏感性较高）",
            ),
            AbnormalMetric(
                name="CEA",
                value="4.8 ng/mL",
                reference_range="0 - 5.0 ng/mL",
                severity="mild",
                description="癌胚抗原接近上限，腺癌相关标志物，建议2周后复查",
            ),
            AbnormalMetric(
                name="NSE",
                value="16.2 ng/mL",
                reference_range="0 - 16.3 ng/mL",
                severity="mild",
                description="神经元特异性烯醇化酶，正常上限，排除小细胞肺癌可能性较大",
            ),
        ]

        elapsed = time.time() - start
        logger.info("[%s] 血液分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

        output = BaseAgentOutput(
            agent_name=self.AGENT_NAME,
            agent_display_name=self.DISPLAY_NAME,
            status=AgentStatus.SUCCESS,
            findings=findings,
            abnormal_metrics=abnormal_metrics,
            confidence=0.82,
            processing_time=round(elapsed, 2),
            raw_data={
                "blood_routine": {
                    "WBC": {"value": 7.2, "status": "normal"},
                    "RBC": {"value": 4.8, "status": "normal"},
                    "HGB": {"value": 145, "status": "normal"},
                    "PLT": {"value": 220, "status": "normal"},
                },
                "tumor_markers": {
                    "CEA": {"value": 4.8, "status": "borderline"},
                    "NSE": {"value": 16.2, "status": "borderline"},
                    "CYFRA21-1": {"value": 3.8, "status": "high"},
                },
                "infection_markers": {
                    "CRP": {"value": 2.1, "status": "normal"},
                },
            },
        )

        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.SUCCESS)

        return output
