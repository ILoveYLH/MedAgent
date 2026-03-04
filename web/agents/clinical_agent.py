"""
临床诊断 Agent (Mock)

根据患者主诉和病史，进行初步临床诊断评估。
当前为 Mock 实现，使用 time.sleep 模拟处理耗时，返回预设结构化数据。

未来升级：
    - 接入真实的临床诊断推理引擎
    - 结合患者电子病历数据库 (EMR)
    - 接入临床决策支持系统 (CDSS)
"""

import logging
import time
from typing import Any, Optional

from web.agents.base import AbnormalMetric, AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)


class ClinicalAgent:
    """
    临床诊断 Agent

    职责：
        - 根据患者主诉、症状和既往病史进行综合分析
        - 输出初步临床印象和需要关注的异常指标
        - 提供鉴别诊断建议

    参数:
        simulate_delay: 模拟处理延迟时间（秒），用于 Demo 演示
    """

    AGENT_NAME = "ClinicalAgent"
    DISPLAY_NAME = "🩺 临床诊断"

    def __init__(self, simulate_delay: float = 2.5) -> None:
        self.simulate_delay = simulate_delay

    def run(
        self,
        text: str,
        patient_profile: Optional[dict[str, Any]] = None,
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行临床诊断分析

        参数:
            text: 用户输入的症状描述或就诊诉求
            patient_profile: 患者基本档案（来自 mock_data.json）
            on_status: 状态回调函数 (agent_name, status) -> None

        返回:
            BaseAgentOutput: 统一格式的临床诊断结果
        """
        logger.info("[%s] 开始临床分析...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()
        # ── 模拟处理耗时 ──
        time.sleep(self.simulate_delay)

        # ── Mock 诊断逻辑 ──
        findings = [
            "患者男性，58岁，有30年吸烟史（已戒3年），属肺癌高危人群",
            "右肺上叶结节随访1年内从6mm增大至8mm，增长速率偏高",
            "间断性干咳2周，无咯血，症状非特异性但需警惕",
            "结合肿瘤标志物（CEA 4.8 接近上限，CYFRA21-1 3.8 偏高），需高度关注",
            "建议完善薄层CT增强扫描及PET-CT检查，必要时行穿刺活检",
        ]

        abnormal_metrics = [
            AbnormalMetric(
                name="肺结节增长速率",
                value="6mm → 8mm / 10个月",
                reference_range="<1mm/年 为低风险",
                severity="moderate",
                description="实性成分增大，需排除恶性可能",
            ),
            AbnormalMetric(
                name="CYFRA21-1",
                value="3.8 ng/mL",
                reference_range="0 - 3.3 ng/mL",
                severity="mild",
                description="细胞角蛋白片段升高，常见于非小细胞肺癌",
            ),
            AbnormalMetric(
                name="CEA",
                value="4.8 ng/mL",
                reference_range="0 - 5.0 ng/mL",
                severity="mild",
                description="癌胚抗原接近上限，需动态监测",
            ),
        ]

        elapsed = time.time() - start
        logger.info("[%s] 临床分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

        output = BaseAgentOutput(
            agent_name=self.AGENT_NAME,
            agent_display_name=self.DISPLAY_NAME,
            status=AgentStatus.SUCCESS,
            findings=findings,
            abnormal_metrics=abnormal_metrics,
            confidence=0.78,
            processing_time=round(elapsed, 2),
            raw_data={
                "risk_level": "中高风险",
                "primary_suspicion": "右肺上叶结节 — 待排非小细胞肺癌",
                "differential_diagnosis": [
                    "肺腺癌（原位/微浸润）",
                    "不典型腺瘤样增生（AAH）",
                    "炎性假瘤",
                    "肺结核球",
                ],
            },
        )

        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.SUCCESS)

        return output
