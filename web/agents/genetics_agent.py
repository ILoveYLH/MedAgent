"""
基因检测分析 Agent

分析患者基因突变检测结果，辅助靶向治疗方案制定。
调用 LLM（Qwen API）基于用户真实输入进行专业分析。
"""

import json
import logging
import os
import re
import time
from typing import Any

from web.agents.base import AbnormalMetric, AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)

GENETICS_SYSTEM_PROMPT = """你是一位基因检测与分子病理科医师，负责分析患者提供的基因检测数据，识别驱动突变并给出靶向治疗建议。

请根据用户输入，提取基因检测相关信息并进行专业分析。以 JSON 格式返回结果，结构如下：
{
  "findings": ["发现1", "发现2", "发现3"],
  "abnormal_metrics": [
    {
      "name": "基因/标志物名称（如 EGFR 19del、PD-L1 TPS）",
      "value": "检测结果",
      "reference_range": "正常参考值",
      "severity": "moderate",
      "description": "临床意义和治疗建议"
    }
  ],
  "confidence": 0.85,
  "recommended_targets": ["推荐靶向药1", "推荐靶向药2"]
}

severity 只能是 mild、moderate 或 severe 之一。
只返回 JSON 对象，不要有其他文字。findings 至少提供3条专业分析。
如果用户输入中没有明确的基因检测数据，请根据其描述推断可能的相关检测建议。"""


class GeneticsAgent:
    """
    基因检测分析 Agent

    职责：
        - 分析常见驱动基因突变（EGFR, ALK, ROS1, KRAS 等）
        - 评估靶向治疗适应症
        - 提供免疫治疗相关生物标志物信息
    """

    AGENT_NAME = "GeneticsAgent"
    DISPLAY_NAME = "🧬 基因检测"

    def __init__(self) -> None:
        pass

    def run(
        self,
        text: str = "",
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行基因检测分析

        参数:
            text: 用户描述或基因检测数据
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的基因检测结果
        """
        logger.info("[%s] 开始基因检测分析...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()

        qwen_key = os.getenv("QWEN_API_KEY", "")
        if not qwen_key:
            elapsed = time.time() - start
            if on_status:
                on_status(self.AGENT_NAME, AgentStatus.FAILED)
            return BaseAgentOutput(
                agent_name=self.AGENT_NAME,
                agent_display_name=self.DISPLAY_NAME,
                status=AgentStatus.FAILED,
                processing_time=round(elapsed, 2),
                error_message="LLM 不可用：未配置 QWEN_API_KEY",
            )

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
                    {"role": "system", "content": GENETICS_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=1500,
            )

            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                match = re.search(r'\{.*\}', content, re.DOTALL)
                if not match:
                    raise ValueError(f"无法从 LLM 响应中提取 JSON: {content[:200]}")
                content = match.group(0)

            data = json.loads(content)

            findings = data.get("findings", [])
            abnormal_metrics = [
                AbnormalMetric(
                    name=m.get("name", ""),
                    value=m.get("value", ""),
                    reference_range=m.get("reference_range", ""),
                    severity=m.get("severity", "mild"),
                    description=m.get("description", ""),
                )
                for m in data.get("abnormal_metrics", [])
            ]
            confidence = float(data.get("confidence", 0.85))
            raw_data = {
                "recommended_targets": data.get("recommended_targets", []),
                "raw_response": data,
            }

            elapsed = time.time() - start
            logger.info("[%s] 基因检测分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

            output = BaseAgentOutput(
                agent_name=self.AGENT_NAME,
                agent_display_name=self.DISPLAY_NAME,
                status=AgentStatus.SUCCESS,
                findings=findings,
                abnormal_metrics=abnormal_metrics,
                confidence=confidence,
                processing_time=round(elapsed, 2),
                raw_data=raw_data,
            )

            if on_status:
                on_status(self.AGENT_NAME, AgentStatus.SUCCESS)

            return output

        except Exception as exc:
            elapsed = time.time() - start
            logger.error("[%s] 分析失败: %s", self.AGENT_NAME, exc)
            if on_status:
                on_status(self.AGENT_NAME, AgentStatus.FAILED)
            return BaseAgentOutput(
                agent_name=self.AGENT_NAME,
                agent_display_name=self.DISPLAY_NAME,
                status=AgentStatus.FAILED,
                processing_time=round(elapsed, 2),
                error_message=str(exc),
            )
