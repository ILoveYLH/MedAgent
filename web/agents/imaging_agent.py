"""
影像分析 Agent

根据用户描述的影像信息进行分析，或对文本描述的CT/MRI等影像学发现进行专业解读。
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

IMAGING_SYSTEM_PROMPT = """你是一位经验丰富的影像科医师，负责根据患者提供的影像学描述或检查结果进行专业分析。

请根据用户输入，提取影像相关信息并进行专业分析。以 JSON 格式返回结果，结构如下：
{
  "findings": ["发现1", "发现2", "发现3"],
  "abnormal_metrics": [
    {
      "name": "影像指标名称（如 结节大小、CT值）",
      "value": "测量值",
      "reference_range": "正常参考范围",
      "severity": "moderate",
      "description": "影像学意义说明"
    }
  ],
  "confidence": 0.82,
  "classification": "影像学诊断印象",
  "malignancy_probability": "恶性概率估计（如 40-60%）"
}

severity 只能是 mild、moderate 或 severe 之一。
只返回 JSON 对象，不要有其他文字。findings 至少提供3条专业影像学分析。
如果用户输入中包含CT、MRI、X光等影像学描述，请重点分析病灶特征、大小、位置和性质。"""


class ImagingAgent:
    """
    影像分析 Agent

    职责：
        - 根据用户描述的影像学信息进行专业解读
        - 分析病灶特征（大小、密度、边缘等）
        - 评估恶性概率并给出随访建议
    """

    AGENT_NAME = "ImagingAgent"
    DISPLAY_NAME = "🔬 影像分析"

    def __init__(self) -> None:
        pass

    def run(
        self,
        text: str = "",
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行影像分析

        参数:
            text: 用户输入的影像学描述或检查结果
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的影像分析结果
        """
        logger.info("[%s] 开始影像分析...", self.AGENT_NAME)
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
                    {"role": "system", "content": IMAGING_SYSTEM_PROMPT},
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
            confidence = float(data.get("confidence", 0.82))
            raw_data = {
                "classification": data.get("classification", ""),
                "malignancy_probability": data.get("malignancy_probability", ""),
                "raw_response": data,
            }

            elapsed = time.time() - start
            logger.info("[%s] 影像分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

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
