"""
血液检验分析 Agent

分析患者血液检验结果，识别异常指标并给出临床提示。
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

BLOOD_SYSTEM_PROMPT = """你是一位血液检验科医师，负责分析患者提供的血液检验数据，识别异常指标并给出临床提示。

请根据用户输入，提取血液相关检验指标并进行专业分析。以 JSON 格式返回结果，结构如下：
{
  "findings": ["发现1", "发现2", "发现3"],
  "abnormal_metrics": [
    {
      "name": "指标名称（如 CEA、CYFRA21-1、WBC）",
      "value": "指标值（含单位）",
      "reference_range": "参考范围",
      "severity": "mild",
      "description": "临床意义说明"
    }
  ],
  "confidence": 0.80
}

severity 只能是 mild、moderate 或 severe 之一。
只返回 JSON 对象，不要有其他文字。findings 至少提供3条专业分析。
如果用户输入中没有明确的血液指标数据，请根据其描述推断可能的异常并说明。"""


class BloodAgent:
    """
    血液检验分析 Agent

    职责：
        - 解析血常规、肿瘤标志物、生化指标
        - 标记异常值并给出临床参考意义
        - 辅助鉴别诊断
    """

    AGENT_NAME = "BloodAgent"
    DISPLAY_NAME = "🩸 血液分析"

    def __init__(self) -> None:
        pass

    def run(
        self,
        text: str = "",
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行血液检验分析

        参数:
            text: 用户描述或血液检验数据
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的血液分析结果
        """
        logger.info("[%s] 开始血液检验分析...", self.AGENT_NAME)
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
                    {"role": "system", "content": BLOOD_SYSTEM_PROMPT},
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
            confidence = float(data.get("confidence", 0.80))

            elapsed = time.time() - start
            logger.info("[%s] 血液分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

            output = BaseAgentOutput(
                agent_name=self.AGENT_NAME,
                agent_display_name=self.DISPLAY_NAME,
                status=AgentStatus.SUCCESS,
                findings=findings,
                abnormal_metrics=abnormal_metrics,
                confidence=confidence,
                processing_time=round(elapsed, 2),
                raw_data={"raw_response": data},
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
