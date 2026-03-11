"""
临床诊断 Agent

根据患者主诉和病史，进行初步临床诊断评估。
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

CLINICAL_SYSTEM_PROMPT = """你是一位经验丰富的临床医师，负责根据患者提供的症状、病史和检查信息进行临床诊断分析。

请根据用户输入，提取关键临床信息并进行专业分析。以 JSON 格式返回结果，结构如下：
{
  "findings": ["发现1", "发现2", "发现3"],
  "abnormal_metrics": [
    {
      "name": "指标名称",
      "value": "指标值",
      "reference_range": "参考范围",
      "severity": "mild",
      "description": "临床意义说明"
    }
  ],
  "confidence": 0.75,
  "risk_level": "中高风险",
  "primary_suspicion": "初步诊断印象",
  "differential_diagnosis": ["鉴别诊断1", "鉴别诊断2"]
}

severity 只能是 mild、moderate 或 severe 之一。
只返回 JSON 对象，不要有其他文字。findings 至少提供3条专业分析。"""


class ClinicalAgent:
    """
    临床诊断 Agent

    职责：
        - 根据患者主诉、症状和既往病史进行综合分析
        - 输出初步临床印象和需要关注的异常指标
        - 提供鉴别诊断建议
    """

    AGENT_NAME = "ClinicalAgent"
    DISPLAY_NAME = "🩺 临床诊断"

    def __init__(self) -> None:
        pass

    def run(
        self,
        text: str,
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行临床诊断分析

        参数:
            text: 用户输入的症状描述或就诊诉求
            on_status: 状态回调函数 (agent_name, status) -> None

        返回:
            BaseAgentOutput: 统一格式的临床诊断结果
        """
        logger.info("[%s] 开始临床分析...", self.AGENT_NAME)
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
                    {"role": "system", "content": CLINICAL_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=1500,
            )

            content = response.choices[0].message.content.strip()
            # 去除可能的 markdown 代码块包裹
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
            confidence = float(data.get("confidence", 0.75))
            raw_data = {
                "risk_level": data.get("risk_level", ""),
                "primary_suspicion": data.get("primary_suspicion", ""),
                "differential_diagnosis": data.get("differential_diagnosis", []),
            }

            elapsed = time.time() - start
            logger.info("[%s] 临床分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

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
