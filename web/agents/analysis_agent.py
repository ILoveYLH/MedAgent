"""
全能医学分析 Agent (AnalysisAgent)

替代原来的 Clinical/Blood/Genetics 三个 Mock Agent。
接收用户在聊天框中输入的所有文本数据（血液指标、基因检测结果、症状描述等），
由 LLM（Qwen）自行决定从临床、血液、基因等角度进行综合分析。
"""

import json
import logging
import os
import time
from typing import Any, Optional

from web.agents.base import AbnormalMetric, AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """你是一位经验丰富的三甲医院多学科主任医师，擅长临床诊断、血液检验解读和基因检测分析。

用户会给你提供医疗信息（可能包括：症状描述、血液检验数据、基因检测结果、病史等）。

请你：
1. **临床角度**：分析症状、体征和病史，给出初步临床印象
2. **血液角度**（如有血液数据）：解读血常规、肿瘤标志物等指标，标注异常值
3. **基因角度**（如有基因数据）：分析驱动基因突变情况，评估靶向治疗适应症
4. **综合判断**：给出风险评估和诊疗建议

请以 JSON 格式返回结果，格式如下：
{
  "findings": ["发现1", "发现2", ...],
  "abnormal_metrics": [
    {
      "name": "指标名称",
      "value": "指标值",
      "reference_range": "参考范围",
      "severity": "mild|moderate|severe",
      "description": "临床意义"
    }
  ],
  "confidence": 0.85,
  "risk_level": "低风险|中风险|中高风险|高风险",
  "primary_impression": "主要临床印象"
}

只返回 JSON，不要返回其他内容。如果某项数据不存在，对应字段返回空列表或默认值。"""


class AnalysisAgent:
    """
    全能医学分析 Agent

    职责：
        - 综合分析用户提供的所有医疗数据（症状、血液、基因等）
        - 由 LLM 自行判断需要从哪些角度进行分析
        - 输出结构化的诊断发现和异常指标
    """

    AGENT_NAME = "AnalysisAgent"
    DISPLAY_NAME = "🩺 综合分析"

    def __init__(self) -> None:
        pass

    def run(
        self,
        text: str = "",
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行综合医学分析

        参数:
            text: 用户输入的医疗数据描述
            on_status: 状态回调函数

        返回:
            BaseAgentOutput: 统一格式的分析结果
        """
        logger.info("[%s] 开始综合医学分析...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()

        # 尝试 LLM 分析
        llm_result = self._run_with_llm(text)

        if llm_result:
            findings = llm_result.get("findings", [])
            raw_abnormal = llm_result.get("abnormal_metrics", [])
            confidence = float(llm_result.get("confidence", 0.75))
            raw_data = {
                "risk_level": llm_result.get("risk_level", "未知"),
                "primary_impression": llm_result.get("primary_impression", ""),
            }

            abnormal_metrics = []
            for m in raw_abnormal:
                try:
                    abnormal_metrics.append(
                        AbnormalMetric(
                            name=m.get("name", ""),
                            value=str(m.get("value", "")),
                            reference_range=m.get("reference_range", ""),
                            severity=m.get("severity", "mild"),
                            description=m.get("description", ""),
                        )
                    )
                except Exception:
                    pass

            status = AgentStatus.SUCCESS
            error_message = None
        else:
            # 降级：仅返回提示信息
            findings = ["LLM 分析服务暂不可用，请配置 QWEN_API_KEY 以启用智能分析。"]
            abnormal_metrics = []
            confidence = 0.0
            raw_data = {}
            status = AgentStatus.FAILED
            error_message = "LLM 服务不可用"

        elapsed = time.time() - start
        logger.info("[%s] 分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

        output = BaseAgentOutput(
            agent_name=self.AGENT_NAME,
            agent_display_name=self.DISPLAY_NAME,
            status=status,
            findings=findings,
            abnormal_metrics=abnormal_metrics,
            confidence=confidence,
            processing_time=round(elapsed, 2),
            raw_data=raw_data,
            error_message=error_message,
        )

        if on_status:
            on_status(self.AGENT_NAME, output.status)

        return output

    def _run_with_llm(self, text: str) -> Optional[dict[str, Any]]:
        """使用 LLM 进行分析，失败时返回 None"""
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
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            content = response.choices[0].message.content.strip()
            logger.debug("[%s] LLM 原始返回: %s", self.AGENT_NAME, content[:200])

            # 提取 JSON（处理可能的 markdown 代码块）
            import re
            if "```" in content or not content.startswith("{"):
                match = re.search(r"\{.*\}", content, re.DOTALL)
                if match:
                    content = match.group(0)

            return json.loads(content)

        except json.JSONDecodeError as exc:
            logger.warning("[%s] LLM 返回 JSON 解析失败: %s", self.AGENT_NAME, exc)
            return None
        except Exception as exc:
            logger.warning("[%s] LLM 调用失败: %s", self.AGENT_NAME, exc)
            return None
