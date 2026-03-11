"""
LLM 流式报告生成模块

将「用户输入 + 并发 Agent 结构化数据 + RAG 指南」
拼接为完整的 Prompt，调用 Qwen/OpenAI 兼容 API，以流式方式输出。

如果 API Key 不可用，降级为模板报告。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Generator, Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from web.agents.base import BaseAgentOutput
from web.rag_reducer import format_guidelines_for_prompt

logger = logging.getLogger(__name__)


# ── 主治医师 Prompt 模板 ──
ATTENDING_PHYSICIAN_PROMPT = """你是一位经验丰富的三甲医院主治医师，正在根据多学科团队（MDT）的会诊结果，为患者撰写综合诊疗报告。

## 患者当前主诉
{user_query}

## 多学科会诊结果

{agent_results_text}

## 医学知识参考
{rag_guidelines}

## 撰写要求
请以主治医师的口吻，撰写一份**综合诊疗报告**，要求：
1. **综合分析**：整合所有 Agent 的发现（影像、症状、血液、基因等）
2. **风险评估**：给出明确的风险分级和依据
3. **诊疗建议**：按优先级列出下一步检查和治疗方案
4. **随访计划**：制定具体的随访时间表
5. 使用 Markdown 格式，结构清晰
6. 在报告末尾添加免责声明

当前日期：{current_date}
"""


def _format_agent_results(results: dict[str, BaseAgentOutput]) -> str:
    """将所有 Agent 结果格式化为 Prompt 文本段"""
    sections = []
    for agent_name, output in results.items():
        if output.status.value == "failed":
            continue
        section = f"### {output.agent_display_name or agent_name}\n"
        if output.confidence > 0:
            section += f"**置信度**: {output.confidence:.0%}\n\n"

        if output.findings:
            section += "**主要发现**：\n"
            for f in output.findings:
                section += f"- {f}\n"

        if output.abnormal_metrics:
            section += "\n**异常指标**：\n"
            for m in output.abnormal_metrics:
                section += (
                    f"- **{m.name}**: {m.value}"
                    + (f"（参考: {m.reference_range}）" if m.reference_range else "")
                    + (f" — {m.description}" if m.description else "")
                    + "\n"
                )

        if output.raw_data:
            extra = []
            if "risk_level" in output.raw_data:
                extra.append(f"风险等级：{output.raw_data['risk_level']}")
            if "primary_impression" in output.raw_data:
                extra.append(f"临床印象：{output.raw_data['primary_impression']}")
            if extra:
                section += "\n" + "\n".join(extra) + "\n"

        sections.append(section)

    return "\n\n".join(sections) if sections else "暂无 Agent 分析结果"


def assemble_prompt(
    user_query: str,
    patient_profile: Optional[dict[str, Any]],
    agent_results: dict[str, BaseAgentOutput],
    rag_guidelines: list[dict[str, str]],
) -> str:
    """
    组装完整的 LLM Prompt

    参数:
        user_query: 用户原始输入
        patient_profile: 患者基本信息（可为 None）
        agent_results: 各 Agent 的分析结果
        rag_guidelines: RAG 检索到的医学知识

    返回:
        str: 完整的 Prompt 文本
    """
    return ATTENDING_PHYSICIAN_PROMPT.format(
        user_query=user_query,
        agent_results_text=_format_agent_results(agent_results),
        rag_guidelines=format_guidelines_for_prompt(rag_guidelines),
        current_date=datetime.now().strftime("%Y年%m月%d日 %H:%M"),
    )


def stream_report(prompt: str) -> Generator[str, None, None]:
    """
    调用 LLM API 流式生成报告

    优先使用 Qwen API，降级为 Google Gemini，最终降级为模板报告。

    参数:
        prompt: 完整的 LLM Prompt

    返回:
        Generator[str]: 每次 yield 一小段文本（流式输出）
    """
    # ─────── 尝试 Qwen API ───────
    qwen_key = os.getenv("QWEN_API_KEY", "")
    if qwen_key:
        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=qwen_key,
                base_url=os.getenv(
                    "QWEN_BASE_URL",
                    "https://dashscope.aliyuncs.com/compatible-mode/v1",
                ),
            )
            model = os.getenv("QWEN_MODEL_NAME", "qwen-plus")

            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                stream=True,
            )

            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

            logger.info("Qwen 流式报告生成完成")
            return
        except Exception as exc:
            logger.warning("Qwen API 调用失败: %s，尝试降级", exc)

    # ─────── 尝试 Google Gemini ───────
    google_key = os.getenv("GOOGLE_API_KEY", "")
    if google_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=google_key)
            model_name = os.getenv("MODEL_NAME", "gemini-2.0-flash")
            model = genai.GenerativeModel(model_name)

            response = model.generate_content(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    yield chunk.text

            logger.info("Gemini 流式报告生成完成")
            return
        except Exception as exc:
            logger.warning("Gemini API 调用失败: %s，使用模板报告", exc)

    # ─────── 降级：模板报告 ───────
    yield from _template_report_generator()


def _template_report_generator() -> Generator[str, None, None]:
    """降级模板报告生成器"""
    import time

    report = f"""# 📋 综合诊疗报告

## 报告信息
- **报告日期**：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- **报告类型**：多学科会诊综合报告（模板）
- **提示**：请配置 QWEN_API_KEY 或 GOOGLE_API_KEY 以启用 AI 智能报告生成

## 综合分析

本报告基于您提供的数据生成。由于 LLM 服务暂不可用，以下为通用建议。

## 诊疗建议

1. 建议到专科门诊进行进一步评估
2. 完善相关实验室检查和影像学检查
3. 如有异常指标，请及时就医

## 随访计划

- 根据具体情况制定个体化随访方案
- 如有不适，随时就诊

---
⚠️ **免责声明**：本报告由 MedAgent AI 辅助生成，所有分析结果和诊疗建议仅供参考，不能替代专业医师的临床诊断和治疗决策。请务必咨询您的主治医师获取个体化的诊疗方案。
"""
    paragraphs = report.split("\n")
    for para in paragraphs:
        if para.strip():
            yield para + "\n"
            time.sleep(0.03)
        else:
            yield "\n"
            time.sleep(0.01)
