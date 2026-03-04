"""
LLM 流式报告生成模块

将「用户档案 + Router 问题 + 并发 Agent 结构化数据 + RAG 指南」
拼接为完整的 Prompt，调用 Qwen/OpenAI 兼容 API，以流式方式输出。

如果 API Key 不可用，降级为模板报告。
"""

import logging
import os
import sys
from datetime import datetime
from typing import Any, Generator, Optional

# 让 web/ 可以引用 app/ 模块
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from web.agents.base import BaseAgentOutput
from web.rag_reducer import format_guidelines_for_prompt

logger = logging.getLogger(__name__)


# ── 主治医师 Prompt 模板 ──
ATTENDING_PHYSICIAN_PROMPT = """你是一位经验丰富的三甲医院主治医师，正在根据多学科团队（MDT）的会诊结果，为患者撰写综合诊疗报告。

## 患者基本信息
{patient_info}

## 患者当前主诉
{user_query}

## 多学科会诊结果

{agent_results_text}

## 医学知识参考
{rag_guidelines}

## 撰写要求
请以主治医师的口吻，撰写一份**综合诊疗报告**，要求：
1. **综合分析**：整合临床、影像、血液、基因各方面的发现
2. **风险评估**：给出明确的风险分级和依据
3. **诊疗建议**：按优先级列出下一步检查和治疗方案
4. **随访计划**：制定具体的随访时间表
5. 使用 Markdown 格式，结构清晰
6. 在报告末尾添加免责声明

当前日期：{current_date}
"""


def _format_patient_info(patient_profile: Optional[dict[str, Any]]) -> str:
    """将患者档案格式化为 Prompt 文本"""
    if not patient_profile:
        return "暂无患者详细信息"

    p = patient_profile
    lines = [
        f"- 姓名：{p.get('name', '未知')}",
        f"- 性别：{p.get('gender', '未知')}",
        f"- 年龄：{p.get('age', '未知')}岁",
        f"- 血型：{p.get('blood_type', '未知')}",
        f"- 吸烟史：{p.get('smoking_history', '未知')}",
        f"- 过敏史：{'、'.join(p.get('allergies', ['无']))}",
    ]
    return "\n".join(lines)


def _format_agent_results(results: dict[str, BaseAgentOutput]) -> str:
    """将所有 Agent 结果格式化为 Prompt 文本段"""
    sections = []
    for agent_name, output in results.items():
        section = f"### {output.agent_display_name or agent_name}\n"
        section += f"**置信度**: {output.confidence:.0%}\n\n"

        if output.findings:
            section += "**主要发现**：\n"
            for f in output.findings:
                section += f"- {f}\n"

        if output.abnormal_metrics:
            section += "\n**异常指标**：\n"
            for m in output.abnormal_metrics:
                section += f"- **{m.name}**: {m.value}（参考: {m.reference_range}）— {m.description}\n"

        sections.append(section)

    return "\n\n".join(sections)


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
        patient_profile: 患者基本信息
        agent_results: 各 Agent 的分析结果
        rag_guidelines: RAG 检索到的医学知识

    返回:
        str: 完整的 Prompt 文本
    """
    return ATTENDING_PHYSICIAN_PROMPT.format(
        patient_info=_format_patient_info(patient_profile),
        user_query=user_query,
        agent_results_text=_format_agent_results(agent_results),
        rag_guidelines=format_guidelines_for_prompt(rag_guidelines),
        current_date=datetime.now().strftime("%Y年%m月%d日 %H:%M"),
    )


def stream_report(prompt: str) -> Generator[str, None, None]:
    """
    调用 LLM API 流式生成报告

    优先使用 Qwen（通义千问）API，降级为 Google Gemini，最终降级为模板报告。

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
    """降级模板报告生成器，模拟打字机效果"""
    import time

    report = f"""# 📋 综合诊疗报告

## 基本信息
- **报告日期**：{datetime.now().strftime("%Y年%m月%d日 %H:%M")}
- **报告类型**：多学科会诊综合报告
- **报告系统**：MedAgent AI 辅助诊断系统 v2.0

## 综合分析

### 影像学评估
右肺上叶后段可见一枚混合密度结节，大小约 8×7mm，较前次检查（2023年6月，6×6mm）有所增大。结节边缘可见短毛刺征，内部密度不均匀，实性成分占比约30%，周围可见轻度磨玻璃晕征。纵隔淋巴结未见明显增大。

### 血液指标评估
血常规各项正常，感染指标（CRP）正常。肿瘤标志物 CYFRA21-1 轻度升高（3.8 ng/mL），CEA 接近正常上限（4.8 ng/mL），NSE 正常上限。

### 基因检测评估
EGFR 外显子19缺失突变阳性（突变丰度15.2%），ALK/ROS1/KRAS阴性，PD-L1 TPS 30%。

## 风险评估

**综合风险等级：中高风险** ⚠️

依据：
1. 结节10个月内增大2mm，增长速率偏高
2. 混合密度结节伴毛刺征，恶性概率约40-60%
3. 肿瘤标志物 CYFRA21-1 升高
4. EGFR 驱动基因突变阳性
5. 患者有30年吸烟史，属高危人群

## 诊疗建议

### 立即执行（1-2周内）
1. **薄层CT增强扫描**：进一步评估结节血供特征和纵隔淋巴结情况
2. **PET-CT检查**：评估结节代谢活性和全身是否存在其他病灶

### 后续计划（2-4周内）
3. **CT引导下经皮穿刺活检**：获取组织病理学诊断
4. **完善术前评估**：肺功能检查、心脏评估

### 如确诊早期NSCLC
5. **手术治疗**：建议胸腔镜肺叶切除术 + 系统淋巴结清扫
6. **靶向治疗**：EGFR 19del 阳性，术后辅助/晚期一线推荐奥希替尼

## 随访计划
| 时间节点 | 检查项目 |
|---------|---------|
| 2周内 | 薄层CT增强 + PET-CT |
| 1个月 | 穿刺活检病理结果 |
| 术后3个月 | 胸部CT + 肿瘤标志物 |
| 术后6个月 | 胸部CT + 全面复查 |
| 术后1-2年 | 每6个月CT + 肿瘤标志物 |

---
⚠️ **免责声明**：本报告由 MedAgent AI 辅助生成，所有分析结果和诊疗建议仅供参考，不能替代专业医师的临床诊断和治疗决策。请务必咨询您的主治医师获取个体化的诊疗方案。
"""
    # 模拟打字机效果
    paragraphs = report.split("\n")
    for para in paragraphs:
        if para.strip():
            yield para + "\n"
            time.sleep(0.05)
        else:
            yield "\n"
            time.sleep(0.02)
