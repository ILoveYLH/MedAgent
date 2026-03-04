"""
诊断报告生成器

根据CT分析结果生成结构化诊断报告。
支持多 LLM 提供商（Google Gemini / 通义千问），通过 .env 中的 LLM_PROVIDER 切换。
"""


import json
import logging
from datetime import datetime
from typing import Any, Optional

from app.agent.prompts import REPORT_GENERATION_PROMPT
from app.config import settings

logger = logging.getLogger(__name__)


def _call_llm(prompt: str) -> str:
    """
    调用 LLM 生成文本（根据 LLM_PROVIDER 自动选择提供商）

    参数:
        prompt: 提示词文本

    返回:
        LLM 生成的文本
    """
    provider = settings.llm_provider.lower()

    if provider == "google":
        import google.generativeai as genai
        genai.configure(api_key=settings.google_api_key)
        model = genai.GenerativeModel(settings.model_name)
        response = model.generate_content(prompt)
        return response.text

    elif provider == "qwen":
        from openai import OpenAI
        client = OpenAI(
            api_key=settings.qwen_api_key,
            base_url=settings.qwen_base_url,
        )
        response = client.chat.completions.create(
            model=settings.qwen_model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return response.choices[0].message.content

    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")


def _format_classification_result(result: dict[str, Any]) -> str:
    """将分类结果格式化为可读字符串"""
    if not result or result.get("error"):
        return f"分类失败: {result.get('error', '未知错误')}"

    lines = [
        f"- 主要诊断: {result.get('disease_type', '未知')}",
        f"- 置信度: {result.get('confidence', 0):.1%}",
    ]

    all_probs = result.get("all_probabilities", {})
    if all_probs:
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        probs_str = "、".join(f"{k}({v:.1%})" for k, v in sorted_probs[:3])
        lines.append(f"- 概率分布（前3）: {probs_str}")

    lines.append(f"- 模型版本: {result.get('model_version', 'unknown')}")
    if result.get("is_mock"):
        lines.append("- ⚠️ 注：当前为模拟结果，非真实模型输出")

    return "\n".join(lines)


def _format_detection_result(result: dict[str, Any]) -> str:
    """将检测结果格式化为可读字符串"""
    if not result or result.get("error"):
        return f"检测失败: {result.get('error', '未知错误')}"

    total = result.get("total_count", 0)
    if total == 0:
        return "未发现明显病灶"

    lines = [f"共发现 {total} 处病灶："]
    for lesion in result.get("lesions", []):
        lines.append(
            f"\n  病灶 {lesion['lesion_id']}:"
            f"\n  - 类型: {lesion.get('type', '未知')}"
            f"\n  - 位置: {lesion.get('location', '未知')}"
            f"\n  - 大小: {lesion.get('size_mm', 0)} mm"
            f"\n  - 置信度: {lesion.get('confidence', 0):.1%}"
        )

    if result.get("is_mock"):
        lines.append("\n⚠️ 注：当前为模拟结果，非真实模型输出")

    return "\n".join(lines)


def _format_rag_knowledge(knowledge_chunks: list[str]) -> str:
    """将RAG检索结果格式化为可读字符串"""
    if not knowledge_chunks:
        return "暂无相关医学知识检索结果"

    lines = []
    for i, chunk in enumerate(knowledge_chunks, 1):
        lines.append(f"**参考{i}**:\n{chunk[:300]}{'...' if len(chunk) > 300 else ''}")

    return "\n\n".join(lines)


def generate_diagnosis_report(
    user_request: str,
    classification_result: dict[str, Any],
    detection_result: dict[str, Any],
    rag_knowledge: list[str],
    patient_info: Optional[dict] = None,
) -> str:
    """
    生成结构化CT诊断报告

    参数:
        user_request: 用户原始请求文本
        classification_result: CT分类结果字典
        detection_result: 病灶检测结果字典
        rag_knowledge: RAG检索到的相关知识列表
        patient_info: 患者信息字典（可选）

    返回:
        Markdown格式的结构化诊断报告字符串
    """
    current_date = datetime.now().strftime("%Y年%m月%d日 %H:%M")

    # 格式化输入数据
    cls_text = _format_classification_result(classification_result)
    det_text = _format_detection_result(detection_result)
    rag_text = _format_rag_knowledge(rag_knowledge)

    # 构建报告生成提示词
    prompt = REPORT_GENERATION_PROMPT.format(
        user_request=user_request,
        classification_result=cls_text,
        detection_result=det_text,
        rag_knowledge=rag_text,
        current_date=current_date,
    )

    # 调用 LLM 生成报告（根据 LLM_PROVIDER 自动选择）
    if settings.active_api_key:
        try:
            report = _call_llm(prompt)
            logger.info("LLM 报告生成成功 (provider=%s)", settings.llm_provider)
            return report
        except Exception as exc:
            logger.error("LLM API 调用失败: %s，使用模板生成报告", exc)

    # 降级方案：使用模板直接生成报告（无需API）
    disease_type = classification_result.get("disease_type", "未知")
    confidence = classification_result.get("confidence", 0)
    total_lesions = detection_result.get("total_count", 0)
    lesion_advice = (
        f"发现 {total_lesions} 处病灶，建议结合临床进一步评估。"
        if total_lesions > 0
        else "未发现明显病灶，建议定期复查。"
    )

    report = f"""# CT影像诊断报告

## 基本信息
- **检查日期**：{current_date}
- **检查类型**：胸部CT平扫
- **报告系统**：MedAgent AI辅助诊断系统 v1.0

## 用户请求
{user_request}

## CT影像描述
根据CT图像分析，影像表现如下：
- 肺窗：病变区域密度改变，符合{disease_type}的影像学特征
- 纵隔窗：纵隔结构可见

## 诊断结果

### 疾病分类
{cls_text}

### 病灶检测
{det_text}

## 医学知识参考
{rag_text}

## 诊断意见
基于AI分析，影像学表现符合**{disease_type}**（置信度: {confidence:.1%}），
{lesion_advice}

## 建议
1. 请结合患者临床症状、实验室检查及其他影像资料综合判断
2. 如有疑问，建议请专科医师会诊
3. 定期随访复查，观察病变动态变化
4. 本报告仅供参考，不作为最终诊断依据

---
⚠️ **免责声明**：本报告由 AI 辅助生成，仅供参考，不能替代专业医师的诊断意见。请务必就医咨询专业医师。
"""
    return report
