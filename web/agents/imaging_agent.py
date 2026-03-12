"""
影像分析 Agent

接收上传的 2D 图像 / DICOM 文件，调用分割调度器进行真实分析。
当没有真实分割引擎可用时降级为纯 LLM 文本分析。

分割引擎优先级：
  1. nnU-Net v2（有模型时）
  2. MedSAM（有模型时）
  3. 纯 LLM 文本分析（降级）
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from web.agents.base import AbnormalMetric, AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)


class ImagingAgent:
    """
    影像分析 Agent

    职责：
        - 接收上传的 CT/MRI 图像
        - 调用 segmentation_dispatcher 智能选择最优分割引擎
        - 从分割结果中提取 findings 和 abnormal_metrics
        - 无图片时返回提示信息
    """

    AGENT_NAME = "ImagingAgent"
    DISPLAY_NAME = "🔬 影像分析"

    def __init__(self) -> None:
        pass

    def run(
        self,
        image_path: Optional[str] = None,
        text: str = "",
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行影像分析

        参数:
            image_path: 上传的 CT 图片路径（.jpg/.png/.dcm）
            text: 附加文本描述
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的影像分析结果
        """
        logger.info("[%s] 开始影像分析...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()

        # ── 无图片时返回提示 ──
        if not image_path or not Path(image_path).exists():
            elapsed = time.time() - start
            if on_status:
                on_status(self.AGENT_NAME, AgentStatus.FAILED)
            return BaseAgentOutput(
                agent_name=self.AGENT_NAME,
                agent_display_name=self.DISPLAY_NAME,
                status=AgentStatus.FAILED,
                findings=["未上传影像文件，无法进行影像分析。"],
                abnormal_metrics=[],
                confidence=0.0,
                processing_time=round(elapsed, 2),
                error_message="未上传影像",
            )

        # ── 调用分割调度器 ──
        try:
            from app.skills.segmentation_dispatcher import smart_segment

            seg_result = smart_segment(image_path)
            is_not_mock = not seg_result.get("is_mock", True)
            has_valid_engine = seg_result.get("engine", "none") != "none"
            is_real = is_not_mock and has_valid_engine
        except Exception as exc:
            logger.warning("[%s] 分割调度器调用失败: %s，降级为 Mock", self.AGENT_NAME, exc)
            # 降级：使用 ct_classifier + lesion_detector mock
            seg_result = self._run_mock_pipeline(image_path)
            is_real = False

        # ── 提取 findings 和 abnormal_metrics ──
        if seg_result.get("total_count", 0) > 0 or is_real:
            findings, abnormal_metrics, confidence, raw_data = self._parse_seg_result(
                seg_result, image_path, is_real
            )
        else:
            # 分割结果为空 or 仍为 Mock → 降级为 LLM 文本分析
            findings, abnormal_metrics, confidence, raw_data = self._llm_fallback(
                text, image_path, seg_result
            )

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

    # ──────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────

    def _run_mock_pipeline(self, image_path: str) -> dict[str, Any]:
        """降级：调用 Mock 分类 + 检测管线"""
        try:
            from app.skills.lesion_detector import detect_lesions
            return detect_lesions(image_path)
        except Exception:
            return {"lesions": [], "total_count": 0, "is_mock": True, "image_path": image_path}

    def _parse_seg_result(
        self,
        seg_result: dict[str, Any],
        image_path: str,
        is_real: bool,
    ) -> tuple[list[str], list[AbnormalMetric], float, dict[str, Any]]:
        """从分割结果中提取 findings 和 abnormal_metrics"""
        lesions = seg_result.get("lesions", [])
        engine = seg_result.get("engine", "mock")
        cross_validated = seg_result.get("cross_validated", False)
        consistency = seg_result.get("consistency_score")

        findings = []
        abnormal_metrics = []

        if not lesions:
            findings.append("影像分析未发现明显异常病灶")
            return findings, abnormal_metrics, 0.9, seg_result

        findings.append(f"影像分析共发现 {len(lesions)} 处疑似病灶")
        if is_real:
            engine_label = {"nnunet": "nnU-Net v2", "medsam": "MedSAM", "nnunet+medsam": "nnU-Net + MedSAM"}.get(engine, engine)
            findings.append(f"分割引擎：{engine_label}")
            if cross_validated and consistency is not None:
                level = "高" if consistency >= 0.8 else ("中" if consistency >= 0.5 else "低")
                findings.append(f"双引擎交叉验证一致性：{level}（{consistency:.0%}）")

        total_confidence = 0.0
        for lesion in lesions:
            size = lesion.get("size_mm", 0)
            loc = lesion.get("location", "未知位置")
            ltype = lesion.get("type", "病灶")
            conf = lesion.get("confidence", 0.8)
            total_confidence += conf

            finding = f"病灶{lesion.get('lesion_id', '?')}：{loc}发现{ltype}，大小约 {size:.1f}mm，置信度 {conf:.0%}"
            findings.append(finding)

            # 中等及以上风险的病灶标记为 abnormal
            severity = "severe" if size > 20 else ("moderate" if size > 8 else "mild")
            abnormal_metrics.append(
                AbnormalMetric(
                    name=f"病灶{lesion.get('lesion_id', '?')} 大小",
                    value=f"{size:.1f}mm",
                    reference_range="<6mm 低风险",
                    severity=severity,
                    description=f"{loc}{ltype}，{_risk_text(size)}",
                )
            )

        avg_confidence = total_confidence / len(lesions) if lesions else 0.8
        return findings, abnormal_metrics, round(avg_confidence, 3), seg_result

    def _llm_fallback(
        self,
        text: str,
        image_path: str,
        seg_result: dict[str, Any],
    ) -> tuple[list[str], list[AbnormalMetric], float, dict[str, Any]]:
        """无真实分割结果时，用 LLM 基于文本描述进行影像分析"""
        qwen_key = os.getenv("QWEN_API_KEY", "")
        if not qwen_key:
            findings = [
                "已接收影像文件，但当前无可用的自动分割引擎。",
                "建议配置 nnU-Net / MedSAM 模型权重以启用真实影像分析。",
                "如需文字报告，请在聊天框中描述影像所见。",
            ]
            return findings, [], 0.5, seg_result

        try:
            from openai import OpenAI

            client = OpenAI(
                api_key=qwen_key,
                base_url=os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            )

            prompt = (
                "你是一位放射科医师。用户上传了医学影像，并提供了如下描述：\n"
                f"{text or '（无文字描述）'}\n\n"
                "请根据描述，给出影像学角度的初步分析，包括可能的影像特征和建议检查。"
                "如果描述不足，请说明需要哪些额外信息。"
                "回复要简洁，以中文回答，每条发现单独一行，最多5条。"
            )

            response = client.chat.completions.create(
                model=os.getenv("QWEN_MODEL_NAME", "qwen-plus"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            content = response.choices[0].message.content.strip()
            findings = [line.strip("- •·") for line in content.split("\n") if line.strip()][:5]
            return findings, [], 0.65, seg_result

        except Exception as exc:
            logger.warning("[%s] LLM 降级分析失败: %s", self.AGENT_NAME, exc)
            return ["影像文件已接收，但自动分析暂不可用，请稍后重试。"], [], 0.0, seg_result

    # ──────────────────────────────────────
    # 保留原有的 DICOM 接口（未实现，保持兼容）
    # ──────────────────────────────────────

    def process_dicom_series(self, dir_path: str, **kwargs: Any) -> dict[str, Any]:
        """处理 DICOM 序列文件夹（TODO）"""
        raise NotImplementedError("DICOM 序列处理尚未实现")

    def generate_3d_reconstruction(self, volume_data: Any = None, **kwargs: Any) -> str:
        """生成三维重建（TODO）"""
        raise NotImplementedError("三维重建尚未实现")


def _risk_text(size_mm: float) -> str:
    """根据大小返回风险描述"""
    if size_mm > 20:
        return "肿块级别，高风险，建议立即就医"
    if size_mm > 8:
        return "中等大小，中高风险，建议3个月内随访"
    if size_mm > 6:
        return "小结节，低-中风险，建议6个月随访"
    return "微小结节，低风险，建议年度随访"
