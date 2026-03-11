"""
智能分割调度器 (Segmentation Dispatcher)

自动选择最优分割引擎：
  - 有 nnU-Net 模型时 → nnU-Net 为主力
  - 无 nnU-Net 但有 MedSAM → MedSAM 兜底
  - 两者都有 → 交叉验证，标记一致性
  - 两者都不可用 → 返回空结果，交给 LLM 文本分析

使用方式：
    result = smart_segment(image_path)
    if result["total_count"] > 0:
        # 使用真实分割结果
    else:
        # 降级到纯文本分析
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def smart_segment(
    image_path: str,
    bbox_prompt: Optional[list[int]] = None,
) -> dict[str, Any]:
    """
    智能分割入口：自动选择最优引擎

    参数:
        image_path: 输入图像路径
        bbox_prompt: 可选的边界框提示 [x1, y1, x2, y2]

    返回:
        dict: 标准分割结果，包含 engine, cross_validated, lesions 等字段
    """
    nnunet_result = _try_nnunet(image_path)
    medsam_result = _try_medsam(image_path, bbox_prompt)

    has_nnunet = nnunet_result is not None
    has_medsam = medsam_result is not None

    if has_nnunet and has_medsam:
        # 两者都可用 → 交叉验证
        result = _cross_validate(nnunet_result, medsam_result)
        logger.info(
            "[SegDispatcher] 双引擎交叉验证完成，一致性=%.2f",
            result.get("consistency_score", 0.0),
        )
    elif has_nnunet:
        result = nnunet_result
        result["engine"] = "nnunet"
        result["cross_validated"] = False
        logger.info("[SegDispatcher] 使用 nnU-Net 单引擎")
    elif has_medsam:
        result = medsam_result
        result["engine"] = "medsam"
        result["cross_validated"] = False
        logger.info("[SegDispatcher] 使用 MedSAM 单引擎")
    else:
        # 两者都不可用
        result = _empty_result(image_path)
        logger.info("[SegDispatcher] 无可用分割引擎，返回空结果")

    return result


# ──────────────────────────────────────────────
# 私有实现
# ──────────────────────────────────────────────

def _try_nnunet(image_path: str) -> Optional[dict[str, Any]]:
    """尝试使用 nnU-Net 进行分割，失败或不可用时返回 None"""
    try:
        from app.skills.lesion_detector import detect_lesions

        result = detect_lesions(image_path)
        if result.get("is_mock", True):
            return None  # Mock 结果不算真实 nnU-Net 可用
        return result
    except Exception as exc:
        logger.debug("[SegDispatcher] nnU-Net 不可用: %s", exc)
        return None


def _try_medsam(
    image_path: str,
    bbox_prompt: Optional[list[int]] = None,
) -> Optional[dict[str, Any]]:
    """尝试使用 MedSAM 进行分割，失败或不可用时返回 None"""
    try:
        from app.skills.medsam_segmentor import MedSAMSegmentor

        segmentor = MedSAMSegmentor()
        if not segmentor.is_available():
            return None

        result = segmentor.segment(image_path, bbox_prompt)
        return result
    except Exception as exc:
        logger.debug("[SegDispatcher] MedSAM 不可用: %s", exc)
        return None


def _cross_validate(
    nnunet_result: dict[str, Any],
    medsam_result: dict[str, Any],
) -> dict[str, Any]:
    """
    交叉验证两个引擎的结果

    简单实现：以 nnU-Net 结果为主，用 MedSAM 计算一致性分数
    """
    nnunet_count = nnunet_result.get("total_count", 0)
    medsam_count = medsam_result.get("total_count", 0)

    # 计算数量一致性（简化版）
    if nnunet_count == 0 and medsam_count == 0:
        consistency = 1.0
    elif nnunet_count == 0 or medsam_count == 0:
        consistency = 0.2
    else:
        consistency = min(nnunet_count, medsam_count) / max(nnunet_count, medsam_count)

    result = dict(nnunet_result)
    result["engine"] = "nnunet+medsam"
    result["cross_validated"] = True
    result["consistency_score"] = round(consistency, 3)
    result["medsam_count"] = medsam_count

    return result


def _empty_result(image_path: str) -> dict[str, Any]:
    """返回空分割结果"""
    return {
        "lesions": [],
        "total_count": 0,
        "image_size": {"width": 0, "height": 0},
        "model_version": "none",
        "engine": "none",
        "image_path": image_path,
        "is_mock": False,
        "cross_validated": False,
    }
