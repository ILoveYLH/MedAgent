"""
MedSAM 通用医学影像分割模块

基于 SAM（Segment Anything Model）在医学影像上微调的通用分割模型。
提供零样本分割能力，作为 nnU-Net 的互补方案。

特性：
- 支持自动分割和 bbox 提示分割
- 从掩码提取病灶信息（位置、大小、置信度）
- 模型不存在时优雅降级

使用方式：
    segmentor = MedSAMSegmentor()
    if segmentor.is_available():
        result = segmentor.segment(image_path)
    else:
        # 降级到其他方案
        pass
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MedSAMSegmentor:
    """
    MedSAM 通用医学影像分割器

    参数:
        model_path: MedSAM 模型权重文件路径（.pth），None 则自动搜索
        device: 推理设备 ('cuda', 'mps', 'cpu')，None 则自动检测
    """

    DEFAULT_MODEL_PATHS = [
        "models/medsam_vit_b.pth",
        "models/medsam/medsam_vit_b.pth",
        "medsam_vit_b.pth",
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        self._model = None
        self._device = device or self._detect_device()
        self._model_path = model_path or self._find_model()
        self._available = False

        if self._model_path:
            self._available = self._load_model(self._model_path)

    # ──────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return self._available

    def segment(
        self,
        image_path: str,
        bbox_prompt: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """
        对医学影像进行分割

        参数:
            image_path: 输入图像路径（.jpg/.png/.dcm）
            bbox_prompt: 可选的边界框提示 [x1, y1, x2, y2]

        返回:
            dict: {
                "lesions": 病灶列表,
                "total_count": 病灶数量,
                "engine": "medsam",
                "is_mock": False,
            }
        """
        if not self._available:
            logger.warning("[MedSAM] 模型不可用，无法执行分割")
            return self._empty_result(image_path)

        try:
            return self._run_inference(image_path, bbox_prompt)
        except Exception as exc:
            logger.error("[MedSAM] 推理失败: %s", exc)
            return self._empty_result(image_path, error=str(exc))

    # ──────────────────────────────────────
    # 私有方法
    # ──────────────────────────────────────

    @staticmethod
    def _detect_device() -> str:
        """自动检测可用的推理设备"""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _find_model(self) -> Optional[str]:
        """自动搜索模型文件"""
        project_root = Path(__file__).resolve().parent.parent.parent
        for rel_path in self.DEFAULT_MODEL_PATHS:
            candidate = project_root / rel_path
            if candidate.exists():
                logger.info("[MedSAM] 找到模型: %s", candidate)
                return str(candidate)
        logger.info("[MedSAM] 未找到模型权重，将以降级模式运行")
        return None

    def _load_model(self, model_path: str) -> bool:
        """加载 MedSAM 模型权重"""
        try:
            import torch
            from segment_anything import sam_model_registry

            checkpoint = Path(model_path)
            if not checkpoint.exists():
                logger.warning("[MedSAM] 模型文件不存在: %s", model_path)
                return False

            model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
            sam.to(device=self._device)
            sam.eval()
            self._model = sam
            logger.info("[MedSAM] 模型加载成功 (device=%s)", self._device)
            return True

        except ImportError:
            logger.info("[MedSAM] segment_anything 未安装，跳过模型加载")
            return False
        except Exception as exc:
            logger.warning("[MedSAM] 模型加载失败: %s", exc)
            return False

    def _run_inference(
        self,
        image_path: str,
        bbox_prompt: Optional[list[int]] = None,
    ) -> dict[str, Any]:
        """执行真实推理"""
        import numpy as np

        try:
            from PIL import Image as PILImage
        except ImportError:
            raise RuntimeError("Pillow 未安装，无法读取图像")

        try:
            from segment_anything import SamPredictor, SamAutomaticMaskGenerator
        except ImportError:
            raise RuntimeError("segment_anything 未安装")

        img = PILImage.open(image_path).convert("RGB")
        img_array = np.array(img)

        predictor = SamPredictor(self._model)
        predictor.set_image(img_array)

        if bbox_prompt:
            import torch
            input_box = np.array(bbox_prompt)
            masks, scores, _ = predictor.predict(
                box=input_box[None, :],
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores))
            masks = masks[best_idx:best_idx + 1]
            scores = scores[best_idx:best_idx + 1]
        else:
            # 自动分割
            mask_generator = SamAutomaticMaskGenerator(
                self._model,
                points_per_side=16,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.90,
                min_mask_region_area=100,
            )
            auto_masks = mask_generator.generate(img_array)
            # 过滤过大和过小的 mask（保留中等大小，更可能是病灶）
            h, w = img_array.shape[:2]
            total_pixels = h * w
            lesion_masks = [
                m for m in auto_masks
                if 0.001 * total_pixels < m["area"] < 0.3 * total_pixels
            ]
            if not lesion_masks:
                return self._empty_result(image_path)

            masks = np.stack([m["segmentation"] for m in lesion_masks])
            scores = np.array([m["predicted_iou"] for m in lesion_masks])

        lesions = self._extract_lesion_info(masks, scores, img_array.shape)
        return {
            "lesions": lesions,
            "total_count": len(lesions),
            "image_size": {"width": img_array.shape[1], "height": img_array.shape[0]},
            "model_version": "medsam-vit-b",
            "engine": "medsam",
            "image_path": image_path,
            "is_mock": False,
        }

    def _extract_lesion_info(
        self,
        masks: Any,
        scores: Any,
        image_shape: tuple,
    ) -> list[dict[str, Any]]:
        """从分割掩码中提取病灶信息"""
        import numpy as np

        h, w = image_shape[:2]
        lesions = []

        for idx, (mask, score) in enumerate(zip(masks, scores)):
            if not mask.any():
                continue

            # 计算边界框
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
            x1, x2 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
            bbox_w = max(x2 - x1, 1)
            bbox_h = max(y2 - y1, 1)

            # 估算大小（假设像素间距 ~0.7mm）
            pixel_spacing = 0.7
            size_mm = round(((bbox_w * bbox_h) ** 0.5) * pixel_spacing, 1)

            # 位置描述
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            h_pos = "左侧" if cx < 0.33 else ("中央" if cx < 0.67 else "右侧")
            v_pos = "上叶" if cy < 0.33 else ("中叶" if cy < 0.67 else "下叶")
            location = f"{h_pos}{v_pos}"

            lesions.append({
                "lesion_id": idx + 1,
                "type": "未分类病灶",
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": bbox_w, "height": bbox_h},
                "size_mm": size_mm,
                "confidence": round(float(score), 4),
                "location": location,
                "mask_path": None,
            })

        return lesions

    @staticmethod
    def _empty_result(image_path: str, error: Optional[str] = None) -> dict[str, Any]:
        """返回空结果"""
        result: dict[str, Any] = {
            "lesions": [],
            "total_count": 0,
            "image_size": {"width": 0, "height": 0},
            "model_version": "medsam-vit-b",
            "engine": "medsam",
            "image_path": image_path,
            "is_mock": False,
        }
        if error:
            result["error"] = error
        return result
