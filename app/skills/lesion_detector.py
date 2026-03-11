"""
病灶分割/检测 Skill

支持两种模式：
  1. 真实模式（nnU-Net v2）：有模型权重时自动启用
  2. Mock 模式：模型不存在时自动降级，用于开发/演示

如何启用真实 nnU-Net 推理：
  1. pip install nnunetv2
  2. 将训练好的 nnU-Net 模型放至 models/nnunet/ 目录
  3. 设置环境变量 NNUNET_RESULTS_PATH 指向模型目录

nnU-Net v2 模型目录结构示例：
  models/nnunet/
    Dataset001_Lung/
      nnUNetTrainer__nnUNetPlans__3d_fullres/
        fold_0/checkpoint_final.pth
        fold_1/checkpoint_final.pth
        ...（5-fold ensemble）
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── nnU-Net 配置 ──
_NNUNET_MODEL_DIRS = [
    "models/nnunet",
    os.getenv("NNUNET_RESULTS_PATH", ""),
]

# 每张图最多提取的病灶数
_MAX_LESIONS = 5

# 预定义的病灶类型
LESION_TYPES = [
    "实性结节",
    "磨玻璃影",
    "混合密度结节",
    "肿块",
    "斑片影",
    "条索影",
]

# 模拟图像尺寸（像素）
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512


def _generate_bounding_box() -> dict[str, int]:
    """生成随机的病灶边界框坐标"""
    x1 = random.randint(50, 350)
    y1 = random.randint(50, 350)
    width = random.randint(20, 100)
    height = random.randint(20, 100)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x1 + width,
        "y2": y1 + height,
        "width": width,
        "height": height,
    }


def _mock_detect(image_path: str) -> dict[str, Any]:
    """
    Mock检测实现：返回随机模拟的病灶检测结果

    ⚠️ 替换此函数以接入真实CT病灶检测模型
    """
    # 随机决定病灶数量（0-3个）
    num_lesions = random.randint(0, 3)

    lesions = []
    for i in range(num_lesions):
        bbox = _generate_bounding_box()
        lesion_type = random.choice(LESION_TYPES)
        # 从边界框估算等效直径（mm），假设像素间距 0.7mm
        pixel_spacing = 0.7
        diameter_mm = round(
            ((bbox["width"] * bbox["height"]) ** 0.5) * pixel_spacing, 1
        )

        lesion = {
            "lesion_id": i + 1,
            "type": lesion_type,
            "bounding_box": bbox,
            "size_mm": diameter_mm,
            "confidence": round(random.uniform(0.65, 0.95), 4),
            "location": _get_location_description(bbox),
            "mask_path": None,  # Mock时不生成实际mask文件
        }
        lesions.append(lesion)

    return {
        "lesions": lesions,
        "total_count": num_lesions,
        "image_size": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT},
        "model_version": "mock-v1.0",
        "image_path": image_path,
        "is_mock": True,
    }


def _get_location_description(bbox: dict[str, int]) -> str:
    """根据边界框坐标生成病灶位置描述"""
    cx = bbox["x1"] + bbox["width"] / 2
    cy = bbox["y1"] + bbox["height"] / 2

    # 水平位置
    if cx < IMAGE_WIDTH / 3:
        h_pos = "左侧"
    elif cx < IMAGE_WIDTH * 2 / 3:
        h_pos = "中央"
    else:
        h_pos = "右侧"

    # 垂直位置
    if cy < IMAGE_HEIGHT / 3:
        v_pos = "上叶"
    elif cy < IMAGE_HEIGHT * 2 / 3:
        v_pos = "中叶"
    else:
        v_pos = "下叶"

    return f"{h_pos}{v_pos}"


def detect_lesions(image_path: str) -> dict[str, Any]:
    """
    病灶检测/分割主函数

    优先尝试 nnU-Net v2 真实推理，失败时自动降级为 Mock。

    参数:
        image_path: CT图片文件路径（支持 .jpg/.png/.dcm 等格式）

    返回:
        包含以下字段的字典：
        - lesions: 病灶列表
        - total_count: 总病灶数量
        - image_size: 图像尺寸
        - model_version: 模型版本
        - image_path: 输入图片路径
        - is_mock: 是否为mock结果
    """
    logger.info("开始病灶检测，图片路径: %s", image_path)

    path = Path(image_path)
    if not path.exists():
        logger.warning("图片文件不存在: %s，使用mock数据", image_path)
        result = _mock_detect(image_path)
        logger.info("病灶检测完成（Mock），发现 %d 处病灶", result["total_count"])
        return result

    # 尝试 nnU-Net 真实推理
    nnunet_result = _run_nnunet_inference(image_path)
    if nnunet_result is not None:
        logger.info(
            "病灶检测完成（nnU-Net），发现 %d 处病灶", nnunet_result["total_count"]
        )
        return nnunet_result

    # 降级为 Mock
    logger.info("nnU-Net 不可用，降级为 Mock 检测")
    result = _mock_detect(image_path)
    logger.info("病灶检测完成（Mock），发现 %d 处病灶", result["total_count"])
    return result


def _run_nnunet_inference(image_path: str) -> Optional[dict[str, Any]]:
    """
    使用 nnU-Net v2 进行真实推理

    支持 5-fold ensemble + TTA（Test Time Augmentation）。
    模型不存在或依赖未安装时返回 None。
    """
    # 检查 nnU-Net 是否可用
    nnunet_model_path = _find_nnunet_model()
    if nnunet_model_path is None:
        return None

    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import numpy as np
    except ImportError:
        logger.debug("nnunetv2 未安装，跳过真实推理")
        return None

    try:
        device_str = _detect_device()
        import torch

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device(device_str),
            verbose=False,
            allow_tqdm=False,
        )

        predictor.initialize_from_trained_model_folder(
            str(nnunet_model_path),
            use_folds="all",
            checkpoint_name="checkpoint_final.pth",
        )

        # 读取并预处理图像
        img_array, spacing = _load_image_for_nnunet(image_path)
        if img_array is None:
            return None

        # 执行推理
        seg_mask, prob_map = predictor.predict_single_npy_array(
            img_array,
            {"spacing": spacing},
            None,
            None,
            True,
        )

        # 从掩码提取病灶
        lesions = _extract_lesions_from_mask(seg_mask, prob_map, spacing)
        h, w = img_array.shape[-2], img_array.shape[-1]

        return {
            "lesions": lesions,
            "total_count": len(lesions),
            "image_size": {"width": int(w), "height": int(h)},
            "model_version": f"nnunet-v2-{nnunet_model_path.name}",
            "image_path": image_path,
            "is_mock": False,
        }

    except Exception as exc:
        logger.warning("nnU-Net 推理失败: %s", exc)
        return None


def _find_nnunet_model() -> Optional[Path]:
    """搜索可用的 nnU-Net 模型目录"""
    project_root = Path(__file__).resolve().parent.parent.parent
    for model_dir in _NNUNET_MODEL_DIRS:
        if not model_dir:
            continue
        candidate = Path(model_dir)
        if not candidate.is_absolute():
            candidate = project_root / model_dir
        if candidate.exists() and candidate.is_dir():
            # 检查是否有实际的模型文件
            checkpoints = list(candidate.rglob("checkpoint_final.pth"))
            if checkpoints:
                # 返回包含 fold_all 或第一个 fold 的目录
                trainer_dir = checkpoints[0].parent.parent
                logger.info("[nnU-Net] 找到模型: %s", trainer_dir)
                return trainer_dir
    logger.debug("[nnU-Net] 未找到模型权重")
    return None


def _load_image_for_nnunet(image_path: str) -> tuple[Any, list[float]]:
    """将图像加载并格式化为 nnU-Net 输入格式"""
    try:
        import numpy as np
        from PIL import Image as PILImage

        img = PILImage.open(image_path).convert("L")  # 转为灰度
        arr = np.array(img, dtype=np.float32)
        # nnU-Net 期望 (C, D, H, W) 或 (C, H, W) 格式
        arr = arr[np.newaxis, np.newaxis, ...]  # (1, 1, H, W)
        spacing = [1.0, 1.0, 1.0]
        return arr, spacing
    except Exception as exc:
        logger.warning("图像加载失败: %s", exc)
        return None, []


def _extract_lesions_from_mask(
    seg_mask: Any,
    prob_map: Any,
    spacing: list[float],
) -> list[dict[str, Any]]:
    """从分割掩码中提取病灶信息（连通域分析）"""
    try:
        import numpy as np
        from scipy import ndimage

        # 确保为 2D/3D numpy 数组
        mask = np.squeeze(np.array(seg_mask) > 0)
        if mask.ndim == 3:
            mask = mask[mask.shape[0] // 2]  # 取中间切片

        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            return []

        # 提取每个连通域的信息
        lesions = []
        pixel_spacing = spacing[-1] if spacing else 0.7
        h, w = mask.shape[:2]

        for i in range(1, min(num_features + 1, _MAX_LESIONS + 1)):
            component = labeled == i
            rows = np.any(component, axis=1)
            cols = np.any(component, axis=0)
            if not rows.any() or not cols.any():
                continue

            y1 = int(np.argmax(rows))
            y2 = int(len(rows) - 1 - np.argmax(rows[::-1]))
            x1 = int(np.argmax(cols))
            x2 = int(len(cols) - 1 - np.argmax(cols[::-1]))
            bbox_w = max(x2 - x1, 1)
            bbox_h = max(y2 - y1, 1)
            size_mm = round(((bbox_w * bbox_h) ** 0.5) * pixel_spacing, 1)

            # 置信度：从概率图取均值（若有）
            if prob_map is not None:
                prob = np.squeeze(np.array(prob_map))
                if prob.ndim == 3:
                    prob = prob[prob.shape[0] // 2]
                if prob.shape == component.shape:
                    confidence = float(np.mean(prob[component]))
                else:
                    confidence = 0.80
            else:
                confidence = 0.80

            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            h_pos = "左侧" if cx < 0.33 else ("中央" if cx < 0.67 else "右侧")
            v_pos = "上叶" if cy < 0.33 else ("中叶" if cy < 0.67 else "下叶")

            lesions.append({
                "lesion_id": i,
                "type": LESION_TYPES[0],  # nnU-Net 输出类型标签，此处简化
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": bbox_w, "height": bbox_h},
                "size_mm": size_mm,
                "confidence": round(confidence, 4),
                "location": f"{h_pos}{v_pos}",
                "mask_path": None,
            })

        return lesions

    except ImportError:
        logger.debug("scipy 未安装，跳过连通域分析")
        return []
    except Exception as exc:
        logger.warning("掩码解析失败: %s", exc)
        return []


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


# ─────────────────────────────────────────
# LangChain Tool 注册（标准化封装）
# ─────────────────────────────────────────

@tool
def detect_lesion_tool(image_path: str) -> str:
    """CT病灶检测工具。

    当你需要在CT图片上找出病灶的具体位置、大小和类型时，必须调用这个工具。
    比如要定位肺结节在哪里、有多大、是实性结节还是磨玻璃影，都靠它。
    它返回一个JSON，包含每个病灶的坐标框(bounding_box)、类型、大小(mm)和置信度。

    参数:
        image_path: CT图片在本地磁盘上的文件路径，比如 /tmp/ct_scan.jpg
    """
    result = detect_lesions(image_path)
    lines = [
        f"病灶检测结果：共发现 {result['total_count']} 处病灶",
    ]
    for lesion in result.get("lesions", []):
        lines.append(
            f"  病灶{lesion['lesion_id']}：{lesion['type']}，"
            f"位置={lesion['location']}，大小={lesion['size_mm']}mm，"
            f"置信度={lesion['confidence']:.2%}"
        )
    lines.append(f"图像尺寸：{result['image_size']['width']}×{result['image_size']['height']}")
    lines.append(f"模型版本：{result.get('model_version', 'unknown')}")
    lines.append(f"是否Mock：{result.get('is_mock', True)}")
    return "\n".join(lines)


# 保留旧名称兼容（别名）
lesion_detection_tool = detect_lesion_tool
