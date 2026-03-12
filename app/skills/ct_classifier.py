"""
CT疾病分类 Skill

支持两种模式：
  1. 真实模式（MONAI EfficientNet-B4）：有模型权重时自动启用
  2. Mock 模式：模型不存在时自动降级，用于开发/演示

如何启用真实分类推理：
  1. pip install monai
  2. 将训练好的 EfficientNet-B4 权重放至 models/ct_classifier.pth
  3. 模型将自动加载并用于推理

肺窗预处理参数（DICOM）：
  - 窗位 (Window Level, WL) = -600 HU
  - 窗宽 (Window Width, WW) = 1500 HU
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── 模型配置 ──
_DEFAULT_MODEL_PATHS = [
    "models/ct_classifier.pth",
    "models/ct_classifier/model.pth",
]
_CT_WINDOW_LEVEL = -600   # 肺窗窗位（HU）
_CT_WINDOW_WIDTH = 1500   # 肺窗窗宽（HU）
_IMAGE_SIZE = (224, 224)  # EfficientNet-B4 输入尺寸

# 预定义的疾病分类列表（与真实临床分类对应）
DISEASE_CLASSES = [
    "正常",
    "肺炎",
    "肺结节",
    "肺癌",
    "肺气肿",
    "胸腔积液",
]

# 各疾病的模拟置信度范围（用于生成更真实的mock数据）
DISEASE_CONFIDENCE_RANGES = {
    "正常": (0.75, 0.98),
    "肺炎": (0.65, 0.95),
    "肺结节": (0.70, 0.92),
    "肺癌": (0.60, 0.88),
    "肺气肿": (0.68, 0.90),
    "胸腔积液": (0.72, 0.93),
}


def _mock_classify(image_path: str) -> dict:
    """
    Mock分类实现：返回随机模拟的分类结果

    ⚠️ 替换此函数以接入真实CT分类模型
    """
    # 随机选择主要疾病
    disease = random.choice(DISEASE_CLASSES)
    conf_range = DISEASE_CONFIDENCE_RANGES[disease]
    confidence = round(random.uniform(*conf_range), 4)

    # 生成所有类别的概率分布（模拟softmax输出）
    remaining = 1.0 - confidence
    other_diseases = [d for d in DISEASE_CLASSES if d != disease]
    other_probs = sorted([random.uniform(0, remaining) for _ in other_diseases])
    # 归一化其他概率
    other_sum = sum(other_probs)
    if other_sum > 0:
        other_probs = [round(p / other_sum * remaining, 4) for p in other_probs]

    all_probs = {disease: confidence}
    for d, p in zip(other_diseases, other_probs):
        all_probs[d] = p

    return {
        "disease_type": disease,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "model_version": "mock-v1.0",
        "image_path": image_path,
        "is_mock": True,
    }


def classify_ct(image_path: str) -> dict:
    """
    CT疾病分类主函数

    优先尝试 MONAI EfficientNet-B4 真实推理，失败时自动降级为 Mock。

    参数:
        image_path: CT图片文件路径（支持 .jpg/.png/.dcm 等格式）

    返回:
        包含以下字段的字典：
        - disease_type: 预测的疾病类型（str）
        - confidence: 置信度（float, 0-1）
        - all_probabilities: 所有类别的概率（dict）
        - model_version: 模型版本（str）
        - image_path: 输入图片路径（str）
        - is_mock: 是否为mock结果（bool）
    """
    logger.info("开始CT分类，图片路径: %s", image_path)

    path = Path(image_path)
    if path.exists():
        # 尝试真实推理
        real_result = _run_monai_inference(image_path)
        if real_result is not None:
            logger.info(
                "CT分类完成（MONAI）: %s (%.2f%%)",
                real_result["disease_type"],
                real_result["confidence"] * 100,
            )
            return real_result
    else:
        logger.warning("图片文件不存在: %s，使用mock数据", image_path)

    # 降级为 Mock
    result = _mock_classify(image_path)
    logger.info(
        "CT分类完成（Mock）: %s (置信度: %.2f%%)",
        result["disease_type"],
        result["confidence"] * 100,
    )
    return result


def _find_model_path() -> Optional[Path]:
    """搜索可用的分类模型权重文件"""
    project_root = Path(__file__).resolve().parent.parent.parent
    for rel_path in _DEFAULT_MODEL_PATHS:
        candidate = Path(rel_path)
        if not candidate.is_absolute():
            candidate = project_root / rel_path
        if candidate.exists():
            logger.info("[CT分类] 找到模型: %s", candidate)
            return candidate
    logger.debug("[CT分类] 未找到模型权重，将使用 Mock")
    return None


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


def _run_monai_inference(image_path: str) -> Optional[dict]:
    """使用 MONAI EfficientNet-B4 进行真实推理，失败时返回 None"""
    model_path = _find_model_path()
    if model_path is None:
        return None

    try:
        import numpy as np
        import torch
        from monai.networks.nets import EfficientNetBN
        from PIL import Image as PILImage
    except ImportError:
        logger.debug("monai/PIL 未安装，跳过真实推理")
        return None

    try:
        device = torch.device(_detect_device())
        model = EfficientNetBN(
            "efficientnet-b4",
            pretrained=False,
            num_classes=len(DISEASE_CLASSES),
        )
        state = torch.load(str(model_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        # 预处理图像（应用肺窗）
        img = PILImage.open(image_path).convert("L")
        img = img.resize(_IMAGE_SIZE)
        arr = np.array(img, dtype=np.float32)

        # 肺窗归一化 (HU → 0-1)
        wl, ww = _CT_WINDOW_LEVEL, _CT_WINDOW_WIDTH
        lower = wl - ww / 2
        upper = wl + ww / 2
        arr = np.clip(arr, lower, upper)
        arr = (arr - lower) / (upper - lower)

        # 构造 3 通道输入
        arr = np.stack([arr, arr, arr], axis=0)[np.newaxis]  # (1, 3, H, W)
        tensor = torch.from_numpy(arr).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        best_idx = int(np.argmax(probs))
        all_probs = {cls: float(probs[i]) for i, cls in enumerate(DISEASE_CLASSES)}

        return {
            "disease_type": DISEASE_CLASSES[best_idx],
            "confidence": float(probs[best_idx]),
            "all_probabilities": all_probs,
            "model_version": f"efficientnet-b4-{model_path.stem}",
            "image_path": image_path,
            "is_mock": False,
        }

    except Exception as exc:
        logger.warning("[CT分类] MONAI 推理失败: %s", exc)
        return None


# ─────────────────────────────────────────
# LangChain Tool 注册（标准化封装）
# ─────────────────────────────────────────

@tool
def analyze_ct_tool(image_path: str) -> str:
    """CT疾病分类诊断工具。

    当你拿到了病人的CT图片路径，需要判断这张CT片子里有没有肺炎、肺结节、
    肺癌、肺气肿、胸腔积液等病变时，就必须调用这个工具。
    它会返回一个JSON，里面包含疾病分类类型、置信度和各类别概率分布。

    参数:
        image_path: CT图片在本地磁盘上的文件路径，比如 /tmp/ct_scan.jpg
    """
    result = classify_ct(image_path)
    data = json.loads(json.dumps(result, ensure_ascii=False))
    lines = [
        "CT分类结果：",
        f"疾病类型：{data['disease_type']}",
        f"置信度：{data['confidence']:.2%}",
        "各类别概率：",
    ]
    for disease, prob in data.get("all_probabilities", {}).items():
        lines.append(f"  {disease}: {prob:.2%}")
    lines.append(f"模型版本：{data.get('model_version', 'unknown')}")
    lines.append(f"是否Mock：{data.get('is_mock', True)}")
    return "\n".join(lines)


# 保留旧名称兼容（别名）
ct_classification_tool = analyze_ct_tool
