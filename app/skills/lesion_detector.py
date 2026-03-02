"""
病灶分割/检测 Skill（Mock实现）

当前使用随机模拟结果，预留真实模型接口。

如何替换为真实模型：
  1. 安装检测模型依赖（如 ultralytics, monai, nnunet）
  2. 在 _load_real_detector() 中加载预训练权重
  3. 在 detect_lesions() 中替换 _mock_detect() 调用为真实推理代码
  4. 真实模型示例：
     - YOLOv8：ultralytics.YOLO('yolov8n-seg.pt')
     - U-Net：monai.networks.nets.UNet(...)
     - nnU-Net：nnunet.inference.predictor.nnUNetPredictor(...)
"""

import logging
import random
from pathlib import Path
from typing import Any

from langchain.tools import tool

logger = logging.getLogger(__name__)

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

    参数:
        image_path: CT图片文件路径（支持 .jpg/.png/.dcm 等格式）

    返回:
        包含以下字段的字典：
        - lesions: 病灶列表，每个病灶包含：
            - lesion_id: 病灶编号
            - type: 病灶类型
            - bounding_box: 边界框坐标（x1, y1, x2, y2, width, height）
            - size_mm: 等效直径（毫米）
            - confidence: 检测置信度
            - location: 病灶位置描述
            - mask_path: 分割mask文件路径（None表示无mask）
        - total_count: 总病灶数量
        - image_size: 图像尺寸
        - model_version: 模型版本
        - image_path: 输入图片路径
        - is_mock: 是否为mock结果

    替换为真实模型的步骤：
        1. 加载预训练检测模型：model = YOLO('ct_lesion_detector.pt')
        2. 推理：results = model.predict(image_path, conf=0.5)
        3. 解析结果：boxes = results[0].boxes
        4. 提取病灶信息：
           for box in boxes:
               lesion = {
                   'bounding_box': box.xyxy[0].tolist(),
                   'confidence': float(box.conf[0]),
                   ...
               }
        5. 如有分割模型，生成mask文件并填写 mask_path
    """
    logger.info("开始病灶检测，图片路径: %s", image_path)

    # 验证输入路径（允许mock模式下路径不存在）
    path = Path(image_path)
    if not path.exists():
        logger.warning("图片文件不存在: %s，使用mock数据", image_path)

    # ── 真实模型接口预留 ──────────────────────────
    # TODO: 取消下面注释以启用真实模型
    # from app.skills._real_models import load_lesion_detector
    # model = load_lesion_detector()
    # result = model.predict(image_path)
    # return result
    # ─────────────────────────────────────────────

    result = _mock_detect(image_path)
    logger.info("病灶检测完成，发现 %d 处病灶", result["total_count"])
    return result


# ─────────────────────────────────────────
# LangChain Tool 注册
# ─────────────────────────────────────────

@tool
def lesion_detection_tool(image_path: str) -> str:
    """
    CT病灶检测工具（LangChain Tool）

    对给定的CT图片进行病灶检测和分割分析，返回病灶位置、大小和类型信息。

    参数:
        image_path: CT图片文件路径

    返回:
        包含检测结果的格式化字符串
    """
    result = detect_lesions(image_path)
    if result["total_count"] == 0:
        return "病灶检测结果：未发现明显病灶"

    lines = [f"病灶检测结果：共发现 {result['total_count']} 处病灶\n"]
    for lesion in result["lesions"]:
        lines.append(
            f"  病灶 {lesion['lesion_id']}:\n"
            f"    - 类型: {lesion['type']}\n"
            f"    - 位置: {lesion['location']}\n"
            f"    - 大小: {lesion['size_mm']} mm\n"
            f"    - 置信度: {lesion['confidence']:.1%}\n"
        )
    return "\n".join(lines)
