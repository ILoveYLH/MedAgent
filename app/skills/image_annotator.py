"""
CT 图像标注可视化模块

使用 Pillow 将病灶检测结果（bounding box、类型、置信度）
直接绘制在用户上传的原始 CT 图片上，生成"诊断标注图"。
"""

import logging
import os
import tempfile
from typing import Any

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# 标注样式配置
BOX_COLOR = (255, 0, 0)          # 红框颜色 (R, G, B)
BOX_WIDTH = 3                     # 边框线宽
LABEL_BG_COLOR = (255, 0, 0)     # 标签背景色
LABEL_TEXT_COLOR = (255, 255, 255) # 标签文字色（白色）
LABEL_FONT_SIZE = 14              # 标签字号


def _get_font(size: int = LABEL_FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """获取字体，优先尝试系统中文字体，否则使用默认字体"""
    # 尝试常见中文字体路径
    font_candidates = [
        # Windows
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",     # 黑体
        "C:/Windows/Fonts/simsun.ttc",     # 宋体
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        # macOS
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
    ]

    for font_path in font_candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue

    # 降级到 Pillow 默认字体
    logger.warning("未找到中文字体，使用默认字体（中文标签可能显示异常）")
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def annotate_ct_image(
    image_path: str,
    lesions: list[dict[str, Any]],
    output_path: str | None = None,
) -> str:
    """
    在 CT 图片上绘制病灶检测的 bounding box 和标签

    参数:
        image_path: 原始CT图片路径
        lesions: 病灶列表，每个病灶应包含:
            - bounding_box: {"x1", "y1", "x2", "y2"}
            - type: 病灶类型（str）
            - confidence: 置信度（float）
            - lesion_id: 病灶编号（int）
        output_path: 输出图片路径，None则自动生成临时文件

    返回:
        标注后图片的文件路径
    """
    if not lesions:
        logger.info("无病灶需要标注，返回原图路径")
        return image_path

    try:
        # 打开原图
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = _get_font()

        for lesion in lesions:
            bbox = lesion.get("bounding_box", {})
            x1 = bbox.get("x1", 0)
            y1 = bbox.get("y1", 0)
            x2 = bbox.get("x2", 0)
            y2 = bbox.get("y2", 0)

            if x2 <= x1 or y2 <= y1:
                logger.warning("无效的边界框坐标: %s", bbox)
                continue

            # 绘制红色矩形框
            draw.rectangle(
                [(x1, y1), (x2, y2)],
                outline=BOX_COLOR,
                width=BOX_WIDTH,
            )

            # 构建标签文字
            lesion_type = lesion.get("type", "未知")
            confidence = lesion.get("confidence", 0)
            lesion_id = lesion.get("lesion_id", "?")
            label_text = f"#{lesion_id} {lesion_type} {confidence:.0%}"

            # 计算标签尺寸
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # 标签位置：在框的上方，如果上方空间不够则放在框内上方
            label_x = x1
            label_y = y1 - text_h - 4
            if label_y < 0:
                label_y = y1 + 2

            # 绘制标签背景
            draw.rectangle(
                [(label_x, label_y), (label_x + text_w + 6, label_y + text_h + 4)],
                fill=LABEL_BG_COLOR,
            )

            # 绘制标签文字
            draw.text(
                (label_x + 3, label_y + 2),
                label_text,
                fill=LABEL_TEXT_COLOR,
                font=font,
            )

            # 绘制尺寸标注（如果有 size_mm 信息）
            size_mm = lesion.get("size_mm")
            if size_mm:
                size_label = f"{size_mm}mm"
                size_bbox = draw.textbbox((0, 0), size_label, font=font)
                size_w = size_bbox[2] - size_bbox[0]
                size_h = size_bbox[3] - size_bbox[1]

                size_x = x2 - size_w - 6
                size_y = y2 + 2

                draw.rectangle(
                    [(size_x, size_y), (size_x + size_w + 6, size_y + size_h + 4)],
                    fill=(0, 120, 255),  # 蓝色背景
                )
                draw.text(
                    (size_x + 3, size_y + 2),
                    size_label,
                    fill=LABEL_TEXT_COLOR,
                    font=font,
                )

        # 保存标注图
        if output_path is None:
            suffix = os.path.splitext(image_path)[1] or ".jpg"
            fd, output_path = tempfile.mkstemp(suffix=suffix, prefix="medagent_annotated_")
            os.close(fd)

        img.save(output_path, quality=95)
        logger.info("标注图已保存: %s，共标注 %d 处病灶", output_path, len(lesions))
        return output_path

    except Exception as exc:
        logger.error("图像标注失败: %s", exc)
        raise
