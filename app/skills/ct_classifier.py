"""
CT疾病分类 Skill（Mock实现）

当前使用随机模拟结果，预留真实模型接口。

如何替换为真实模型：
  1. 安装模型依赖（如 torch, torchvision 或 onnxruntime）
  2. 在 _load_real_model() 中加载预训练权重
  3. 在 classify_ct() 中替换 _mock_classify() 调用为真实推理代码
  4. 真实模型示例：
     - ResNet50：torchvision.models.resnet50(pretrained=True)
     - EfficientNet：timm.create_model('efficientnet_b4', pretrained=True)
     - ONNX模型：onnxruntime.InferenceSession('ct_classifier.onnx')
"""

import logging
import random
from pathlib import Path

from langchain.tools import tool

logger = logging.getLogger(__name__)

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

    替换为真实模型的步骤：
        1. 加载预训练模型：model = load_model('path/to/model.pth')
        2. 预处理图片：img_tensor = preprocess(image_path)
        3. 推理：with torch.no_grad(): output = model(img_tensor)
        4. 后处理：probs = torch.softmax(output, dim=1)
        5. 返回格式化结果
    """
    logger.info("开始CT分类，图片路径: %s", image_path)

    # 验证输入路径（允许mock模式下路径不存在）
    path = Path(image_path)
    if not path.exists():
        logger.warning("图片文件不存在: %s，使用mock数据", image_path)

    # ── 真实模型接口预留 ──────────────────────────
    # TODO: 取消下面注释以启用真实模型
    # from app.skills._real_models import load_ct_classifier
    # model = load_ct_classifier()
    # result = model.predict(image_path)
    # return result
    # ─────────────────────────────────────────────

    result = _mock_classify(image_path)
    logger.info("CT分类完成: %s (置信度: %.2f%%)", result["disease_type"], result["confidence"] * 100)
    return result


# ─────────────────────────────────────────
# LangChain Tool 注册
# ─────────────────────────────────────────

@tool
def ct_classification_tool(image_path: str) -> str:
    """
    CT疾病分类工具（LangChain Tool）

    对给定的CT图片进行疾病分类分析，返回疾病类型和置信度。

    参数:
        image_path: CT图片文件路径

    返回:
        包含分类结果的格式化字符串
    """
    result = classify_ct(image_path)
    return (
        f"CT分类结果:\n"
        f"- 疾病类型: {result['disease_type']}\n"
        f"- 置信度: {result['confidence']:.1%}\n"
        f"- 模型版本: {result['model_version']}\n"
        f"- 所有类别概率: {result['all_probabilities']}"
    )
