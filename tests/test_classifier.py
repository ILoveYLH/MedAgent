"""
CT分类 Skill 测试

测试 Mock 分类功能的返回格式和数据合法性。
"""

import pytest
from app.skills.ct_classifier import (
    DISEASE_CLASSES,
    classify_ct,
    analyze_ct_tool,
    ct_classification_tool,
)


class TestClassifyCT:
    """测试 classify_ct 函数"""

    def test_returns_dict(self):
        """返回值应为字典"""
        result = classify_ct("fake_image.png")
        assert isinstance(result, dict)

    def test_required_keys(self):
        """返回字典应包含所有必需字段"""
        result = classify_ct("fake_image.png")
        required_keys = {"disease_type", "confidence", "all_probabilities", "model_version", "image_path", "is_mock"}
        assert required_keys.issubset(result.keys()), f"缺少字段: {required_keys - result.keys()}"

    def test_disease_type_valid(self):
        """疾病类型应在预定义列表中"""
        result = classify_ct("fake_image.png")
        assert result["disease_type"] in DISEASE_CLASSES, (
            f"未知疾病类型: {result['disease_type']}"
        )

    def test_confidence_range(self):
        """置信度应在 0.0 - 1.0 范围内"""
        for _ in range(10):
            result = classify_ct("fake_image.png")
            assert 0.0 <= result["confidence"] <= 1.0, (
                f"置信度超出范围: {result['confidence']}"
            )

    def test_all_probabilities_is_dict(self):
        """all_probabilities 应为字典"""
        result = classify_ct("fake_image.png")
        assert isinstance(result["all_probabilities"], dict)

    def test_all_probabilities_contains_disease(self):
        """all_probabilities 应包含主要疾病"""
        result = classify_ct("fake_image.png")
        assert result["disease_type"] in result["all_probabilities"]

    def test_all_probabilities_keys(self):
        """all_probabilities 的所有键应在预定义疾病列表中"""
        result = classify_ct("fake_image.png")
        for disease in result["all_probabilities"]:
            assert disease in DISEASE_CLASSES, f"未知疾病: {disease}"

    def test_is_mock_flag(self):
        """is_mock 应为 True（mock实现）"""
        result = classify_ct("fake_image.png")
        assert result["is_mock"] is True

    def test_image_path_preserved(self):
        """返回结果应保留输入的图片路径"""
        path = "/some/test/path/ct.png"
        result = classify_ct(path)
        assert result["image_path"] == path

    def test_model_version_format(self):
        """model_version 应为非空字符串"""
        result = classify_ct("fake_image.png")
        assert isinstance(result["model_version"], str)
        assert len(result["model_version"]) > 0

    def test_multiple_calls_return_valid_results(self):
        """多次调用都应返回合法结果（随机性测试）"""
        for _ in range(20):
            result = classify_ct("test.png")
            assert result["disease_type"] in DISEASE_CLASSES
            assert 0.0 <= result["confidence"] <= 1.0


class TestCTClassificationTool:
    """测试 LangChain Tool 封装"""

    def test_tool_returns_string(self):
        """LangChain Tool 应返回字符串"""
        result = ct_classification_tool.invoke({"image_path": "fake_image.png"})
        assert isinstance(result, str)

    def test_tool_contains_disease_info(self):
        """Tool 返回字符串应包含疾病信息"""
        result = ct_classification_tool.invoke({"image_path": "fake_image.png"})
        assert "CT分类结果" in result
        assert "疾病类型" in result
        assert "置信度" in result
