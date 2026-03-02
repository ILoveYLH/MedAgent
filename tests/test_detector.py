"""
病灶检测 Skill 测试

测试 Mock 检测功能的返回格式和数据合法性。
"""

import pytest
from app.skills.lesion_detector import (
    LESION_TYPES,
    detect_lesions,
    lesion_detection_tool,
)


class TestDetectLesions:
    """测试 detect_lesions 函数"""

    def test_returns_dict(self):
        """返回值应为字典"""
        result = detect_lesions("fake_image.png")
        assert isinstance(result, dict)

    def test_required_keys(self):
        """返回字典应包含所有必需字段"""
        result = detect_lesions("fake_image.png")
        required_keys = {"lesions", "total_count", "image_size", "model_version", "image_path", "is_mock"}
        assert required_keys.issubset(result.keys()), f"缺少字段: {required_keys - result.keys()}"

    def test_lesions_is_list(self):
        """lesions 字段应为列表"""
        result = detect_lesions("fake_image.png")
        assert isinstance(result["lesions"], list)

    def test_total_count_matches_lesions(self):
        """total_count 应与 lesions 列表长度一致"""
        result = detect_lesions("fake_image.png")
        assert result["total_count"] == len(result["lesions"])

    def test_total_count_range(self):
        """病灶数量应在 0-3 范围内（mock实现）"""
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            assert 0 <= result["total_count"] <= 3, (
                f"病灶数量超出范围: {result['total_count']}"
            )

    def test_lesion_keys(self):
        """每个病灶对象应包含必需字段"""
        # 多次调用以确保有病灶返回
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            for lesion in result["lesions"]:
                required_lesion_keys = {
                    "lesion_id", "type", "bounding_box", "size_mm",
                    "confidence", "location", "mask_path"
                }
                assert required_lesion_keys.issubset(lesion.keys()), (
                    f"病灶缺少字段: {required_lesion_keys - lesion.keys()}"
                )

    def test_lesion_type_valid(self):
        """病灶类型应在预定义列表中"""
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            for lesion in result["lesions"]:
                assert lesion["type"] in LESION_TYPES, (
                    f"未知病灶类型: {lesion['type']}"
                )

    def test_bounding_box_keys(self):
        """边界框应包含必需坐标字段"""
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            for lesion in result["lesions"]:
                bbox = lesion["bounding_box"]
                for key in ["x1", "y1", "x2", "y2", "width", "height"]:
                    assert key in bbox, f"边界框缺少字段: {key}"

    def test_bounding_box_valid_coords(self):
        """边界框坐标应合法（x2>x1, y2>y1）"""
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            for lesion in result["lesions"]:
                bbox = lesion["bounding_box"]
                assert bbox["x2"] > bbox["x1"], "x2 应大于 x1"
                assert bbox["y2"] > bbox["y1"], "y2 应大于 y1"
                assert bbox["width"] > 0, "width 应大于 0"
                assert bbox["height"] > 0, "height 应大于 0"

    def test_lesion_confidence_range(self):
        """病灶置信度应在 0.0 - 1.0 范围内"""
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            for lesion in result["lesions"]:
                assert 0.0 <= lesion["confidence"] <= 1.0, (
                    f"置信度超出范围: {lesion['confidence']}"
                )

    def test_lesion_size_positive(self):
        """病灶大小应为正数"""
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            for lesion in result["lesions"]:
                assert lesion["size_mm"] > 0, f"病灶大小应为正数: {lesion['size_mm']}"

    def test_mask_path_is_none(self):
        """Mock实现中 mask_path 应为 None"""
        for _ in range(20):
            result = detect_lesions("fake_image.png")
            for lesion in result["lesions"]:
                assert lesion["mask_path"] is None

    def test_is_mock_flag(self):
        """is_mock 应为 True"""
        result = detect_lesions("fake_image.png")
        assert result["is_mock"] is True

    def test_image_size_keys(self):
        """image_size 应包含 width 和 height"""
        result = detect_lesions("fake_image.png")
        assert "width" in result["image_size"]
        assert "height" in result["image_size"]


class TestLesionDetectionTool:
    """测试 LangChain Tool 封装"""

    def test_tool_returns_string(self):
        """LangChain Tool 应返回字符串"""
        result = lesion_detection_tool.invoke({"image_path": "fake_image.png"})
        assert isinstance(result, str)

    def test_tool_contains_detection_info(self):
        """Tool 返回字符串应包含检测信息"""
        result = lesion_detection_tool.invoke({"image_path": "fake_image.png"})
        assert "病灶检测结果" in result
