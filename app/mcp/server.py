"""
MCP Server 模块

使用 MCP (Model Context Protocol) SDK 将 CT分类 和 病灶检测 技能
暴露为标准的 MCP Tools，支持通过 MCP 协议调用。

启动方式：
    python -m app.mcp.server

连接方式（stdio）：
    使用支持 MCP 协议的客户端（如 Claude Desktop、MCP Inspector）连接
"""

import json
import logging

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from app.skills.ct_classifier import classify_ct
from app.skills.lesion_detector import detect_lesions

logger = logging.getLogger(__name__)

# 创建 MCP Server 实例
server = Server("medagent-mcp-server")


# ─────────────────────────────────────────
# MCP Tools 定义
# ─────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[Tool]:
    """列出所有可用的 MCP Tools"""
    return [
        Tool(
            name="ct_classify",
            description=(
                "对CT图片进行疾病分类分析。"
                "输入CT图片路径，返回疾病类型、置信度及各类别概率分布。"
                "支持识别：正常、肺炎、肺结节、肺癌、肺气肿、胸腔积液等。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "CT图片文件的绝对或相对路径",
                    }
                },
                "required": ["image_path"],
            },
        ),
        Tool(
            name="detect_lesions",
            description=(
                "对CT图片进行病灶检测和分割分析。"
                "输入CT图片路径，返回病灶的位置（边界框）、大小、类型和置信度。"
                "支持检测：实性结节、磨玻璃影、混合密度结节、肿块等病灶类型。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "CT图片文件的绝对或相对路径",
                    }
                },
                "required": ["image_path"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """
    处理 MCP Tool 调用请求

    参数:
        name: 工具名称（ct_classify 或 detect_lesions）
        arguments: 工具调用参数字典

    返回:
        包含结果文本的 TextContent 列表
    """
    image_path = arguments.get("image_path", "")

    if name == "ct_classify":
        logger.info("MCP Tool 调用: ct_classify, image_path=%s", image_path)
        result = classify_ct(image_path)
        result_text = (
            f"CT分类结果:\n"
            f"- 疾病类型: {result['disease_type']}\n"
            f"- 置信度: {result['confidence']:.1%}\n"
            f"- 模型版本: {result['model_version']}\n"
            f"- 各类别概率:\n"
        )
        for disease, prob in sorted(result["all_probabilities"].items(), key=lambda x: x[1], reverse=True):
            result_text += f"  • {disease}: {prob:.1%}\n"

        if result.get("is_mock"):
            result_text += "\n⚠️ 注：当前为模拟结果，非真实模型输出"

        return [TextContent(type="text", text=result_text)]

    elif name == "detect_lesions":
        logger.info("MCP Tool 调用: detect_lesions, image_path=%s", image_path)
        result = detect_lesions(image_path)

        if result["total_count"] == 0:
            result_text = "病灶检测结果：未发现明显病灶"
        else:
            lines = [f"病灶检测结果：共发现 {result['total_count']} 处病灶\n"]
            for lesion in result["lesions"]:
                bbox = lesion["bounding_box"]
                lines.append(
                    f"病灶 {lesion['lesion_id']}:\n"
                    f"  - 类型: {lesion['type']}\n"
                    f"  - 位置: {lesion['location']}\n"
                    f"  - 大小: {lesion['size_mm']} mm\n"
                    f"  - 边界框: ({bbox['x1']},{bbox['y1']}) - ({bbox['x2']},{bbox['y2']})\n"
                    f"  - 置信度: {lesion['confidence']:.1%}\n"
                )
            if result.get("is_mock"):
                lines.append("⚠️ 注：当前为模拟结果，非真实模型输出")
            result_text = "\n".join(lines)

        return [TextContent(type="text", text=result_text)]

    else:
        error_msg = f"未知工具: {name}。可用工具: ct_classify, detect_lesions"
        logger.warning(error_msg)
        return [TextContent(type="text", text=error_msg)]


async def main():
    """启动 MCP Server（stdio 模式）"""
    logger.info("启动 MedAgent MCP Server...")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
