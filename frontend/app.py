"""
Gradio 前端界面

MedAgent - 智能医学CT诊断助手
界面功能：
  - CT图片上传和分析
  - 疾病分类结果展示
  - 病灶检测结果展示
  - 结构化诊断报告展示（Markdown渲染）
  - 医学知识库管理
"""

import os
import sys
from pathlib import Path

import gradio as gr

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.agent.orchestrator import run_agent
from app.config import settings
from app.rag.knowledge_base import build_knowledge_base, get_collection_info
from app.skills.ct_classifier import classify_ct
from app.skills.lesion_detector import detect_lesions


# ─────────────────────────────────────────
# 核心功能函数
# ─────────────────────────────────────────

def analyze_ct(image, request_text: str):
    """
    完整CT分析流程

    参数:
        image: Gradio上传的图片（PIL Image 或 numpy array）
        request_text: 用户诊断请求文本

    返回:
        (classification_text, detection_text, report_text)
    """
    if image is None:
        return "❌ 请先上传CT图片", "❌ 请先上传CT图片", "❌ 请先上传CT图片"

    # 保存图片到临时文件
    import tempfile
    import numpy as np
    from PIL import Image as PILImage

    if isinstance(image, np.ndarray):
        pil_image = PILImage.fromarray(image)
    else:
        pil_image = image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = run_agent(
            image_path=tmp_path,
            user_request=request_text or "请分析这张CT图片",
        )

        # 格式化分类结果
        cls = result.get("classification", {})
        if cls.get("error"):
            cls_text = f"❌ 分类失败: {cls['error']}"
        else:
            cls_lines = [
                f"**疾病类型**: {cls.get('disease_type', '未知')}",
                f"**置信度**: {cls.get('confidence', 0):.1%}",
                "",
                "**各类别概率**:",
            ]
            all_probs = cls.get("all_probabilities", {})
            for disease, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                bar = "█" * int(prob * 20)
                cls_lines.append(f"- {disease}: {prob:.1%} {bar}")
            if cls.get("is_mock"):
                cls_lines.append("\n> ⚠️ 当前为模拟结果，非真实模型输出")
            cls_text = "\n".join(cls_lines)

        # 格式化检测结果
        det = result.get("detection", {})
        if det.get("error"):
            det_text = f"❌ 检测失败: {det['error']}"
        elif det.get("total_count", 0) == 0:
            det_text = "✅ **未发现明显病灶**"
        else:
            det_lines = [f"**共发现 {det['total_count']} 处病灶**\n"]
            for lesion in det.get("lesions", []):
                bbox = lesion.get("bounding_box", {})
                det_lines.append(
                    f"**病灶 {lesion['lesion_id']}**\n"
                    f"- 类型: {lesion.get('type', '未知')}\n"
                    f"- 位置: {lesion.get('location', '未知')}\n"
                    f"- 大小: {lesion.get('size_mm', 0)} mm\n"
                    f"- 置信度: {lesion.get('confidence', 0):.1%}\n"
                    f"- 边界框: ({bbox.get('x1',0)},{bbox.get('y1',0)}) ~ ({bbox.get('x2',0)},{bbox.get('y2',0)})\n"
                )
            if det.get("is_mock"):
                det_lines.append("> ⚠️ 当前为模拟结果，非真实模型输出")
            det_text = "\n".join(det_lines)

        report = result.get("report", "报告生成失败")

        return cls_text, det_text, report

    except Exception as exc:
        error_msg = f"❌ 分析失败: {str(exc)}"
        return error_msg, error_msg, error_msg
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def quick_classify(image):
    """快速CT分类"""
    if image is None:
        return "❌ 请先上传CT图片"

    import tempfile
    import numpy as np
    from PIL import Image as PILImage

    if isinstance(image, np.ndarray):
        pil_image = PILImage.fromarray(image)
    else:
        pil_image = image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = classify_ct(tmp_path)
        lines = [
            f"**疾病类型**: {result['disease_type']}",
            f"**置信度**: {result['confidence']:.1%}",
        ]
        if result.get("is_mock"):
            lines.append("\n> ⚠️ 当前为模拟结果")
        return "\n".join(lines)
    finally:
        os.unlink(tmp_path)


def build_kb():
    """构建知识库"""
    try:
        count = build_knowledge_base()
        return f"✅ 知识库构建成功！共存入 {count} 个文本块。"
    except Exception as exc:
        return f"❌ 知识库构建失败: {str(exc)}"


def get_kb_status():
    """获取知识库状态"""
    info = get_collection_info()
    return (
        f"**知识库状态**: {info['status']}\n"
        f"**集合名称**: {info['name']}\n"
        f"**文档片段数**: {info['count']}"
    )


def search_kb(query: str, top_k: int):
    """检索知识库"""
    from app.rag.knowledge_base import query_knowledge

    if not query.strip():
        return "❌ 请输入搜索内容"

    results = query_knowledge(query, top_k=int(top_k))
    if not results:
        return "未找到相关知识片段，请先构建知识库。"

    output = []
    for i, chunk in enumerate(results, 1):
        output.append(f"### 参考片段 {i}\n\n{chunk}\n\n---")
    return "\n\n".join(output)


# ─────────────────────────────────────────
# Gradio 界面构建
# ─────────────────────────────────────────

def create_demo():
    """创建 Gradio 界面"""

    with gr.Blocks(
        title="🏥 MedAgent - 智能医学CT诊断助手",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
        css="""
            .header-title { text-align: center; margin-bottom: 10px; }
            .disclaimer { color: #888; font-size: 0.85em; margin-top: 10px; }
        """,
    ) as demo:

        # 标题
        gr.HTML("""
            <div class="header-title">
                <h1>🏥 MedAgent - 智能医学CT诊断助手</h1>
                <p>基于 AI 的医学CT影像智能分析系统 | 支持疾病分类、病灶检测与诊断报告生成</p>
            </div>
        """)

        with gr.Tabs():

            # ── 主分析标签页 ──────────────────────────────
            with gr.TabItem("🔬 CT综合分析"):
                gr.Markdown("### 上传CT图片进行完整诊断分析")

                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="CT图片",
                            type="pil",
                            height=300,
                        )
                        request_input = gr.Textbox(
                            label="诊断请求",
                            placeholder="请描述您的诊断需求，例如：请分析这张胸部CT，重点关注肺部病变",
                            value="请分析这张胸部CT图片，生成完整诊断报告",
                            lines=3,
                        )
                        analyze_btn = gr.Button(
                            "🚀 开始分析",
                            variant="primary",
                            size="lg",
                        )

                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("📊 疾病分类"):
                                cls_output = gr.Markdown(label="分类结果")
                            with gr.TabItem("🎯 病灶检测"):
                                det_output = gr.Markdown(label="检测结果")
                            with gr.TabItem("📋 诊断报告"):
                                report_output = gr.Markdown(label="完整诊断报告")

                analyze_btn.click(
                    fn=analyze_ct,
                    inputs=[image_input, request_input],
                    outputs=[cls_output, det_output, report_output],
                    show_progress=True,
                )

                gr.HTML('<p class="disclaimer">⚠️ 免责声明：本系统仅供参考，不能替代专业医师的诊断意见。</p>')

            # ── 快速分类标签页 ──────────────────────────────
            with gr.TabItem("⚡ 快速分类"):
                gr.Markdown("### 快速CT疾病分类")

                with gr.Row():
                    with gr.Column():
                        quick_image = gr.Image(label="CT图片", type="pil", height=250)
                        quick_btn = gr.Button("⚡ 快速分类", variant="secondary")
                    with gr.Column():
                        quick_output = gr.Markdown(label="分类结果")

                quick_btn.click(
                    fn=quick_classify,
                    inputs=[quick_image],
                    outputs=[quick_output],
                )

            # ── 知识库管理标签页 ──────────────────────────────
            with gr.TabItem("📚 知识库管理"):
                gr.Markdown("### 医学知识库管理")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 知识库状态")
                        kb_status = gr.Markdown()
                        refresh_btn = gr.Button("🔄 刷新状态")
                        build_btn = gr.Button("🏗️ 构建/重建知识库", variant="primary")
                        build_output = gr.Textbox(label="构建结果", interactive=False)

                    with gr.Column():
                        gr.Markdown("#### 知识库检索")
                        search_input = gr.Textbox(
                            label="搜索查询",
                            placeholder="例如：肺结节的良恶性判断标准",
                        )
                        top_k_slider = gr.Slider(
                            label="返回结果数",
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                        )
                        search_btn = gr.Button("🔍 搜索", variant="secondary")
                        search_output = gr.Markdown(label="搜索结果")

                refresh_btn.click(fn=get_kb_status, outputs=[kb_status])
                build_btn.click(fn=build_kb, outputs=[build_output])
                search_btn.click(
                    fn=search_kb,
                    inputs=[search_input, top_k_slider],
                    outputs=[search_output],
                )

                # 页面加载时自动获取知识库状态
                demo.load(fn=get_kb_status, outputs=[kb_status])

            # ── 系统信息标签页 ──────────────────────────────
            with gr.TabItem("ℹ️ 系统信息"):
                gr.Markdown(f"""
## MedAgent 系统信息

### 技术栈
- **LLM**: Google Gemini ({settings.model_name})
- **Agent框架**: LangChain + LangGraph
- **向量数据库**: ChromaDB
- **Embedding**: Google Embedding API
- **前端**: Gradio
- **后端**: FastAPI

### 当前配置
- **模型**: `{settings.model_name}`
- **Embedding模型**: `{settings.embedding_model}`
- **知识库路径**: `{settings.chroma_db_path}`
- **API Key状态**: {"✅ 已配置" if settings.google_api_key else "❌ 未配置（使用Mock模式）"}

### 系统架构
```
用户输入（CT图片 + 文字）
        ↓
  意图识别（Gemini）
        ↓
  CT分类 → 病灶检测
        ↓
  RAG知识库检索
        ↓
  诊断报告生成（Gemini）
        ↓
    最终结果输出
```

### 注意事项
⚠️ 本系统所有分析结果**仅供参考**，不能替代专业医师的临床诊断。
当前使用**Mock模式**（随机模拟结果），如需真实分析请配置Google API Key并接入真实CT分析模型。
""")

    return demo


# ─────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name=settings.gradio_host,
        server_port=settings.gradio_port,
        share=False,
        show_error=True,
    )
