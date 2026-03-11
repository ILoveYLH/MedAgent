"""
MedAgent — Streamlit GPT 风格聊天界面

主入口文件。实现：
  - 简洁的 GPT 风格聊天界面
  - 侧边栏：标题 + 新对话按钮 + 对话历史
  - 主区域：消息历史 + 文件上传 + 聊天输入框
  - 完整工作流：路由 → 并发 Agent → RAG → 流式报告

运行方式：
  streamlit run web/app.py --server.port 8501
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

# ── 项目根目录加入 sys.path ──
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from web.agents.base import AgentStatus, BaseAgentOutput
from web.agents.general_agent import GeneralAgent
from web.executor import ParallelExecutor
from web.rag_reducer import retrieve_medical_guidelines
from web.report_stream import assemble_prompt, stream_report
from web.router import ALL_AGENTS, SPECIALIST_AGENTS, route_input

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────

st.set_page_config(
    page_title="MedAgent · 智能医疗助手",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── 简洁深色主题 CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', 'Microsoft YaHei', sans-serif;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #00d4ff !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #c0c0c0;
    }

    /* 标题区域 */
    .main-header {
        text-align: center;
        padding: 10px 0 15px 0;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6f91);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .main-header p {
        color: #888;
        font-size: 0.9rem;
    }

    /* 聊天消息 */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* 对话历史条目 */
    .conv-item {
        padding: 8px 12px;
        border-radius: 8px;
        margin: 4px 0;
        cursor: pointer;
        color: #c0c0c0;
        font-size: 0.9rem;
        border: 1px solid #333366;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .conv-item:hover {
        border-color: #00d4ff;
        background: rgba(0, 212, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────

def init_session_state() -> None:
    """初始化 Session State"""
    if "conversations" not in st.session_state:
        # 对话历史列表，每个元素为一次对话的消息列表
        st.session_state.conversations = [[]]
    if "current_conv" not in st.session_state:
        st.session_state.current_conv = 0
    if "agent_statuses" not in st.session_state:
        st.session_state.agent_statuses = {
            name: AgentStatus.IDLE for name in ALL_AGENTS
        }
    if "agent_results" not in st.session_state:
        st.session_state.agent_results = {}
    if "processing" not in st.session_state:
        st.session_state.processing = False

    # 确保 current_conv 有效
    if st.session_state.current_conv >= len(st.session_state.conversations):
        st.session_state.current_conv = len(st.session_state.conversations) - 1


def get_messages() -> list:
    """获取当前对话的消息列表"""
    idx = st.session_state.current_conv
    return st.session_state.conversations[idx]


def add_message(role: str, content: str, avatar: str = "") -> None:
    """向当前对话添加消息"""
    idx = st.session_state.current_conv
    st.session_state.conversations[idx].append({
        "role": role,
        "content": content,
        "avatar": avatar,
    })


# ─────────────────────────────────────────
# 侧边栏：简洁对话管理
# ─────────────────────────────────────────

def render_sidebar() -> None:
    """渲染侧边栏 — 标题 + 新对话 + 对话历史"""
    with st.sidebar:
        st.markdown("## 🏥 MedAgent")
        st.markdown("*智能多模态诊疗助手*")
        st.markdown("---")

        # ── 新对话按钮 ──
        if st.button("➕ 新对话", use_container_width=True, type="primary"):
            st.session_state.conversations.append([])
            st.session_state.current_conv = len(st.session_state.conversations) - 1
            st.rerun()

        st.markdown("---")
        st.markdown("### 💬 对话历史")

        # ── 对话历史列表 ──
        for i, conv in enumerate(st.session_state.conversations):
            if not conv:
                label = "新对话"
            else:
                # 取第一条用户消息的前20字符作为标题
                first_user = next(
                    (m["content"] for m in conv if m["role"] == "user"), "新对话"
                )
                label = first_user[:20] + ("..." if len(first_user) > 20 else "")

            is_active = i == st.session_state.current_conv
            btn_type = "primary" if is_active else "secondary"
            if st.button(
                f"{'▶ ' if is_active else ''}{label}",
                key=f"conv_{i}",
                use_container_width=True,
                type=btn_type,
            ):
                st.session_state.current_conv = i
                st.rerun()

        st.markdown("---")
        st.markdown(
            "<div style='color:#666;font-size:0.8rem;text-align:center;'>"
            "支持影像分析、血液检验、基因检测<br>综合医学诊断</div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────
# 主聊天区域
# ─────────────────────────────────────────

def render_chat() -> None:
    """渲染聊天区域"""
    # ── 标题 ──
    st.markdown(
        '<div class="main-header">'
        '<h1>🏥 MedAgent</h1>'
        '<p>智能多模态诊疗助手 · 输入症状/数据，或上传影像文件</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── 显示历史消息 ──
    messages = get_messages()
    for msg in messages:
        avatar = msg.get("avatar") or ("👤" if msg["role"] == "user" else "🏥")
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # ── 文件上传区域 ──
    uploaded_file = st.file_uploader(
        "📎 上传医学文件（CT图像、检验报告等）",
        type=["jpg", "jpeg", "png", "bmp", "dcm", "pdf", "txt"],
        key="file_uploader",
        help="支持上传 CT 图片、检验报告等文件",
    )

    # ── 聊天输入 ──
    user_input = st.chat_input(
        "请描述您的症状、检验数据，或提问...",
        key="chat_input",
    )

    if user_input:
        _handle_user_input(user_input, uploaded_file)


def _handle_user_input(text: str, uploaded_file=None) -> None:
    """处理用户输入并触发诊疗流程"""
    # ── 1. 显示用户消息 ──
    add_message("user", text, "👤")
    with st.chat_message("user", avatar="👤"):
        st.markdown(text)

    # ── 2. 保存上传文件 ──
    image_path = None
    attachments = []
    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        attachments.append(tmp_path)
        if suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".dcm"}:
            image_path = tmp_path

    # ── 3. 路由分发 ──
    activated_agents = route_input(text, attachments)

    # ── 4. 重置 Agent 状态 ──
    for name in ALL_AGENTS:
        st.session_state.agent_statuses[name] = AgentStatus.IDLE

    # ── 5. 判断路径 ──
    is_general_only = (
        len(activated_agents) == 1 and activated_agents[0] == "GeneralAgent"
    )

    if is_general_only:
        # ════════════════
        # 路径 A：通用对话
        # ════════════════
        with st.chat_message("assistant", avatar="💬"):
            general_agent = GeneralAgent()
            reply_placeholder = st.empty()
            full_reply = ""
            for chunk in general_agent.stream_reply(text):
                full_reply += chunk
                reply_placeholder.markdown(full_reply)

        add_message("assistant", full_reply, "💬")

    else:
        # ═══════════════════════
        # 路径 B：专业诊断流水线
        # ═══════════════════════
        specialist_list = [a for a in activated_agents if a != "GeneralAgent"]

        with st.chat_message("assistant", avatar="🏥"):
            st.markdown(
                f"📡 已激活 **{len(specialist_list)}** 个专业 Agent："
                + "、".join(f"`{a}`" for a in specialist_list)
            )

            # ── 6. 并行 Agent 处理 ──
            with st.spinner("⚡ 多 Agent 并行分析中..."):
                executor = ParallelExecutor(max_workers=4)
                input_data = {
                    "text": text,
                    "image_path": image_path,
                }
                results = executor.run(
                    agent_names=specialist_list,
                    input_data=input_data,
                )

            # ── 展示 Agent 结果摘要 ──
            st.session_state.agent_results = results
            for agent_name, result in results.items():
                if result.status == AgentStatus.SUCCESS and result.findings:
                    with st.expander(
                        f"{'✅' if result.status == AgentStatus.SUCCESS else '❌'} "
                        f"{result.agent_display_name or agent_name} "
                        f"({result.processing_time:.1f}s)",
                        expanded=False,
                    ):
                        for finding in result.findings[:3]:
                            st.markdown(f"• {finding}")
                        if result.abnormal_metrics:
                            st.markdown(f"⚠️ 发现 **{len(result.abnormal_metrics)}** 项异常指标")
                elif result.status == AgentStatus.FAILED:
                    st.warning(
                        f"⚠️ {result.agent_display_name or agent_name} 未能完成分析"
                        + (f"：{result.error_message}" if result.error_message else "")
                    )

            # ── 7. RAG 知识融合 ──
            with st.spinner("📚 检索医学知识库..."):
                guidelines = retrieve_medical_guidelines(
                    query=text,
                    agent_results=results,
                )

            # ── 8. 流式报告生成 ──
            st.markdown("---")
            st.markdown("📝 **生成综合诊疗报告...**")

            prompt = assemble_prompt(
                user_query=text,
                patient_profile=None,
                agent_results=results,
                rag_guidelines=guidelines,
            )

            report_placeholder = st.empty()
            full_report = ""
            for chunk in stream_report(prompt):
                full_report += chunk
                report_placeholder.markdown(full_report)

        # ── 9. 保存到消息历史 ──
        add_message("assistant", full_report, "🏥")

    # 清理临时文件
    for path in attachments:
        try:
            os.unlink(path)
        except OSError:
            pass


# ─────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────

def main() -> None:
    """Streamlit 应用主入口"""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__page__":
    main()

main()
