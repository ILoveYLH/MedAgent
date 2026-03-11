"""
MedAgent — GPT 风格简洁聊天界面

主入口文件。实现：
  - 简洁聊天界面（类似 ChatGPT）
  - 用户直接在聊天框输入真实医疗数据
  - Agent 自动路由、并行处理、流式报告输出

运行方式：
  cd /path/to/MedAgent
  streamlit run web/app.py --server.port 8501
"""

import logging
import os
import sys
from pathlib import Path

# ── 项目根目录加入 sys.path ──
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st

# 加载 .env 环境变量
from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from web.agents.base import AgentStatus
from web.agents.general_agent import GeneralAgent
from web.executor import ParallelExecutor
from web.rag_reducer import retrieve_medical_guidelines
from web.report_stream import assemble_prompt, stream_report
from web.router import route_input

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# 页面配置
# ─────────────────────────────────────────

st.set_page_config(
    page_title="MedAgent · 智能诊疗助手",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── 自定义 CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', 'Microsoft YaHei', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .main-header h1 {
        background: linear-gradient(90deg, #00d4ff, #7b2ff7, #ff6f91);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .main-header p {
        color: #888;
        font-size: 0.95rem;
    }

    .stChatMessage {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# Session State 初始化
# ─────────────────────────────────────────

def init_session_state() -> None:
    """初始化 Session State"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing" not in st.session_state:
        st.session_state.processing = False


# ─────────────────────────────────────────
# 主聊天界面
# ─────────────────────────────────────────

def render_chat() -> None:
    """渲染聊天区域"""
    st.markdown(
        '<div class="main-header">'
        '<h1>🏥 MedAgent</h1>'
        '<p>智能多模态诊疗助手 — 请直接输入您的症状或检查数据</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.markdown(msg["content"])

    # 聊天输入
    user_input = st.chat_input(
        "请描述您的症状或粘贴检查数据（如：患者男性58岁，干咳两周，CT显示右肺上叶8mm结节...）",
    )

    if user_input:
        _handle_user_input(user_input)


def _handle_user_input(text: str) -> None:  # noqa: C901
    """处理用户输入并触发诊疗流程"""
    # ── 1. 显示用户消息 ──
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "avatar": "👤",
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(text)

    # ── 2. 路由分发 ──
    activated_agents = route_input(text)

    # ── 3. 判断走哪条路径 ──
    is_general_only = (
        len(activated_agents) == 1 and activated_agents[0] == "GeneralAgent"
    )

    if is_general_only:
        # ════════════════════════════════════
        # 路径 A：通用对话（GeneralAgent）
        # ════════════════════════════════════
        with st.chat_message("assistant", avatar="💬"):
            general_agent = GeneralAgent()
            reply_placeholder = st.empty()
            full_reply = ""
            for chunk in general_agent.stream_reply(text):
                full_reply += chunk
                reply_placeholder.markdown(full_reply)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_reply,
            "avatar": "💬",
        })

    else:
        # ════════════════════════════════════
        # 路径 B：专业诊断流水线
        # ════════════════════════════════════
        specialist_list = [a for a in activated_agents if a != "GeneralAgent"]

        with st.chat_message("assistant", avatar="🏥"):
            st.markdown(
                f"📡 **已激活 {len(specialist_list)} 个专业 Agent**：{' · '.join(specialist_list)}"
            )

            # ── 4. 并行 Agent 处理 ──
            with st.spinner("🔍 正在分析..."):
                executor = ParallelExecutor(max_workers=4)
                results = executor.run(
                    agent_names=specialist_list,
                    input_data={"text": text},
                    status_callback=None,
                )

            # 简洁展示各 Agent 结果摘要
            success_count = sum(
                1 for r in results.values() if r.status == AgentStatus.SUCCESS
            )
            st.markdown(f"✅ **{success_count}/{len(specialist_list)} 个 Agent 分析完成**")

            # ── 5. RAG 知识融合 ──
            with st.spinner("📚 检索医学知识库..."):
                guidelines = retrieve_medical_guidelines(
                    query=text,
                    agent_results=results,
                )
            st.markdown(f"📖 匹配到 **{len(guidelines)}** 条相关指南")

            # ── 6. 流式报告生成 ──
            st.markdown("---")
            st.markdown("📝 **正在生成综合诊疗报告...**")

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

        # ── 7. 保存到消息历史 ──
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_report,
            "avatar": "🏥",
        })


# ─────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────

def main() -> None:
    """Streamlit 应用主入口"""
    init_session_state()
    render_chat()


if __name__ == "__page__":
    main()

main()
