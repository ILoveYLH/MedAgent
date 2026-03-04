"""
MedAgent — Streamlit 多 Agent 可视化诊疗 Demo

主入口文件。实现：
  - 侧边栏：患者数字医疗档案
  - 主区域上方：多 Agent 并行状态看板
  - 主区域下方：Chat UI（文本 + 文件上传）
  - 完整五阶段工作流：路由 → 并发 Agent → RAG → 流式报告

运行方式：
  cd d:/Project/Project_AI/MedAgent
  streamlit run web/app.py --server.port 8501
"""

import json
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
    page_title="MedAgent · 智能多模态诊疗系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 自定义 CSS ──
st.markdown("""
<style>
    /* 全局字体 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', 'Microsoft YaHei', sans-serif;
    }

    /* 侧边栏美化 */
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

    /* Agent 状态卡片 */
    .agent-card {
        background: linear-gradient(135deg, #1e1e2f, #2a2a4a);
        border-radius: 12px;
        padding: 16px;
        margin: 4px 0;
        border: 1px solid #333366;
        transition: all 0.3s ease;
    }
    .agent-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.15);
    }
    .agent-card-running {
        border-color: #ffa500 !important;
        box-shadow: 0 0 20px rgba(255, 165, 0, 0.2);
        animation: pulse-border 1.5s ease-in-out infinite;
    }
    .agent-card-success {
        border-color: #00e676 !important;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.15);
    }
    .agent-card-failed {
        border-color: #ff5252 !important;
    }
    @keyframes pulse-border {
        0%, 100% { border-color: #ffa500; }
        50% { border-color: #ff6f00; }
    }

    /* 标题区域 */
    .main-header {
        text-align: center;
        padding: 10px 0 20px 0;
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

    /* 状态标签 */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-idle { background: #333; color: #888; }
    .status-running { background: #4a3800; color: #ffa500; }
    .status-success { background: #003d20; color: #00e676; }
    .status-failed { background: #3d0000; color: #ff5252; }

    /* 聊天区域微调 */
    .stChatMessage {
        border-radius: 12px !important;
    }

    /* 文件上传区域 */
    .stFileUploader > div {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────

@st.cache_data
def load_mock_data() -> dict:
    """加载虚拟病人数据"""
    mock_path = Path(__file__).parent / "mock_data.json"
    with open(mock_path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_session_state() -> None:
    """初始化 Session State"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_statuses" not in st.session_state:
        st.session_state.agent_statuses = {
            name: AgentStatus.IDLE for name in ALL_AGENTS
        }
    if "agent_results" not in st.session_state:
        st.session_state.agent_results = {}
    if "patient_data" not in st.session_state:
        st.session_state.patient_data = load_mock_data()
    if "processing" not in st.session_state:
        st.session_state.processing = False


# ─────────────────────────────────────────
# 侧边栏：患者数字医疗档案
# ─────────────────────────────────────────

def render_sidebar() -> None:
    """渲染侧边栏 — 患者数字医疗档案"""
    data = st.session_state.patient_data
    patient = data.get("patient", {})

    with st.sidebar:
        st.markdown("## 🏥 MedAgent")
        st.markdown("---")

        # ── 基本信息 ──
        st.markdown("### 📋 患者信息")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**姓名**：{patient.get('name', '-')}")
            st.markdown(f"**年龄**：{patient.get('age', '-')}岁")
            st.markdown(f"**血型**：{patient.get('blood_type', '-')}")
        with col2:
            st.markdown(f"**性别**：{patient.get('gender', '-')}")
            st.markdown(f"**BMI**：{patient.get('bmi', '-')}")
            st.markdown(f"**ID**：{patient.get('id', '-')}")

        # ── 吸烟/过敏 ──
        st.markdown(f"🚬 **吸烟史**：{patient.get('smoking_history', '未知')}")
        allergies = patient.get("allergies", [])
        if allergies:
            st.markdown(f"⚠️ **过敏史**：{'、'.join(allergies)}")

        st.markdown("---")

        # ── 当前症状 ──
        symptoms = data.get("recent_symptoms", [])
        if symptoms:
            st.markdown("### 🩹 当前症状")
            for s in symptoms:
                st.markdown(f"- {s}")

        st.markdown("---")

        # ── 既往病史 ──
        history = data.get("medical_history", [])
        if history:
            st.markdown("### 📂 既往病史")
            for h in history:
                with st.expander(f"📅 {h['date']} — {h['type']}", expanded=False):
                    st.markdown(f"**科室**：{h.get('department', '-')}")
                    st.markdown(f"**诊断**：{h.get('diagnosis', '-')}")
                    st.markdown(f"**处置**：{h.get('treatment', '-')}")
                    st.markdown(f"**医师**：{h.get('doctor', '-')}")

        st.markdown("---")

        # ── 历史报告 ──
        reports = data.get("historical_reports", [])
        if reports:
            st.markdown("### 📄 历史报告")
            for r in reports:
                st.markdown(
                    f"- **{r['date']}** {r['type']}\n"
                    f"  {r.get('summary', '')[:60]}..."
                )

        st.markdown("---")

        # ── 近期化验 ──
        lab = data.get("recent_lab_results", {})
        if lab:
            st.markdown(f"### 🧪 近期化验 ({lab.get('date', '')})")
            items = lab.get("items", {})
            for name, info in items.items():
                status_icon = "✅" if info["status"] == "normal" else "⚠️"
                st.markdown(
                    f"{status_icon} **{name}**: {info['value']} {info['unit']}"
                )


# ─────────────────────────────────────────
# 主区域上方：Agent 状态看板
# ─────────────────────────────────────────

def render_agent_dashboard() -> None:
    """渲染 Agent 并行处理状态看板"""
    st.markdown(
        '<div class="main-header">'
        '<h1>🏥 MedAgent 智能多模态诊疗系统</h1>'
        '<p>Multi-Agent Parallel Diagnostic Pipeline — AI 辅助多学科会诊</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    agent_info = {
        "GeneralAgent": ("💬", "智能助手", "对话与导诊"),
        "ClinicalAgent": ("🩺", "临床诊断", "综合分析症状与病史"),
        "ImagingAgent": ("🔬", "影像分析", "CT / MRI 影像解读"),
        "BloodAgent": ("🩸", "血液分析", "血常规与肿瘤标志物"),
        "GeneticsAgent": ("🧬", "基因检测", "驱动突变与靶向药"),
    }

    cols = st.columns(5)
    for idx, (agent_name, (icon, label, desc)) in enumerate(agent_info.items()):
        status = st.session_state.agent_statuses.get(agent_name, AgentStatus.IDLE)

        # 状态样式
        status_map = {
            AgentStatus.IDLE: ("⏳ 待机", "status-idle", "agent-card"),
            AgentStatus.RUNNING: ("⚡ 处理中", "status-running", "agent-card agent-card-running"),
            AgentStatus.SUCCESS: ("✅ 完成", "status-success", "agent-card agent-card-success"),
            AgentStatus.FAILED: ("❌ 失败", "status-failed", "agent-card agent-card-failed"),
        }
        status_text, badge_cls, card_cls = status_map.get(
            status, ("⏳ 待机", "status-idle", "agent-card")
        )

        with cols[idx]:
            # 获取结果摘要
            result = st.session_state.agent_results.get(agent_name)
            confidence_text = ""
            if result and status == AgentStatus.SUCCESS:
                confidence_text = f"<br><span style='color:#00d4ff;font-size:1.2em;font-weight:700;'>{result.confidence:.0%}</span> 置信度"
                time_text = f"<br><span style='color:#888;font-size:0.8em;'>⏱ {result.processing_time:.1f}s</span>"
            else:
                time_text = ""

            st.markdown(
                f'<div class="{card_cls}">'
                f'  <div style="font-size:2em;text-align:center;">{icon}</div>'
                f'  <div style="text-align:center;font-weight:600;margin:6px 0;">{label}</div>'
                f'  <div style="text-align:center;color:#888;font-size:0.8em;">{desc}</div>'
                f'  <div style="text-align:center;margin-top:8px;">'
                f'    <span class="status-badge {badge_cls}">{status_text}</span>'
                f'    {confidence_text}{time_text}'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")


# ─────────────────────────────────────────
# 主区域下方：Chat UI
# ─────────────────────────────────────────

def render_chat() -> None:
    """渲染聊天区域"""
    # 显示历史消息
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar=msg.get("avatar")):
            st.markdown(msg["content"])

    # 文件上传区域
    uploaded_file = st.file_uploader(
        "📎 上传医学文件（CT图像、检验报告等）",
        type=["jpg", "jpeg", "png", "bmp", "dcm", "pdf", "txt"],
        key="file_uploader",
        help="支持上传 CT 图片、检验报告等文件",
    )

    # 聊天输入
    user_input = st.chat_input(
        "请描述您的症状或诊疗需求...",
        key="chat_input",
    )

    if user_input:
        _handle_user_input(user_input, uploaded_file)


def _handle_user_input(text: str, uploaded_file=None) -> None:  # noqa: C901
    """处理用户输入并触发诊疗流程"""
    # ── 1. 显示用户消息 ──
    st.session_state.messages.append({
        "role": "user",
        "content": text,
        "avatar": "👤",
    })
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

    # ── 3. 路由分发（LLM 智能路由）──
    activated_agents = route_input(text, attachments)

    # ── 4. 初始化状态 ──
    for name in ALL_AGENTS:
        st.session_state.agent_statuses[name] = AgentStatus.IDLE

    # ── 5. 判断走哪条路径 ──
    is_general_only = (
        len(activated_agents) == 1 and activated_agents[0] == "GeneralAgent"
    )

    if is_general_only:
        # ════════════════════════════════════
        # 路径 A：通用对话（GeneralAgent）
        # ════════════════════════════════════
        st.session_state.agent_statuses["GeneralAgent"] = AgentStatus.RUNNING

        with st.chat_message("assistant", avatar="💬"):
            general_agent = GeneralAgent()
            patient_profile = st.session_state.patient_data.get("patient")

            reply_placeholder = st.empty()
            full_reply = ""
            for chunk in general_agent.stream_reply(text, patient_profile):
                full_reply += chunk
                reply_placeholder.markdown(full_reply)

        st.session_state.agent_statuses["GeneralAgent"] = AgentStatus.SUCCESS
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_reply,
            "avatar": "💬",
        })

    else:
        # ════════════════════════════════════
        # 路径 B：专业诊断流水线
        # ════════════════════════════════════
        # 过滤掉 GeneralAgent（不参与诊断流水线）
        specialist_list = [a for a in activated_agents if a != "GeneralAgent"]

        with st.chat_message("assistant", avatar="🏥"):
            st.markdown(
                f"📡 **智能路由完成** — 已激活 **{len(specialist_list)}** 个专业 Agent："
            )
            st.markdown("、".join(f"`{a}`" for a in specialist_list))

            # ── 6. 并行 Agent 处理 ──
            st.markdown("---")
            st.markdown("⚡ **多 Agent 并行处理中...**")

            status_containers = {}
            status_cols = st.columns(len(specialist_list))
            for idx, agent_name in enumerate(specialist_list):
                with status_cols[idx]:
                    status_containers[agent_name] = st.status(
                        f"⚡ 正在启动 {agent_name}...",
                        expanded=True,
                        state="running",
                    )

            executor = ParallelExecutor(max_workers=4)
            input_data = {
                "text": text,
                "image_path": image_path,
                "attachments": attachments,
                "patient_profile": st.session_state.patient_data.get("patient"),
            }

            results = executor.run(
                agent_names=specialist_list,
                input_data=input_data,
                status_callback=None,
            )

            # ── 主线程更新 session_state 和 UI ──
            st.session_state.agent_results = results
            for agent_name in specialist_list:
                result = results.get(agent_name)
                if result:
                    st.session_state.agent_statuses[agent_name] = result.status
                    container = status_containers.get(agent_name)
                    if container and result.status == AgentStatus.SUCCESS:
                        container.update(
                            label=f"✅ {result.agent_display_name or agent_name} — 完成 ({result.processing_time:.1f}s)",
                            state="complete",
                        )
                        with container:
                            for finding in result.findings[:3]:
                                st.markdown(f"• {finding}")
                            if result.abnormal_metrics:
                                st.markdown(
                                    f"⚠️ 发现 **{len(result.abnormal_metrics)}** 项异常指标"
                                )
                    elif container and result.status == AgentStatus.FAILED:
                        container.update(
                            label=f"❌ {result.agent_display_name or agent_name} — 失败",
                            state="error",
                        )
                        with container:
                            st.error(result.error_message or "未知错误")

            # ── 7. RAG 知识融合 ──
            st.markdown("---")
            st.markdown("📚 **正在检索医学知识库...**")

            guidelines = retrieve_medical_guidelines(
                query=text,
                agent_results=results,
            )
            st.markdown(f"✅ 匹配到 **{len(guidelines)}** 条相关指南")

            # ── 8. 流式报告生成 ──
            st.markdown("---")
            st.markdown("📝 **主治医师正在生成综合诊疗报告...**")
            st.markdown("")

            patient_profile = st.session_state.patient_data.get("patient")
            prompt = assemble_prompt(
                user_query=text,
                patient_profile=patient_profile,
                agent_results=results,
                rag_guidelines=guidelines,
            )

            report_placeholder = st.empty()
            full_report = ""
            for chunk in stream_report(prompt):
                full_report += chunk
                report_placeholder.markdown(full_report)

        # ── 9. 保存到消息历史 ──
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_report,
            "avatar": "🏥",
        })

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
    render_agent_dashboard()
    render_chat()


if __name__ == "__page__":
    main()

main()
