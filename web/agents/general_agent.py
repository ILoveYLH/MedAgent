"""
通用对话 Agent (General Agent)

处理日常问候、闲聊、通用医学咨询等非专业诊断请求。
当用户输入"你好"、"谢谢"、一般性问题时，由此 Agent 回复，
而非直接调用专业诊断 Agent。

未来升级：
    - 接入更强的对话模型，支持多轮上下文
    - 集成医学常识 FAQ 知识库
    - 支持导诊功能（根据对话引导用户提供必要信息）
"""

import logging
import os
import time
from typing import Any, Generator, Optional

from web.agents.base import AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)


class GeneralAgent:
    """
    通用对话 Agent

    职责：
        - 回应日常问候和礼貌用语
        - 回答通用医学常识问题
        - 引导用户提供更具体的诊疗需求

    参数:
        simulate_delay: 模拟处理延迟时间（秒）
    """

    AGENT_NAME = "GeneralAgent"
    DISPLAY_NAME = "💬 智能助手"

    def __init__(self, simulate_delay: float = 0.5) -> None:
        self.simulate_delay = simulate_delay

    def run(
        self,
        text: str = "",
        patient_profile: Optional[dict[str, Any]] = None,
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行通用对话

        参数:
            text: 用户输入文本
            patient_profile: 患者档案（可选）
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的回复结果
        """
        logger.info("[%s] 处理通用对话...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()
        time.sleep(self.simulate_delay)

        elapsed = time.time() - start

        output = BaseAgentOutput(
            agent_name=self.AGENT_NAME,
            agent_display_name=self.DISPLAY_NAME,
            status=AgentStatus.SUCCESS,
            findings=[],
            abnormal_metrics=[],
            confidence=1.0,
            processing_time=round(elapsed, 2),
            raw_data={"original_text": text},
        )

        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.SUCCESS)

        return output

    def stream_reply(self, text: str, patient_profile: Optional[dict[str, Any]] = None) -> Generator[str, None, None]:
        """
        流式生成通用回复（直接调用 LLM）

        参数:
            text: 用户输入
            patient_profile: 患者档案

        返回:
            Generator[str]: 流式文本输出
        """
        qwen_key = os.getenv("QWEN_API_KEY", "")
        if qwen_key:
            try:
                from openai import OpenAI

                client = OpenAI(
                    api_key=qwen_key,
                    base_url=os.getenv(
                        "QWEN_BASE_URL",
                        "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    ),
                )

                patient_context = ""
                if patient_profile:
                    patient_context = (
                        f"\n\n当前患者信息：{patient_profile.get('name', '未知')}，"
                        f"{patient_profile.get('age', '未知')}岁，"
                        f"{patient_profile.get('gender', '未知')}，"
                        f"既往有肺结节病史。"
                    )

                system_prompt = (
                    "你是 MedAgent 智能医疗助手，一个友好、专业的医疗问诊引导系统。\n"
                    "- 如果用户打招呼或闲聊，请友好回应，并简要介绍你的能力\n"
                    "- 如果用户问通用医学问题，请简要回答并建议就医\n"
                    "- 如果用户的问题涉及具体诊断需求，请引导他们提供更多信息"
                    "（如症状描述、上传CT图片等）\n"
                    "- 回复要简洁友好，使用中文，适当使用 emoji\n"
                    "- 不要生成很长的报告，保持对话式风格"
                    f"{patient_context}"
                )

                response = client.chat.completions.create(
                    model=os.getenv("QWEN_MODEL_NAME", "qwen-plus"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
                    ],
                    temperature=0.7,
                    stream=True,
                )

                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as exc:
                logger.warning("GeneralAgent LLM 调用失败: %s", exc)

        # 降级：预设回复
        yield from self._fallback_reply(text)

    def _fallback_reply(self, text: str) -> Generator[str, None, None]:
        """降级预设回复"""
        import time as _time

        reply = (
            "您好！👋 我是 **MedAgent 智能医疗助手**。\n\n"
            "我可以帮您进行以下服务：\n"
            "- 🔬 **CT 影像分析** — 上传 CT 图片进行智能解读\n"
            "- 🩸 **血液检验分析** — 解读血常规和肿瘤标志物\n"
            "- 🧬 **基因检测分析** — 驱动突变和靶向药评估\n"
            "- 🩺 **综合临床诊断** — 多学科会诊报告\n\n"
            "请描述您的症状，或直接上传检查报告，我将为您进行专业分析。 😊"
        )
        for char in reply:
            yield char
            _time.sleep(0.01)
