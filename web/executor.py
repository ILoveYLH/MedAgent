"""
并行调度器 (Parallel Executor)

使用 ThreadPoolExecutor 并发调用多个 Agent，
通过 status_callback 实时通知前端更新状态看板。

Streamlit 运行在同步环境，因此使用线程池而非 asyncio。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from web.agents.base import AgentStatus, BaseAgentOutput
from web.agents.blood_agent import BloodAgent
from web.agents.clinical_agent import ClinicalAgent
from web.agents.general_agent import GeneralAgent
from web.agents.genetics_agent import GeneticsAgent
from web.agents.imaging_agent import ImagingAgent

logger = logging.getLogger(__name__)

# ── Agent 注册表 ──
AGENT_REGISTRY: dict[str, type] = {
    "GeneralAgent": GeneralAgent,
    "ClinicalAgent": ClinicalAgent,
    "ImagingAgent": ImagingAgent,
    "BloodAgent": BloodAgent,
    "GeneticsAgent": GeneticsAgent,
}


class ParallelExecutor:
    """
    多 Agent 并行调度器

    使用线程池并发运行多个 Agent，收集结果并实时推送状态更新。

    参数:
        max_workers: 最大并发线程数
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers

    def run(
        self,
        agent_names: list[str],
        input_data: dict[str, Any],
        status_callback: Optional[Callable[[str, AgentStatus], None]] = None,
    ) -> dict[str, BaseAgentOutput]:
        """
        并行调用指定的 Agent 列表

        参数:
            agent_names: 需要运行的 Agent 名称列表
            input_data: 传递给每个 Agent 的输入数据
                - "text": 用户输入文本
                - "image_path": 上传的图片路径 (可选)
                - "attachments": 附件路径列表 (可选)
                - "patient_profile": 患者档案 (可选)
            status_callback: 状态回调函数 (agent_name, status) -> None

        返回:
            dict[str, BaseAgentOutput]: Agent 名称 → 输出结果的映射
        """
        results: dict[str, BaseAgentOutput] = {}

        # 初始化所有 Agent 状态为 IDLE
        if status_callback:
            for name in agent_names:
                status_callback(name, AgentStatus.IDLE)

        def _run_single(agent_name: str) -> tuple[str, BaseAgentOutput]:
            """运行单个 Agent"""
            agent_cls = AGENT_REGISTRY.get(agent_name)
            if not agent_cls:
                logger.error("未知 Agent: %s", agent_name)
                return agent_name, BaseAgentOutput(
                    agent_name=agent_name,
                    status=AgentStatus.FAILED,
                    error_message=f"未注册的 Agent: {agent_name}",
                )

            agent = agent_cls()
            try:
                output = agent.run(
                    text=input_data.get("text", ""),
                    **(
                        {"image_path": input_data.get("image_path")}
                        if agent_name == "ImagingAgent"
                        else {}
                    ),
                    **(
                        {"patient_profile": input_data.get("patient_profile")}
                        if agent_name in ("ClinicalAgent", "BloodAgent", "GeneticsAgent")
                        else {}
                    ),
                    on_status=status_callback,
                )
                return agent_name, output
            except Exception as exc:
                logger.error("[%s] 运行失败: %s", agent_name, exc)
                if status_callback:
                    status_callback(agent_name, AgentStatus.FAILED)
                return agent_name, BaseAgentOutput(
                    agent_name=agent_name,
                    status=AgentStatus.FAILED,
                    error_message=str(exc),
                )

        # ── 并行执行 ──
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(_run_single, name): name
                for name in agent_names
            }

            for future in as_completed(futures):
                agent_name, output = future.result()
                results[agent_name] = output
                logger.info(
                    "[%s] 完成 → status=%s, confidence=%.2f",
                    agent_name, output.status.value, output.confidence,
                )

        return results
