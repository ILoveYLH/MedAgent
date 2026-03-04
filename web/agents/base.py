"""
统一 Agent 输出协议 (JSON Schema)

定义所有专业 Agent 必须遵循的输出数据结构。
使用 Pydantic v2 BaseModel 确保类型安全和自动序列化。

未来扩展：
    - 增加 `evidence_links` 字段，支持引用医学文献
    - 增加 `visualization_data` 字段，支持前端可视化
    - 增加 `dicom_metadata` 字段，支持 DICOM 元信息传递
"""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AgentStatus(str, Enum):
    """Agent 运行状态枚举"""
    IDLE = "idle"           # 等待中
    RUNNING = "running"     # 处理中
    SUCCESS = "success"     # 已完成
    FAILED = "failed"       # 失败


class AbnormalMetric(BaseModel):
    """异常指标数据模型"""
    name: str = Field(..., description="指标名称，如 'CEA'、'结节大小'")
    value: str = Field(..., description="指标值，如 '4.8 ng/mL'")
    reference_range: str = Field(default="", description="参考范围")
    severity: str = Field(
        default="mild",
        description="严重程度: mild / moderate / severe",
    )
    description: str = Field(default="", description="临床意义简述")


class BaseAgentOutput(BaseModel):
    """
    所有 Agent 的统一输出协议

    任何专业 Agent（临床、影像、血液、基因）都必须返回此格式的 JSON。
    前端状态看板和报告生成器均依赖此协议进行解析。
    """
    agent_name: str = Field(
        ..., description="Agent 名称，如 'ClinicalAgent'"
    )
    agent_display_name: str = Field(
        default="", description="Agent 显示名称（中文），如 '临床诊断'"
    )
    status: AgentStatus = Field(
        default=AgentStatus.IDLE,
        description="当前处理状态",
    )
    findings: list[str] = Field(
        default_factory=list,
        description="诊断发现列表，每条为一句中文描述",
    )
    abnormal_metrics: list[AbnormalMetric] = Field(
        default_factory=list,
        description="异常指标列表",
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="综合置信度 (0.0 ~ 1.0)",
    )
    processing_time: float = Field(
        default=0.0,
        description="处理耗时（秒）",
    )
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="原始结构化数据（供下游模块使用）",
    )
    error_message: Optional[str] = Field(
        default=None,
        description="错误信息（仅 status=failed 时填写）",
    )

    # ── 未来扩展字段 ──
    # evidence_links: list[str] = Field(default_factory=list)
    # visualization_data: dict = Field(default_factory=dict)
    # dicom_metadata: Optional[dict] = Field(default=None)
"""
Agent 协议层 — 统一输出格式

所有专业 Agent 必须：
1. 继承或使用 BaseAgentOutput 作为返回值
2. 填充 agent_name, status, findings, abnormal_metrics, confidence
3. 将原始返回数据放入 raw_data
"""
