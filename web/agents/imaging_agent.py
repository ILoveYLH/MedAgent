"""
影像分析 Agent (Mock + 冗余接口)

读取上传的 2D 图像 / DICOM 序列，进行影像学分析。
当前 MVP：复用 app/skills 的 mock 分类+检测逻辑。

未来升级：
    - process_dicom_series(): 接入基于 VTK/ITK 的三维重建管线
    - generate_3d_reconstruction(): 输出三维体渲染结果
    - 接入真实 ResNet/EfficientNet CT 分类模型
    - 接入真实 YOLOv8/nnU-Net 病灶检测模型
"""

import logging
import time
from pathlib import Path
from typing import Any, Optional

from web.agents.base import AbnormalMetric, AgentStatus, BaseAgentOutput

logger = logging.getLogger(__name__)


class ImagingAgent:
    """
    影像分析 Agent

    职责：
        - 2D CT 图像的疾病分类
        - 病灶检测与定位
        - (预留) DICOM 序列三维重建

    参数:
        simulate_delay: 模拟处理延迟时间（秒）
    """

    AGENT_NAME = "ImagingAgent"
    DISPLAY_NAME = "🔬 影像分析"

    def __init__(self, simulate_delay: float = 3.0) -> None:
        self.simulate_delay = simulate_delay

    def run(
        self,
        image_path: Optional[str] = None,
        text: str = "",
        *,
        on_status: Any = None,
    ) -> BaseAgentOutput:
        """
        执行影像分析

        参数:
            image_path: 上传的 CT 图片路径（.jpg/.png/.dcm）
            text: 附加文本描述
            on_status: 状态回调

        返回:
            BaseAgentOutput: 统一格式的影像分析结果
        """
        logger.info("[%s] 开始影像分析...", self.AGENT_NAME)
        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.RUNNING)

        start = time.time()
        time.sleep(self.simulate_delay)

        # ── Mock 影像分析结果 ──
        has_image = image_path and Path(image_path).exists()

        findings = [
            "右肺上叶后段可见一枚混合密度结节影，大小约 8×7mm",
            "结节边缘可见短毛刺征，内部密度不均匀",
            "结节周围可见轻度磨玻璃晕征（halo sign）",
            "余肺野清晰，未见实变影或胸腔积液",
            "纵隔淋巴结未见明显增大（短径 < 10mm）",
        ]

        abnormal_metrics = [
            AbnormalMetric(
                name="结节最大径",
                value="8mm",
                reference_range="<6mm 低风险",
                severity="moderate",
                description="较前次检查增大2mm，需进一步评估",
            ),
            AbnormalMetric(
                name="结节CT值",
                value="-320 ~ +45 HU",
                reference_range="纯GGO: <-400HU",
                severity="moderate",
                description="混合密度结节，实性成分占比约30%",
            ),
        ]

        elapsed = time.time() - start
        logger.info("[%s] 影像分析完成 (%.1fs)", self.AGENT_NAME, elapsed)

        output = BaseAgentOutput(
            agent_name=self.AGENT_NAME,
            agent_display_name=self.DISPLAY_NAME,
            status=AgentStatus.SUCCESS,
            findings=findings,
            abnormal_metrics=abnormal_metrics,
            confidence=0.85,
            processing_time=round(elapsed, 2),
            raw_data={
                "classification": {
                    "disease_type": "肺结节",
                    "confidence": 0.85,
                    "all_probabilities": {
                        "肺结节": 0.85,
                        "肺炎": 0.06,
                        "正常": 0.04,
                        "肺癌": 0.03,
                        "肺气肿": 0.01,
                        "胸腔积液": 0.01,
                    },
                },
                "detection": {
                    "lesions": [
                        {
                            "lesion_id": 1,
                            "type": "混合密度结节",
                            "location": "右侧上叶",
                            "size_mm": 8.0,
                            "bounding_box": {"x1": 185, "y1": 120, "x2": 240, "y2": 170},
                            "confidence": 0.88,
                        }
                    ],
                    "total_count": 1,
                },
                "has_uploaded_image": has_image,
                "image_path": image_path,
            },
        )

        if on_status:
            on_status(self.AGENT_NAME, AgentStatus.SUCCESS)

        return output

    # ──────────────────────────────────────
    # 未来冗余接口（三维重建管线）
    # ──────────────────────────────────────

    def process_dicom_series(
        self,
        dir_path: str,
        *,
        slice_thickness: Optional[float] = None,
        window_center: int = -600,
        window_width: int = 1500,
    ) -> dict[str, Any]:
        """
        处理 DICOM 序列文件夹，提取三维体数据

        # TODO: 接入基于 VTK/ITK 的三维重建管线
        # 实现步骤：
        #   1. 使用 pydicom 读取 .dcm 文件列表
        #   2. 基于 SliceLocation / InstanceNumber 排序
        #   3. 使用 SimpleITK 构建 3D Volume
        #   4. 应用窗宽窗位 (WL/WW) 进行灰度映射
        #   5. 使用 VTK 进行 Marching Cubes 面绘制或 Volume Rendering 体绘制

        参数:
            dir_path: 包含 .dcm 文件的文件夹路径
            slice_thickness: 层厚（mm），None 则从 DICOM 元数据自动获取
            window_center: 窗位（HU），默认肺窗 -600
            window_width: 窗宽（HU），默认肺窗 1500

        返回:
            dict: {
                "volume_shape": (D, H, W),
                "voxel_spacing": (sz, sy, sx),
                "series_uid": str,
                "num_slices": int,
                "reconstruction_path": str  # 三维重建结果文件路径
            }
        """
        raise NotImplementedError(
            "DICOM 序列处理尚未实现。TODO: 接入基于 VTK/ITK 的三维重建管线"
        )

    def generate_3d_reconstruction(
        self,
        volume_data: Any = None,
        *,
        method: str = "volume_rendering",
        iso_value: float = -300.0,
        output_format: str = "glb",
    ) -> str:
        """
        根据三维体数据生成三维重建结果

        # TODO: 接入基于 VTK/ITK 的三维重建管线
        # 支持的重建方式：
        #   - "volume_rendering": VTK vtkSmartVolumeMapper 体渲染
        #   - "surface_rendering": Marching Cubes 面渲染
        #   - "mip": 最大密度投影 (Maximum Intensity Projection)

        参数:
            volume_data: 三维体数据（numpy ndarray 或 SimpleITK Image）
            method: 重建方式 ("volume_rendering" / "surface_rendering" / "mip")
            iso_value: 等值面阈值（HU），用于面渲染
            output_format: 输出格式 ("glb" / "obj" / "stl" / "png")

        返回:
            str: 重建结果文件路径
        """
        raise NotImplementedError(
            "三维重建尚未实现。TODO: 接入基于 VTK 的 Volume Rendering 管线"
        )
