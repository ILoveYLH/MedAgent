"""
FastAPI 主入口

提供 RESTful API 接口，集成 Agent 调度器。

API 端点：
    POST /api/analyze    - 完整CT分析（分类 + 检测 + 报告）
    POST /api/classify   - 单独CT分类
    POST /api/detect     - 单独病灶检测
    GET  /api/knowledge/search - 知识库检索
    GET  /api/health     - 健康检查
"""

import logging
import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="MedAgent API",
    description="智能医学CT诊断Agent系统 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# 响应模型
# ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


class ClassifyResponse(BaseModel):
    disease_type: str
    confidence: float
    all_probabilities: dict
    model_version: str
    is_mock: bool


class DetectResponse(BaseModel):
    lesions: list
    total_count: int
    model_version: str
    is_mock: bool


class AnalyzeResponse(BaseModel):
    classification: dict
    detection: dict
    knowledge: list
    report: str
    intents: list


class KnowledgeSearchResponse(BaseModel):
    query: str
    results: list[str]
    count: int


# ─────────────────────────────────────────
# API 端点
# ─────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["系统"])
async def health_check():
    """健康检查接口"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model=settings.model_name,
    )


@app.post("/api/classify", response_model=ClassifyResponse, tags=["CT分析"])
async def classify_ct_image(
    file: UploadFile = File(..., description="CT图片文件"),
):
    """
    CT疾病分类接口

    上传CT图片，返回疾病分类结果（疾病类型、置信度、各类别概率）。
    """
    from app.skills.ct_classifier import classify_ct

    # 保存上传文件到临时路径
    suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = classify_ct(tmp_path)
        return ClassifyResponse(
            disease_type=result["disease_type"],
            confidence=result["confidence"],
            all_probabilities=result["all_probabilities"],
            model_version=result["model_version"],
            is_mock=result.get("is_mock", False),
        )
    finally:
        os.unlink(tmp_path)


@app.post("/api/detect", response_model=DetectResponse, tags=["CT分析"])
async def detect_lesions_in_image(
    file: UploadFile = File(..., description="CT图片文件"),
):
    """
    病灶检测接口

    上传CT图片，返回病灶检测结果（病灶位置、大小、类型）。
    """
    from app.skills.lesion_detector import detect_lesions

    suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = detect_lesions(tmp_path)
        return DetectResponse(
            lesions=result["lesions"],
            total_count=result["total_count"],
            model_version=result["model_version"],
            is_mock=result.get("is_mock", False),
        )
    finally:
        os.unlink(tmp_path)


@app.post("/api/analyze", response_model=AnalyzeResponse, tags=["CT分析"])
async def analyze_ct_image(
    file: UploadFile = File(..., description="CT图片文件"),
    request_text: str = Form(default="请分析这张CT图片", description="用户诊断请求文本"),
):
    """
    完整CT分析接口

    上传CT图片并提供诊断请求，系统将依次执行：
    1. 疾病分类
    2. 病灶检测
    3. 医学知识检索
    4. 诊断报告生成

    返回完整的分析结果和结构化诊断报告。
    """
    from app.agent.orchestrator import run_agent

    suffix = os.path.splitext(file.filename or "image.png")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_agent(
            image_path=tmp_path,
            user_request=request_text,
        )
        return AnalyzeResponse(
            classification=result["classification"],
            detection=result["detection"],
            knowledge=result["knowledge"],
            report=result["report"],
            intents=result["intents"],
        )
    except Exception as exc:
        logger.error("CT分析失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(exc)}")
    finally:
        os.unlink(tmp_path)


@app.get("/api/knowledge/search", response_model=KnowledgeSearchResponse, tags=["知识库"])
async def search_knowledge(
    q: str = Query(..., description="搜索查询文本"),
    top_k: int = Query(default=3, ge=1, le=10, description="返回结果数量"),
):
    """
    医学知识库检索接口

    根据查询文本检索相关的医学知识片段。
    """
    from app.rag.knowledge_base import query_knowledge

    try:
        results = query_knowledge(q, top_k=top_k)
        return KnowledgeSearchResponse(
            query=q,
            results=results,
            count=len(results),
        )
    except Exception as exc:
        logger.error("知识库检索失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"检索失败: {str(exc)}")


@app.post("/api/knowledge/build", tags=["知识库"])
async def build_knowledge():
    """
    构建/重建医学知识库

    从 docs/ 目录加载文档，分块向量化后存入 ChromaDB。
    """
    from app.rag.knowledge_base import build_knowledge_base

    try:
        count = build_knowledge_base()
        return {"message": f"知识库构建成功，共存入 {count} 个文本块", "count": count}
    except Exception as exc:
        logger.error("知识库构建失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"构建失败: {str(exc)}")


# ─────────────────────────────────────────
# 启动入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
        log_level="info",
    )
