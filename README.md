# 🏥 MedAgent - 智能医学CT诊断Agent系统

基于 AI Agent 的医学CT影像智能分析系统，集成疾病分类、病灶检测和诊断报告生成功能。

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   用户输入层                              │
│         CT图片 + 文字请求（Gradio前端）                   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 FastAPI 后端 API                          │
│    /api/analyze  /api/classify  /api/detect  /api/health │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│             LangGraph Agent 调度器                        │
│                                                          │
│  意图识别 → CT分类 → 病灶检测 → RAG检索 → 报告生成        │
│     (Gemini)  (Mock)    (Mock)  (ChromaDB) (Gemini)      │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼───────────┐
          │            │           │
┌─────────▼──┐  ┌──────▼──┐  ┌───▼─────────┐
│ CT分类Skill│  │病灶检测  │  │ RAG知识库   │
│ (ct_class- │  │Skill     │  │ ChromaDB +  │
│  ifier.py) │  │(lesion_  │  │ Google      │
│            │  │detector) │  │ Embedding   │
└────────────┘  └─────────┘  └─────────────┘
          │
┌─────────▼──────────────────────────────────┐
│           MCP Server                        │
│   ct_classify tool  |  detect_lesions tool  │
└────────────────────────────────────────────┘
```

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM | Google Gemini 2.0 Flash |
| Agent框架 | LangChain + LangGraph |
| 向量数据库 | ChromaDB |
| Embedding | Google Embedding API |
| 前端 | Gradio 5.x |
| 后端 | FastAPI |
| CT分析 | Mock（预留真实模型接口） |
| MCP协议 | mcp Python SDK |

## 目录结构

```
MedAgent/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI 入口 & API路由
│   ├── config.py                # 全局配置（pydantic-settings）
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # LangGraph Agent调度器
│   │   └── prompts.py           # 提示词模板
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── ct_classifier.py     # CT疾病分类（Mock + LangChain Tool）
│   │   └── lesion_detector.py   # 病灶检测（Mock + LangChain Tool）
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py    # ChromaDB知识库构建与检索
│   │   ├── embeddings.py        # Google Embedding封装
│   │   └── docs/                # 示例医学知识文档
│   │       ├── lung_cancer.md
│   │       ├── pneumonia.md
│   │       └── pulmonary_nodule.md
│   ├── report/
│   │   ├── __init__.py
│   │   └── generator.py         # Gemini诊断报告生成器
│   └── mcp/
│       ├── __init__.py
│       └── server.py            # MCP Server
├── frontend/
│   └── app.py                   # Gradio前端界面
├── tests/
│   ├── __init__.py
│   ├── test_classifier.py       # CT分类测试
│   ├── test_detector.py         # 病灶检测测试
│   └── test_rag.py              # RAG知识库测试
├── data/
│   └── chroma_db/               # ChromaDB存储（自动创建）
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 安装与运行

### 本地安装

#### 1. 克隆仓库

```bash
git clone https://github.com/ILoveYLH/MedAgent.git
cd MedAgent
```

#### 2. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 Google API Key
```

#### 5. 构建医学知识库

```bash
python -c "from app.rag.knowledge_base import build_knowledge_base; build_knowledge_base()"
```

#### 6. 启动 FastAPI 后端

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 7. 启动 Gradio 前端

```bash
python frontend/app.py
```

访问地址：
- Gradio界面：http://localhost:7860
- FastAPI文档：http://localhost:8000/docs

### Docker 部署

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API Key

# 2. 构建并启动
docker-compose up -d

# 3. 查看日志
docker-compose logs -f
```

## API 文档

### `GET /api/health` - 健康检查

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model": "gemini-2.0-flash"
}
```

### `POST /api/classify` - CT疾病分类

请求：`multipart/form-data`，包含 `file`（CT图片）

响应：
```json
{
  "disease_type": "肺结节",
  "confidence": 0.856,
  "all_probabilities": {"肺结节": 0.856, "正常": 0.09, ...},
  "model_version": "mock-v1.0",
  "is_mock": true
}
```

### `POST /api/detect` - 病灶检测

请求：`multipart/form-data`，包含 `file`（CT图片）

响应：
```json
{
  "lesions": [
    {
      "lesion_id": 1,
      "type": "实性结节",
      "bounding_box": {"x1": 120, "y1": 85, "x2": 178, "y2": 143, "width": 58, "height": 58},
      "size_mm": 12.3,
      "confidence": 0.82,
      "location": "右侧上叶",
      "mask_path": null
    }
  ],
  "total_count": 1,
  "model_version": "mock-v1.0",
  "is_mock": true
}
```

### `POST /api/analyze` - 完整CT分析

请求：`multipart/form-data`，包含 `file` 和 `request_text`

响应：包含分类、检测、知识检索和完整诊断报告。

### `GET /api/knowledge/search` - 知识库检索

参数：`q`（查询文本），`top_k`（返回数量，默认3）

## 如何替换 Mock 模型为真实模型

### CT分类模型替换

编辑 `app/skills/ct_classifier.py`：

```python
# 1. 安装依赖
# pip install torch torchvision timm

# 2. 在 classify_ct() 函数中替换以下注释块：
# from app.skills._real_models import load_ct_classifier
# model = load_ct_classifier()
# result = model.predict(image_path)
# return result

# 3. 实现真实分类逻辑（示例：使用 EfficientNet）
import torch
import timm
from torchvision import transforms

def load_ct_classifier():
    model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=6)
    model.load_state_dict(torch.load('models/ct_classifier.pth'))
    model.eval()
    return model
```

### 病灶检测模型替换

编辑 `app/skills/lesion_detector.py`：

```python
# 使用 YOLOv8 进行病灶检测
from ultralytics import YOLO

def detect_lesions_real(image_path: str) -> dict:
    model = YOLO('models/ct_lesion_detector.pt')
    results = model.predict(image_path, conf=0.5)
    # 解析结果...
```

## 如何扩展知识库

1. 将新的医学知识文档（`.md` 或 `.txt` 格式）放入 `app/rag/docs/` 目录
2. 调用重建API：`POST /api/knowledge/build`
3. 或直接运行：`python -c "from app.rag.knowledge_base import build_knowledge_base; build_knowledge_base()"`

## 运行测试

```bash
pytest tests/ -v
```

## MCP Server 使用

```bash
# 启动 MCP Server（stdio模式）
python -m app.mcp.server
```

支持通过 MCP 协议调用的工具：
- `ct_classify`: CT疾病分类
- `detect_lesions`: 病灶检测

## 后续开发计划

- [ ] 接入真实CT分类模型（ResNet50/EfficientNet）
- [ ] 接入真实病灶检测模型（YOLOv8/U-Net）
- [ ] 支持DICOM格式CT文件
- [ ] 添加图像标注可视化（在图片上标出病灶位置）
- [ ] 扩展知识库（加入更多疾病和诊断指南）
- [ ] 添加用户认证和报告存储
- [ ] 支持多模态输入（CT + 临床信息）
- [ ] 部署到云平台（GCP/AWS）

## 免责声明

⚠️ **重要提示**：本系统仅用于研究和学习目的，所有分析结果**仅供参考**，不能替代专业医师的临床诊断意见。在实际医疗场景中使用前，必须经过严格的临床验证和监管审批。
