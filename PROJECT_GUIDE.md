# 🏥 MedAgent 项目实战指南

## 1. 项目概览 (Project Overview)

**核心目标**：
用一句话概括：**MedAgent 是一个基于 AI Agent 架构的智能医学 CT 诊断辅助系统，它通过串联疾病分类、病灶检测和基于外部知识库的增强生成（RAG），最终为医生或用户生成参考性诊断报告。**

**技术栈**：
*   **大语言模型 (LLM)**: Google Gemini 2.0 Flash (充当系统的大脑，负责意图识别和报告生成)
*   **Agent 编排框架**: LangChain + LangGraph (用于定义和控制工作流节点)
*   **向量数据库**: ChromaDB (用于本地存储医学知识库，支持 RAG 检索)
*   **文本向量化**: Google Embedding API
*   **后端框架**: FastAPI (提供高性能的 RESTful API)
*   **前端界面**: Gradio 5.x (用于快速构建可交互的 Web 演示界面)
*   **跨服务通信**: MCP (Model Context Protocol) SDK (为模型提供标准化工具调用接口)

**运行机制简述**：
1. **用户输入**：用户通过 Gradio 前端上传一张 CT 图片，并用文字输入他们的请求。
2. **API 接收**：FastAPI 后端接收请求，并将任务转发给 LangGraph 构建的 Agent 调度器。
3. **Agent 编排与执行**：
   * **意图识别**：LLM 思考用户的需求。
   * **图像分析**：调用 `CT分类` 和 `病灶检测` 两个工具（目前为预留的 Mock 接口）提取图片里的关键医学指标。
   * **知识检索**：根据识别出的疾病或病灶类型，去 ChromaDB 知识库中查找相关的医学指南或知识（RAG）。
   * **报告生成**：LLM 将图像特征、检测结果和检索到的专业知识融合在一起，生成最终的医学诊断报告。
4. **结果返回**：前端展示最终报告及各步骤的详细数据。

---

## 2. 项目架构剖析 (Architecture Deep Dive)

**目录结构树**：

```text
MedAgent/
├── app/                     # 核心后端代码目录
│   ├── main.py              # 路由器：FastAPI 入口，定义对外暴露的接口。
│   ├── config.py            # 大管家：管理所有环境变量和系统全局配置。
│   ├── agent/               # 大脑中枢：Agent 编排层
│   │   ├── orchestrator.py  # LangGraph 工作流定义，控制各个节点的流转。
│   │   └── prompts.py       # 提示词库，教大模型如何扮演医生。
│   ├── skills/              # 感知器官：具体干活的医学视觉工具
│   │   ├── ct_classifier.py # 负责判断得什么病（目前是Mock数据，预留真实模型入口）。
│   │   └── lesion_detector.py # 负责圈出病在哪儿（也是Mock，预留真实模型入口）。
│   ├── rag/                 # 记忆外挂：外部知识库系统
│   │   ├── knowledge_base.py# ChromaDB的增删改查逻辑。
│   │   └── docs/            # 本地 Markdown 医学文献，作为知识源。
│   ├── report/              # 嘴巴：输出层
│   │   └── generator.py     # 将所有数据组装发给 Gemini 生成最终诊断报告。
│   └── mcp/                 # 通讯员：模型上下文协议服务
│       └── server.py        # 暴露工具给符合 MCP 标准的客户端使用。
├── frontend/                # 面子：用户界面
│   └── app.py               # 简单的 Gradio 交互界面。
└── data/                    # 仓库：存放持久化数据（如ChromaDB向量文件）。
```

**模块依赖关系**：
你可以把系统想象成一家医院的看病流程：
`前端 (Gradio)` -> `导诊台 (FastAPI)` -> `主治医生 (LangGraph Agent)`
主治医生在看病时会调用多个科室：
*   去 `影像科` 拿结果 -> `skills模块`
*   翻阅 `医学词典` 查资料 -> `rag模块`
*   最后自己写 `病历本` -> `report模块`

**设计模式识别**：
1.  **Agent 工作流模式 (Workflow/Pipeline)**：通过 LangGraph 实现，将复杂任务拆解为有向无环图 (DAG) 的节点。它把大模型的不可控变为了步骤可控。
2.  **工具/插件模式 (Plugin/Tool Pattern)**：把 `ct_classifier` 和 `lesion_detector` 封装成大模型可以调用的工具（Tools）。这保证了核心逻辑的解耦。
3.  **Facade (外观模式) 雏形**：`app/main.py` 作为整个系统对外的唯一入口，隐藏了后端 Agent 和各个组件交互的复杂性。

---

## 3. 核心业务流程与时序 (Core Workflows)

我们来看看系统最核心的**完整 CT 分析流程** (`/api/analyze`)：

1. **接收请求**：前端传入 `CT 图片` 和 `描述文本` 给 FastAPI。
2. **初始化 Agent 状态**：LangGraph 接收任务，创建一个包含初始状态（图片路径、用户问题）的上下文对象。
3. **节点1: 视觉特征提取 (Vision Analysis Node)**
   * Agent 同时或顺序调用 `ct_classifier.py` 和 `lesion_detector.py`。
   * 获取结构化数据（例：疾病=肺结节，置信度=0.85，位置=右侧上叶）。
4. **节点2: 知识检索 (RAG Node)**
   * 拿着提取到的关键字（如"肺结节"），去 `rag/knowledge_base.py` 检索。
   * 把查到的专业 Markdown 文本追加到 Agent 的状态对象中。
5. **节点3: 报告生成 (Report Generation Node)**
   * `report/generator.py` 将用户的原始问题、视觉提取的指标、RAG 检索出的医学背景知识，组装成超级 Prompt。
   * 请求 Gemini 2.0 Flash。
6. **返回响应**：把 Gemini 输出的完整诊断报告和中间步骤数据打包成 JSON，返回给前端展示。

---

## 4. 关键类与核心方法详解 (Key Classes & Methods)

### A. 调度中枢模块 (`app/agent/orchestrator.py`)
*   **职责**：定义并执行 LangGraph 的图结构（StateGraph）。

### B. 知识检索模块 (`app/rag/knowledge_base.py`)
*   **职责**：把文本变成向量存起来，并提供搜索功能。

### C. 视觉技能模块 (`app/skills/ct_classifier.py`)
*   **职责**：识别 CT 图像中的疾病（目前为 Mock 占位，等待接入真实的 PyTorch 模型）。

---

## 5. 数据库或存储设计 (Storage & Models)

*   **ChromaDB 存储**：保存在 `data/chroma_db/` 目录下。这是一个 SQLite 风格的本地持久化向量库。
*   **核心数据结构**：系统在内存中通过 Pydantic 或 TypedDict（Python 类型提示）来传递状态。典型的内部模型包含 `CTClassificationResult`, `Lesion`, `AgentState` 等。

---

## 6. 给新手的阅读建议与踩坑指南 (Tips & Pitfalls)

**🎯 阅读顺序建议：**
1. **先看皮囊**：看 `frontend/app.py` 和 `app/main.py`。
2. **再看工具**：看 `app/skills/` 里的两个 mock 文件。
3. **核心攻克**：深入阅读 `app/agent/orchestrator.py` 和 `app/report/generator.py`。
4. **最后看存储**：看 `app/rag/knowledge_base.py`。

**🚧 踩坑指南与难点解析：**

### 重点攻克：LangGraph 的 State 传递（最容易懵圈的地方）

在标准的 Python 程序中，我们习惯了这样的函数调用：`result = step2(step1(input))`。数据是通过 `return` 显式地传递给下一个函数的。但在 LangGraph 中，数据传递靠的是一个**全局的共享状态（State）**。你可以把它想象成医院里病人的**"实体病历本"**。

#### 1. State 是什么？
通常是在代码里定义的一个 `TypedDict` 或 `Pydantic` 类，例如 `AgentState`。它规定了病历本上有哪些格子（字段），比如 `user_input` (病人自述), `image_features` (影像科报告), `retrieved_docs` (文献参考), `final_report` (最终诊断)。

#### 2. 节点（Node）如何工作？
每个节点（Node，比如图像识别节点、文档检索节点）就像是医院里的一个科室（或医生）。当流程流转到某个节点时，LangGraph 会把当前的**整本病历本（完整的 State 字典）**交给这个节点。
节点读取里面的信息，完成自己的工作，然后**仅仅返回一个字典**（包含它想更新的字段）。

#### 3. 状态更新（Reducer 机制）
**最容易懵圈的地方来了：节点返回的字典并不会直接覆盖整个病历本！** LangGraph 会按照事先定义好的规则（Reducer），把节点返回的数据**合并（Merge）**到现有的病历本上。
*   **覆盖更新（默认）**：如果节点返回 `{"image_features": "发现结节"}`，那么病历本上的 `image_features` 格子就会被擦除，重新填上"发现结节"。
*   **追加更新（Append）**：如果某个字段在定义时指定了 reducer（比如对于消息列表指定了 `operator.add`），那么当节点返回 `{"messages": ["新消息"]}` 时，"新消息" 会被**追加**到原来的消息列表后面，而不会清空以前的聊天记录。

#### 4. 为什么新手容易懵？
因为你在代码里看不到 `A -> B` 的直接传参！你看到的是 `orchestrator.py` 里定义了图的节点和连线（Edges），然后每个节点函数内部都在默默地读取 `state["xxx"]`，执行完逻辑后返回 `{"yyy": zzz}`。

**💡 破解方法**：在阅读 LangGraph 代码时，**一定要先找到并看懂 `AgentState` 的结构定义！** 可以在脑海中（或在纸上）画出这个字典。看着它在每个节点执行后，哪个字段被覆盖了，哪个字段被追加了，整个数据流向就会瞬间清晰！
