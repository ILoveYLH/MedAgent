# MedAgent - 智能医学CT诊断系统
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建数据目录
RUN mkdir -p /app/data/chroma_db

# 暴露端口
EXPOSE 8000 7860

# 设置环境变量
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# 启动脚本（同时启动 FastAPI 和 Gradio）
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 & python frontend/app.py"]
