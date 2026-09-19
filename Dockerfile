# RAG 知识库问答 Agent - 生产镜像
FROM python:3.12-slim

WORKDIR /app

# 先装依赖（利用层缓存，代码改动不重装依赖）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# pgvector 驱动：镜像内的默认生产向量库后端（连不上 DB 仍自动降级 Memory）
RUN pip install --no-cache-dir "psycopg[binary]>=3.1" "psycopg_pool>=3.2"

COPY . .

# 非 root 运行（安全）
RUN useradd -m rag
RUN chown -R rag:rag /app
USER rag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "chat_service:app", "--host", "0.0.0.0", "--port", "8000"]
