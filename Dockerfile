FROM python:3.13-slim

WORKDIR /app

# 系統依賴（sentence-transformers 需要 libgomp）
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴（排除 pywin32，那是 Windows-only COM 套件）
COPY requirements.txt .
RUN grep -v "pywin32" requirements.txt > requirements-linux.txt \
    && pip install --no-cache-dir -r requirements-linux.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
