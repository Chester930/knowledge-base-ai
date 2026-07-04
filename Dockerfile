FROM python:3.13-slim

WORKDIR /app

# 系統依賴
# libgomp1: sentence-transformers
# libgl1 + libglib2.0-0: PaddleOCR / OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        wget \
    && rm -rf /var/lib/apt/lists/*

# 先裝 CPU-only 版 torch：sentence-transformers/accelerate/datasets 都會拉 torch，
# 若不先裝好 CPU 版，pip 預設會裝 GPU 版並額外下載一整包 nvidia_* CUDA 函式庫
# （cublas/cufft/cusparse/cudnn 等，累計數 GB），這個 container 沒有 GPU，白白拖慢
# build 時間、增加 image 體積。torch 先滿足需求後，pip 裝其餘套件時不會再重裝/替換。
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 安裝 Python 依賴（排除 pywin32，那是 Windows-only COM 套件）
COPY requirements.txt .
RUN grep -v "pywin32" requirements.txt > requirements-linux.txt \
    && pip install --no-cache-dir -r requirements-linux.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
