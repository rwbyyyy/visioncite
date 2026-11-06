# 说明：本 Dockerfile 未在本机构建验证（开发机没有 Docker）。
# 它的作用是给出可复现的部署形态；CI 只跑测试，不构建镜像。

FROM python:3.12-slim

# rapidocr / onnxruntime 需要 libgomp；字体用于证据裁图里的中文渲染（可选）
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY src/visioncite/web ./src/visioncite/web

# 只装运行必需 + OCR（视觉模型按需挂在宿主机，见 README）
RUN pip install --no-cache-dir ".[ocr]"

ENV VISIONCITE_CACHE=/data
VOLUME ["/data"]
EXPOSE 8080

CMD ["uvicorn", "visioncite.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
