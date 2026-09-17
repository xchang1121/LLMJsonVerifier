FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/data/huggingface
COPY pyproject.toml README.md ./
COPY src ./src
COPY requirements/gateway-constraints.txt ./requirements/gateway-constraints.txt
RUN python -m pip install -c requirements/gateway-constraints.txt .
COPY configs ./configs
ENTRYPOINT ["llmjv"]
CMD ["serve", "--config", "configs/qwen3.8-27b.toml"]
