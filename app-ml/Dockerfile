FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake python3-dev \
        libfreetype6-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first to leverage Docker layer cache
COPY app-ml/requirements.txt ./app-ml/requirements.txt
RUN pip install --no-cache-dir -r app-ml/requirements.txt

# Copy source code
COPY app-ml/ ./app-ml/
COPY common/ ./common/
COPY config/ ./config/
COPY data/ ./data/

# models/ is a named Docker volume shared with app-ui

ENV PYTHONPATH="/app:/app/app-ml/src"
