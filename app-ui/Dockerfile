FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc g++ cmake python3-dev \
        libfreetype6-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first to leverage Docker layer cache
COPY app-ui/requirements.txt ./app-ui/requirements.txt
RUN pip install --no-cache-dir -r app-ui/requirements.txt

# Copy source code
COPY app-ui/ ./app-ui/
COPY app-ml/src/ ./app-ml/src/
COPY common/ ./common/
COPY config/ ./config/
COPY data/ ./data/

# models/ is a named Docker volume shared with app-ml-train

ENV PYTHONPATH="/app:/app/app-ml/src"

EXPOSE 8050

# --chdir sets CWD and adds app-ui/ to sys.path so gunicorn finds app:server
CMD ["gunicorn", "--chdir", "/app/app-ui", "app:server", \
     "--bind", "0.0.0.0:8050", "--workers", "1", "--timeout", "120"]
