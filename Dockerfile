FROM python:3.11-slim

WORKDIR /app

# System deps: git for cloning repos
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + embedding model (models/ is excluded from git but included in Docker build)
COPY src/ ./src/
COPY models/ ./models/
COPY .env.example ./

# Force offline mode — model is baked into the image, no network needed for embeddings
ENV HF_HUB_OFFLINE=1
ENV EMBEDDING_MODEL_PATH=/app/models/bge-small-zh-v1.5

EXPOSE 8501

CMD ["streamlit", "run", "src/main.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]