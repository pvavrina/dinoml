FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build helpers
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU-only version and other dependencies safely
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    fastapi==0.110.0 \
    uvicorn==0.28.0 \
    pillow==10.2.0 \
    python-multipart==0.0.9 \
    opentelemetry-api==1.45.2 \
    opentelemetry-sdk==1.45.2 \
    opentelemetry-exporter-otlp==1.45.2 \
    opentelemetry-instrumentation-fastapi==0.66b2

COPY main.py /app/main.py

EXPOSE 8080

CMD ["python", "/app/main.py"]
