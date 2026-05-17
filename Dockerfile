FROM python:3.11-slim

# System libs required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights
COPY app.py detector.py distance_utils.py ./
COPY yolov8s-worldv2.pt yolov8n.pt ./
COPY templates/ ./templates/

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT} --timeout-keep-alive 75"]
