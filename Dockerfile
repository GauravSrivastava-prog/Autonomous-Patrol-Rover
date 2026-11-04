# syntax=docker/dockerfile:1
FROM cyberbotics/webots:R2025a-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV WEBOTS_HOME=/usr/local/webots
ENV PYTHONPATH="${WEBOTS_HOME}/lib/controller/python"
ENV LD_LIBRARY_PATH="${WEBOTS_HOME}/lib/controller:${LD_LIBRARY_PATH}"
ENV DISPLAY=:99
ENV PORT=7860
ENV HOST=0.0.0.0
ENV YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

# Install system dependencies & Xvfb for headless rendering
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    git \
    wget \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Setup virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy all project source files
COPY . /app

# Ensure proper permissions for Hugging Face non-root user
RUN chmod -R 777 /app && \
    chmod -R 777 /tmp

# Curl command to downlooad yolov8n
RUN curl -fsSL -o yolov8n.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt

# Pre-download and cache model weights
RUN python3 -c "import clip; clip.load('ViT-B/32', device='cpu')" && \
    python3 -c "from ultralytics import YOLO; YOLO('/app/yolov8n.pt')"

EXPOSE 7860

# Launch application via app.py
CMD ["python3", "/app/app.py"]

