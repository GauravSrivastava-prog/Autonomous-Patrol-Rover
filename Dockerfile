# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOST=0.0.0.0

WORKDIR /app

# Install system dependencies for headless OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project source files
COPY . /app

# Non-root user setup for Hugging Face Docker SDK security
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

EXPOSE 7860

# Launch high-performance mission control
CMD ["python", "app.py"]
