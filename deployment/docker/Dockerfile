# Dockerfile for YOLOv11-ConvNeXt deployment

FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and code
COPY models/ ./models/
COPY inference/ ./inference/
COPY deployment/ ./deployment/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run server
CMD ["python", "-m", "deployment.server", "--model", "weights/best.pt", "--port", "8000"]
