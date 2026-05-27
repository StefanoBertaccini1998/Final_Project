FROM python:3.11-slim

WORKDIR /app

# System libs needed by OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps — PyTorch CPU wheel from pytorch.org
COPY requirements-app.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-app.txt

# Copy project (source, checkpoints, examples)
COPY src/ ./src/
COPY outputs/checkpoints/ ./outputs/checkpoints/
COPY examples/ ./examples/
COPY app.py .

# Railway injects $PORT; Gradio reads it at runtime
EXPOSE 7860
CMD ["python", "app.py"]
