FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY ml-mobileclip/ ./ml-mobileclip/

# Create directories
RUN mkdir -p models temp

# Download model during build
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='pcuenq/MobileCLIP-S1', filename='mobileclip_s1.pt', local_dir='models', local_dir_use_symlinks=False)"

# Set environment variables
ENV PYTHONPATH="/app/src:/app/ml-mobileclip"
ENV MODEL_PATH="/app/models/mobileclip_s1.pt"
ENV PORT=5000

# Clean up
RUN pip cache purge && rm -rf ~/.cache

# Expose port
EXPOSE 5000

# Start command
CMD ["python", "src/routes.py"]