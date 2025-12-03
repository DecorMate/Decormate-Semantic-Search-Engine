FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

    # Install PyTorch CPU version first with specific index
    RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
    
    # Install remaining dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy source code
    COPY src/ ./src/
    
    # Clone MobileCLIP repository
    RUN git clone https://github.com/apple/ml-mobileclip.git

# Create directories
RUN mkdir -p models temp

    # Copy download script and run it
    COPY download_model.py .
    RUN python download_model.py

# Set environment variables
ENV PYTHONPATH="/app/src:/app/ml-mobileclip"
ENV MODEL_PATH="/app/models/mobileclip_s1.pt"
ENV PORT=5000

# Clean up
RUN pip cache purge && rm -rf ~/.cache && rm -rf /tmp/*

# Expose port
EXPOSE 5000

# Start command
CMD ["python", "src/routes.py"]