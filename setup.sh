#!/bin/bash
echo "🚀 Setting up Semantic Search Engine..."

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install CPU version of PyTorch for Railway (more reliable)
echo "🔥 Installing PyTorch (CPU)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models temp

# Download MobileCLIP model
echo "🤖 Downloading MobileCLIP model..."
python3 -c "
import os
from huggingface_hub import hf_hub_download
print('Downloading MobileCLIP-S1...')
try:
    hf_hub_download(
        repo_id='pcuenq/MobileCLIP-S1',
        filename='mobileclip_s1.pt',
        local_dir='models',
        local_dir_use_symlinks=False
    )
    print('✅ Model downloaded successfully!')
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"

# Verify model exists
if [ -f "models/mobileclip_s1.pt" ]; then
    echo "✅ Model verification passed"
    ls -lh models/
else
    echo "❌ Model file not found!"
    exit 1
fi

# Set up environment
echo "🔧 Setting up environment..."
export PYTHONPATH="/app/src:/app/ml-mobileclip"
export MODEL_PATH="/app/models/mobileclip_s1.pt"

echo "🎉 Setup completed successfully!"
echo "Model path: $MODEL_PATH"
echo "Python path: $PYTHONPATH"