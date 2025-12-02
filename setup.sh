#!/bin/bash
echo "🚀 Setting up Semantic Search Engine (Optimized)..."

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --no-cache-dir --upgrade pip

# Install minimal PyTorch CPU version
echo "🔥 Installing PyTorch (CPU, minimal)..."
pip install --no-cache-dir torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install only essential dependencies (no cache)
echo "📚 Installing minimal dependencies..."
pip install --no-cache-dir \
    open-clip-torch==2.20.0 \
    timm==0.9.5 \
    pinecone==5.0.1 \
    flask==3.0.0 \
    flask-cors==4.0.0 \
    python-dotenv==1.0.0 \
    pillow==10.0.1 \
    huggingface-hub==0.20.0 \
    numpy==1.24.4

# Create minimal directories
echo "📁 Creating directories..."
mkdir -p models temp

# Download only the model file we need
echo "🤖 Downloading MobileCLIP model (minimal)..."
python3 -c "
import os
from huggingface_hub import hf_hub_download
print('Downloading only mobileclip_s1.pt...')
try:
    hf_hub_download(
        repo_id='pcuenq/MobileCLIP-S1',
        filename='mobileclip_s1.pt',
        local_dir='models',
        local_dir_use_symlinks=False
    )
    print('✅ Model downloaded!')
    
    # Clean up any cache
    import shutil
    cache_dir = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print('🧹 Cleaned HF cache')
        
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"

# Clean up pip cache
echo "🧹 Cleaning up caches..."
pip cache purge
rm -rf ~/.cache/pip
rm -rf /tmp/*

# Verify model exists
if [ -f "models/mobileclip_s1.pt" ]; then
    echo "✅ Setup completed! Model size:"
    ls -lh models/mobileclip_s1.pt
else
    echo "❌ Model file not found!"
    exit 1
fi

echo "🎉 Optimized setup completed!"