#!/bin/bash
echo "🚀 Setting up Semantic Search Engine (Ultra Memory Optimized)..."

# Set memory limits
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Upgrade pip with minimal memory
echo "📦 Upgrading pip..."
pip install --no-cache-dir --upgrade pip
pip cache purge

# Install PyTorch CPU with minimal memory footprint
echo "🔥 Installing PyTorch (CPU only, minimal)..."
pip install --no-cache-dir --no-deps torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip cache purge
rm -rf /tmp/* 2>/dev/null || true

echo "🔥 Installing torchvision..."
pip install --no-cache-dir --no-deps torchvision==0.16.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip cache purge
rm -rf /tmp/* 2>/dev/null || true

# Install dependencies one by one with cleanup
echo "📚 Installing dependencies (one by one)..."

echo "Installing numpy..."
pip install --no-cache-dir numpy==1.24.4
pip cache purge

echo "Installing pillow..."
pip install --no-cache-dir pillow==10.0.1
pip cache purge

echo "Installing flask..."
pip install --no-cache-dir flask==3.0.0
pip cache purge

echo "Installing flask-cors..."
pip install --no-cache-dir flask-cors==4.0.0
pip cache purge

echo "Installing python-dotenv..."
pip install --no-cache-dir python-dotenv==1.0.0
pip cache purge

echo "Installing pinecone..."
pip install --no-cache-dir pinecone==5.0.1
pip cache purge

echo "Installing huggingface-hub..."
pip install --no-cache-dir huggingface-hub==0.20.0
pip cache purge

echo "Installing psutil (memory monitoring)..."
pip install --no-cache-dir psutil==5.9.5
pip cache purge

# Install timm without dependencies to avoid conflicts
echo "Installing timm (minimal)..."
pip install --no-cache-dir --no-deps timm==0.9.5
pip cache purge

# Install open-clip last (biggest package)
echo "Installing open-clip-torch (minimal)..."
pip install --no-cache-dir --no-deps open-clip-torch==2.20.0
pip cache purge

# Aggressive cleanup
echo "🧹 Aggressive memory cleanup..."
pip cache purge
rm -rf ~/.cache/pip 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true

# Clone MobileCLIP if not present (lightweight)
if [ ! -d "ml-mobileclip" ]; then
    echo "📥 Cloning MobileCLIP repository..."
    git clone --depth 1 https://github.com/apple/ml-mobileclip.git
fi

# Create minimal directories
echo "📁 Creating directories..."
mkdir -p models temp

# Use download_model.py with retry logic instead of inline Python
echo "🤖 Downloading MobileCLIP model..."
if [ -f "download_model.py" ]; then
    python3 download_model.py
else
    # Fallback inline download with minimal memory
    python3 -c "
import os, gc
from huggingface_hub import hf_hub_download
print('Downloading mobileclip_s1.pt...')
try:
    hf_hub_download(
        repo_id='pcuenq/MobileCLIP-S1',
        filename='mobileclip_s1.pt',
        local_dir='models',
        local_dir_use_symlinks=False
    )
    print('✅ Model downloaded!')
    gc.collect()  # Force garbage collection
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"
fi

# Final aggressive cleanup
echo "🧹 Final cleanup..."
pip cache purge
rm -rf ~/.cache 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true

# Verify model exists
if [ -f "models/mobileclip_s1.pt" ]; then
    echo "✅ Setup completed! Model size:"
    ls -lh models/mobileclip_s1.pt
else
    echo "❌ Model file not found!"
    exit 1
fi

echo "🎉 Ultra-optimized setup completed!"
echo "💾 Memory usage minimized for Railway deployment"