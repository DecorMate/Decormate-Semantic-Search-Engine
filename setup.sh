#!/bin/bash
echo "🚀 EMERGENCY Memory Setup (Railway 512MB limit)..."

# Set extremely restrictive memory limits
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1

# Function to monitor memory
check_memory() {
    echo "💾 Memory: $(free -m | awk 'NR==2{printf "%.1fMB used of %.1fMB (%.1f%%)", $3,$2,$3*100/$2 }')"
}

check_memory

# Upgrade pip minimal
echo "📦 Upgrading pip (minimal)..."
pip install --no-cache-dir --upgrade pip --quiet
pip cache purge
check_memory

# Install only CORE dependencies first
echo "🔥 Installing PyTorch CPU (ultra minimal)..."
pip install --no-cache-dir --quiet torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip cache purge
rm -rf /tmp/* 2>/dev/null || true
check_memory

echo "Installing numpy (minimal)..."
pip install --no-cache-dir --quiet numpy==1.24.4
pip cache purge
check_memory

echo "Installing pillow..."
pip install --no-cache-dir --quiet pillow==10.0.1
pip cache purge
check_memory

echo "Installing flask..."
pip install --no-cache-dir --quiet flask==3.0.0
pip cache purge
check_memory

echo "Installing flask-cors..."
pip install --no-cache-dir --quiet flask-cors==4.0.0
pip cache purge
check_memory

echo "Installing python-dotenv..."
pip install --no-cache-dir --quiet python-dotenv==1.0.0
pip cache purge
check_memory

echo "Installing pinecone..."
pip install --no-cache-dir --quiet pinecone==5.0.1
pip cache purge
check_memory

echo "Installing huggingface-hub..."
pip install --no-cache-dir --quiet huggingface-hub==0.20.0
pip cache purge
check_memory

# Skip timm and open-clip for now - we'll use a minimal CLIP implementation
echo "⚠️ Skipping heavy packages (timm, open-clip) to save memory"

# Clone MobileCLIP with minimal history
echo "📥 Cloning MobileCLIP (minimal)..."
git clone --depth 1 --single-branch https://github.com/apple/ml-mobileclip.git
rm -rf ml-mobileclip/.git  # Remove git history to save space
check_memory

# Create directories
mkdir -p models temp

# Download model file directly without heavy dependencies
echo "🤖 Downloading model file only..."
python3 -c "
import os, gc
from huggingface_hub import hf_hub_download
print('Downloading model...')
try:
    hf_hub_download(
        repo_id='pcuenq/MobileCLIP-S1',
        filename='mobileclip_s1.pt',
        local_dir='models',
        local_dir_use_symlinks=False
    )
    print('✅ Model downloaded!')
    gc.collect()
except Exception as e:
    print(f'❌ Download failed: {e}')
    exit(1)
"

# Ultra-aggressive cleanup
echo "🧹 Ultra cleanup..."
pip cache purge
rm -rf ~/.cache 2>/dev/null || true
rm -rf /tmp/* 2>/dev/null || true
rm -rf /var/tmp/* 2>/dev/null || true
rm -rf /root/.cache 2>/dev/null || true

check_memory

# Verify model
if [ -f "models/mobileclip_s1.pt" ]; then
    echo "✅ Emergency setup complete!"
    ls -lh models/mobileclip_s1.pt
else
    echo "❌ Model not found!"
    exit 1
fi

echo "🚨 Emergency memory-optimized setup done!"