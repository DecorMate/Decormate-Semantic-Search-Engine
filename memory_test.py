#!/usr/bin/env python3
"""
Memory test script to verify optimizations work
"""
import sys
import os
import gc
import psutil

sys.path.append('src')

def get_memory():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except:
        return 0

def main():
    print("🧪 Memory Test for Railway Deployment")
    print(f"💾 Starting memory: {get_memory():.1f} MB")
    
    # Test 1: Import basic modules
    print("\n1️⃣ Testing basic imports...")
    from indexer import SimpleIndexer
    print(f"💾 After imports: {get_memory():.1f} MB")
    
    # Test 2: Create indexer (Pinecone only)
    print("\n2️⃣ Testing Pinecone connection...")
    indexer = SimpleIndexer()
    print(f"💾 After Pinecone: {get_memory():.1f} MB")
    
    # Test 3: Load model (this is the big test)
    print("\n3️⃣ Testing model loading...")
    try:
        model, preprocess, tokenizer = indexer._get_model()
        print(f"💾 After model load: {get_memory():.1f} MB")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False
    
    # Test 4: Simple search
    print("\n4️⃣ Testing search functionality...")
    try:
        results = indexer.search("test query", limit=1)
        print(f"💾 After search: {get_memory():.1f} MB")
        print(f"✅ Search returned {len(results)} results")
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False
    
    # Memory check
    final_memory = get_memory()
    print(f"\n📊 Final memory usage: {final_memory:.1f} MB")
    
    if final_memory < 800:  # Railway limit is usually around 1GB
        print("✅ Memory usage looks good for Railway!")
        return True
    else:
        print("⚠️ Memory usage might be too high for Railway")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)