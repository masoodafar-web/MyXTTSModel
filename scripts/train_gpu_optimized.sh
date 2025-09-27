#!/bin/bash
# GPU Utilization Optimized Training Launch Script
# ===============================================

echo "🚀 Starting MyXTTS training with GPU utilization optimization..."

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected. This optimization requires CUDA."
    exit 1
fi

# Display GPU info
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Set optimal environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install required packages if needed
echo "📦 Checking required packages..."
pip install pynvml psutil --quiet

# Run training with GPU optimization
echo "🎯 Starting optimized training..."
python3 train_main.py \
    --config config_gpu_utilization_optimized.yaml \
    --model-size tiny \
    --train-data ./data/train.csv \
    --val-data ./data/val.csv \
    --optimization-level enhanced \
    --apply-fast-convergence \
    --enable-evaluation \
    --checkpoint-dir ./checkpoints_gpu_optimized \
    --num-workers auto \
    --batch-size auto \
    --verbose

echo "✅ Training completed!"
