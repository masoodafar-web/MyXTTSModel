#!/bin/bash

# Quick Training Restart Script
# اسکریپت برای restart سریع training

echo "🔄 Stopping current training process..."

# Kill current training process
pkill -f "python3 train_main.py" || echo "No training process found"

# Wait a moment
sleep 2

# Check GPU memory
echo "📊 GPU Status After Stop:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader,nounits

# Clear GPU memory if possible
echo "🧹 Attempting GPU memory cleanup..."
python3 -c "
import tensorflow as tf
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.reset_memory_growth(gpu)
        print('✅ GPU memory reset attempted')
except Exception as e:
    print(f'Memory reset note: {e}')
"

echo ""
echo "🚀 Starting training with debug configuration..."
echo "   - Single GPU mode"
echo "   - Smaller model size"
echo "   - Reduced batch size"
echo ""

# Start training with single GPU and debug config
MYXTTS_SIMPLE_LOSS=1 \
CUDA_VISIBLE_DEVICES=0 \
python3 train_main.py \
    --config config_debug_fast.yaml \
    --batch-size 8 \
    --grad-accum 1 \
    --epochs 1 \
    --optimization-level basic \
    --model-size small

echo "Training started with PID: $!"