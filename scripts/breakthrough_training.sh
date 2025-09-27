#!/bin/bash
# راه‌حل فوری برای loss plateau

echo "🚀 Starting Loss Plateau Breakthrough Training..."
echo "=================================="

# متوقف کردن training فعلی (اگر داره اجرا میشه)
echo "⏹️  Stopping current training (if running)..."
pkill -f "python3 train_main.py" 2>/dev/null || true
sleep 3

# شروع training با تنظیمات plateau_breaker
echo "🔧 Starting with PLATEAU_BREAKER optimization..."
python3 train_main.py \
    --optimization-level plateau_breaker \
    --batch-size 24 \
    --epochs 50 \
    --lr 1.5e-5 \
    --enable-gpu-stabilizer \
    --train-data ../dataset/dataset_train \
    --val-data ../dataset/dataset_eval

echo "✅ Training started with loss breakthrough configuration"
echo "Expected: Loss should break below 2.5 within 10-20 epochs"