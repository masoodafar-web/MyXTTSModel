#!/bin/bash
# Optimized Environment for MyXTTS Training

# CUDA Settings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=1

# Python Settings  
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONUNBUFFERED=1

# Memory Settings
export TF_ENABLE_ONEDNN_OPTS=1
export TF_ENABLE_GPU_GARBAGE_COLLECTION=1

# Training with optimized settings
python3 train_main.py \
    --model-size normal \
    --optimization-level enhanced \
    --batch-size 24 \
    --epochs 100 \
    "$@"
