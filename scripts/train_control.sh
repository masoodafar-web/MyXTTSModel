#!/bin/bash

# MyXTTS Training Control Script
# اسکریپت کنترل training برای MyXTTS

echo "🎯 MyXTTS Training Control Script"
echo "================================="

# Default values
BATCH_SIZE=4
EPOCHS=1
MODEL_SIZE="tiny"
OPTIMIZATION_LEVEL="basic"
CUDA_DEVICES="0"

# Function to show usage
show_usage() {
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --batch-size SIZE          Batch size (default: 4)"
    echo "  --epochs NUM               Number of epochs (default: 1)"
    echo "  --model-size SIZE          Model size: tiny, small, normal, big (default: tiny)"
    echo "  --optimization LEVEL       Optimization level: basic, enhanced, experimental (default: basic)"
    echo "  --cuda-devices DEVICES     CUDA devices to use (default: 0)"
    echo "  --reset-training           Reset training from scratch"
    echo "  --help, -h                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Basic training"
    echo "  $0 --batch-size 8 --model-size small # Larger batch and model"
    echo "  $0 --optimization enhanced            # Enhanced optimization"
    echo ""
}

# Parse command line arguments
RESET_TRAINING=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --model-size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --optimization)
            OPTIMIZATION_LEVEL="$2"
            shift 2
            ;;
        --cuda-devices)
            CUDA_DEVICES="$2"
            shift 2
            ;;
        --reset-training)
            RESET_TRAINING="--reset-training"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Display configuration
echo ""
echo "📋 Training Configuration:"
echo "  Batch Size: $BATCH_SIZE"
echo "  Epochs: $EPOCHS"
echo "  Model Size: $MODEL_SIZE"
echo "  Optimization: $OPTIMIZATION_LEVEL"
echo "  CUDA Devices: $CUDA_DEVICES"
echo "  Reset Training: $([ -n "$RESET_TRAINING" ] && echo 'YES' || echo 'NO')"
echo ""

# Confirm before starting
read -p "Continue with training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Set environment variables
export MYXTTS_SIMPLE_LOSS=1
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Start training
echo ""
echo "🚀 Starting training..."
echo ""

python3 train_main.py \
    --batch-size $BATCH_SIZE \
    --grad-accum 1 \
    --epochs $EPOCHS \
    --optimization-level $OPTIMIZATION_LEVEL \
    --model-size $MODEL_SIZE \
    $RESET_TRAINING

echo ""
echo "✅ Training script completed!"