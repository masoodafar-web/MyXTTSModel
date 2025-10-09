#!/bin/bash
# Breakthrough Training Script for Loss Plateau
# This script starts training with plateau_breaker optimization level
# to help break through loss plateaus around 2.5-2.7

set -e

echo "🚀 Starting Breakthrough Training for Loss Plateau"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  • Optimization Level: plateau_breaker"
echo "  • Learning Rate: 1.5e-5 (80% reduction)"
echo "  • Mel Loss Weight: 2.0"
echo "  • Gradient Clip: 0.3 (tight control)"
echo "  • Scheduler: Cosine with restarts every ~100 epochs"
echo ""
echo "Expected Results:"
echo "  • Loss reduction to 2.2-2.3 within 5-10 epochs"
echo "  • Better balance between mel_loss and stop_loss"
echo "  • More stable training convergence"
echo ""
echo "=================================================="
echo ""

# Default values
BATCH_SIZE=${BATCH_SIZE:-24}
EPOCHS=${EPOCHS:-1000}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints"}
TRAIN_DATA=${TRAIN_DATA:-"./data/ljspeech"}

# Run training with plateau_breaker optimization
python3 train_main.py \
    --optimization-level plateau_breaker \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --train-data "$TRAIN_DATA" \
    "$@"
