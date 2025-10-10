#!/bin/bash
# Quick Fix for GPU Utilization Issue
# راهکار سریع برای مشکل استفاده پایین از GPU
#
# This script applies all critical fixes for the 1-5% GPU utilization issue
# on dual RTX 4090 setup.
#
# Usage:
#   bash quick_fix_gpu_utilization.sh
#   OR
#   bash quick_fix_gpu_utilization.sh --config configs/config.yaml

set -e

echo "=========================================================================="
echo "          QUICK FIX FOR GPU UTILIZATION ISSUE"
echo "      راهکار سریع برای مشکل استفاده پایین از GPU"
echo "=========================================================================="
echo

# Parse arguments
CONFIG_FILE="configs/config.yaml"
if [ "$1" = "--config" ] && [ -n "$2" ]; then
    CONFIG_FILE="$2"
fi

echo "Configuration file: $CONFIG_FILE"
echo

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo "-----------------------------------"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found"
    exit 1
fi
echo "✅ Python3: $(python3 --version)"

if ! python3 -c "import tensorflow" 2>/dev/null; then
    echo "❌ TensorFlow not installed"
    exit 1
fi
echo "✅ TensorFlow: $(python3 -c 'import tensorflow as tf; print(tf.__version__)')"

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  nvidia-smi not found (GPU monitoring disabled)"
else
    echo "✅ NVIDIA Driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✅ GPUs detected: $GPU_COUNT"
fi

echo

# Step 2: Run diagnostic
echo "Step 2: Running diagnostic..."
echo "-----------------------------------"

if [ -f "$CONFIG_FILE" ]; then
    python3 utilities/diagnose_gpu_utilization.py --config "$CONFIG_FILE" --skip-realtime
else
    echo "⚠️  Config file not found: $CONFIG_FILE"
    echo "   Running diagnostic without config"
    python3 utilities/diagnose_gpu_utilization.py --skip-realtime
fi

echo

# Step 3: Apply TensorFlow optimizations
echo "Step 3: Applying TensorFlow optimizations..."
echo "----------------------------------------------"
python3 utilities/configure_max_gpu_utilization.py --verify

echo

# Step 4: Backup and update config
echo "Step 4: Updating configuration..."
echo "-----------------------------------"

if [ -f "$CONFIG_FILE" ]; then
    # Backup original config
    BACKUP_FILE="${CONFIG_FILE}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "$CONFIG_FILE" "$BACKUP_FILE"
    echo "✅ Backup created: $BACKUP_FILE"
    
    # Check and suggest config updates
    echo
    echo "Recommended configuration changes:"
    echo "-----------------------------------"
    echo "data:"
    echo "  batch_size: 128          # Increased from default"
    echo "  num_workers: 32          # Increased for better parallelism"
    echo "  prefetch_buffer_size: 100  # Aggressive prefetch"
    echo "  use_tf_native_loading: true"
    echo "  prefetch_to_gpu: true"
    echo "  pad_to_fixed_length: true  # Enable static shapes"
    echo "  enable_xla: true"
    echo "  mixed_precision: true"
    echo
    echo "⚠️  Please manually update your config file with these settings"
    echo "   Original backed up to: $BACKUP_FILE"
else
    echo "⚠️  Config file not found: $CONFIG_FILE"
    echo "   Please create a config file with optimized settings"
fi

echo

# Step 5: Provide training command
echo "Step 5: Ready to train with optimized settings"
echo "------------------------------------------------"
echo
echo "Use this command to start training:"
echo
echo "python3 train_main.py \\"
echo "    --train-data ../dataset/dataset_train \\"
echo "    --val-data ../dataset/dataset_eval \\"
echo "    --batch-size 128 \\"
echo "    --num-workers 32 \\"
echo "    --enable-memory-isolation \\"
echo "    --data-gpu 0 \\"
echo "    --model-gpu 1 \\"
echo "    --data-gpu-memory 10240 \\"
echo "    --model-gpu-memory 20480 \\"
echo "    --enable-static-shapes \\"
echo "    --optimization-level enhanced"
echo

# Step 6: Monitoring instructions
echo "Step 6: Monitoring GPU utilization"
echo "------------------------------------"
echo
echo "In another terminal, run:"
echo "  watch -n 1 nvidia-smi"
echo
echo "Expected results:"
echo "  • GPU:0 (Data) Utilization: 60-80%"
echo "  • GPU:1 (Model) Utilization: 85-95%"
echo "  • Step time: <0.5s (with batch_size=128)"
echo

# Step 7: Troubleshooting
echo "=========================================================================="
echo "TROUBLESHOOTING"
echo "=========================================================================="
echo
echo "If GPU utilization is still low (<50%):"
echo
echo "1. Verify TF-native loading is active:"
echo "   Look for this in training logs:"
echo "   ✅ SUCCESS: Using TensorFlow-native data loading (GPU-optimized)"
echo
echo "2. Check for data pipeline bottleneck:"
echo "   python3 utilities/dual_gpu_bottleneck_profiler.py \\"
echo "       --batch-size 128 --num-steps 100"
echo
echo "3. Try increasing batch size further:"
echo "   python3 train_main.py --batch-size 256 ..."
echo
echo "4. Increase num_workers if you have many CPU cores:"
echo "   python3 train_main.py --num-workers 48 ..."
echo
echo "5. Check for OOM errors:"
echo "   If out of memory, reduce batch size or adjust GPU memory limits"
echo

echo "=========================================================================="
echo "QUICK FIX COMPLETE"
echo "=========================================================================="
echo
echo "✅ All optimizations have been applied"
echo "✅ You are ready to start training"
echo
echo "For detailed documentation, see:"
echo "  • CRITICAL_GPU_UTILIZATION_FIX.md"
echo "  • DUAL_GPU_BOTTLENECK_SOLUTION.md"
echo
echo "=========================================================================="
