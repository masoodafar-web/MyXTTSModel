#!/bin/bash
# 📦 MyXTTS Dependencies Installer

echo "📦 Installing MyXTTS Dependencies..."
echo "===================================="

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "🐍 Python version: $python_version"

if [[ "$python_version" < "3.8" ]]; then
    echo "❌ Error: Python 3.8+ is required"
    exit 1
fi

# Install core dependencies
echo "📦 Installing core dependencies..."
pip3 install -r requirements.txt

# Try to install GPU monitoring packages
echo "🎮 Installing GPU monitoring packages..."

# Try GPUtil first (preferred)
if pip3 install GPUtil>=1.4.0; then
    echo "✅ GPUtil installed successfully"
else
    echo "⚠️  GPUtil installation failed, will use nvidia-ml-py fallback"
fi

# Ensure nvidia-ml-py is available (usually comes with nvidia drivers)
if python3 -c "import pynvml; print('✅ pynvml available')" 2>/dev/null; then
    echo "✅ nvidia-ml-py is available"
else
    echo "⚠️  nvidia-ml-py not available, installing..."
    pip3 install nvidia-ml-py3
fi

# Install additional plotting libraries
echo "📊 Installing visualization libraries..."
pip3 install seaborn plotly

# Test installations
echo ""
echo "🧪 Testing installations..."

# Test basic imports
python3 -c "
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print('✅ Core libraries working')
"

# Test GPU utilities
python3 -c "
try:
    import GPUtil
    gpus = GPUtil.getGPUs()
    print(f'✅ GPUtil working - Found {len(gpus)} GPU(s)')
except ImportError:
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print(f'✅ nvidia-ml-py working - Found {count} GPU(s)')
    except:
        print('⚠️  No GPU monitoring available (will use nvidia-smi fallback)')
"

# Test TensorFlow GPU
python3 -c "
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(f'✅ TensorFlow GPU: {len(gpus)} device(s) found')
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Ready to run benchmarks:"
echo "  bash scripts/benchmark_params.sh"
echo "  python3 scripts/quick_param_test.py"
echo "  python3 utilities/benchmark_hyperparameters.py --quick-test"