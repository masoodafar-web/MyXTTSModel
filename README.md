# 🎙️ MyXTTS - Advanced Text-to-Speech Training Framework

A comprehensive, production-ready Text-to-Speech training framework with advanced voice cloning capabilities, GPU optimization, and plateau breakthrough techniques.

## ⚡ GPU Optimization Alert (New!)

**Having GPU utilization issues (2-40%)? Retracing warnings?** 

✅ **SOLVED!** We've implemented a complete fix for tf.function retracing issues.

```bash
# Quick validation before training
./validate_setup.sh configs/config.yaml

# Or run the diagnostic tool
python utilities/diagnose_retracing.py --config configs/config.yaml
```

📖 **See**: [`RETRACING_COMPLETE_SOLUTION.md`](RETRACING_COMPLETE_SOLUTION.md) for the complete fix

**Results**: 70-90% GPU utilization (stable), 30x faster training

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd MyXTTSModel

# Install dependencies
pip install -r requirements.txt

# Validate your setup (recommended!)
./validate_setup.sh configs/config.yaml

# Basic training - NOW WITH SMART DEFAULTS! 🎯
# Uses: tiny model, batch-size 16, static shapes enabled
python3 train_main.py --train-data ../dataset/dataset_train --val-data ../dataset/dataset_eval

# Or even simpler (uses default dataset paths):
python3 train_main.py

# Override specific parameters as needed:
python3 train_main.py --model-size small --batch-size 24

# Advanced training with all optimizations
python3 train_main.py --model-size normal --optimization-level enhanced
```

### ✨ New Smart Defaults

The training script now comes with sensible defaults that work out of the box:
- **Model size**: `tiny` (great for learning and quick iterations)
- **Batch size**: `16` (automatically adjusted based on your GPU memory)
- **Static shapes**: `enabled` (prevents GPU utilization issues)
- **GPU mode**: `single-GPU` (automatically switches to multi-GPU when you specify `--data-gpu` and `--model-gpu`)
- **Workers**: `8` (automatically adjusted based on your system)

You can override any of these with command-line arguments!

## 📊 TensorBoard Monitoring (New!)

**Track training progress visually with images and audio!**

This feature allows you to monitor training by logging:
- 🖼️ **Images** from `training_samples` directory (spectrograms, visualizations)
- 🔊 **Audio samples** from `training_samples` directory
- 🎵 **Generated audio** samples during training
- 📈 **Training metrics** and loss curves

### Quick Setup

```bash
# 1. Create samples directory and add your files
mkdir training_samples
cp your_spectrograms/*.png training_samples/
cp your_reference_audio/*.wav training_samples/

# 2. Start training (samples logged automatically)
python train_main.py --train-data ../dataset/dataset_train

# 3. View in TensorBoard
tensorboard --logdir=logs  # or ./checkpointsmain/tensorboard
```

Then open `http://localhost:6006` to see:
- **IMAGES tab**: Your sample images and spectrograms
- **AUDIO tab**: Your audio files and generated samples

📖 **Full Guide**: See [TENSORBOARD_SAMPLES_GUIDE.md](TENSORBOARD_SAMPLES_GUIDE.md) for complete documentation
🇮🇷 **راهنمای فارسی**: [TENSORBOARD_SAMPLES_GUIDE_FA.md](TENSORBOARD_SAMPLES_GUIDE_FA.md)

### Configuration Options

```bash
# Custom samples directory
python train_main.py --training-samples-dir ./my_samples

# Change logging frequency (default: every 100 steps)
python train_main.py --training-samples-log-interval 50

# Custom TensorBoard directory
python train_main.py --tensorboard-log-dir ./logs
```

## 📁 Project Structure

```
MyXTTSModel/
├── 📜 Core Files
│   ├── train_main.py          # Main training script
│   ├── inference_main.py      # Inference and voice cloning script
│   ├── fixed_inference.py     # Fixed inference implementation
│   ├── manage.sh              # Project management script
│   ├── setup.py               # Package setup and installation
│   └── requirements.txt       # Python dependencies
│
├── 🧠 myxtts/                 # Core model package
│   ├── config/                # Configuration classes
│   ├── models/                # XTTS model implementations
│   ├── training/              # Training classes and utilities
│   └── utils/                 # Common utilities
│
├── ⚙️ configs/                # Configuration files
│   ├── config.yaml            # Main configuration
│   └── example_config.yaml    # Example configuration
│
├── 📋 scripts/                # Utility scripts
│   ├── install_dependencies.sh   # Dependency installation
│   ├── quick_restart.sh          # Quick restart utility
│   └── train_gpu_optimized.sh    # GPU-optimized training
│
├── 🛠️ utilities/              # Utility scripts
│   ├── memory_optimizer.py    # Memory optimization tools
│   ├── evaluate_tts.py        # TTS quality evaluation
│   └── optimize_model.py      # Model optimization for deployment
│
├── 📘 examples/               # Usage examples and demos
│   └── Various usage examples
│
├── 🧪 tests/                  # Test suite
│   └── Comprehensive test files
│
├── 📓 notebooks/              # Jupyter notebooks
│   └── Training and evaluation notebooks
│
├── 📚 docs/                   # Documentation
│   └── Technical documentation
│
└── 🗃️ data/               # Data directories
    ├── checkpointsmain/             # Training checkpoints
    ├── dataset/                     # Training datasets
    └── small_dataset_test/          # Test datasets
```

## 🎯 Key Features

### 🧠 **Advanced Model Architecture**
- **Multiple Model Sizes**: tiny, small, normal, big
- **Voice Cloning**: Advanced speaker conditioning and adaptation
- **Global Style Tokens (GST)**: Prosody and emotion control
- **Multi-language Support**: 16+ languages with NLLB tokenization

### ⚡ **Optimization Levels**
- **Basic**: Stable, conservative settings
- **Enhanced**: Recommended optimizations with model-size-aware tuning (default)
- **Experimental**: Bleeding-edge features
- **Plateau Breaker**: Special config for stuck loss (around 2.5-2.8)

> **New in Latest Version**: Enhanced level now automatically adjusts learning rate, gradient clipping, and other parameters based on model size for better convergence and plateau prevention.

### 🔧 **GPU Optimization**
- **Memory Management**: Efficient VRAM usage
- **Single GPU Training**: Optimized for single GPU or CPU training
- **Intelligent GPU Pipeline**: 🆕 Automatic Multi-GPU and Single-GPU Buffered modes
  - Multi-GPU Mode: Separate GPUs for data and model
  - Single-GPU Buffered Mode: Smart prefetching with configurable buffer
  - See [`docs/INTELLIGENT_GPU_PIPELINE.md`](docs/INTELLIGENT_GPU_PIPELINE.md) for details

### 📈 **Training Enhancements**
- **Fast Convergence**: 2-3x faster loss convergence
- **Adaptive Loss Weights**: Auto-adjusting loss components
- **Plateau Detection**: Automatic learning rate adjustment
- **Enhanced Monitoring**: Real-time training metrics
- **Text-to-Audio Evaluation**: 🆕 Automatic audio generation during training
  - Generates audio samples every N steps (default: 200)
  - Saves WAV files for quality comparison
  - TensorBoard integration for real-time listening
  - See [`docs/TEXT2AUDIO_EVAL_GUIDE.md`](docs/TEXT2AUDIO_EVAL_GUIDE.md) for details

## 🚀 Usage Examples

### Basic Training
```bash
# Simplest command - uses smart defaults (tiny model, batch-size 16, static shapes enabled)
python3 train_main.py

# Quick test with even smaller batch
python3 train_main.py --batch-size 4 --epochs 10

# Production training with larger model
python3 train_main.py \
    --model-size normal \
    --optimization-level enhanced \
    --batch-size 32 \
    --epochs 500
```



### Advanced Voice Cloning
```bash
# Enable Global Style Tokens for prosody control
python3 train_main.py \
    --enable-gst \
    --gst-num-style-tokens 12 \
    --model-size normal \
    --optimization-level enhanced
```

### Multi-GPU Training (OPTIMIZED! ⚡)

**NEW v2.0**: Async pipeline with triple buffering for 2-3x faster training!

```bash
# Memory-Isolated Dual-GPU (RECOMMENDED - Optimized v2.0)
python3 train_main.py \
    --model-size tiny \
    --batch-size 16 \
    --data-gpu 0 \
    --model-gpu 1 \
    --enable-memory-isolation \
    --enable-static-shapes \
    --data-gpu-memory 8192 \
    --model-gpu-memory 16384 \
    --epochs 500

# Legacy Multi-GPU (older method)
python3 train_main.py \
    --data-gpu 0 \
    --model-gpu 1 \
    --buffer-size 100 \
    --batch-size 64 \
    --epochs 500

# Single-GPU with buffer (for users with 1 GPU)
python3 train_main.py \
    --buffer-size 100 \
    --batch-size 32 \
    --epochs 500
```

📊 **Performance improvement**: The new memory-isolated mode achieves 80-95% GPU utilization (vs 50-70% before)
📖 **Documentation**: See [DUAL_GPU_BOTTLENECK_FIX.md](./DUAL_GPU_BOTTLENECK_FIX.md) for details
🔧 **Profiler**: Use `utilities/dual_gpu_bottleneck_profiler.py` to diagnose bottlenecks

### Model Evaluation and Optimization
```bash
# Training with automatic evaluation
python3 train_main.py \
    --enable-evaluation \
    --evaluation-interval 25 \
    --create-optimized-model
```

## 🔧 Configuration

### Optimization Levels
- **`--optimization-level basic`**: Conservative, stable settings for compatibility
- **`--optimization-level enhanced`**: Recommended optimizations with model-size-aware tuning (default)
  - Automatically adjusts learning rate and gradient clipping based on model size
  - Tiny model: lr=3e-5, clip=0.5 | Small: lr=5e-5, clip=0.7 | Normal/Big: lr=8e-5, clip=0.8
  - Includes warnings for suboptimal configurations
- **`--optimization-level plateau_breaker`**: For persistent plateaus at 2.5-2.8

### Model Sizes
- **`--model-size tiny`**: Fast training, lower quality (256/768 dims)
  - Best with: `--batch-size 8` or `16`, `--optimization-level enhanced`
- **`--model-size small`**: Balanced quality vs speed (384/1024 dims)
  - Best with: `--batch-size 16`, `--optimization-level enhanced`
- **`--model-size normal`**: High quality, default (512/1536 dims)
  - Best with: `--batch-size 32`, `--optimization-level enhanced`
- **`--model-size big`**: Maximum quality (768/2048 dims)
  - Best with: `--batch-size 8` or `16`, `--optimization-level enhanced`

## 🔧 Troubleshooting

### Loss Plateau Issues

If your loss plateaus and stops decreasing (e.g., stuck at 2.8):

**For Tiny Model:**
```bash
# Option 1: Use recommended batch size (RECOMMENDED)
python3 train_main.py --model-size tiny --optimization-level enhanced --batch-size 16

# Option 2: Use plateau_breaker if still stuck
python3 train_main.py --model-size tiny --optimization-level plateau_breaker --batch-size 16

# Option 3: Upgrade to small model for better capacity
python3 train_main.py --model-size small --optimization-level enhanced --batch-size 16
```

**Common Causes:**
- Batch size too large for model size (tiny model with batch_size > 16)
- Model capacity insufficient (tiny model may underfit)
- Learning rate too high (now auto-adjusted in enhanced level)

**Solution Path:**
1. Try with recommended batch size for your model size
2. If still stuck, use `--optimization-level plateau_breaker`
3. Consider upgrading to a larger model size
4. Check training logs for warnings and recommendations

See: `docs/LOSS_PLATEAU_2.8_TINY_ENHANCED_FIX.md` for detailed troubleshooting.

## 📊 Monitoring and Validation

### Quick Validation
```bash
# Run comprehensive model validation
python3 utilities/validate_model_correctness.py

# Run end-to-end tests
python3 tests/test_end_to_end_validation.py

# Quick validation (both commands)
python3 utilities/validate_model_correctness.py && python3 tests/test_end_to_end_validation.py
```

### Full Validation Suite
```bash
# Validate model functionality
python3 utilities/validate_enhancements.py

# Memory optimization testing
python3 utilities/validate_memory_fixes.py

# Complete system validation
python3 utilities/comprehensive_validation.py --data-path YOUR_DATA --quick-test
```

### Validation Results
- ✓ Model Architecture: PASS (100%)
- ✓ Loss Functions: PASS (100%)  
- ✓ Gradient Flow: PASS (100%)
- ✓ Training Pipeline: PASS (100%)
- ✓ Inference Mode: PASS (100%)

See [Validation Guide](docs/VALIDATION_GUIDE.md) for details.

## 🛠️ Development and Testing

### Running Tests
```bash
# Basic functionality tests
python3 tests/test_basic_functionality.py

# Enhanced model tests
python3 tests/test_enhanced_model.py

# GPU optimization tests
python3 tests/test_gpu_optimization.py
```

### Utilities
```bash
# Model optimization for deployment
python3 utilities/optimize_model.py

# TTS quality evaluation
python3 utilities/evaluate_tts.py

# Memory usage optimization
python3 utilities/memory_optimizer.py
```

## 📚 Documentation

Comprehensive guides available in the `docs/` directory:

### Validation and Testing
- **[Validation Guide](docs/VALIDATION_GUIDE.md)** - Complete validation and testing guide
- **[Validation Quick Reference](docs/VALIDATION_QUICK_REFERENCE.md)** - Quick reference for validation
- **[Model Validation Summary](docs/MODEL_VALIDATION_SUMMARY.md)** - Validation results summary
- **[Coqui XTTS Comparison](docs/COQUI_XTTS_COMPARISON.md)** - Detailed comparison with Coqui XTTS

### Architecture and Features
- **[Architecture](docs/ARCHITECTURE.md)** - Model architecture documentation
- **[Fast Convergence Solution](docs/FAST_CONVERGENCE_SOLUTION.md)** - Training optimization guide
- **[Enhanced Voice Conditioning Guide](docs/ENHANCED_VOICE_CONDITIONING.md)** - Voice cloning features

### Optimization Guides
- **[Intelligent GPU Pipeline System](docs/INTELLIGENT_GPU_PIPELINE.md)** 🆕 - Multi-GPU and Single-GPU optimization
- **[Intelligent GPU Pipeline (Persian)](docs/INTELLIGENT_GPU_PIPELINE_FA.md)** 🆕 - راهنمای فارسی
- **[Advanced Memory Optimization Guide](docs/ADVANCED_MEMORY_OPTIMIZATION_GUIDE.md)**
- **[GPU Utilization Fix Guide](docs/GPU_UTILIZATION_FIX_GUIDE.md)**
- **[Plateau Breakthrough Guide](docs/PLATEAU_BREAKTHROUGH_GUIDE.md)**
- **[Loss Plateau 2.8 Fix](docs/LOSS_PLATEAU_2.8_TINY_ENHANCED_FIX.md)** - Fix for loss plateau with tiny+enhanced
- **[Loss Plateau 2.7 Solution](docs/LOSS_PLATEAU_SOLUTION_2.7.md)** - Previous plateau fix

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on the XTTS architecture
- Optimized for production use cases
- Community-driven improvements and bug fixes

---

**🎯 Ready to train high-quality voice cloning models? Start with the quick start guide above!**