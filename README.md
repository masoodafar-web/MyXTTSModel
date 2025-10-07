# 🎙️ MyXTTS - Advanced Text-to-Speech Training Framework

A comprehensive, production-ready Text-to-Speech training framework with advanced voice cloning capabilities, GPU optimization, and plateau breakthrough techniques.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd MyXTTSModel

# Install dependencies
pip install -r requirements.txt

# Basic training
python3 train_main.py --train-data ../dataset/dataset_train --val-data ../dataset/dataset_eval

# Advanced training with optimizations
python3 train_main.py --optimization-level enhanced
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
- **Enhanced**: Recommended optimizations (default)
- **Experimental**: Bleeding-edge features
- **Plateau Breaker**: Special config for stuck loss (around 2.5)

### 🔧 **GPU Optimization**
- **Memory Management**: Efficient VRAM usage
- **Single GPU Training**: Optimized for single GPU or CPU training

### 📈 **Training Enhancements**
- **Fast Convergence**: 2-3x faster loss convergence
- **Adaptive Loss Weights**: Auto-adjusting loss components
- **Plateau Detection**: Automatic learning rate adjustment
- **Enhanced Monitoring**: Real-time training metrics

## 🚀 Usage Examples

### Basic Training
```bash
# Quick start with tiny model
python3 train_main.py --model-size tiny --batch-size 4 --epochs 10

# Production training with GPU optimization
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
- **`--optimization-level enhanced`**: Recommended optimizations (default)

### Model Sizes
- **`--model-size tiny`**: Fast training, lower quality (256/768 dims)
- **`--model-size small`**: Balanced quality vs speed (384/1024 dims)
- **`--model-size normal`**: High quality, default (512/1536 dims)
- **`--model-size big`**: Maximum quality (768/2048 dims)

## 📊 Monitoring and Validation

### Validation and Testing
```bash
# Validate model functionality
python3 utilities/validate_enhancements.py

# Memory optimization testing
python3 utilities/validate_memory_fixes.py
```

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

- **[Advanced Memory Optimization Guide](docs/ADVANCED_MEMORY_OPTIMIZATION_GUIDE.md)**
- **[GPU Utilization Fix Guide](docs/GPU_UTILIZATION_FIX_GUIDE.md)**
- **[Plateau Breakthrough Guide](docs/PLATEAU_BREAKTHROUGH_GUIDE.md)**
- **[Enhanced Voice Conditioning Guide](docs/ENHANCED_VOICE_CONDITIONING.md)**

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