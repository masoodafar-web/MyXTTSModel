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

# Advanced training with GPU stabilizer
python3 train_main.py --optimization-level enhanced --enable-gpu-stabilizer
```

## 📁 Project Structure

```
MyXTTSModel/
├── 📜 Core Files
│   ├── train_main.py          # Main training script with comprehensive options
│   ├── inference_main.py      # Inference and voice cloning script
│   ├── setup.py              # Package setup and installation
│   └── requirements.txt      # Python dependencies
│
├── 🧠 myxtts/               # Core model package
│   ├── config/              # Configuration classes
│   ├── models/              # XTTS model implementations
│   ├── training/            # Training classes and utilities
│   └── utils/               # Common utilities
│
├── 📋 scripts/              # Training and utility scripts
│   ├── train_control.sh     # Training control script
│   ├── breakthrough_training.sh  # Plateau breakthrough script
│   └── quick_restart.sh     # Quick restart utility
│
├── ⚙️ configs/              # Configuration files
│   ├── config.yaml          # Main configuration
│   ├── config_enhanced.yaml # Enhanced optimization config
│   ├── config_gpu_optimized.yaml  # GPU-optimized settings
│   └── config_plateau_breaker.yaml  # Plateau breaking config
│
├── 🔧 optimization/         # Optimization modules
│   ├── advanced_gpu_stabilizer.py     # GPU utilization optimizer
│   ├── enhanced_training_monitor.py   # Training monitoring
│   ├── fast_convergence_config.py     # Fast convergence optimizations
│   ├── loss_breakthrough_config.py    # Loss plateau solutions
│   └── gpu_utilization_optimizer.py   # GPU utilization tools
│
├── 📊 monitoring/           # Monitoring and debugging
│   ├── monitor_gpu_live.py  # Real-time GPU monitoring
│   ├── monitor_training.py  # Training process monitoring
│   └── debug_cpu_usage.py   # CPU usage debugging
│
├── 🛠️ utilities/            # Utility scripts
│   ├── memory_optimizer.py  # Memory optimization tools
│   ├── evaluate_tts.py      # TTS quality evaluation
│   ├── optimize_model.py    # Model optimization for deployment
│   └── validate_*.py        # Various validation scripts
│
├── 📘 examples/             # Usage examples and demos
│   ├── demo_enhanced_features.py    # Enhanced features demo
│   ├── example_usage.py             # Basic usage examples
│   ├── usage_examples_enhanced.py   # Advanced usage examples
│   └── gradient_fix_usage_example.py  # Gradient fix examples
│
├── 🧪 tests/               # Test suite
│   ├── test_basic_functionality.py   # Basic functionality tests
│   ├── test_enhanced_model.py        # Enhanced model tests
│   ├── test_gpu_optimization.py      # GPU optimization tests
│   └── test_*.py                     # Various component tests
│
├── 📓 notebooks/           # Jupyter notebooks
│   ├── MyXTTSTrain.ipynb             # Main training notebook
│   ├── evaluation_and_optimization_demo.ipynb  # Evaluation demo
│   └── *.ipynb                       # Additional notebooks
│
├── 📋 reports/             # Training reports and logs
│   ├── gpu_training_analysis_*.json  # GPU training analysis
│   ├── training.log                  # Training logs
│   └── *.json                        # Various reports
│
├── 🎨 assets/              # Media assets
│   ├── speaker.wav                   # Reference audio files
│   ├── gpu_utilization_comparison.png  # Visualization assets
│   └── *.png                         # Additional media
│
├── 📤 outputs/             # Generated outputs
│   └── inference_outputs/            # Inference results
│
├── 📚 docs/                # Documentation
│   ├── ADVANCED_MEMORY_OPTIMIZATION_GUIDE.md
│   ├── GPU_UTILIZATION_FIX_GUIDE.md
│   ├── PLATEAU_BREAKTHROUGH_GUIDE.md
│   └── *.md                          # Additional documentation
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
- **Advanced GPU Stabilizer**: Consistent 90%+ GPU utilization
- **Memory Management**: Efficient VRAM usage
- **Single GPU Training**: Optimized for single GPU or CPU training
- **Real-time Monitoring**: GPU usage tracking and optimization

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
    --enable-gpu-stabilizer \
    --batch-size 32 \
    --epochs 500
```

### Plateau Breakthrough
```bash
# When loss gets stuck around 2.5
python3 train_main.py --optimization-level plateau_breaker --batch-size 24

# Or use the convenience script
bash scripts/breakthrough_training.sh
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
- **`--optimization-level basic`**: Conservative, stable settings
- **`--optimization-level enhanced`**: Recommended optimizations
- **`--optimization-level experimental`**: Latest features
- **`--optimization-level plateau_breaker`**: For stuck loss around 2.5

### GPU Stabilizer Control
- **`--enable-gpu-stabilizer`**: Enable GPU optimization with logs
- **`--disable-gpu-stabilizer`**: Clean training without extra logs

### Model Sizes
- **`--model-size tiny`**: Fast training, lower quality (256/768 dims)
- **`--model-size small`**: Balanced quality vs speed (384/1024 dims)
- **`--model-size normal`**: High quality, default (512/1536 dims)
- **`--model-size big`**: Maximum quality (768/2048 dims)

## 📊 Monitoring and Debugging

### Real-time Monitoring
```bash
# GPU utilization monitoring
python3 monitoring/monitor_gpu_live.py

# Training process monitoring
python3 monitoring/monitor_training.py
```

### Validation and Testing
```bash
# Validate model functionality
python3 utilities/validate_enhancements.py

# Test GPU optimization
python3 utilities/validate_gpu_optimization.py

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