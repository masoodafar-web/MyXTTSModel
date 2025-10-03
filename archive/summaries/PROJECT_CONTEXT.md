# 🎤 MyXTTS Project Context & Reference Guide

**استفاده برای GPT**: این فایل رو به عنوان context به هر GPT ارجاع بدید تا بتونه راهنمایی دقیق و مفصل برای پروژتون ارائه بده.

---

## 📋 **Project Overview**

### **What is MyXTTS?**
MyXTTS is an advanced Text-to-Speech (TTS) training framework built on TensorFlow that specializes in:
- **High-quality voice synthesis** with neural vocoders
- **Voice cloning capabilities** with advanced conditioning
- **GPU optimization** for efficient training
- **Loss plateau breakthrough** techniques
- **Multilingual support** (16+ languages)
- **Advanced optimization strategies** for faster convergence

### **Current Status:**
- **Version**: 1.0.0 (Production Ready)
- **Framework**: TensorFlow 2.x + PyTorch components
- **Languages**: Python 3.8+ 
- **Platform**: Linux (optimized for CUDA GPUs)
- **Architecture**: Transformer-based XTTS with enhanced conditioning

---

## 🎯 **Core Problems Solved**

### **Historical Issues Addressed:**
1. **Loss Plateau Problem**: Loss getting stuck around 2.5 → **SOLVED** with plateau_breaker optimization
2. **GPU Utilization Issues**: Fluctuating 40%-2% usage → **SOLVED** with Advanced GPU Stabilizer
3. **NaN Loss Issues**: Training crashes → **SOLVED** with enhanced loss stability
4. **Memory Optimization**: OOM errors → **SOLVED** with memory-efficient configurations
5. **Voice Cloning Quality**: Poor voice similarity → **SOLVED** with advanced conditioning
6. **Slow Convergence**: 3x faster training achieved → **SOLVED** with fast convergence config

### **Key Innovations:**
- **Plateau Breakthrough**: Automatic detection and resolution of loss plateaus
- **GPU Stabilizer**: Real-time GPU utilization optimization
- **Adaptive Loss Weights**: Dynamic loss component balancing
- **Enhanced Voice Conditioning**: Superior voice cloning with GST tokens
- **Multi-size Architecture**: Scalable from tiny to big model sizes

---

## 🏗️ **Technical Architecture**

### **Model Architecture:**
```
Text Input → Text Encoder (Transformer) → 
Audio Conditioning → Audio Encoder (Enhanced) → 
Decoder (Large Transformer) → Mel Spectrogram → 
Neural Vocoder → Audio Output
```

### **Key Components:**
- **Text Encoder**: 4-10 layers, 256-768 dims (multilingual NLLB)
- **Audio Encoder**: 4-10 layers, 256-1024 dims (voice conditioning)
- **Decoder**: 8-20 layers, 768-2048 dims (synthesis core)
- **Voice Conditioning**: GST tokens, speaker embeddings
- **Neural Vocoders**: Griffin-Lim, HiFiGAN, BigVGAN support

### **Optimization Features:**
- **Gradient Checkpointing**: Memory efficiency
- **Mixed Precision**: Speed optimization
- **Adaptive Learning**: Dynamic rate adjustment
- **Loss Scheduling**: Cosine restarts with plateau detection

---

## 📂 **Project Structure (Organized)**

```
MyXTTSModel/
├── 🧠 myxtts/                 # Core package (models, training, config)
├── 📜 train_main.py           # Main training script (START HERE)
├── 📜 inference_main.py       # Voice synthesis script
├── 📋 scripts/                # Training automation scripts
│   ├── train_control.sh       # Easy training control
│   ├── breakthrough_training.sh  # Plateau breakthrough
│   └── quick_restart.sh       # Training restart
├── ⚙️ configs/                # Configuration files
│   ├── config.yaml           # Main config
│   ├── config_enhanced.yaml  # Enhanced optimization
│   ├── config_plateau_breaker.yaml  # Plateau fixes
│   └── [15+ specialized configs]
├── 🔧 optimization/           # Advanced optimization modules
│   ├── advanced_gpu_stabilizer.py    # GPU optimization
│   ├── loss_breakthrough_config.py   # Loss plateau fixes
│   ├── fast_convergence_config.py    # Speed optimization
│   └── [20+ optimization tools]
├── 📊 monitoring/             # Real-time monitoring
├── 🛠️ utilities/             # Helper tools & validation
├── 📘 examples/               # Usage examples & demos
├── 🧪 tests/                  # Comprehensive test suite
├── 📓 notebooks/              # Jupyter analysis notebooks
├── 📋 reports/                # Training analysis & logs
├── 🎨 assets/                 # Reference audio & visualizations
├── 📤 outputs/                # Generated audio outputs
└── 📚 docs/                   # Complete documentation (50+ guides)
```

---

## 🚀 **Usage Patterns & Commands**

### **Basic Training:**
```bash
# Quick start (recommended)
python3 train_main.py --model-size normal --optimization-level enhanced

# Small test
python3 train_main.py --model-size tiny --batch-size 4 --epochs 10
```

### **Advanced Scenarios:**
```bash
# Loss stuck around 2.5? (COMMON ISSUE)
python3 train_main.py --optimization-level plateau_breaker --batch-size 24

# Maximum GPU utilization
python3 train_main.py --enable-gpu-stabilizer --optimization-level enhanced

# Voice cloning training
python3 train_main.py --enable-gst --gst-num-style-tokens 12 --model-size normal

# Production training
python3 train_main.py \
    --model-size big \
    --optimization-level enhanced \
    --enable-gpu-stabilizer \
    --enable-evaluation \
    --epochs 500
```

### **Convenience Scripts:**
```bash
# Easy management
bash scripts/train_control.sh --enable-gpu-stabilizer --batch-size 32

# Automatic plateau fixing
bash scripts/breakthrough_training.sh
```

---

## ⚙️ **Configuration System**

### **Optimization Levels:**
- **`basic`**: Conservative, stable settings (learning_rate=1e-5)
- **`enhanced`**: Recommended production settings (learning_rate=8e-5)
- **`experimental`**: Bleeding-edge optimizations
- **`plateau_breaker`**: Special config for stuck loss (learning_rate=1.5e-5)

### **Model Sizes:**
- **`tiny`**: 256-768 dims, fast training, lower quality
- **`small`**: 384-1024 dims, balanced quality/speed
- **`normal`**: 512-1536 dims, high quality (default)
- **`big`**: 768-2048 dims, maximum quality, requires high-end GPU

### **Key Parameters:**
```python
# Loss weights (critical for convergence)
mel_loss_weight: 2.5      # Primary synthesis loss
kl_loss_weight: 1.8       # Regularization
voice_similarity_loss: 3.0 # Voice cloning

# Learning schedule
learning_rate: 8e-5       # Enhanced default
scheduler: "cosine"       # With restarts
gradient_clip_norm: 0.8   # Stability
```

---

## ⚡ **Common Issues & Solutions**

### **🔴 Issue: Loss Stuck at 2.5**
**Symptoms**: Training loss plateaus around 2.5-2.6, no improvement
**Solution**: 
```bash
python3 train_main.py --optimization-level plateau_breaker --batch-size 24
```
**Why**: Learning rate too high, loss components unbalanced

### **🔴 Issue: Low GPU Utilization (40%-2%)**
**Symptoms**: GPU usage fluctuating, slow training
**Solution**:
```bash
python3 train_main.py --enable-gpu-stabilizer --optimization-level enhanced
```
**Why**: Inefficient data loading, CPU-GPU synchronization issues

### **🔴 Issue: Out of Memory**
**Symptoms**: CUDA OOM errors
**Solution**:
```bash
python3 train_main.py --model-size tiny --batch-size 4
```
**Why**: Model too large for available GPU memory

### **🔴 Issue: NaN Loss**
**Symptoms**: Loss becomes NaN, training crashes
**Solution**:
```bash
python3 train_main.py --optimization-level basic --use-huber-loss
```
**Why**: Gradient explosion, aggressive learning rate

### **🔴 Issue: Poor Voice Cloning**
**Symptoms**: Generated voice doesn't match reference
**Solution**:
```bash
python3 train_main.py --enable-gst --gst-num-style-tokens 15 --voice-similarity-loss-weight 3.0
```
**Why**: Insufficient voice conditioning, weak similarity loss

---

## 🔧 **Development Context**

### **Important Files to Know:**
- **`train_main.py`**: Main entry point, comprehensive options
- **`myxtts/training/trainer.py`**: Core training logic
- **`myxtts/models/xtts.py`**: Model architecture
- **`optimization/advanced_gpu_stabilizer.py`**: GPU optimization
- **`configs/config.yaml`**: Base configuration

### **Import Structure:**
```python
from myxtts.config.config import XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.training.trainer import XTTSTrainer
```

### **Key Classes:**
- **`XTTSConfig`**: Configuration management
- **`XTTS`**: Main model class
- **`XTTSTrainer`**: Training orchestrator
- **`AdvancedGPUStabilizer`**: GPU optimization

---

## 📊 **Performance Metrics**

### **Typical Training Performance:**
- **Tiny Model**: ~2-3 seconds/batch, 4GB VRAM
- **Normal Model**: ~3-4 seconds/batch, 8-12GB VRAM  
- **Big Model**: ~5-6 seconds/batch, 16-20GB VRAM

### **Convergence Expectations:**
- **Basic Training**: Loss ~2.8-3.0 after 100 epochs
- **Enhanced Training**: Loss ~2.2-2.5 after 100 epochs
- **Plateau Breaker**: Loss <2.0 after breakthrough

### **Hardware Requirements:**
- **Minimum**: GTX 1080 (8GB), 16GB RAM
- **Recommended**: RTX 3090/4090 (24GB), 32GB RAM
- **Optimal**: Multi-GPU setup with 40GB+ VRAM

---

## 🎯 **Current Challenges & Goals**

### **Ongoing Optimization Areas:**
1. **Loss Convergence**: Further improvements below 2.0
2. **Voice Similarity**: Even better voice cloning accuracy
3. **Inference Speed**: Real-time synthesis optimization
4. **Memory Efficiency**: Larger models on smaller GPUs
5. **Multi-GPU Training**: Distributed training enhancements

### **Recent Achievements:**
- ✅ Solved loss plateau at 2.5 with plateau_breaker
- ✅ 90%+ GPU utilization with stabilizer
- ✅ 3x faster convergence with enhanced optimization
- ✅ Superior voice cloning with GST integration
- ✅ Comprehensive documentation and organization

---

## 🤝 **GPT Guidance Instructions**

### **When providing help:**

1. **Always consider the specific optimization level** being used
2. **Reference the exact file structure** when suggesting changes
3. **Provide complete command examples** with all necessary flags
4. **Consider GPU memory constraints** when recommending settings
5. **Suggest appropriate config files** for different scenarios
6. **Reference the comprehensive documentation** in `docs/` folder

### **Common user intents to recognize:**
- **"Loss is stuck"** → plateau_breaker optimization
- **"Low GPU usage"** → GPU stabilizer enablement  
- **"Out of memory"** → Smaller model size or batch size
- **"Poor quality"** → Larger model or enhanced optimization
- **"Voice cloning issues"** → GST enablement and voice conditioning
- **"Training too slow"** → GPU optimization and enhanced settings

### **Important context to remember:**
- User has RTX 4090 GPUs (high-end hardware)
- Project is production-ready with comprehensive tooling
- All major training issues have been solved with specific solutions
- Extensive monitoring and validation tools are available
- Multiple convenience scripts exist for common tasks

---

## 📞 **Quick Reference Commands**

```bash
# Emergency plateau fix
bash scripts/breakthrough_training.sh

# GPU optimization check
python3 optimization/gpu_monitor.py

# Validation suite
python3 utilities/comprehensive_validation.py

# Memory test
python3 utilities/quick_memory_test.py

# Generate training report
python3 utilities/solution_summary.py
```

---

**🎯 Use this file as your complete project context when asking any GPT for help with MyXTTS!**