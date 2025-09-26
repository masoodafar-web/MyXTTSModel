#!/usr/bin/env python3
"""
GPU Utilization Training Configuration
=====================================

پیکربندی بهینه‌شده برای حل مسئله GPU utilization که بین 40% و 2% نوسان می‌کند.
این configuration شامل تنظیمات زیر است:

1. Optimized DataLoader settings
2. Async data prefetching
3. Memory management optimizations
4. GPU utilization monitoring
5. Enhanced batch processing

استفاده:
python3 train_main.py --config config_gpu_utilization_optimized.yaml --model-size tiny
"""

import yaml
from pathlib import Path

def create_gpu_optimized_config():
    """ایجاد configuration بهینه‌شده برای GPU utilization"""
    
    config = {
        "model": {
            "model_type": "xtts",
            "architecture": "transformer",
            
            # Model size optimizations for GPU utilization
            "text_encoder": {
                "dim": 256,
                "layers": 4,
                "heads": 8,
                "head_dim": 32,
                "dropout": 0.1
            },
            
            "audio_encoder": {
                "dim": 512,
                "layers": 6,
                "heads": 8,
                "head_dim": 64,
                "dropout": 0.1
            },
            
            "decoder": {
                "dim": 768,
                "layers": 8,
                "heads": 12,
                "head_dim": 64,
                "dropout": 0.1
            },
            
            "speaker_embedding": {
                "dim": 256,
                "conditioning_layers": 3
            },
            
            # Memory optimizations
            "enable_gradient_checkpointing": True,
            "mixed_precision": True,
            "attention_implementation": "flash_attention_2"
        },
        
        "training": {
            # Core training parameters
            "epochs": 500,
            "learning_rate": 1e-4,
            "weight_decay": 1e-6,
            "gradient_clip_norm": 1.0,
            
            # GPU utilization optimizations
            "batch_size": 16,  # Will be auto-adjusted based on GPU
            "gradient_accumulation_steps": 2,
            "max_attention_sequence_length": 384,
            
            # Loss function settings
            "mel_loss_weight": 1.0,
            "kl_loss_weight": 1.0,
            "feature_loss_weight": 1.0,
            
            # Advanced training features
            "scheduler": "cosine_with_restarts",
            "warmup_steps": 1000,
            "use_adaptive_loss_weights": True,
            "use_label_smoothing": True,
            "label_smoothing_factor": 0.1,
            
            # Checkpoint management
            "checkpoint_interval": 1000,
            "validation_interval": 500,
            "early_stopping_patience": 10,
            
            # Memory management
            "max_memory_fraction": 0.75,
            "enable_memory_efficient_attention": True
        },
        
        "data": {
            # GPU utilization optimized data loading
            "num_workers": 8,  # Will be auto-adjusted
            "prefetch_buffer_size": 24,
            "shuffle_buffer_multiplier": 36,
            "prefetch_factor": 4,
            "persistent_workers": True,
            "pin_memory": True,
            "drop_last": True,
            "multiprocessing_context": "spawn",
            
            # Data processing optimizations
            "cache_data": True,
            "preload_data": True,
            "async_data_loading": True,
            "enable_memory_mapping": True,
            "cache_verification": True,
            
            # Audio processing
            "sample_rate": 22050,
            "hop_length": 256,
            "win_length": 1024,
            "n_fft": 1024,
            "n_mels": 80,
            "max_audio_length": 11.0,  # seconds
            "min_audio_length": 0.5,   # seconds
            
            # Text processing
            "max_text_length": 200,
            "min_text_length": 10,
            "enable_phonemes": True,
            "phoneme_language": "en",
            
            # Preprocessing
            "preprocessing_mode": "precompute",
            "enable_audio_normalization": True,
            "enable_text_normalization": True
        },
        
        "gpu_optimization": {
            # GPU utilization specific settings
            "enable_async_prefetch": True,
            "max_prefetch_batches": 8,
            "monitoring_interval": 50,
            "target_gpu_utilization": 85.0,
            "min_gpu_utilization_warning": 50.0,
            
            # Memory optimization
            "enable_memory_pool": True,
            "memory_pool_size": "auto",
            "enable_memory_defragmentation": True,
            "cleanup_interval": 100,
            
            # Performance monitoring
            "enable_performance_logging": True,
            "log_gpu_stats": True,
            "log_memory_stats": True,
            "log_timing_stats": True
        },
        
        "paths": {
            "checkpoint_dir": "./checkpoints_gpu_optimized",
            "log_dir": "./logs_gpu_optimized",
            "cache_dir": "./cache_gpu_optimized"
        },
        
        "logging": {
            "level": "INFO",
            "log_gpu_utilization": True,
            "log_memory_usage": True,
            "log_data_loading_time": True,
            "save_training_plots": True
        }
    }
    
    return config

def save_config_file(config_path: str = "config_gpu_utilization_optimized.yaml"):
    """ذخیره configuration file"""
    config = create_gpu_optimized_config()
    
    with open(config_path, 'w', encoding='utf-8') as f:
        # Add header comment
        f.write(f"""# GPU Utilization Optimized Configuration for MyXTTS
# =====================================================
# 
# این configuration برای رفع مسئله GPU utilization که بین 40% و 2% نوسان می‌کند طراحی شده است.
# 
# ویژگی‌های کلیدی:
# - Async data prefetching برای کاهش GPU idle time
# - Optimized DataLoader settings
# - Memory management بهینه‌شده
# - Real-time GPU monitoring
# - Enhanced batch processing
#
# استفاده:
# python3 train_main.py --config {config_path} --model-size tiny
#
# Generated automatically by gpu_utilization_config.py
# Date: {Path(__file__).stat().st_mtime}

""")
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
    
    print(f"✅ GPU utilization optimized config saved to: {config_path}")
    return config_path

def create_launch_script():
    """ایجاد launch script بهینه‌شده"""
    script_content = '''#!/bin/bash
# GPU Utilization Optimized Training Launch Script
# ===============================================

echo "🚀 Starting MyXTTS training with GPU utilization optimization..."

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA GPU not detected. This optimization requires CUDA."
    exit 1
fi

# Display GPU info
echo "📊 GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Set optimal environment variables
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Install required packages if needed
echo "📦 Checking required packages..."
pip install pynvml psutil --quiet

# Run training with GPU optimization
echo "🎯 Starting optimized training..."
python3 train_main.py \\
    --config config_gpu_utilization_optimized.yaml \\
    --model-size tiny \\
    --train-data ./data/train.csv \\
    --val-data ./data/val.csv \\
    --optimization-level enhanced \\
    --apply-fast-convergence \\
    --enable-evaluation \\
    --checkpoint-dir ./checkpoints_gpu_optimized \\
    --num-workers auto \\
    --batch-size auto \\
    --verbose

echo "✅ Training completed!"
'''
    
    with open("train_gpu_optimized.sh", "w") as f:
        f.write(script_content)
    
    # Make executable
    import os
    os.chmod("train_gpu_optimized.sh", 0o755)
    
    print("✅ Launch script saved to: train_gpu_optimized.sh")

if __name__ == "__main__":
    # Create optimized configuration
    config_path = save_config_file()
    
    # Create launch script
    create_launch_script()
    
    print(f"""
🎯 GPU Utilization Optimization Setup Complete!

Files created:
- {config_path} (optimized configuration)
- train_gpu_optimized.sh (launch script)

Usage:
1. Direct: python3 train_main.py --config {config_path} --model-size tiny
2. Script: ./train_gpu_optimized.sh

Key optimizations:
✅ Async data prefetching
✅ Optimized DataLoader settings  
✅ GPU memory management
✅ Real-time utilization monitoring
✅ Enhanced batch processing

This should resolve the GPU utilization fluctuation between 40% and 2%.
""")