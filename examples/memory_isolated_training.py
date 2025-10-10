#!/usr/bin/env python3
"""
Example: Memory-Isolated Dual-GPU Training

This example demonstrates how to use the memory-isolated trainer
for optimal dual-GPU training performance.

Usage:
    python examples/memory_isolated_training.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# IMPORTANT: Setup GPU memory isolation BEFORE importing TensorFlow
from myxtts.utils.gpu_memory import (
    setup_gpu_memory_isolation,
    get_optimal_memory_limits,
    get_gpu_memory_info
)

def main():
    """Main function demonstrating memory-isolated training."""
    
    print("=" * 70)
    print("Memory-Isolated Dual-GPU Training Example")
    print("=" * 70)
    
    # Step 1: Check GPU availability
    print("\n1. Checking GPU availability...")
    try:
        gpu_info = get_gpu_memory_info()
        if 'gpus' in gpu_info:
            print(f"   Found {len(gpu_info['gpus'])} GPU(s)")
            for gpu in gpu_info['gpus']:
                print(f"   GPU {gpu['gpu_id']}: {gpu['total_mb']}MB total")
        else:
            print("   ⚠️  Could not detect GPUs (pynvml not available)")
            print("   Continuing with default configuration...")
    except Exception as e:
        print(f"   ⚠️  Error checking GPUs: {e}")
    
    # Step 2: Calculate optimal memory limits
    print("\n2. Calculating optimal memory limits...")
    data_gpu_id = 0
    model_gpu_id = 1
    
    try:
        data_limit, model_limit = get_optimal_memory_limits(
            data_gpu_id=data_gpu_id,
            model_gpu_id=model_gpu_id,
            data_fraction=0.33,
            model_fraction=0.67
        )
        print(f"   Data GPU {data_gpu_id}: {data_limit}MB (recommended)")
        print(f"   Model GPU {model_gpu_id}: {model_limit}MB (recommended)")
    except Exception as e:
        print(f"   ⚠️  Could not calculate optimal limits: {e}")
        data_limit = 8192
        model_limit = 16384
        print(f"   Using defaults: {data_limit}MB (data), {model_limit}MB (model)")
    
    # Step 3: Setup memory isolation
    print("\n3. Setting up memory isolation...")
    success = setup_gpu_memory_isolation(
        data_gpu_id=data_gpu_id,
        model_gpu_id=model_gpu_id,
        data_gpu_memory_limit=data_limit,
        model_gpu_memory_limit=model_limit
    )
    
    if not success:
        print("   ❌ Failed to setup memory isolation")
        print("   This might be because:")
        print("      - You have less than 2 GPUs")
        print("      - TensorFlow was already initialized")
        print("   Please check the requirements and try again.")
        return
    
    # Step 4: Now we can import TensorFlow and create trainer
    print("\n4. Importing TensorFlow and creating configuration...")
    try:
        import tensorflow as tf
        from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
        from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
        
        print("   ✅ TensorFlow imported successfully")
        
        # Create configuration
        config = XTTSConfig(
            model=ModelConfig(
                text_encoder_dim=256,
                decoder_dim=512,
            ),
            data=DataConfig(
                batch_size=32,
                num_workers=8,
                data_gpu=data_gpu_id,
                model_gpu=model_gpu_id,
            ),
            training=TrainingConfig(
                learning_rate=8e-5,
                num_epochs=100,
            )
        )
        print("   ✅ Configuration created")
        
        # Step 5: Create memory-isolated trainer
        print("\n5. Creating memory-isolated trainer...")
        trainer = MemoryIsolatedDualGPUTrainer(
            config=config,
            data_gpu_id=data_gpu_id,
            model_gpu_id=model_gpu_id,
            data_gpu_memory_limit=data_limit,
            model_gpu_memory_limit=model_limit,
            enable_monitoring=True
        )
        print("   ✅ Trainer created successfully")
        
        # Step 6: Get memory stats
        print("\n6. Current memory status:")
        stats = trainer.get_memory_stats()
        print(f"   Data GPU {data_gpu_id} ({stats['data_gpu']['device']}):")
        print(f"      Limit: {stats['data_gpu']['limit_mb']}MB")
        if stats['data_gpu']['info']:
            info = stats['data_gpu']['info']
            print(f"      Used: {info['used_mb']}MB / {info['total_mb']}MB")
        
        print(f"   Model GPU {model_gpu_id} ({stats['model_gpu']['device']}):")
        print(f"      Limit: {stats['model_gpu']['limit_mb']}MB")
        if stats['model_gpu']['info']:
            info = stats['model_gpu']['info']
            print(f"      Used: {info['used_mb']}MB / {info['total_mb']}MB")
        
        # Step 7: Ready for training
        print("\n7. Ready for training!")
        print("   You can now call trainer.train(train_dataset, val_dataset)")
        print("\n   Example:")
        print("   >>> train_dataset = dataset.create_tf_dataset(...)")
        print("   >>> trainer.train(train_dataset, val_dataset, epochs=100)")
        
        print("\n" + "=" * 70)
        print("Memory-Isolated Trainer Setup Complete!")
        print("=" * 70)
        
        return trainer
        
    except ImportError as e:
        print(f"   ❌ Failed to import required modules: {e}")
        print("   Make sure TensorFlow and all dependencies are installed.")
        return None
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # Run the example
    trainer = main()
    
    if trainer:
        print("\n✅ Example completed successfully!")
        print("\nTo actually train a model, use:")
        print("   python train_main.py --enable-memory-isolation --data-gpu 0 --model-gpu 1 ...")
    else:
        print("\n❌ Example failed. Please check the error messages above.")
        sys.exit(1)
