#!/usr/bin/env python3
"""
Validation script for Memory-Isolated Dual-GPU Training.

This script checks if your system is ready for memory-isolated training
and provides recommendations.

Usage:
    python validate_memory_isolation.py --data-gpu 0 --model-gpu 1
"""

import sys
import os
import argparse

def check_prerequisites():
    """Check basic prerequisites."""
    print("\n1. Checking prerequisites...")
    
    # Check NVIDIA driver
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            gpu_count = sum(1 for line in lines if '|' in line and 'MiB' in line)
            print(f"   ✅ NVIDIA driver installed ({gpu_count} GPUs detected)")
            
            if gpu_count < 2:
                print(f"   ❌ Need at least 2 GPUs, found {gpu_count}")
                return False
        else:
            print("   ❌ NVIDIA driver not found")
            return False
    except FileNotFoundError:
        print("   ❌ nvidia-smi not found - NVIDIA driver may not be installed")
        return False
    
    # Check Python version
    if sys.version_info >= (3, 8):
        print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    else:
        print(f"   ⚠️  Python {sys.version_info.major}.{sys.version_info.minor} (3.8+ recommended)")
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow {tf.__version__}")
        
        # Check if TensorFlow can see GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) >= 2:
            print(f"   ✅ TensorFlow can see {len(gpus)} GPUs")
        else:
            print(f"   ❌ TensorFlow can only see {len(gpus)} GPU(s)")
            return False
    except ImportError:
        print("   ❌ TensorFlow not installed")
        return False
    
    # Check pynvml (optional but recommended)
    try:
        import pynvml
        print("   ✅ pynvml available (for detailed memory monitoring)")
    except ImportError:
        print("   ⚠️  pynvml not available (install with: pip install pynvml)")
        print("      Memory monitoring will have limited functionality")
    
    print()
    return True


def validate_gpu_indices(data_gpu, model_gpu):
    """Validate GPU indices."""
    print("2. Validating GPU indices...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if data_gpu < 0 or data_gpu >= len(gpus):
            print(f"   ❌ Invalid data_gpu={data_gpu}, must be 0-{len(gpus)-1}")
            return False
        
        if model_gpu < 0 or model_gpu >= len(gpus):
            print(f"   ❌ Invalid model_gpu={model_gpu}, must be 0-{len(gpus)-1}")
            return False
        
        if data_gpu == model_gpu:
            print(f"   ❌ data_gpu and model_gpu must be different")
            return False
        
        print(f"   ✅ GPU indices valid: data_gpu={data_gpu}, model_gpu={model_gpu}")
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Error validating GPU indices: {e}")
        return False


def check_gpu_memory():
    """Check GPU memory availability."""
    print("3. Checking GPU memory...")
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            
            total_gb = info.total / (1024 ** 3)
            free_gb = info.free / (1024 ** 3)
            
            print(f"   GPU {i}: {name}")
            print(f"      Total: {total_gb:.1f}GB")
            print(f"      Free: {free_gb:.1f}GB")
            
            # Recommendations
            if total_gb >= 20:
                print(f"      ✅ Excellent for memory-isolated training")
                print(f"      Recommended: --data-gpu-memory 8192 --model-gpu-memory 16384")
            elif total_gb >= 12:
                print(f"      ✅ Good for memory-isolated training")
                print(f"      Recommended: --data-gpu-memory 6144 --model-gpu-memory 10240")
            elif total_gb >= 8:
                print(f"      ⚠️  Limited memory - use smaller batch size")
                print(f"      Recommended: --data-gpu-memory 4096 --model-gpu-memory 6144")
            else:
                print(f"      ❌ Insufficient memory for memory-isolated training")
        
        print()
        return True
        
    except ImportError:
        print("   ⚠️  pynvml not available - cannot check detailed memory info")
        print("   Install with: pip install pynvml")
        print()
        return True
    except Exception as e:
        print(f"   ⚠️  Error checking GPU memory: {e}")
        print()
        return True


def test_memory_isolation(data_gpu, model_gpu):
    """Test memory isolation setup."""
    print("4. Testing memory isolation setup...")
    
    try:
        from myxtts.utils.gpu_memory import setup_gpu_memory_isolation
        
        success = setup_gpu_memory_isolation(
            data_gpu_id=data_gpu,
            model_gpu_id=model_gpu,
            data_gpu_memory_limit=2048,  # Small test allocation
            model_gpu_memory_limit=4096
        )
        
        if success:
            print("   ✅ Memory isolation setup successful")
            print()
            return True
        else:
            print("   ❌ Memory isolation setup failed")
            print()
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing memory isolation: {e}")
        print()
        return False


def test_trainer_import():
    """Test if memory-isolated trainer can be imported."""
    print("5. Testing trainer import...")
    
    try:
        from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
        print("   ✅ MemoryIsolatedDualGPUTrainer can be imported")
        print()
        return True
    except ImportError as e:
        print(f"   ❌ Cannot import MemoryIsolatedDualGPUTrainer: {e}")
        print()
        return False


def generate_command(data_gpu, model_gpu):
    """Generate recommended command."""
    print("6. Recommended command:")
    print()
    print("   python train_main.py \\")
    print(f"       --data-gpu {data_gpu} \\")
    print(f"       --model-gpu {model_gpu} \\")
    print("       --enable-memory-isolation \\")
    print("       --data-gpu-memory 8192 \\")
    print("       --model-gpu-memory 16384 \\")
    print("       --batch-size 32 \\")
    print("       --train-data ../dataset/dataset_train \\")
    print("       --val-data ../dataset/dataset_eval")
    print()


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate system for memory-isolated dual-GPU training"
    )
    parser.add_argument(
        "--data-gpu",
        type=int,
        default=0,
        help="GPU ID for data processing (default: 0)"
    )
    parser.add_argument(
        "--model-gpu",
        type=int,
        default=1,
        help="GPU ID for model training (default: 1)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Memory-Isolated Dual-GPU Training Validation")
    print("=" * 70)
    
    # Run validation steps
    checks = []
    
    checks.append(("Prerequisites", check_prerequisites()))
    checks.append(("GPU Indices", validate_gpu_indices(args.data_gpu, args.model_gpu)))
    checks.append(("GPU Memory", check_gpu_memory()))
    checks.append(("Memory Isolation", test_memory_isolation(args.data_gpu, args.model_gpu)))
    checks.append(("Trainer Import", test_trainer_import()))
    
    # Summary
    print("=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    all_passed = True
    for name, passed in checks:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("✅ System is ready for memory-isolated dual-GPU training!")
        print()
        generate_command(args.data_gpu, args.model_gpu)
        return 0
    else:
        print("❌ System is not ready. Please fix the issues above.")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
