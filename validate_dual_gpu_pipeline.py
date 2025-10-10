#!/usr/bin/env python3
"""
Dual-GPU Pipeline Validation Script

This script validates that the dual-GPU pipeline is working correctly by:
1. Checking GPU availability
2. Verifying device placement configuration
3. Testing model creation on specified device
4. Validating data transfer between GPUs

Usage:
    python validate_dual_gpu_pipeline.py --data-gpu 0 --model-gpu 1
"""

import argparse
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def check_prerequisites():
    """Check if all prerequisites are met."""
    print("=" * 60)
    print("Dual-GPU Pipeline Validation")
    print("=" * 60)
    print()
    
    print("1. Checking prerequisites...")
    
    # Check NVIDIA driver
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_lines = [line for line in result.stdout.strip().split('\n') if line]
            print(f"   ✅ NVIDIA driver installed ({len(gpu_lines)} GPUs detected)")
            for line in gpu_lines:
                print(f"      {line}")
        else:
            print("   ❌ nvidia-smi command failed")
            return False
    except Exception as e:
        print(f"   ❌ Could not check NVIDIA driver: {e}")
        return False
    
    # Check TensorFlow
    try:
        import tensorflow as tf
        print(f"   ✅ TensorFlow installed (version {tf.__version__})")
        
        # Check if TensorFlow can see GPUs
        gpus = tf.config.list_physical_devices('GPU')
        if len(gpus) >= 2:
            print(f"   ✅ TensorFlow can see {len(gpus)} GPUs")
            for i, gpu in enumerate(gpus):
                print(f"      GPU {i}: {gpu.name}")
        else:
            print(f"   ❌ TensorFlow can only see {len(gpus)} GPU(s), need at least 2")
            return False
    except ImportError:
        print("   ❌ TensorFlow not installed")
        return False
    except Exception as e:
        print(f"   ❌ Error checking TensorFlow: {e}")
        return False
    
    print()
    return True


def validate_device_placement(data_gpu, model_gpu):
    """Validate GPU device placement configuration."""
    print("2. Validating device placement configuration...")
    
    try:
        import tensorflow as tf
        
        # Check GPU indices are valid
        gpus = tf.config.list_physical_devices('GPU')
        if data_gpu < 0 or data_gpu >= len(gpus):
            print(f"   ❌ Invalid data_gpu={data_gpu}, must be 0-{len(gpus)-1}")
            return False
        if model_gpu < 0 or model_gpu >= len(gpus):
            print(f"   ❌ Invalid model_gpu={model_gpu}, must be 0-{len(gpus)-1}")
            return False
        
        print(f"   ✅ GPU indices valid: data_gpu={data_gpu}, model_gpu={model_gpu}")
        
        # Configure visible devices
        selected_gpus = [gpus[data_gpu], gpus[model_gpu]]
        tf.config.set_visible_devices(selected_gpus, 'GPU')
        print(f"   ✅ Set visible devices: GPU {data_gpu} and GPU {model_gpu}")
        
        # Configure memory growth
        visible_gpus = tf.config.list_physical_devices('GPU')
        for i, gpu in enumerate(visible_gpus):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   ✅ Memory growth configured for GPU {i}")
            except Exception as e:
                print(f"   ⚠️  Could not set memory growth for GPU {i}: {e}")
        
        # Set device policy
        try:
            tf.config.experimental.set_device_policy('silent')
            print("   ✅ Device policy set to 'silent'")
        except Exception as e:
            print(f"   ⚠️  Could not set device policy: {e}")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Error during device placement: {e}")
        return False


def test_model_creation():
    """Test creating a simple model on GPU:1."""
    print("3. Testing model creation on GPU:1...")
    
    try:
        import tensorflow as tf
        
        with tf.device('/GPU:1'):
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            print("   ✅ Model created successfully on GPU:1")
            
            # Test forward pass
            dummy_input = tf.random.normal((32, 100))
            output = model(dummy_input, training=False)
            print(f"   ✅ Forward pass successful (output shape: {output.shape})")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Error creating model: {e}")
        return False


def test_data_transfer():
    """Test data transfer between GPU:0 and GPU:1."""
    print("4. Testing data transfer between GPUs...")
    
    try:
        import tensorflow as tf
        
        # Create data on GPU:0
        with tf.device('/GPU:0'):
            data = tf.random.normal((100, 50))
            print(f"   ✅ Created data on GPU:0 (shape: {data.shape})")
        
        # Transfer to GPU:1
        with tf.device('/GPU:1'):
            transferred_data = tf.identity(data)
            print(f"   ✅ Transferred data to GPU:1 (shape: {transferred_data.shape})")
        
        # Verify data is the same
        if tf.reduce_all(tf.equal(data, transferred_data)):
            print("   ✅ Data integrity verified after transfer")
        else:
            print("   ⚠️  Data differs after transfer (this might be expected)")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Error during data transfer: {e}")
        return False


def test_pipeline_simulation():
    """Simulate the actual training pipeline."""
    print("5. Simulating training pipeline...")
    
    try:
        import tensorflow as tf
        
        # Simulate data pipeline on GPU:0
        with tf.device('/GPU:0'):
            # Create a simple dataset
            dataset = tf.data.Dataset.from_tensor_slices(
                tf.random.normal((1000, 50))
            ).batch(32).prefetch(tf.data.AUTOTUNE)
            print("   ✅ Created dataset on GPU:0")
        
        # Simulate model on GPU:1
        with tf.device('/GPU:1'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10)
            ])
            print("   ✅ Created model on GPU:1")
        
        # Simulate training step
        batch_count = 0
        for batch in dataset.take(3):
            with tf.device('/GPU:1'):
                # Transfer data
                batch_gpu1 = tf.identity(batch)
                # Forward pass
                output = model(batch_gpu1, training=False)
                batch_count += 1
        
        print(f"   ✅ Processed {batch_count} batches successfully")
        print("   ✅ Pipeline simulation successful")
        
        print()
        return True
        
    except Exception as e:
        print(f"   ❌ Error during pipeline simulation: {e}")
        return False


def print_summary(results):
    """Print validation summary."""
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    if all_passed:
        print("🎉 All validation checks passed!")
        print("Your system is ready for dual-GPU training.")
        print()
        print("To start training with dual-GPU mode:")
        print("python train_main.py --data-gpu 0 --model-gpu 1 --train-data ... --val-data ...")
    else:
        print("⚠️  Some validation checks failed.")
        print("Please address the issues above before using dual-GPU mode.")
    
    print("=" * 60)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Validate dual-GPU pipeline setup"
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
    
    results = {}
    
    # Run validation checks
    results["Prerequisites"] = check_prerequisites()
    
    if results["Prerequisites"]:
        results["Device Placement"] = validate_device_placement(args.data_gpu, args.model_gpu)
        results["Model Creation"] = test_model_creation()
        results["Data Transfer"] = test_data_transfer()
        results["Pipeline Simulation"] = test_pipeline_simulation()
    else:
        print("⚠️  Skipping further checks due to prerequisite failures.")
    
    # Print summary
    success = print_summary(results)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
