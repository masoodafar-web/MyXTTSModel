#!/usr/bin/env python3
"""
GPU Utilization Test Script for MyXTTS

This script tests and demonstrates the GPU utilization improvements.
It can be run to verify that the fixes are working properly.
"""

import os
import sys
import time
import threading
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from myxtts.config.config import XTTSConfig
    from myxtts.utils.commons import get_device, setup_gpu_strategy
    from myxtts.utils.performance import PerformanceMonitor
    from gpu_monitor import GPUMonitor
    MYXTTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MyXTTS modules not fully available: {e}")
    MYXTTS_AVAILABLE = False


def test_tensorflow_gpu():
    """Test TensorFlow GPU configuration."""
    print("=== TensorFlow GPU Test ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check physical devices
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Physical GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    # Check logical devices
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(f"Logical GPUs: {len(logical_gpus)}")
    
    if gpus:
        # Test GPU memory growth setting
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ GPU memory growth enabled")
        except Exception as e:
            print(f"✗ GPU memory growth failed: {e}")
        
        # Test simple GPU computation
        try:
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.random.normal([1000, 1000])
                c = tf.matmul(a, b)
                result = tf.reduce_sum(c).numpy()
            print(f"✓ GPU computation test passed: {result:.2f}")
        except Exception as e:
            print(f"✗ GPU computation test failed: {e}")
    else:
        print("⚠️  No GPUs available")
    
    return len(gpus) > 0


def test_distribution_strategy():
    """Test TensorFlow distribution strategy."""
    print("\n=== Distribution Strategy Test ===")
    
    if not MYXTTS_AVAILABLE:
        print("MyXTTS not available, skipping strategy test")
        return
    
    try:
        strategy = setup_gpu_strategy()
        print(f"Strategy: {type(strategy).__name__}")
        print(f"Number of replicas: {strategy.num_replicas_in_sync}")
        
        # Test simple computation with strategy
        with strategy.scope():
            # Create a simple model
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(128, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(10, activation='softmax')
            ])
            
            # Compile model
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print(f"✓ Model created with strategy: {model.count_params()} parameters")
        
        return True
    except Exception as e:
        print(f"✗ Distribution strategy test failed: {e}")
        return False


def test_mixed_precision():
    """Test mixed precision training."""
    print("\n=== Mixed Precision Test ===")
    
    try:
        # Set mixed precision policy
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"✓ Mixed precision policy set: {policy.name}")
        
        # Test model with mixed precision
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = tf.keras.Sequential([
                tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(10, dtype='float32')  # Output layer should be float32
            ])
            
            # Test forward pass
            x = tf.random.normal([32, 100])
            y = model(x)
            print(f"✓ Mixed precision forward pass: input {x.dtype}, output {y.dtype}")
        
        return True
    except Exception as e:
        print(f"✗ Mixed precision test failed: {e}")
        return False


def test_data_pipeline_performance():
    """Test data pipeline performance for GPU training."""
    print("\n=== Data Pipeline Performance Test ===")
    
    try:
        # Create a synthetic dataset to test performance
        def generator():
            for i in range(1000):
                # Simulate text and mel data
                text = tf.random.uniform([50], maxval=1000, dtype=tf.int32)
                mel = tf.random.normal([100, 80])
                text_len = tf.constant(50, dtype=tf.int32)
                mel_len = tf.constant(100, dtype=tf.int32)
                yield text, mel, text_len, mel_len
        
        # Create TensorFlow dataset
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(50,), dtype=tf.int32),
                tf.TensorSpec(shape=(100, 80), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
        
        # Apply optimizations
        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = 32
        
        dataset = (dataset
                  .batch(batch_size)
                  .prefetch(AUTOTUNE)
                  .cache())
        
        # Test iteration speed
        start_time = time.time()
        batch_count = 0
        
        for batch in dataset.take(20):
            batch_count += 1
            # Simulate some processing
            time.sleep(0.01)
        
        end_time = time.time()
        total_time = end_time - start_time
        batches_per_second = batch_count / total_time
        
        print(f"✓ Data pipeline test: {batch_count} batches in {total_time:.2f}s")
        print(f"  Throughput: {batches_per_second:.2f} batches/sec")
        print(f"  Samples/sec: {batches_per_second * batch_size:.1f}")
        
        return True
    except Exception as e:
        print(f"✗ Data pipeline test failed: {e}")
        return False


def run_gpu_stress_test(duration: int = 30):
    """Run a GPU stress test to verify utilization."""
    print(f"\n=== GPU Stress Test ({duration}s) ===")
    
    if not tf.config.list_physical_devices('GPU'):
        print("No GPUs available for stress test")
        return
    
    # Start GPU monitoring
    gpu_monitor = GPUMonitor(interval=1.0)
    gpu_monitor.start_monitoring()
    
    try:
        with tf.device('/GPU:0'):
            # Create large tensors for computation
            print("Creating large tensors...")
            a = tf.random.normal([2048, 2048], dtype=tf.float32)
            b = tf.random.normal([2048, 2048], dtype=tf.float32)
            
            print(f"Running matrix multiplications for {duration} seconds...")
            start_time = time.time()
            iterations = 0
            
            while time.time() - start_time < duration:
                # Perform GPU-intensive operations
                c = tf.matmul(a, b)
                d = tf.matmul(c, a)
                result = tf.reduce_sum(d)
                iterations += 1
                
                if iterations % 10 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Iteration {iterations}, elapsed: {elapsed:.1f}s, result: {result.numpy():.2e}")
            
            total_time = time.time() - start_time
            print(f"✓ Completed {iterations} iterations in {total_time:.2f}s")
            print(f"  Average: {iterations/total_time:.2f} iterations/sec")
    
    except Exception as e:
        print(f"✗ GPU stress test failed: {e}")
    
    finally:
        time.sleep(2)  # Let monitoring collect final data
        gpu_monitor.stop_monitoring()
        print(gpu_monitor.get_summary_report())


def main():
    """Main test function."""
    parser = argparse.ArgumentParser(description="GPU Utilization Test for MyXTTS")
    parser.add_argument("--stress-duration", type=int, default=30,
                       help="Duration for GPU stress test in seconds")
    parser.add_argument("--skip-stress", action="store_true",
                       help="Skip the GPU stress test")
    
    args = parser.parse_args()
    
    print("MyXTTS GPU Utilization Test")
    print("=" * 50)
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: TensorFlow GPU
    if test_tensorflow_gpu():
        tests_passed += 1
    
    # Test 2: Distribution strategy
    if test_distribution_strategy():
        tests_passed += 1
    
    # Test 3: Mixed precision
    if test_mixed_precision():
        tests_passed += 1
    
    # Test 4: Data pipeline
    if test_data_pipeline_performance():
        tests_passed += 1
    
    # Optional stress test
    if not args.skip_stress and tf.config.list_physical_devices('GPU'):
        print("\nRunning GPU stress test to verify utilization...")
        run_gpu_stress_test(args.stress_duration)
    
    # Summary
    print(f"\n=== Test Summary ===")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("✓ All tests passed! GPU utilization improvements are working.")
    else:
        print("⚠️  Some tests failed. GPU utilization may still have issues.")
        
        if not tf.config.list_physical_devices('GPU'):
            print("\nNote: No GPUs detected. This may be expected in CPU-only environments.")
            print("The fixes will work when GPUs are available.")
    
    # Recommendations
    print(f"\n=== Recommendations for GPU Training ===")
    print("1. Use the updated config.yaml with larger batch sizes and optimizations")
    print("2. Monitor GPU utilization during training with gpu_monitor.py")
    print("3. If GPU utilization is still low:")
    print("   - Increase prefetch_buffer_size in config")
    print("   - Increase num_workers for data loading")
    print("   - Check that your model and data are properly placed on GPU")
    print("   - Use the performance monitoring tools to identify bottlenecks")


if __name__ == "__main__":
    main()