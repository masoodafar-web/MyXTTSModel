#!/usr/bin/env python3
"""
GPU Utilization Benchmark Script

This script compares data loading performance with and without optimizations
to demonstrate the improvement in GPU utilization and reduction in CPU bottlenecks.
"""

import os
import sys
import time
import psutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def benchmark_data_loading():
    """Benchmark data loading with different configurations."""
    try:
        import tensorflow as tf
        from myxtts.config.config import DataConfig
        from myxtts.data.ljspeech import LJSpeechDataset
    except ImportError:
        print("Required packages not available for benchmark")
        return
    
    print("=== GPU Utilization Benchmark ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print(f"CPU cores: {psutil.cpu_count()}")
    
    # Test configurations
    configs = {
        "Original (with bottlenecks)": DataConfig(
            dataset_path="./data/ljspeech",
            batch_size=16,
            num_workers=4,
            use_tf_native_loading=False,  # Disable optimizations
            enhanced_gpu_prefetch=False,
            optimize_cpu_gpu_overlap=False,
            prefetch_buffer_size=2,
            preprocessing_mode="auto"
        ),
        "Optimized (bottleneck-free)": DataConfig(
            dataset_path="./data/ljspeech",
            batch_size=32,  # Larger batch size possible with optimizations
            num_workers=8,  # More workers
            use_tf_native_loading=True,   # Enable TF-native loading
            enhanced_gpu_prefetch=True,   # Enhanced prefetching
            optimize_cpu_gpu_overlap=True, # CPU-GPU overlap
            prefetch_buffer_size=8,       # Larger buffer
            preprocessing_mode="precompute"
        )
    }
    
    results = {}
    
    for config_name, config in configs.items():
        print(f"\n--- Testing {config_name} ---")
        
        try:
            # Create dataset
            dataset = LJSpeechDataset(
                data_path="./data/ljspeech", 
                config=config, 
                subset="train",
                download=False,
                preprocess=False
            )
            
            # Create TensorFlow dataset
            tf_dataset = dataset.create_tf_dataset(
                batch_size=config.batch_size,
                shuffle=True,
                repeat=False,
                use_cache_files=(config.preprocessing_mode == "precompute"),
                num_parallel_calls=config.num_workers
            )
            
            # Benchmark iteration
            start_time = time.time()
            cpu_percent_start = psutil.cpu_percent()
            
            batch_count = 0
            max_batches = 10  # Test 10 batches
            
            for batch in tf_dataset.take(max_batches):
                batch_count += 1
                # Simulate some GPU work
                text_seq, mel_spec, text_len, mel_len = batch
                
                # Force computation to trigger data loading
                tf.reduce_mean(text_seq)
                tf.reduce_mean(mel_spec)
            
            end_time = time.time()
            cpu_percent_end = psutil.cpu_percent()
            
            total_time = end_time - start_time
            avg_time_per_batch = total_time / batch_count
            throughput = config.batch_size / avg_time_per_batch  # samples per second
            
            results[config_name] = {
                "avg_time_per_batch": avg_time_per_batch,
                "throughput": throughput,
                "total_time": total_time,
                "cpu_usage": (cpu_percent_start + cpu_percent_end) / 2,
                "batch_size": config.batch_size
            }
            
            print(f"  ✅ Completed {batch_count} batches")
            print(f"  ⏱️  Average time per batch: {avg_time_per_batch:.3f}s")
            print(f"  🚀 Throughput: {throughput:.1f} samples/sec")
            print(f"  💻 CPU usage: {results[config_name]['cpu_usage']:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results[config_name] = {"error": str(e)}
    
    # Compare results
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    if len(results) == 2:
        original_key = "Original (with bottlenecks)"
        optimized_key = "Optimized (bottleneck-free)"
        
        if "error" not in results[original_key] and "error" not in results[optimized_key]:
            original = results[original_key]
            optimized = results[optimized_key]
            
            speed_improvement = original["avg_time_per_batch"] / optimized["avg_time_per_batch"]
            throughput_improvement = optimized["throughput"] / original["throughput"]
            
            print(f"Speed improvement: {speed_improvement:.2f}x faster")
            print(f"Throughput improvement: {throughput_improvement:.2f}x higher")
            print(f"CPU usage change: {original['cpu_usage']:.1f}% → {optimized['cpu_usage']:.1f}%")
            
            if speed_improvement > 1.5:
                print("🎉 Significant performance improvement achieved!")
            elif speed_improvement > 1.2:
                print("✅ Good performance improvement")
            else:
                print("⚠️  Modest improvement - check if optimizations are working")
        
        else:
            print("❌ Could not complete comparison due to errors")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS FOR MAXIMUM GPU UTILIZATION")
    print("=" * 60)
    print("1. Use preprocessing_mode='precompute' to eliminate runtime CPU work")
    print("2. Enable use_tf_native_loading=True (eliminates Python function bottlenecks)")
    print("3. Increase batch_size as much as GPU memory allows")
    print("4. Set num_workers to 2x CPU cores for data loading")
    print("5. Use enhanced_gpu_prefetch=True for better CPU-GPU overlap")
    print("6. Set prefetch_buffer_size=8 or higher for sustained GPU utilization")
    print("\nExample configuration:")
    print("  python trainTestFile.py --config config_gpu_bottleneck_fix.yaml")


if __name__ == "__main__":
    benchmark_data_loading()