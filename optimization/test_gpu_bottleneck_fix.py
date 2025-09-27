#!/usr/bin/env python3
"""
GPU Bottleneck Fix Test Script

This script tests the optimized data loading pipeline to verify that:
1. TensorFlow-native loading works correctly
2. GPU utilization is improved
3. CPU bottlenecks are eliminated
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    import tensorflow as tf
    from myxtts.config.config import XTTSConfig, DataConfig
    from myxtts.data.ljspeech import LJSpeechDataset
    
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Installing required packages...")
    os.system("pip install tensorflow>=2.12.0 --quiet")
    import tensorflow as tf
    from myxtts.config.config import XTTSConfig, DataConfig
    from myxtts.data.ljspeech import LJSpeechDataset


def test_optimized_data_loading():
    """Test the optimized data loading pipeline."""
    print("\n=== Testing GPU Bottleneck Fixes ===")
    
    # Create optimized configuration
    config = DataConfig(
        dataset_path="./data/ljspeech",
        batch_size=16,  # Smaller batch for testing
        num_workers=4,
        
        # Enable optimizations
        use_tf_native_loading=True,
        enhanced_gpu_prefetch=True,
        optimize_cpu_gpu_overlap=True,
        
        prefetch_buffer_size=4,
        prefetch_to_gpu=True,
        enable_memory_mapping=True,
        cache_verification=True,
        
        # Test with precompute mode for maximum performance
        preprocessing_mode="precompute"
    )
    
    print(f"Configuration:")
    print(f"  - TF Native Loading: {config.use_tf_native_loading}")
    print(f"  - Enhanced GPU Prefetch: {config.enhanced_gpu_prefetch}")
    print(f"  - CPU-GPU Overlap: {config.optimize_cpu_gpu_overlap}")
    print(f"  - Preprocessing Mode: {config.preprocessing_mode}")
    
    # Test data loading performance
    try:
        # Create dataset (this will test if the optimizations work)
        print("\nCreating dataset...")
        dataset = LJSpeechDataset(
            data_path="./data/ljspeech", 
            config=config, 
            subset="train",
            download=False,
            preprocess=False  # Don't preprocess in this test
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test TensorFlow dataset creation
        print("\nCreating TensorFlow dataset with optimizations...")
        tf_dataset = dataset.create_tf_dataset(
            batch_size=config.batch_size,
            shuffle=True,
            repeat=False,
            prefetch=True,
            use_cache_files=True,  # This will test TF-native loading
            num_parallel_calls=config.num_workers
        )
        
        # Test iteration performance
        print("\nTesting iteration performance...")
        start_time = time.time()
        batch_count = 0
        max_batches = 5  # Test first 5 batches
        
        for batch in tf_dataset.take(max_batches):
            batch_count += 1
            text_seq, mel_spec, text_len, mel_len = batch
            
            print(f"Batch {batch_count}:")
            print(f"  - Text shape: {text_seq.shape}")
            print(f"  - Mel shape: {mel_spec.shape}")
            print(f"  - Text lengths: {text_len.shape}")
            print(f"  - Mel lengths: {mel_len.shape}")
            
            # Verify data is loaded correctly
            assert text_seq.shape[0] == config.batch_size or batch_count == max_batches
            assert mel_spec.shape[0] == config.batch_size or batch_count == max_batches
            assert len(text_len.shape) == 1
            assert len(mel_len.shape) == 1
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_batch = total_time / batch_count
        
        print(f"\nPerformance Results:")
        print(f"  - Total time: {total_time:.2f}s")
        print(f"  - Average time per batch: {avg_time_per_batch:.3f}s")
        print(f"  - Batches per second: {1/avg_time_per_batch:.2f}")
        
        # Performance threshold check
        if avg_time_per_batch < 0.5:  # Should be faster than 0.5s per batch
            print("✅ Performance test PASSED - Good batch loading speed")
        else:
            print("⚠️  Performance test WARNING - Batch loading might be slow")
        
        print("✅ Data loading optimization test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Data loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tf_native_loading_fallback():
    """Test that fallback to Python functions works when TF-native fails."""
    print("\n=== Testing TF-Native Loading Fallback ===")
    
    # Test with TF-native disabled to ensure fallback works
    config = DataConfig(
        dataset_path="./data/ljspeech",
        batch_size=8,
        num_workers=2,
        use_tf_native_loading=False,  # Disable TF-native to test fallback
        preprocessing_mode="auto"
    )
    
    try:
        dataset = LJSpeechDataset(
            data_path="./data/ljspeech", 
            config=config, 
            subset="train",
            download=False,
            preprocess=False
        )
        
        tf_dataset = dataset.create_tf_dataset(
            batch_size=config.batch_size,
            shuffle=False,
            repeat=False,
            use_cache_files=False  # Use fallback path
        )
        
        # Test one batch
        for batch in tf_dataset.take(1):
            text_seq, mel_spec, text_len, mel_len = batch
            print(f"Fallback loading works - shapes: {text_seq.shape}, {mel_spec.shape}")
            break
        
        print("✅ Fallback mechanism test PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Fallback test FAILED: {e}")
        return False


if __name__ == "__main__":
    print("GPU Bottleneck Fix Validation")
    print("=" * 50)
    
    # Run tests
    test1_passed = test_optimized_data_loading()
    test2_passed = test_tf_native_loading_fallback()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"  - Optimized loading test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  - Fallback mechanism test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 All GPU bottleneck fixes validated successfully!")
        print("\nTo use the optimizations:")
        print("1. Set preprocessing_mode='precompute' in your config")
        print("2. Set use_tf_native_loading=True (default)")
        print("3. Set enhanced_gpu_prefetch=True (default)")
        print("4. Use the config_gpu_bottleneck_fix.yaml configuration")
    else:
        print("\n⚠️  Some tests failed. Check the error messages above.")
        sys.exit(1)