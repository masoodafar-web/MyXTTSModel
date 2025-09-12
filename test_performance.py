#!/usr/bin/env python3
"""
Performance Test Script for MyXTTS Data Loading Optimizations

This script tests and demonstrates the performance improvements made to
address CPU bottlenecks in the data loading pipeline.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig, DataConfig
from myxtts.data.ljspeech import LJSpeechDataset
from myxtts.utils.performance import PerformanceMonitor, start_performance_monitoring, stop_performance_monitoring, print_performance_report
import tensorflow as tf
import numpy as np


def create_test_dataset(data_path: str, num_samples: int = 100) -> Path:
    """
    Create a small test dataset for performance testing.
    
    Args:
        data_path: Path where to create the test dataset
        num_samples: Number of test samples to create
        
    Returns:
        Path to the created dataset
    """
    test_dir = Path(data_path)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata file
    metadata_file = test_dir / "metadata.csv"
    
    if not metadata_file.exists():
        print(f"Creating test dataset with {num_samples} samples...")
        
        # Create wavs directory
        wavs_dir = test_dir / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        # Generate test metadata and dummy audio files
        with open(metadata_file, 'w') as f:
            for i in range(num_samples):
                sample_id = f"test_{i:04d}"
                text = f"This is test sample number {i}"
                normalized_text = text.lower()
                
                # Write metadata
                f.write(f"{sample_id}|{text}|{normalized_text}\n")
                
                # Create dummy audio file (short sine wave instead of silence)
                audio_file = wavs_dir / f"{sample_id}.wav"
                if not audio_file.exists():
                    # Create 1 second of sine wave at 22050 Hz to avoid silence issues
                    import soundfile as sf
                    t = np.linspace(0, 1, 22050, dtype=np.float32)
                    # Generate a sine wave with frequency varying by sample number
                    freq = 440 + (i % 100) * 2  # Vary frequency from 440-640 Hz
                    dummy_audio = 0.1 * np.sin(2 * np.pi * freq * t)
                    sf.write(audio_file, dummy_audio, 22050)
        
        print(f"Test dataset created at {test_dir}")
    
    return test_dir


def test_data_loading_performance(dataset_path: str, batch_size: int = 16, num_batches: int = 50):
    """
    Test data loading performance with and without optimizations.
    
    Args:
        dataset_path: Path to test dataset
        batch_size: Batch size for testing
        num_batches: Number of batches to process
    """
    print("=" * 60)
    print("MyXTTS Data Loading Performance Test")
    print("=" * 60)
    
    # Create optimized configuration
    config = DataConfig(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=4,
        prefetch_buffer_size=4,
        shuffle_buffer_multiplier=10,
        enable_memory_mapping=True,
        cache_verification=True
    )
    
    print(f"Configuration:")
    print(f"  Dataset path: {config.dataset_path}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Workers: {config.num_workers}")
    print(f"  Memory mapping: {config.enable_memory_mapping}")
    print()
    
    # Start performance monitoring
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # Create dataset
        print("Creating dataset...")
        dataset = LJSpeechDataset(
            data_path=dataset_path,
            config=config,
            subset="train",
            download=False
        )
        
        print(f"Dataset loaded: {len(dataset)} samples")
        
        # Test preprocessing and caching
        print("\nTesting cache generation...")
        cache_start = time.perf_counter()
        
        # Generate caches with parallel processing
        dataset.precompute_mels(num_workers=config.num_workers, overwrite=False)
        dataset.precompute_tokens(num_workers=config.num_workers, overwrite=False)
        
        # Verify and fix caches
        cache_report = dataset.verify_and_fix_cache(fix=True)
        print(f"Cache verification: {cache_report}")
        
        # Filter to valid cached items
        valid_items = dataset.filter_items_by_cache()
        print(f"Valid cached items: {valid_items}")
        
        cache_end = time.perf_counter()
        print(f"Cache generation time: {cache_end - cache_start:.2f}s")
        
        # Test TensorFlow dataset creation with optimizations
        print("\nTesting optimized TensorFlow dataset...")
        tf_dataset = dataset.create_tf_dataset(
            batch_size=batch_size,
            shuffle=True,
            repeat=False,
            prefetch=True,
            use_cache_files=True,
            memory_cache=False,
            num_parallel_calls=config.num_workers,
            buffer_size_multiplier=config.shuffle_buffer_multiplier
        )
        
        # Test data loading speed
        print(f"\nTesting data loading speed ({num_batches} batches)...")
        data_loading_times = []
        
        for i, batch in enumerate(tf_dataset.take(num_batches)):
            batch_start = time.perf_counter()
            
            # Force evaluation by accessing the data
            text_seq, mel_spec, text_len, mel_len = batch
            _ = text_seq.numpy()
            _ = mel_spec.numpy()
            
            batch_end = time.perf_counter()
            batch_time = batch_end - batch_start
            data_loading_times.append(batch_time)
            
            if i % 10 == 0:
                print(f"  Batch {i+1}/{num_batches}: {batch_time*1000:.1f}ms")
        
        # Calculate statistics
        avg_time = np.mean(data_loading_times)
        std_time = np.std(data_loading_times)
        min_time = np.min(data_loading_times)
        max_time = np.max(data_loading_times)
        samples_per_second = batch_size / avg_time
        
        print(f"\nData Loading Performance:")
        print(f"  Average time per batch: {avg_time*1000:.1f}ms ±{std_time*1000:.1f}ms")
        print(f"  Min/Max time: {min_time*1000:.1f}ms / {max_time*1000:.1f}ms")
        print(f"  Samples per second: {samples_per_second:.1f}")
        print(f"  Throughput: {samples_per_second * 60:.0f} samples/minute")
        
        # Get performance report
        print("\nData Loading Profiler Report:")
        print(dataset.get_performance_report())
        
    finally:
        # Stop monitoring and get final report
        monitor.stop_monitoring()
        print("\nSystem Performance Report:")
        print(monitor.get_summary_report())


def test_bottleneck_detection(dataset_path: str):
    """Test the bottleneck detection capabilities."""
    print("\n" + "=" * 60)
    print("Bottleneck Detection Test")
    print("=" * 60)
    
    config = DataConfig(dataset_path=dataset_path, batch_size=8, num_workers=2)
    dataset = LJSpeechDataset(data_path=dataset_path, config=config, subset="train", download=False)
    
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    # Simulate data loading with timing
    tf_dataset = dataset.create_tf_dataset(batch_size=8, shuffle=False, repeat=False)
    
    for i, batch in enumerate(tf_dataset.take(20)):
        # Simulate data loading time
        with monitor.time_operation("data_loading"):
            time.sleep(0.1)  # Simulate slow data loading
        
        # Simulate model computation time
        with monitor.time_operation("model_compute"):
            time.sleep(0.05)  # Simulate faster computation
    
    monitor.stop_monitoring()
    
    # Get bottleneck analysis
    analysis = monitor.get_bottleneck_analysis()
    print("Bottleneck Analysis:")
    print(f"  Detected bottlenecks: {analysis.get('bottlenecks', [])}")
    print(f"  Recommendations:")
    for i, rec in enumerate(analysis.get('recommendations', []), 1):
        print(f"    {i}. {rec}")


def main():
    """Main function for the performance test."""
    parser = argparse.ArgumentParser(description="Test MyXTTS data loading performance optimizations")
    parser.add_argument("--test-data-path", default="/tmp/myxtts_test_data", 
                       help="Path for test dataset")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for testing")
    parser.add_argument("--num-batches", type=int, default=50, 
                       help="Number of batches to test")
    parser.add_argument("--create-test-data", action="store_true", 
                       help="Create test dataset")
    parser.add_argument("--test-bottleneck-detection", action="store_true",
                       help="Test bottleneck detection")
    
    args = parser.parse_args()
    
    # Create test dataset if requested
    if args.create_test_data:
        create_test_dataset(args.test_data_path, num_samples=200)
    
    # Test performance if dataset exists
    if Path(args.test_data_path).exists():
        test_data_loading_performance(
            args.test_data_path, 
            args.batch_size, 
            args.num_batches
        )
        
        if args.test_bottleneck_detection:
            test_bottleneck_detection(args.test_data_path)
    else:
        print(f"Test dataset not found at {args.test_data_path}")
        print("Use --create-test-data to create a test dataset first")


if __name__ == "__main__":
    main()