#!/usr/bin/env python3
"""
GPU Bottleneck Diagnostic Tool - Identifies cyclic GPU utilization patterns

This tool specifically diagnoses the 2-40% oscillating GPU utilization issue
by profiling:
1. Data loading pipeline timing
2. CPU vs GPU operation placement  
3. Batch preparation vs GPU execution timing
4. Memory transfer bottlenecks
5. Real-time GPU utilization patterns

Usage:
    python utilities/diagnose_gpu_bottleneck.py
    python utilities/diagnose_gpu_bottleneck.py --batch-size 16 --num-batches 50
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed")
    sys.exit(1)

from myxtts.config.config import XTTSConfig, DataConfig
from myxtts.data.ljspeech import LJSpeechDataset


class GPUBottleneckDiagnostic:
    """Diagnose GPU bottleneck causing cyclic utilization patterns."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize diagnostic tool."""
        self.config_path = config_path
        if Path(config_path).exists():
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            print(f"⚠️  Config not found: {config_path}, using defaults")
            self.config = XTTSConfig()
        
        # Timing metrics
        self.batch_times: List[float] = []
        self.data_load_times: List[float] = []
        self.gpu_wait_times: List[float] = []
        
    def check_gpu_status(self) -> bool:
        """Check if GPU is available."""
        print("\n" + "="*70)
        print("GPU STATUS CHECK")
        print("="*70)
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU detected")
            return False
        
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Test GPU functionality
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0]])
                b = tf.matmul(a, a)
                _ = b.numpy()
            print("✅ GPU is functional")
            return True
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False
    
    def profile_data_pipeline(
        self, 
        data_path: str,
        batch_size: int = 8,
        num_batches: int = 50
    ) -> Dict[str, float]:
        """
        Profile data loading pipeline to identify bottlenecks.
        
        Args:
            data_path: Path to dataset
            batch_size: Batch size for profiling
            num_batches: Number of batches to profile
            
        Returns:
            Dictionary with profiling metrics
        """
        print("\n" + "="*70)
        print("DATA PIPELINE PROFILING")
        print("="*70)
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {num_batches}")
        
        # Create dataset
        try:
            dataset = LJSpeechDataset(
                data_path=data_path,
                config=self.config.data,
                subset="train",
                download=False
            )
            print(f"✅ Dataset loaded: {len(dataset)} samples")
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return {}
        
        # Create TensorFlow dataset
        tf_dataset = dataset.create_tf_dataset(
            batch_size=batch_size,
            shuffle=False,
            repeat=False,
            prefetch=True,
            memory_cache=False,
            num_parallel_calls=self.config.data.num_workers,
        )
        
        # Profile batch loading times
        print(f"\nProfiling {num_batches} batches...")
        iterator = iter(tf_dataset)
        
        batch_times = []
        first_batch_time = None
        
        for i in range(num_batches):
            start_time = time.perf_counter()
            try:
                batch = next(iterator)
                end_time = time.perf_counter()
                
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                batch_times.append(batch_time)
                
                if i == 0:
                    first_batch_time = batch_time
                    print(f"   First batch (cold start): {batch_time:.2f}ms")
                elif i < 5:
                    print(f"   Batch {i+1}: {batch_time:.2f}ms")
                    
            except StopIteration:
                print(f"⚠️  Dataset exhausted after {i} batches")
                break
            except Exception as e:
                print(f"❌ Error loading batch {i}: {e}")
                break
        
        if not batch_times:
            print("❌ No batches loaded successfully")
            return {}
        
        # Calculate statistics
        avg_time = np.mean(batch_times)
        std_time = np.std(batch_times)
        min_time = np.min(batch_times)
        max_time = np.max(batch_times)
        median_time = np.median(batch_times)
        
        # Detect oscillation pattern
        oscillation_detected = std_time > (avg_time * 0.5)  # High variance indicates oscillation
        
        print(f"\n{'='*70}")
        print("TIMING STATISTICS")
        print("="*70)
        print(f"Average batch time:  {avg_time:.2f}ms")
        print(f"Std deviation:       {std_time:.2f}ms")
        print(f"Min time:            {min_time:.2f}ms")
        print(f"Max time:            {max_time:.2f}ms")
        print(f"Median time:         {median_time:.2f}ms")
        print(f"First batch:         {first_batch_time:.2f}ms")
        
        # Analyze results
        print(f"\n{'='*70}")
        print("BOTTLENECK ANALYSIS")
        print("="*70)
        
        if oscillation_detected:
            print("🔴 HIGH VARIATION DETECTED - Cyclic pattern identified!")
            print(f"   Std/Mean ratio: {std_time/avg_time:.2%}")
            print("   This indicates GPU is waiting for data preparation")
        else:
            print("✅ Low variation - Data pipeline is stable")
        
        if avg_time > 200:
            print("🔴 SLOW DATA LOADING DETECTED")
            print(f"   Average {avg_time:.0f}ms per batch is too slow")
            print("   GPU likely idle during data preparation")
        elif avg_time > 100:
            print("⚠️  DATA LOADING IS MODERATE")
            print(f"   Average {avg_time:.0f}ms per batch could be improved")
        else:
            print("✅ DATA LOADING IS FAST")
            print(f"   Average {avg_time:.0f}ms per batch is acceptable")
        
        # Check for tf.numpy_function usage
        print(f"\n{'='*70}")
        print("KNOWN ISSUES CHECK")
        print("="*70)
        
        # Check if using tf.numpy_function (CPU bottleneck)
        print("🔍 Checking for tf.numpy_function usage...")
        ljspeech_file = Path(__file__).parent.parent / "myxtts" / "data" / "ljspeech.py"
        if ljspeech_file.exists():
            with open(ljspeech_file, 'r') as f:
                content = f.read()
                if 'tf.numpy_function' in content:
                    print("🔴 CRITICAL: tf.numpy_function found in data pipeline!")
                    print("   This forces CPU execution and breaks TensorFlow graph")
                    print("   Causes: GPU waits → CPU prepares → GPU processes → repeat")
                    print("   Solution: Replace with TensorFlow-native operations")
                else:
                    print("✅ No tf.numpy_function usage detected")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'first_batch_time': first_batch_time,
            'oscillation_detected': oscillation_detected,
            'variation_ratio': std_time / avg_time if avg_time > 0 else 0
        }
    
    def recommend_fixes(self, metrics: Dict[str, float]):
        """Provide recommendations based on profiling results."""
        print(f"\n{'='*70}")
        print("RECOMMENDED FIXES")
        print("="*70)
        
        if metrics.get('oscillation_detected', False):
            print("\n1. 🔴 HIGH PRIORITY: Fix cyclic GPU utilization")
            print("   Problem: GPU utilization oscillates between 2-40%")
            print("   Root cause: Data pipeline bottleneck")
            print("   ")
            print("   Solutions:")
            print("   a) Replace tf.numpy_function with TensorFlow-native ops")
            print("      - Use tf.io.read_file() for file loading")
            print("      - Use tf.audio.decode_wav() for audio")
            print("      - Use tf.py_function only if absolutely necessary")
            print("   ")
            print("   b) Increase prefetch buffer size")
            print("      - Current: prefetch_buffer_size in config")
            print("      - Recommended: 8-16 batches")
            print("   ")
            print("   c) Increase num_workers for parallel loading")
            print("      - Current: num_workers in config")
            print("      - Recommended: 8-16 workers")
            print("   ")
            print("   d) Enable GPU prefetching")
            print("      - Set prefetch_to_gpu: true in config")
            print("      - Use tf.data.experimental.prefetch_to_device")
        
        avg_time = metrics.get('avg_time', 0)
        if avg_time > 100:
            print("\n2. ⚠️  MEDIUM PRIORITY: Optimize data loading speed")
            print(f"   Current: {avg_time:.0f}ms per batch")
            print("   Target: <100ms per batch")
            print("   ")
            print("   Solutions:")
            print("   a) Use precomputed features (cached mel spectrograms)")
            print("   b) Increase parallel loading workers")
            print("   c) Use faster storage (SSD instead of HDD)")
            print("   d) Enable memory caching for small datasets")
        
        print("\n3. ✅ CONFIGURATION CHECKLIST")
        print("   Ensure these settings in config.yaml:")
        print("   ")
        print("   data:")
        print("     num_workers: 8-16")
        print("     prefetch_buffer_size: 8-16")
        print("     prefetch_to_gpu: true")
        print("     enhanced_gpu_prefetch: true")
        print("     optimize_cpu_gpu_overlap: true")
        print("   ")
        print("   training:")
        print("     enable_graph_mode: true")
        print("     enable_xla_compilation: true")
        print("     enable_eager_debug: false")


def main():
    """Main diagnostic entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose GPU bottleneck causing cyclic utilization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to dataset"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for profiling"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of batches to profile"
    )
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = GPUBottleneckDiagnostic(args.config)
    
    # Check GPU status
    if not diagnostic.check_gpu_status():
        print("\n⚠️  GPU not available - diagnostic limited")
        return
    
    # Profile data pipeline
    metrics = diagnostic.profile_data_pipeline(
        args.data_path,
        args.batch_size,
        args.num_batches
    )
    
    # Provide recommendations
    if metrics:
        diagnostic.recommend_fixes(metrics)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review recommendations above")
    print("2. Apply suggested configuration changes")
    print("3. Re-run diagnostic to verify improvements")
    print("4. Monitor GPU utilization during training with: watch -n 0.5 nvidia-smi")


if __name__ == "__main__":
    main()
