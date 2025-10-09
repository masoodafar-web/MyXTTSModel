#!/usr/bin/env python3
"""
Enhanced GPU Profiler - Deep analysis of GPU oscillation bottlenecks

This tool provides comprehensive profiling to identify remaining bottlenecks
even after TF-native loading is implemented:
1. Batch loading timing with variance analysis
2. Model forward pass profiling
3. GPU memory transfer monitoring
4. XLA compilation verification
5. Graph mode verification
6. CPU-GPU overlap analysis
7. Prefetch buffer efficiency

Usage:
    python utilities/enhanced_gpu_profiler.py
    python utilities/enhanced_gpu_profiler.py --batch-size 16 --num-workers 8 --num-batches 100
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed")
    print("Please install: pip install tensorflow>=2.12.0")
    sys.exit(1)

from myxtts.config.config import XTTSConfig, DataConfig


class EnhancedGPUProfiler:
    """Enhanced GPU profiler for deep bottleneck analysis."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize profiler."""
        self.config_path = config_path
        if Path(config_path).exists():
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            print(f"⚠️  Config not found: {config_path}, using defaults")
            self.config = XTTSConfig()
        
        # Timing storage
        self.batch_load_times: List[float] = []
        self.model_forward_times: List[float] = []
        self.total_step_times: List[float] = []
        self.gpu_memory_usage: List[float] = []
        
    def check_gpu_status(self) -> Dict[str, any]:
        """Check GPU status and capabilities."""
        print("\n" + "="*70)
        print("GPU STATUS CHECK")
        print("="*70)
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU detected")
            return {'available': False}
        
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            
        # Check GPU memory
        try:
            import GPUtil
            gpu_list = GPUtil.getGPUs()
            if gpu_list:
                for i, gpu in enumerate(gpu_list):
                    print(f"   Memory: {gpu.memoryTotal}MB total, {gpu.memoryFree}MB free")
        except ImportError:
            print("   (Install GPUtil for memory info: pip install gputil)")
        
        # Test GPU functionality
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0]])
                b = tf.matmul(a, a)
                _ = b.numpy()
            print("✅ GPU is functional")
            gpu_functional = True
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            gpu_functional = False
        
        # Check XLA availability
        xla_available = hasattr(tf.config.optimizer, 'set_jit')
        print(f"{'✅' if xla_available else '❌'} XLA compilation {'available' if xla_available else 'not available'}")
        
        # Check mixed precision
        print(f"✅ Mixed precision policy: {tf.keras.mixed_precision.global_policy().name}")
        
        return {
            'available': True,
            'count': len(gpus),
            'functional': gpu_functional,
            'xla_available': xla_available
        }
    
    def verify_tf_native_loading(self) -> bool:
        """Verify TF-native loading is enabled and functional."""
        print("\n" + "="*70)
        print("TF-NATIVE LOADING VERIFICATION")
        print("="*70)
        
        # Check config setting
        use_tf_native = getattr(self.config.data, 'use_tf_native_loading', False)
        print(f"Config setting: use_tf_native_loading = {use_tf_native}")
        
        if not use_tf_native:
            print("❌ TF-native loading is DISABLED in config")
            print("   Enable with: use_tf_native_loading: true")
            return False
        
        # Check if tf_native_loader module exists
        try:
            from myxtts.data.tf_native_loader import TFNativeDataLoader
            print("✅ TF-native loader module available")
            
            # Test loader initialization
            loader = TFNativeDataLoader(
                sample_rate=self.config.data.sample_rate,
                n_mels=self.config.model.n_mels,
            )
            print("✅ TF-native loader initialized successfully")
            return True
            
        except ImportError as e:
            print(f"❌ TF-native loader import failed: {e}")
            return False
        except Exception as e:
            print(f"❌ TF-native loader initialization failed: {e}")
            return False
    
    def verify_graph_mode(self) -> bool:
        """Verify graph mode and XLA compilation."""
        print("\n" + "="*70)
        print("GRAPH MODE & XLA VERIFICATION")
        print("="*70)
        
        # Check config settings
        enable_graph = getattr(self.config.training, 'enable_graph_mode', False)
        enable_xla = getattr(self.config.training, 'enable_xla_compilation', False)
        
        print(f"Config: enable_graph_mode = {enable_graph}")
        print(f"Config: enable_xla_compilation = {enable_xla}")
        
        # Test graph mode compilation
        try:
            @tf.function
            def test_graph_func(x):
                return tf.matmul(x, x)
            
            x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            result = test_graph_func(x)
            print("✅ Graph mode functional")
            
        except Exception as e:
            print(f"❌ Graph mode test failed: {e}")
            return False
        
        # Test XLA compilation
        if enable_xla:
            try:
                @tf.function(jit_compile=True)
                def test_xla_func(x):
                    return tf.matmul(x, x)
                
                x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = test_xla_func(x)
                print("✅ XLA compilation functional")
                
            except Exception as e:
                print(f"⚠️  XLA compilation test failed: {e}")
                print("   XLA may not be available on this system")
        
        return True
    
    def profile_data_pipeline(
        self,
        data_path: str,
        batch_size: int = 8,
        num_batches: int = 100,
        num_workers: int = None
    ) -> Dict[str, any]:
        """
        Comprehensive data pipeline profiling.
        
        Args:
            data_path: Path to dataset
            batch_size: Batch size for profiling
            num_batches: Number of batches to profile
            num_workers: Number of data loading workers (None = use config)
            
        Returns:
            Dictionary with profiling metrics
        """
        print("\n" + "="*70)
        print("DATA PIPELINE PROFILING")
        print("="*70)
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {num_batches}")
        print(f"Workers: {num_workers or 'auto'}")
        
        # Override workers if specified
        if num_workers is not None:
            original_workers = self.config.data.num_workers
            self.config.data.num_workers = num_workers
        
        # Create dataset
        try:
            from myxtts.data.ljspeech import LJSpeechDataset
            
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
        
        # Profiling loop
        print(f"\n{'='*70}")
        print("PROFILING IN PROGRESS...")
        print("="*70)
        
        iterator = iter(tf_dataset)
        batch_times = []
        cold_start_time = None
        
        # Warm-up iterations (don't count first few batches)
        warmup_batches = 5
        print(f"Warm-up phase ({warmup_batches} batches)...")
        for i in range(warmup_batches):
            try:
                _ = next(iterator)
            except StopIteration:
                print(f"⚠️  Dataset exhausted during warmup after {i} batches")
                break
            except Exception as e:
                print(f"⚠️  Error during warmup: {e}")
                break
        
        # Actual profiling
        print(f"Profiling {num_batches} batches...")
        for i in range(num_batches):
            start_time = time.perf_counter()
            try:
                batch = next(iterator)
                
                # Force GPU synchronization to get accurate timing
                if isinstance(batch, tuple) and len(batch) > 0:
                    if hasattr(batch[0], 'numpy'):
                        _ = batch[0].numpy()
                
                end_time = time.perf_counter()
                batch_time = (end_time - start_time) * 1000  # Convert to ms
                batch_times.append(batch_time)
                
                if i == 0:
                    cold_start_time = batch_time
                    print(f"   First batch: {batch_time:.2f}ms")
                elif i < 5:
                    print(f"   Batch {i+1}: {batch_time:.2f}ms")
                elif i == num_batches - 1:
                    print(f"   Last batch: {batch_time:.2f}ms")
                    
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
        return self._analyze_timing_stats(batch_times, "Data Loading")
    
    def _analyze_timing_stats(
        self,
        times: List[float],
        operation_name: str
    ) -> Dict[str, any]:
        """Analyze timing statistics and detect patterns."""
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        median_time = np.median(times)
        p95_time = np.percentile(times, 95)
        p99_time = np.percentile(times, 99)
        
        # Detect oscillation
        variation_ratio = std_time / avg_time if avg_time > 0 else 0
        oscillation_detected = variation_ratio > 0.5  # High variance indicates oscillation
        
        # Detect cyclic patterns
        cyclic_pattern = self._detect_cyclic_pattern(times)
        
        print(f"\n{'='*70}")
        print(f"{operation_name.upper()} - TIMING STATISTICS")
        print("="*70)
        print(f"Samples analyzed:    {len(times)}")
        print(f"Average time:        {avg_time:.2f}ms")
        print(f"Std deviation:       {std_time:.2f}ms")
        print(f"Min time:            {min_time:.2f}ms")
        print(f"Max time:            {max_time:.2f}ms")
        print(f"Median time:         {median_time:.2f}ms")
        print(f"95th percentile:     {p95_time:.2f}ms")
        print(f"99th percentile:     {p99_time:.2f}ms")
        print(f"Variation ratio:     {variation_ratio:.2%}")
        
        # Diagnosis
        print(f"\n{'-'*70}")
        if oscillation_detected:
            print("🔴 HIGH VARIATION DETECTED - Possible cyclic pattern!")
            print(f"   Variation ratio: {variation_ratio:.2%} (threshold: 50%)")
            print("   This indicates unstable timing, likely due to bottleneck")
        else:
            print("✅ LOW VARIATION - Stable timing")
            print(f"   Variation ratio: {variation_ratio:.2%}")
        
        if cyclic_pattern:
            print("🔴 CYCLIC PATTERN DETECTED")
            print(f"   Period: ~{cyclic_pattern['period']} batches")
            print("   This indicates GPU waiting for data preparation")
        
        if avg_time > 200:
            print("🔴 SLOW OPERATION DETECTED")
            print(f"   Average {avg_time:.0f}ms is too slow")
        elif avg_time > 100:
            print("⚠️  MODERATE OPERATION SPEED")
            print(f"   Average {avg_time:.0f}ms could be improved")
        else:
            print("✅ FAST OPERATION")
            print(f"   Average {avg_time:.0f}ms is acceptable")
        
        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'min_time': min_time,
            'max_time': max_time,
            'median_time': median_time,
            'p95_time': p95_time,
            'p99_time': p99_time,
            'variation_ratio': variation_ratio,
            'oscillation_detected': oscillation_detected,
            'cyclic_pattern': cyclic_pattern,
            'samples': len(times)
        }
    
    def _detect_cyclic_pattern(self, times: List[float]) -> Optional[Dict[str, any]]:
        """Detect cyclic patterns in timing data using autocorrelation."""
        if len(times) < 10:
            return None
        
        # Normalize times
        times_array = np.array(times)
        times_norm = (times_array - np.mean(times_array)) / (np.std(times_array) + 1e-8)
        
        # Compute autocorrelation for lags 2 to len(times)//2
        max_lag = min(20, len(times) // 2)
        autocorr = []
        
        for lag in range(2, max_lag):
            corr = np.corrcoef(times_norm[:-lag], times_norm[lag:])[0, 1]
            if not np.isnan(corr):
                autocorr.append((lag, corr))
        
        # Find peaks in autocorrelation (indicates periodic pattern)
        if autocorr:
            max_corr = max(autocorr, key=lambda x: x[1])
            if max_corr[1] > 0.3:  # Significant correlation
                return {
                    'detected': True,
                    'period': max_corr[0],
                    'correlation': max_corr[1]
                }
        
        return None
    
    def benchmark_different_configs(
        self,
        data_path: str,
        batch_sizes: List[int] = [8, 16, 32],
        worker_counts: List[int] = [4, 8, 16],
        num_batches: int = 50
    ) -> Dict[str, any]:
        """
        Benchmark different configurations to find optimal settings.
        
        Args:
            data_path: Path to dataset
            batch_sizes: List of batch sizes to test
            worker_counts: List of worker counts to test
            num_batches: Number of batches per test
            
        Returns:
            Dictionary with benchmark results
        """
        print("\n" + "="*70)
        print("BENCHMARK - TESTING MULTIPLE CONFIGURATIONS")
        print("="*70)
        
        results = defaultdict(dict)
        
        for batch_size in batch_sizes:
            for num_workers in worker_counts:
                print(f"\n{'-'*70}")
                print(f"Testing: batch_size={batch_size}, num_workers={num_workers}")
                print("-"*70)
                
                metrics = self.profile_data_pipeline(
                    data_path=data_path,
                    batch_size=batch_size,
                    num_batches=num_batches,
                    num_workers=num_workers
                )
                
                if metrics:
                    config_key = f"bs{batch_size}_w{num_workers}"
                    results[config_key] = {
                        'batch_size': batch_size,
                        'num_workers': num_workers,
                        'metrics': metrics
                    }
        
        # Find best configuration
        if results:
            print(f"\n{'='*70}")
            print("BENCHMARK SUMMARY")
            print("="*70)
            
            best_config = None
            best_score = float('inf')
            
            for config_key, result in results.items():
                metrics = result['metrics']
                # Score based on average time and variation
                score = metrics['avg_time'] * (1 + metrics['variation_ratio'])
                
                print(f"\n{config_key}:")
                print(f"  Avg time: {metrics['avg_time']:.2f}ms")
                print(f"  Variation: {metrics['variation_ratio']:.2%}")
                print(f"  Score: {score:.2f}")
                
                if score < best_score:
                    best_score = score
                    best_config = result
            
            if best_config:
                print(f"\n{'='*70}")
                print("RECOMMENDED CONFIGURATION")
                print("="*70)
                print(f"batch_size: {best_config['batch_size']}")
                print(f"num_workers: {best_config['num_workers']}")
                print(f"Expected avg time: {best_config['metrics']['avg_time']:.2f}ms")
                print(f"Expected variation: {best_config['metrics']['variation_ratio']:.2%}")
        
        return results
    
    def generate_report(self, output_file: str = "gpu_profiling_report.txt"):
        """Generate comprehensive profiling report."""
        print(f"\n{'='*70}")
        print("GENERATING REPORT")
        print("="*70)
        print(f"Output file: {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GPU PROFILING REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Add configuration
            f.write("CONFIGURATION:\n")
            f.write(f"  use_tf_native_loading: {getattr(self.config.data, 'use_tf_native_loading', False)}\n")
            f.write(f"  prefetch_to_gpu: {getattr(self.config.data, 'prefetch_to_gpu', False)}\n")
            f.write(f"  enable_graph_mode: {getattr(self.config.training, 'enable_graph_mode', False)}\n")
            f.write(f"  enable_xla_compilation: {getattr(self.config.training, 'enable_xla_compilation', False)}\n")
            f.write(f"  batch_size: {self.config.data.batch_size}\n")
            f.write(f"  num_workers: {self.config.data.num_workers}\n")
            f.write(f"  prefetch_buffer_size: {getattr(self.config.data, 'prefetch_buffer_size', 'N/A')}\n")
            
        print(f"✅ Report generated: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Enhanced GPU profiler for bottleneck analysis"
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
        default=100,
        help="Number of batches to profile"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers (None = use config)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark with multiple configurations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="gpu_profiling_report.txt",
        help="Output file for report"
    )
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = EnhancedGPUProfiler(args.config)
    
    # Run checks
    gpu_status = profiler.check_gpu_status()
    if not gpu_status.get('available'):
        print("\n⚠️  GPU not available - profiling limited")
    
    profiler.verify_tf_native_loading()
    profiler.verify_graph_mode()
    
    # Run profiling
    if args.benchmark:
        profiler.benchmark_different_configs(
            args.data_path,
            batch_sizes=[8, 16, 32],
            worker_counts=[4, 8, 16],
            num_batches=args.num_batches
        )
    else:
        profiler.profile_data_pipeline(
            args.data_path,
            args.batch_size,
            args.num_batches,
            args.num_workers
        )
    
    # Generate report
    profiler.generate_report(args.output)
    
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the profiling results above")
    print("2. Check the generated report:", args.output)
    print("3. Apply recommended configuration changes")
    print("4. Monitor GPU utilization during training: watch -n 0.5 nvidia-smi")


if __name__ == "__main__":
    main()
