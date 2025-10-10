#!/usr/bin/env python3
"""
Advanced Data Pipeline Bottleneck Analyzer

This script performs deep analysis of the data pipeline to identify remaining bottlenecks
that cause GPU oscillation even after TF-native loading is implemented.

Focus Areas:
1. Text preprocessing cache building time
2. Parallel processing efficiency
3. Audio loading with TF-native operations
4. Memory transfer patterns
5. CPU-GPU overlap efficiency

Usage:
    python utilities/analyze_data_pipeline_bottleneck.py
    python utilities/analyze_data_pipeline_bottleneck.py --batch-size 16 --num-samples 100
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
    print("Please install: pip install tensorflow>=2.12.0")
    sys.exit(1)

from myxtts.config.config import XTTSConfig
from myxtts.data.ljspeech import LJSpeechDataset


class DataPipelineBottleneckAnalyzer:
    """Analyze data pipeline for bottlenecks causing GPU oscillation."""
    
    def __init__(self, config_path: str = None):
        """Initialize analyzer with config."""
        if config_path and Path(config_path).exists():
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            # Use default config
            self.config = XTTSConfig()
            # Set default paths if not configured
            if not self.config.data.dataset_path:
                self.config.data.dataset_path = "./data/ljspeech"
        
        print(f"\n{'='*70}")
        print("DATA PIPELINE BOTTLENECK ANALYZER")
        print("="*70)
        print(f"Dataset path: {self.config.data.dataset_path}")
        print(f"TF-native loading: {self.config.data.use_tf_native_loading}")
        print(f"Num workers: {self.config.data.num_workers}")
        print(f"Batch size: {self.config.data.batch_size}")
        print("="*70 + "\n")
    
    def analyze_text_preprocessing(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Analyze text preprocessing performance.
        
        This measures the time taken to build the TF-native cache which includes:
        - Text tokenization
        - Language detection
        - Phone-level normalization
        """
        print(f"\n{'='*70}")
        print("ANALYSIS 1: Text Preprocessing Performance")
        print("="*70)
        
        try:
            # Create dataset (this triggers cache building)
            start_time = time.perf_counter()
            
            dataset = LJSpeechDataset(
                data_path=self.config.data.dataset_path,
                config=self.config.data,
                subset="train",
                download=False,
                preprocess=False
            )
            
            # Limit to num_samples for quick analysis
            actual_samples = min(num_samples, len(dataset))
            
            # Trigger cache building by creating TF dataset
            print(f"\nCreating TF dataset with {actual_samples} samples...")
            cache_start = time.perf_counter()
            
            tf_dataset = dataset.create_tf_dataset(
                batch_size=1,  # Single sample to isolate preprocessing time
                shuffle=False,
                repeat=False,
                prefetch=False
            )
            
            cache_time = time.perf_counter() - cache_start
            total_time = time.perf_counter() - start_time
            
            print(f"\n📊 Text Preprocessing Results:")
            print(f"  Total initialization time: {total_time:.2f}s")
            print(f"  TF dataset creation time: {cache_time:.2f}s")
            print(f"  Per-sample preprocessing: {(cache_time / actual_samples * 1000):.2f}ms")
            
            # Analyze if parallel processing helped
            if cache_time > 5.0:
                print(f"\n⚠️  WARNING: Cache building took {cache_time:.2f}s")
                print(f"  This suggests text preprocessing is a bottleneck")
                print(f"\nRecommendations:")
                print(f"  1. Verify parallel processing is enabled")
                print(f"  2. Check if disk cache is being used")
                print(f"  3. Increase num_workers in config")
            else:
                print(f"\n✅ Text preprocessing is efficient")
            
            return {
                'total_time': total_time,
                'cache_time': cache_time,
                'per_sample_ms': cache_time / actual_samples * 1000
            }
            
        except Exception as e:
            print(f"\n❌ ERROR during text preprocessing analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_audio_loading(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Analyze audio loading performance with TF-native operations.
        
        This measures:
        - Time per batch for audio loading
        - Variance in loading times (indicates CPU bottleneck)
        - TF-native vs numpy loading comparison
        """
        print(f"\n{'='*70}")
        print("ANALYSIS 2: Audio Loading Performance")
        print("="*70)
        
        try:
            dataset = LJSpeechDataset(
                data_path=self.config.data.dataset_path,
                config=self.config.data,
                subset="train",
                download=False,
                preprocess=False
            )
            
            batch_size = self.config.data.batch_size
            
            # Create dataset with TF-native loading
            print(f"\nCreating dataset with batch_size={batch_size}...")
            tf_dataset = dataset.create_tf_dataset(
                batch_size=batch_size,
                shuffle=False,
                repeat=False,
                prefetch=True
            )
            
            # Measure batch loading times
            print(f"Measuring {num_batches} batch loading times...")
            batch_times = []
            
            for i, batch in enumerate(tf_dataset.take(num_batches)):
                batch_start = time.perf_counter()
                
                # Force materialization
                _ = batch[0].numpy()
                _ = batch[1].numpy()
                
                batch_time = time.perf_counter() - batch_start
                batch_times.append(batch_time)
                
                if i % max(1, num_batches // 5) == 0:
                    print(f"  Batch {i+1}/{num_batches}: {batch_time*1000:.1f}ms")
            
            # Calculate statistics
            mean_time = np.mean(batch_times)
            std_time = np.std(batch_times)
            min_time = np.min(batch_times)
            max_time = np.max(batch_times)
            cv = std_time / mean_time  # Coefficient of variation
            
            print(f"\n📊 Audio Loading Results:")
            print(f"  Mean batch time: {mean_time*1000:.1f}ms")
            print(f"  Std deviation: {std_time*1000:.1f}ms")
            print(f"  Min/Max: {min_time*1000:.1f}ms / {max_time*1000:.1f}ms")
            print(f"  Coefficient of variation: {cv:.3f}")
            
            # Diagnose issues
            if cv > 0.3:
                print(f"\n⚠️  HIGH VARIANCE detected (CV={cv:.3f})")
                print(f"  This indicates inconsistent loading performance")
                print(f"  Possible causes:")
                print(f"    • CPU bottleneck in data loading")
                print(f"    • I/O contention (disk speed)")
                print(f"    • Insufficient prefetching")
                print(f"\nRecommendations:")
                print(f"  1. Increase prefetch_buffer_size")
                print(f"  2. Verify TF-native loading is actually being used")
                print(f"  3. Check disk I/O performance")
                print(f"  4. Consider using SSD for dataset")
            else:
                print(f"\n✅ Consistent batch loading performance")
            
            return {
                'mean_batch_ms': mean_time * 1000,
                'std_batch_ms': std_time * 1000,
                'cv': cv,
                'min_batch_ms': min_time * 1000,
                'max_batch_ms': max_time * 1000
            }
            
        except Exception as e:
            print(f"\n❌ ERROR during audio loading analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def analyze_pipeline_efficiency(self, num_batches: int = 20) -> Dict[str, float]:
        """
        Analyze overall pipeline efficiency including prefetching.
        
        This measures:
        - Time between batches (wait time)
        - GPU idle time estimation
        - Prefetch buffer effectiveness
        """
        print(f"\n{'='*70}")
        print("ANALYSIS 3: Pipeline Efficiency")
        print("="*70)
        
        try:
            dataset = LJSpeechDataset(
                data_path=self.config.data.dataset_path,
                config=self.config.data,
                subset="train",
                download=False,
                preprocess=False
            )
            
            batch_size = self.config.data.batch_size
            
            # Create dataset with full pipeline optimizations
            print(f"\nCreating optimized pipeline...")
            tf_dataset = dataset.create_tf_dataset(
                batch_size=batch_size,
                shuffle=True,
                repeat=True,
                prefetch=True
            )
            
            # Measure inter-batch times
            print(f"Measuring inter-batch times for {num_batches} batches...")
            inter_batch_times = []
            last_batch_end = None
            
            for i, batch in enumerate(tf_dataset.take(num_batches)):
                if last_batch_end is not None:
                    wait_time = time.perf_counter() - last_batch_end
                    inter_batch_times.append(wait_time)
                
                # Simulate GPU processing time (minimal)
                batch_start = time.perf_counter()
                _ = batch[0].numpy()
                batch_process_time = time.perf_counter() - batch_start
                
                last_batch_end = time.perf_counter()
                
                if i % max(1, num_batches // 5) == 0:
                    if inter_batch_times:
                        print(f"  Batch {i+1}: wait={inter_batch_times[-1]*1000:.1f}ms, "
                              f"process={batch_process_time*1000:.1f}ms")
            
            if not inter_batch_times:
                print("⚠️  Not enough batches to measure inter-batch times")
                return {}
            
            # Calculate statistics
            mean_wait = np.mean(inter_batch_times)
            std_wait = np.std(inter_batch_times)
            max_wait = np.max(inter_batch_times)
            
            print(f"\n📊 Pipeline Efficiency Results:")
            print(f"  Mean wait time: {mean_wait*1000:.1f}ms")
            print(f"  Std deviation: {std_wait*1000:.1f}ms")
            print(f"  Max wait time: {max_wait*1000:.1f}ms")
            
            # Estimate GPU idle percentage
            # Assuming ~50ms per batch for actual GPU training
            assumed_gpu_time = 0.050
            idle_percentage = (mean_wait / (mean_wait + assumed_gpu_time)) * 100
            
            print(f"  Estimated GPU idle: {idle_percentage:.1f}%")
            
            if mean_wait > 0.010:  # More than 10ms wait
                print(f"\n⚠️  PIPELINE BOTTLENECK detected")
                print(f"  GPU is waiting {mean_wait*1000:.1f}ms between batches")
                print(f"  This causes GPU oscillation and low utilization")
                print(f"\nRecommendations:")
                print(f"  1. Increase prefetch_buffer_size (current: {self.config.data.prefetch_buffer_size})")
                print(f"  2. Increase num_workers (current: {self.config.data.num_workers})")
                print(f"  3. Verify TF-native loading is working correctly")
                print(f"  4. Check if text preprocessing cache is being used")
            else:
                print(f"\n✅ Pipeline is efficiently feeding GPU")
            
            return {
                'mean_wait_ms': mean_wait * 1000,
                'std_wait_ms': std_wait * 1000,
                'max_wait_ms': max_wait * 1000,
                'estimated_gpu_idle_pct': idle_percentage
            }
            
        except Exception as e:
            print(f"\n❌ ERROR during pipeline efficiency analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_report(self):
        """Generate comprehensive bottleneck analysis report."""
        print(f"\n{'='*70}")
        print("COMPREHENSIVE BOTTLENECK ANALYSIS")
        print("="*70)
        
        results = {}
        
        # Run all analyses
        results['text_preprocessing'] = self.analyze_text_preprocessing(num_samples=100)
        results['audio_loading'] = self.analyze_audio_loading(num_batches=10)
        results['pipeline_efficiency'] = self.analyze_pipeline_efficiency(num_batches=20)
        
        # Generate summary
        print(f"\n{'='*70}")
        print("SUMMARY AND RECOMMENDATIONS")
        print("="*70)
        
        critical_issues = []
        
        # Check text preprocessing
        if results['text_preprocessing'].get('cache_time', 0) > 5.0:
            critical_issues.append("Text preprocessing is slow (>5s)")
        
        # Check audio loading variance
        if results['audio_loading'].get('cv', 0) > 0.3:
            critical_issues.append("High variance in audio loading times")
        
        # Check pipeline wait times
        if results['pipeline_efficiency'].get('mean_wait_ms', 0) > 10:
            critical_issues.append("High wait times between batches")
        
        if critical_issues:
            print("\n🔴 CRITICAL ISSUES FOUND:")
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")
            
            print("\n📋 ACTION ITEMS:")
            print("  1. Verify TF-native loading is enabled and working:")
            print("     • Check config: use_tf_native_loading: true")
            print("     • Check logs for TF-native loading success message")
            print("  2. Optimize parallel processing:")
            print("     • Increase num_workers in config (try 16-32)")
            print("     • Verify disk cache is being created and used")
            print("  3. Increase prefetching:")
            print("     • Increase prefetch_buffer_size (try 16-32)")
            print("     • Enable prefetch_to_gpu if not already enabled")
            print("  4. Hardware optimization:")
            print("     • Use SSD instead of HDD for dataset")
            print("     • Check disk I/O with: iostat -x 1")
        else:
            print("\n✅ No critical bottlenecks detected")
            print("   Data pipeline appears to be well optimized")
        
        print("\n" + "="*70)
        print("Analysis complete!")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze data pipeline for bottlenecks"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    analyzer = DataPipelineBottleneckAnalyzer(config_path=args.config)
    analyzer.generate_report()


if __name__ == "__main__":
    main()
