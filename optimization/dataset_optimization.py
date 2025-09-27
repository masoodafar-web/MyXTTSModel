#!/usr/bin/env python3
"""
Dataset Preprocessing Analysis and Optimization Tool

This script analyzes the dataset preprocessing pipeline and provides optimizations
to ensure proper dataset preparation and improved training efficiency.

Usage:
    python dataset_optimization.py --data-path ./data/ljspeech --analyze
    python dataset_optimization.py --data-path ./data/ljspeech --optimize --mode precompute
"""

import os
import sys
import time
import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import psutil

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from myxtts.config.config import XTTSConfig, DataConfig
from myxtts.data.ljspeech import LJSpeechDataset
from myxtts.utils.commons import setup_logging


class DatasetOptimizer:
    """Dataset preprocessing analyzer and optimizer."""
    
    def __init__(self, data_path: str, config_path: Optional[str] = None):
        """
        Initialize dataset optimizer.
        
        Args:
            data_path: Path to dataset
            config_path: Optional path to config file
        """
        self.data_path = data_path
        self.logger = setup_logging()
        self.analysis_results = {}
        self.optimizations = []
        
        # Load or create configuration
        if config_path and os.path.exists(config_path):
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            self.config = XTTSConfig()
            self.config.data.dataset_path = data_path
    
    def analyze_dataset_structure(self) -> Dict[str, Any]:
        """Analyze dataset structure and identify potential issues."""
        self.logger.info("🔍 Analyzing Dataset Structure...")
        
        results = {}
        
        try:
            # Check if dataset exists
            if not os.path.exists(self.data_path):
                results['error'] = f"Dataset path does not exist: {self.data_path}"
                return results
            
            # Analyze directory structure
            path_obj = Path(self.data_path)
            
            # Look for common dataset files
            metadata_files = list(path_obj.glob("**/metadata*.csv")) + list(path_obj.glob("**/metadata*.txt"))
            wav_dirs = [p for p in path_obj.rglob("*") if p.is_dir() and any(p.glob("*.wav"))]
            wav_files = list(path_obj.rglob("*.wav"))
            
            results['metadata_files'] = [str(p) for p in metadata_files]
            results['wav_directories'] = [str(p) for p in wav_dirs]
            results['total_wav_files'] = len(wav_files)
            
            self.logger.info(f"Found {len(metadata_files)} metadata files")
            self.logger.info(f"Found {len(wav_dirs)} directories with WAV files")
            self.logger.info(f"Total WAV files: {len(wav_files)}")
            
            # Analyze WAV file sizes and distribution
            if wav_files:
                sizes = []
                durations = []
                
                # Sample a subset for analysis
                sample_files = wav_files[:min(100, len(wav_files))]
                
                for wav_path in tqdm(sample_files, desc="Analyzing WAV files"):
                    try:
                        # File size
                        size = wav_path.stat().st_size
                        sizes.append(size)
                        
                        # Audio duration (rough estimate from file size)
                        # Assuming 16-bit, 22050 Hz: ~44100 bytes per second
                        estimated_duration = size / (2 * self.config.data.sample_rate)
                        durations.append(estimated_duration)
                        
                    except Exception as e:
                        self.logger.warning(f"Could not analyze {wav_path}: {e}")
                
                if sizes:
                    results['avg_file_size'] = np.mean(sizes)
                    results['min_file_size'] = np.min(sizes)
                    results['max_file_size'] = np.max(sizes)
                    results['avg_duration'] = np.mean(durations)
                    results['min_duration'] = np.min(durations)
                    results['max_duration'] = np.max(durations)
                    
                    # Check for potential issues
                    if np.max(durations) > 10.0:  # Very long audio
                        self.optimizations.append("Some audio files are very long (>10s) - consider chunking")
                    
                    if np.min(durations) < 0.5:  # Very short audio
                        self.optimizations.append("Some audio files are very short (<0.5s) - may affect training")
                    
                    if np.std(durations) > 3.0:  # High variance in duration
                        self.optimizations.append("High variance in audio durations - consider dynamic batching")
            
            # Check metadata format
            if metadata_files:
                try:
                    metadata_path = metadata_files[0]
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        first_lines = [f.readline().strip() for _ in range(3)]
                    
                    results['metadata_sample'] = first_lines
                    
                    # Try to determine format
                    if '|' in first_lines[0]:
                        results['metadata_format'] = 'pipe_separated'
                    elif ',' in first_lines[0]:
                        results['metadata_format'] = 'comma_separated'
                    else:
                        results['metadata_format'] = 'unknown'
                    
                    # Count lines
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        line_count = sum(1 for _ in f)
                    results['metadata_line_count'] = line_count
                    
                    self.logger.info(f"Metadata format: {results.get('metadata_format', 'unknown')}")
                    self.logger.info(f"Metadata lines: {line_count}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not analyze metadata: {e}")
            
            return results
            
        except Exception as e:
            results['error'] = f"Dataset analysis failed: {str(e)}"
            return results
    
    def analyze_preprocessing_performance(self, mode: str = "runtime") -> Dict[str, Any]:
        """Analyze preprocessing performance for different modes."""
        self.logger.info(f"🔍 Analyzing Preprocessing Performance (mode: {mode})...")
        
        results = {}
        
        try:
            # Create dataset with specified mode
            original_mode = self.config.data.preprocessing_mode
            self.config.data.preprocessing_mode = mode
            self.config.data.batch_size = 8  # Small batch for testing
            
            dataset = LJSpeechDataset(
                data_path=self.data_path,
                config=self.config.data,
                subset="train",
                download=False,
                preprocess=(mode == "precompute")
            )
            
            # Get TensorFlow dataset
            tf_dataset = dataset.get_tf_dataset()
            
            # Measure loading performance
            batch_times = []
            memory_usage = []
            cpu_usage = []
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**2  # MB
            
            self.logger.info("Measuring data loading performance...")
            
            for i, batch in enumerate(tf_dataset.take(10)):
                start_time = time.time()
                
                # Force evaluation of batch
                if isinstance(batch, dict):
                    for key, tensor in batch.items():
                        if isinstance(tensor, tf.Tensor):
                            _ = tf.reduce_sum(tensor)
                
                batch_time = time.time() - start_time
                batch_times.append(batch_time)
                
                # Monitor system resources
                current_memory = process.memory_info().rss / 1024**2  # MB
                memory_usage.append(current_memory - initial_memory)
                cpu_usage.append(psutil.cpu_percent(interval=None))
                
                if i == 0:
                    # Log first batch details
                    if isinstance(batch, dict):
                        self.logger.info(f"First batch structure:")
                        for key, tensor in batch.items():
                            if isinstance(tensor, tf.Tensor):
                                self.logger.info(f"  {key}: {tensor.shape} {tensor.dtype}")
            
            # Compute statistics
            results['mode'] = mode
            results['avg_batch_time'] = np.mean(batch_times) if batch_times else 0
            results['min_batch_time'] = np.min(batch_times) if batch_times else 0
            results['max_batch_time'] = np.max(batch_times) if batch_times else 0
            results['std_batch_time'] = np.std(batch_times) if batch_times else 0
            results['avg_memory_usage'] = np.mean(memory_usage) if memory_usage else 0
            results['max_memory_usage'] = np.max(memory_usage) if memory_usage else 0
            results['avg_cpu_usage'] = np.mean(cpu_usage) if cpu_usage else 0
            
            self.logger.info(f"Average batch time: {results['avg_batch_time']:.3f}s")
            self.logger.info(f"Memory usage: {results['max_memory_usage']:.1f}MB")
            self.logger.info(f"CPU usage: {results['avg_cpu_usage']:.1f}%")
            
            # Performance recommendations
            if results['avg_batch_time'] > 0.5:
                self.optimizations.append(f"Slow data loading in {mode} mode ({results['avg_batch_time']:.3f}s/batch)")
                if mode == "runtime":
                    self.optimizations.append("Consider using 'precompute' mode for faster loading")
            
            if results['std_batch_time'] > 0.1:
                self.optimizations.append(f"High variance in batch loading times - indicates bottlenecks")
            
            if results['avg_cpu_usage'] > 80:
                self.optimizations.append(f"High CPU usage ({results['avg_cpu_usage']:.1f}%) - may indicate CPU bottleneck")
            
            # Restore original mode
            self.config.data.preprocessing_mode = original_mode
            
            return results
            
        except Exception as e:
            results['error'] = f"Preprocessing performance analysis failed: {str(e)}"
            self.logger.error(f"❌ Error: {e}")
            return results
    
    def analyze_cache_efficiency(self) -> Dict[str, Any]:
        """Analyze cache file efficiency and integrity."""
        self.logger.info("🔍 Analyzing Cache Efficiency...")
        
        results = {}
        
        try:
            # Look for cache directories
            cache_paths = []
            for root, dirs, files in os.walk(self.data_path):
                for dir_name in dirs:
                    if 'cache' in dir_name.lower() or 'precomputed' in dir_name.lower():
                        cache_paths.append(os.path.join(root, dir_name))
            
            results['cache_directories'] = cache_paths
            
            if not cache_paths:
                results['has_cache'] = False
                self.optimizations.append("No cache directories found - consider precomputing for faster loading")
                return results
            
            results['has_cache'] = True
            
            # Analyze each cache directory
            total_cache_size = 0
            cache_file_counts = {}
            
            for cache_path in cache_paths:
                cache_files = list(Path(cache_path).rglob("*"))
                cache_file_counts[cache_path] = len(cache_files)
                
                # Calculate cache size
                cache_size = 0
                for file_path in cache_files:
                    if file_path.is_file():
                        cache_size += file_path.stat().st_size
                
                total_cache_size += cache_size
                self.logger.info(f"Cache {cache_path}: {len(cache_files)} files, {cache_size / 1024**2:.1f}MB")
            
            results['total_cache_size_mb'] = total_cache_size / 1024**2
            results['cache_file_counts'] = cache_file_counts
            
            # Check cache freshness
            try:
                dataset = LJSpeechDataset(
                    data_path=self.data_path,
                    config=self.config.data,
                    subset="train",
                    download=False,
                    preprocess=False
                )
                
                # Check if dataset has cache validation
                if hasattr(dataset, 'verify_and_fix_cache'):
                    self.logger.info("Running cache verification...")
                    cache_report = dataset.verify_and_fix_cache(fix=False)
                    
                    if isinstance(cache_report, dict):
                        results['cache_report'] = cache_report
                        
                        if cache_report.get('corrupted_files', 0) > 0:
                            self.optimizations.append(f"Found {cache_report['corrupted_files']} corrupted cache files")
                        
                        if cache_report.get('missing_files', 0) > 0:
                            self.optimizations.append(f"Found {cache_report['missing_files']} missing cache files")
                        
                        cache_hit_rate = cache_report.get('cache_hit_rate', 0)
                        if cache_hit_rate < 0.9:
                            self.optimizations.append(f"Low cache hit rate ({cache_hit_rate:.1%}) - consider rebuilding cache")
                
            except Exception as e:
                self.logger.warning(f"Could not verify cache: {e}")
            
            # Cache size recommendations
            if total_cache_size > 10 * 1024**3:  # > 10GB
                self.optimizations.append("Large cache size - monitor disk space")
            
            return results
            
        except Exception as e:
            results['error'] = f"Cache analysis failed: {str(e)}"
            return results
    
    def optimize_configuration(self) -> XTTSConfig:
        """Generate optimized configuration based on analysis."""
        self.logger.info("🔧 Generating Optimized Configuration...")
        
        optimized_config = XTTSConfig()
        
        # Copy base configuration
        optimized_config.model = self.config.model
        optimized_config.data = self.config.data
        optimized_config.training = self.config.training
        
        # Apply optimizations based on analysis
        
        # Preprocessing optimization
        if hasattr(self, 'cache_results') and self.cache_results.get('has_cache'):
            optimized_config.data.preprocessing_mode = "precompute"
            self.logger.info("✅ Set preprocessing_mode to 'precompute' (cache available)")
        else:
            optimized_config.data.preprocessing_mode = "auto"
            self.logger.info("✅ Set preprocessing_mode to 'auto' (fallback to runtime)")
        
        # Data loading optimization
        cpu_count = psutil.cpu_count()
        optimized_config.data.num_workers = min(cpu_count, 16)  # Limit to prevent oversubscription
        optimized_config.data.prefetch_buffer_size = max(8, cpu_count // 2)
        
        # Memory optimization
        available_memory_gb = psutil.virtual_memory().available / 1024**3
        
        if available_memory_gb > 16:
            optimized_config.data.batch_size = 64
            optimized_config.data.enable_memory_mapping = True
        elif available_memory_gb > 8:
            optimized_config.data.batch_size = 48
            optimized_config.data.enable_memory_mapping = True
        else:
            optimized_config.data.batch_size = 32
            optimized_config.data.enable_memory_mapping = False
        
        # GPU optimization
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            optimized_config.data.pin_memory = True
            optimized_config.data.prefetch_to_gpu = True
            optimized_config.data.enable_xla = True
            optimized_config.data.mixed_precision = True
            self.logger.info("✅ Enabled GPU optimizations")
        else:
            optimized_config.data.pin_memory = False
            optimized_config.data.prefetch_to_gpu = False
            optimized_config.data.enable_xla = False
            optimized_config.data.mixed_precision = False
            self.logger.info("✅ Disabled GPU optimizations (CPU mode)")
        
        # Training stability optimization
        optimized_config.training.use_adaptive_loss_weights = True
        optimized_config.training.loss_smoothing_factor = 0.1
        optimized_config.training.use_label_smoothing = True
        optimized_config.training.use_huber_loss = True
        optimized_config.training.gradient_clip_norm = 1.0
        
        # Learning rate optimization
        optimized_config.training.use_warmup_cosine_schedule = True
        optimized_config.training.warmup_steps = 4000
        optimized_config.training.min_learning_rate = 1e-6
        
        self.logger.info("✅ Applied training stability optimizations")
        
        return optimized_config
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run comprehensive dataset analysis."""
        self.logger.info("🚀 Running Dataset Analysis...")
        
        analysis = {}
        
        # Structure analysis
        analysis['structure'] = self.analyze_dataset_structure()
        
        # Performance analysis for different modes
        analysis['performance_runtime'] = self.analyze_preprocessing_performance("runtime")
        analysis['performance_precompute'] = self.analyze_preprocessing_performance("precompute")
        
        # Cache analysis
        analysis['cache'] = self.analyze_cache_efficiency()
        self.cache_results = analysis['cache']  # Store for optimization
        
        return analysis
    
    def run_optimization(self, output_path: Optional[str] = None) -> XTTSConfig:
        """Run optimization and generate optimized configuration."""
        self.logger.info("🚀 Running Dataset Optimization...")
        
        # Run analysis first
        analysis = self.run_analysis()
        
        # Generate optimized configuration
        optimized_config = self.optimize_configuration()
        
        # Save optimized configuration if path provided
        if output_path:
            optimized_config.to_yaml(output_path)
            self.logger.info(f"✅ Saved optimized configuration to: {output_path}")
        
        return optimized_config
    
    def generate_report(self, analysis: Dict[str, Any]) -> None:
        """Generate analysis report."""
        self.logger.info("\n" + "="*80)
        self.logger.info("📋 DATASET ANALYSIS REPORT")
        self.logger.info("="*80)
        
        # Dataset structure
        structure = analysis.get('structure', {})
        if 'error' not in structure:
            self.logger.info(f"\n📁 Dataset Structure:")
            self.logger.info(f"  WAV files: {structure.get('total_wav_files', 0)}")
            self.logger.info(f"  Metadata files: {len(structure.get('metadata_files', []))}")
            
            if 'avg_duration' in structure:
                self.logger.info(f"  Average duration: {structure['avg_duration']:.2f}s")
                self.logger.info(f"  Duration range: {structure['min_duration']:.2f}s - {structure['max_duration']:.2f}s")
        
        # Performance comparison
        runtime_perf = analysis.get('performance_runtime', {})
        precompute_perf = analysis.get('performance_precompute', {})
        
        if 'error' not in runtime_perf and 'error' not in precompute_perf:
            self.logger.info(f"\n⚡ Performance Comparison:")
            self.logger.info(f"  Runtime mode: {runtime_perf.get('avg_batch_time', 0):.3f}s/batch")
            self.logger.info(f"  Precompute mode: {precompute_perf.get('avg_batch_time', 0):.3f}s/batch")
            
            if runtime_perf.get('avg_batch_time', 0) > precompute_perf.get('avg_batch_time', 0):
                speedup = runtime_perf.get('avg_batch_time', 0) / precompute_perf.get('avg_batch_time', 1)
                self.logger.info(f"  Precompute speedup: {speedup:.1f}x faster")
        
        # Cache status
        cache_info = analysis.get('cache', {})
        if 'error' not in cache_info:
            self.logger.info(f"\n💾 Cache Status:")
            self.logger.info(f"  Has cache: {cache_info.get('has_cache', False)}")
            if cache_info.get('has_cache'):
                self.logger.info(f"  Cache size: {cache_info.get('total_cache_size_mb', 0):.1f}MB")
        
        # Optimizations
        if self.optimizations:
            self.logger.info(f"\n💡 Optimizations ({len(self.optimizations)}):")
            for i, opt in enumerate(self.optimizations, 1):
                self.logger.info(f"  {i}. {opt}")
        else:
            self.logger.info(f"\n✅ No optimization recommendations - dataset is well configured!")
        
        self.logger.info("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dataset Preprocessing Analysis and Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-path", 
        required=True,
        help="Path to dataset directory"
    )
    
    parser.add_argument(
        "--config", 
        help="Path to configuration file (optional)"
    )
    
    parser.add_argument(
        "--analyze", 
        action="store_true",
        help="Run dataset analysis"
    )
    
    parser.add_argument(
        "--optimize", 
        action="store_true",
        help="Run optimization and generate optimized config"
    )
    
    parser.add_argument(
        "--output", 
        help="Output path for optimized configuration"
    )
    
    args = parser.parse_args()
    
    if not args.analyze and not args.optimize:
        print("Error: Must specify either --analyze or --optimize")
        sys.exit(1)
    
    # Create optimizer
    optimizer = DatasetOptimizer(args.data_path, args.config)
    
    try:
        if args.analyze:
            # Run analysis
            analysis = optimizer.run_analysis()
            optimizer.generate_report(analysis)
        
        if args.optimize:
            # Run optimization
            output_path = args.output or "optimized_config.yaml"
            optimized_config = optimizer.run_optimization(output_path)
            
            print(f"\n✅ Optimization complete!")
            print(f"Optimized configuration saved to: {output_path}")
            print("You can now use this configuration for improved training performance.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Operation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()