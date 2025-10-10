#!/usr/bin/env python3
"""
GPU Optimization Configuration Validator

This script validates that all necessary optimizations for dual-GPU
pipeline are properly configured to achieve stable 80-95% GPU utilization.

Usage:
    python utilities/validate_gpu_optimization.py
    python utilities/validate_gpu_optimization.py --config configs/config.yaml
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from myxtts.config.config import XTTSConfig


class GPUOptimizationValidator:
    """Validate GPU optimization configuration."""
    
    def __init__(self, config_path: str = None):
        """Initialize validator with config."""
        if config_path and Path(config_path).exists():
            self.config = XTTSConfig.from_yaml(config_path)
            self.config_path = config_path
        else:
            # Use default config
            self.config = XTTSConfig()
            self.config_path = "default (no config file)"
        
        self.warnings = []
        self.errors = []
        self.recommendations = []
    
    def validate_tf_native_loading(self) -> bool:
        """Validate TF-native loading configuration."""
        print("\n" + "="*70)
        print("1. TF-NATIVE LOADING VALIDATION")
        print("="*70)
        
        use_tf_native = getattr(self.config.data, 'use_tf_native_loading', False)
        
        if not use_tf_native:
            self.errors.append("use_tf_native_loading is disabled")
            print("❌ CRITICAL: TF-native loading is DISABLED")
            print("   Current value: False")
            print("   Required value: True")
            print("\n   Impact: CPU bottleneck, GPU oscillation (2-40%)")
            print("   Fix: Set use_tf_native_loading: true in config.yaml")
            return False
        else:
            print("✅ TF-native loading is enabled")
            print("   This eliminates CPU bottleneck from tf.numpy_function")
            return True
    
    def validate_parallel_processing(self) -> bool:
        """Validate parallel processing configuration."""
        print("\n" + "="*70)
        print("2. PARALLEL PROCESSING VALIDATION")
        print("="*70)
        
        num_workers = getattr(self.config.data, 'num_workers', 1)
        
        print(f"Number of workers: {num_workers}")
        
        if num_workers < 8:
            self.warnings.append(f"Low num_workers ({num_workers})")
            print(f"⚠️  WARNING: num_workers is low ({num_workers})")
            print(f"   Recommended: 16-32 for optimal performance")
            print(f"   Impact: Slow text preprocessing cache building")
            self.recommendations.append("Increase num_workers to 16-32")
            return False
        elif num_workers < 16:
            print(f"⚠️  num_workers is moderate ({num_workers})")
            print(f"   Good for most systems, consider increasing to 16-32")
            self.recommendations.append("Consider increasing num_workers to 16-32")
            return True
        else:
            print(f"✅ num_workers is optimal ({num_workers})")
            return True
    
    def validate_prefetching(self) -> bool:
        """Validate prefetching configuration."""
        print("\n" + "="*70)
        print("3. PREFETCHING VALIDATION")
        print("="*70)
        
        all_good = True
        
        # Check prefetch buffer size
        prefetch_buffer = getattr(self.config.data, 'prefetch_buffer_size', 1)
        print(f"Prefetch buffer size: {prefetch_buffer}")
        
        if prefetch_buffer < 8:
            self.warnings.append(f"Low prefetch_buffer_size ({prefetch_buffer})")
            print(f"⚠️  WARNING: prefetch_buffer_size is low ({prefetch_buffer})")
            print(f"   Recommended: 16-32 for smooth GPU feeding")
            self.recommendations.append("Increase prefetch_buffer_size to 16-32")
            all_good = False
        else:
            print(f"✅ prefetch_buffer_size is good")
        
        # Check prefetch to GPU
        prefetch_to_gpu = getattr(self.config.data, 'prefetch_to_gpu', False)
        print(f"\nPrefetch to GPU: {prefetch_to_gpu}")
        
        if not prefetch_to_gpu:
            self.warnings.append("prefetch_to_gpu is disabled")
            print(f"⚠️  WARNING: prefetch_to_gpu is disabled")
            print(f"   Recommended: Enable for better CPU-GPU overlap")
            self.recommendations.append("Enable prefetch_to_gpu")
            all_good = False
        else:
            print(f"✅ prefetch_to_gpu is enabled")
        
        # Check enhanced GPU prefetch
        enhanced_prefetch = getattr(self.config.data, 'enhanced_gpu_prefetch', False)
        print(f"\nEnhanced GPU prefetch: {enhanced_prefetch}")
        
        if not enhanced_prefetch:
            self.warnings.append("enhanced_gpu_prefetch is disabled")
            print(f"⚠️  WARNING: enhanced_gpu_prefetch is disabled")
            self.recommendations.append("Enable enhanced_gpu_prefetch")
            all_good = False
        else:
            print(f"✅ enhanced_gpu_prefetch is enabled")
        
        return all_good
    
    def validate_batch_size(self) -> bool:
        """Validate batch size configuration."""
        print("\n" + "="*70)
        print("4. BATCH SIZE VALIDATION")
        print("="*70)
        
        batch_size = getattr(self.config.data, 'batch_size', 8)
        print(f"Batch size: {batch_size}")
        
        if batch_size < 16:
            self.warnings.append(f"Low batch_size ({batch_size})")
            print(f"⚠️  WARNING: batch_size is low ({batch_size})")
            print(f"   Recommended: 32-64 for better GPU utilization")
            print(f"   Note: Increase if you have enough GPU memory")
            self.recommendations.append("Increase batch_size to 32-64 if memory allows")
            return False
        elif batch_size >= 32:
            print(f"✅ batch_size is good for GPU utilization")
            return True
        else:
            print(f"⚠️  batch_size is moderate ({batch_size})")
            print(f"   Consider increasing to 32-64 for better GPU utilization")
            return True
    
    def validate_fixed_shapes(self) -> bool:
        """Validate fixed shape configuration."""
        print("\n" + "="*70)
        print("5. FIXED SHAPES VALIDATION (Anti-Retracing)")
        print("="*70)
        
        pad_to_fixed = getattr(self.config.data, 'pad_to_fixed_length', False)
        print(f"Pad to fixed length: {pad_to_fixed}")
        
        if not pad_to_fixed:
            self.warnings.append("pad_to_fixed_length is disabled")
            print(f"⚠️  WARNING: pad_to_fixed_length is disabled")
            print(f"   Impact: tf.function retracing, lower GPU utilization")
            print(f"   Fix: Set pad_to_fixed_length: true in config.yaml")
            self.recommendations.append("Enable pad_to_fixed_length to prevent retracing")
            
            # Check if max lengths are defined
            max_text = getattr(self.config.data, 'max_text_length', None)
            max_mel = getattr(self.config.data, 'max_mel_frames', None)
            
            if max_text and max_mel:
                print(f"   Note: max_text_length and max_mel_frames are defined")
                print(f"         Just need to enable pad_to_fixed_length")
            else:
                print(f"   Note: Also need to define max_text_length and max_mel_frames")
            
            return False
        else:
            print(f"✅ pad_to_fixed_length is enabled")
            print(f"   This prevents tf.function retracing")
            return True
    
    def validate_xla(self) -> bool:
        """Validate XLA compilation configuration."""
        print("\n" + "="*70)
        print("6. XLA COMPILATION VALIDATION")
        print("="*70)
        
        enable_xla = getattr(self.config.data, 'enable_xla', False)
        print(f"Enable XLA: {enable_xla}")
        
        if not enable_xla:
            self.warnings.append("XLA compilation is disabled")
            print(f"⚠️  WARNING: XLA compilation is disabled")
            print(f"   Impact: Suboptimal GPU kernel performance")
            print(f"   Benefit: 10-30% speedup with XLA")
            self.recommendations.append("Enable XLA compilation for better performance")
            return False
        else:
            print(f"✅ XLA compilation is enabled")
            print(f"   This optimizes TensorFlow graph execution")
            return True
    
    def validate_dual_gpu_config(self) -> bool:
        """Validate dual-GPU configuration."""
        print("\n" + "="*70)
        print("7. DUAL-GPU CONFIGURATION VALIDATION")
        print("="*70)
        
        data_gpu = getattr(self.config.data, 'data_gpu', None)
        model_gpu = getattr(self.config.data, 'model_gpu', None)
        
        print(f"Data GPU: {data_gpu}")
        print(f"Model GPU: {model_gpu}")
        
        if data_gpu is not None and model_gpu is not None:
            if data_gpu == model_gpu:
                self.warnings.append("data_gpu and model_gpu are the same")
                print(f"⚠️  WARNING: data_gpu and model_gpu are the same ({data_gpu})")
                print(f"   This disables dual-GPU pipeline benefits")
                print(f"   Recommendation: Use different GPUs for data and model")
                return False
            else:
                print(f"✅ Dual-GPU mode is properly configured")
                print(f"   Data preprocessing on GPU:{data_gpu}")
                print(f"   Model training on GPU:{model_gpu}")
                return True
        else:
            print(f"ℹ️  Single-GPU mode (dual-GPU not configured)")
            print(f"   This is fine for single-GPU systems")
            return True
    
    def generate_report(self) -> bool:
        """Generate comprehensive validation report."""
        print("\n" + "="*70)
        print("GPU OPTIMIZATION CONFIGURATION VALIDATOR")
        print("="*70)
        print(f"Config source: {self.config_path}")
        print("="*70)
        
        # Run all validations
        validations = [
            self.validate_tf_native_loading(),
            self.validate_parallel_processing(),
            self.validate_prefetching(),
            self.validate_batch_size(),
            self.validate_fixed_shapes(),
            self.validate_xla(),
            self.validate_dual_gpu_config()
        ]
        
        # Generate summary
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        if self.errors:
            print("\n🔴 CRITICAL ERRORS:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
        
        if self.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(self.recommendations, 1):
                print(f"  {i}. {rec}")
        
        if not self.errors and not self.warnings:
            print("\n✅ ALL CHECKS PASSED")
            print("   Your configuration is optimized for dual-GPU pipeline")
            print("   Expected GPU utilization: 80-95%")
        elif self.errors:
            print("\n❌ CONFIGURATION NEEDS FIXES")
            print("   Critical issues found that will cause poor GPU utilization")
            print("   Fix errors before training")
            return False
        else:
            print("\n⚠️  CONFIGURATION CAN BE IMPROVED")
            print("   No critical issues, but optimizations recommended")
            print("   Current config will work but may not achieve 80-95% GPU usage")
        
        print("\n" + "="*70)
        print("Validation complete!")
        print("="*70 + "\n")
        
        return len(self.errors) == 0


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate GPU optimization configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    validator = GPUOptimizationValidator(config_path=args.config)
    success = validator.generate_report()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
