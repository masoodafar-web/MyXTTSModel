#!/usr/bin/env python3
"""
GPU Utilization Diagnostic Tool

ابزار تشخیص مشکل استفاده پایین از GPU

This tool diagnoses why GPU utilization is low (<50%) and provides
specific recommendations to fix the issue.

Usage:
    python utilities/diagnose_gpu_utilization.py
    python utilities/diagnose_gpu_utilization.py --config configs/config.yaml
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed")
    sys.exit(1)

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("WARNING: GPUtil not installed, GPU monitoring limited")

try:
    import yaml
except ImportError:
    yaml = None


class GPUUtilizationDiagnostic:
    """Diagnose GPU utilization issues."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize diagnostic tool.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.issues = []
        self.recommendations = []
        
    def _load_config(self) -> Optional[Dict]:
        """Load configuration from file."""
        if not self.config_path or not yaml:
            return None
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            return None
    
    def check_gpu_availability(self) -> bool:
        """Check if GPUs are available and properly configured."""
        print("\n" + "="*70)
        print("1. GPU AVAILABILITY CHECK")
        print("="*70)
        
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            self.issues.append("NO_GPUS_DETECTED")
            self.recommendations.append(
                "Install CUDA and cuDNN, ensure GPU drivers are up to date"
            )
            print("❌ No GPUs detected by TensorFlow")
            print("   Check CUDA installation and GPU drivers")
            return False
        
        print(f"✅ {len(gpus)} GPU(s) detected:")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Check if GPUs are actually usable
        try:
            with tf.device('/GPU:0'):
                test = tf.constant([1.0, 2.0, 3.0])
                result = tf.reduce_sum(test)
            print("✅ GPU computation test: PASSED")
        except Exception as e:
            self.issues.append("GPU_NOT_USABLE")
            self.recommendations.append(
                f"GPU detected but not usable: {e}"
            )
            print(f"❌ GPU computation test: FAILED - {e}")
            return False
        
        # Check GPU memory
        if GPUTIL_AVAILABLE:
            gpu_list = GPUtil.getGPUs()
            for i, gpu in enumerate(gpu_list):
                total_mem = gpu.memoryTotal
                print(f"   GPU {i}: {total_mem}MB total memory")
                
                if total_mem < 8000:
                    self.issues.append("LOW_GPU_MEMORY")
                    self.recommendations.append(
                        f"GPU {i} has only {total_mem}MB - may limit batch size"
                    )
        
        return True
    
    def check_tensorflow_configuration(self) -> bool:
        """Check TensorFlow configuration for performance."""
        print("\n" + "="*70)
        print("2. TENSORFLOW CONFIGURATION CHECK")
        print("="*70)
        
        all_good = True
        
        # Check thread configuration
        inter_op = tf.config.threading.get_inter_op_parallelism_threads()
        intra_op = tf.config.threading.get_intra_op_parallelism_threads()
        
        if inter_op == 0:
            self.issues.append("INTER_OP_NOT_CONFIGURED")
            self.recommendations.append(
                "Set inter_op_parallelism_threads: "
                "tf.config.threading.set_inter_op_parallelism_threads(cpu_count)"
            )
            print("⚠️  Inter-op parallelism: Not configured (using defaults)")
            all_good = False
        else:
            print(f"✅ Inter-op parallelism: {inter_op} threads")
        
        if intra_op == 0:
            self.issues.append("INTRA_OP_NOT_CONFIGURED")
            self.recommendations.append(
                "Set intra_op_parallelism_threads: "
                "tf.config.threading.set_intra_op_parallelism_threads(cpu_count // 2)"
            )
            print("⚠️  Intra-op parallelism: Not configured (using defaults)")
            all_good = False
        else:
            print(f"✅ Intra-op parallelism: {intra_op} threads")
        
        # Check XLA
        xla_enabled = os.environ.get('TF_XLA_FLAGS') or tf.config.optimizer.get_jit()
        if xla_enabled:
            print(f"✅ XLA JIT: ENABLED")
        else:
            self.issues.append("XLA_NOT_ENABLED")
            self.recommendations.append(
                "Enable XLA JIT: tf.config.optimizer.set_jit(True)"
            )
            print("⚠️  XLA JIT: NOT ENABLED (significant performance loss)")
            all_good = False
        
        # Check mixed precision
        try:
            policy = tf.keras.mixed_precision.global_policy()
            if 'float16' in str(policy.name):
                print(f"✅ Mixed Precision: {policy.name}")
            else:
                self.issues.append("MIXED_PRECISION_NOT_ENABLED")
                self.recommendations.append(
                    "Enable mixed precision for RTX 4090: "
                    "tf.keras.mixed_precision.set_global_policy('mixed_float16')"
                )
                print(f"⚠️  Mixed Precision: {policy.name} (float16 recommended for RTX 4090)")
                all_good = False
        except Exception:
            print("⚠️  Mixed Precision: Could not verify")
        
        return all_good
    
    def check_dataset_configuration(self) -> bool:
        """Check dataset configuration for performance issues."""
        print("\n" + "="*70)
        print("3. DATASET CONFIGURATION CHECK")
        print("="*70)
        
        if not self.config:
            print("⚠️  No config file provided, skipping dataset checks")
            print("   Use: --config configs/config.yaml")
            return False
        
        data_config = self.config.get('data', {})
        all_good = True
        
        # Check batch size
        batch_size = data_config.get('batch_size', 0)
        if batch_size < 64:
            self.issues.append("BATCH_SIZE_TOO_SMALL")
            self.recommendations.append(
                f"Batch size {batch_size} is too small for RTX 4090. "
                f"Increase to at least 128 for better GPU utilization"
            )
            print(f"⚠️  Batch Size: {batch_size} (too small for RTX 4090)")
            print(f"   Recommendation: Increase to 128-256")
            all_good = False
        else:
            print(f"✅ Batch Size: {batch_size}")
        
        # Check num_workers
        num_workers = data_config.get('num_workers', 0)
        if num_workers < 16:
            self.issues.append("INSUFFICIENT_WORKERS")
            self.recommendations.append(
                f"num_workers {num_workers} is too low. "
                f"Increase to at least 32 for dual-GPU setup"
            )
            print(f"⚠️  Num Workers: {num_workers} (insufficient for fast data loading)")
            print(f"   Recommendation: Increase to 32-48")
            all_good = False
        else:
            print(f"✅ Num Workers: {num_workers}")
        
        # Check TF-native loading
        use_tf_native = data_config.get('use_tf_native_loading', False)
        if not use_tf_native:
            self.issues.append("TF_NATIVE_NOT_ENABLED")
            self.recommendations.append(
                "Enable TF-native loading: set use_tf_native_loading: true in config"
            )
            print(f"❌ TF-Native Loading: DISABLED")
            print(f"   This causes CPU bottleneck and low GPU utilization")
            print(f"   Set: use_tf_native_loading: true")
            all_good = False
        else:
            print(f"✅ TF-Native Loading: ENABLED")
        
        # Check prefetch settings
        prefetch_buffer = data_config.get('prefetch_buffer_size', 0)
        if prefetch_buffer < 50:
            self.issues.append("INSUFFICIENT_PREFETCH")
            self.recommendations.append(
                f"prefetch_buffer_size {prefetch_buffer} is too low. "
                f"Increase to at least 100"
            )
            print(f"⚠️  Prefetch Buffer: {prefetch_buffer} (too small)")
            print(f"   Recommendation: Increase to 100+")
            all_good = False
        else:
            print(f"✅ Prefetch Buffer: {prefetch_buffer}")
        
        # Check GPU prefetch
        prefetch_to_gpu = data_config.get('prefetch_to_gpu', False)
        if not prefetch_to_gpu:
            self.issues.append("GPU_PREFETCH_NOT_ENABLED")
            self.recommendations.append(
                "Enable GPU prefetch: set prefetch_to_gpu: true in config"
            )
            print(f"⚠️  Prefetch to GPU: DISABLED")
            print(f"   Recommendation: Enable for faster data transfer")
            all_good = False
        else:
            print(f"✅ Prefetch to GPU: ENABLED")
        
        # Check static shapes
        pad_to_fixed = data_config.get('pad_to_fixed_length', False)
        if not pad_to_fixed:
            self.issues.append("STATIC_SHAPES_NOT_ENABLED")
            self.recommendations.append(
                "Enable static shapes: set pad_to_fixed_length: true in config"
            )
            print(f"⚠️  Static Shapes: DISABLED")
            print(f"   This causes tf.function retracing and GPU stalls")
            print(f"   Set: pad_to_fixed_length: true")
            all_good = False
        else:
            print(f"✅ Static Shapes: ENABLED")
        
        return all_good
    
    def measure_data_pipeline_speed(self, num_samples: int = 100) -> Tuple[float, float]:
        """
        Measure data pipeline throughput.
        
        Args:
            num_samples: Number of samples to test
            
        Returns:
            Tuple of (samples_per_second, avg_batch_time_ms)
        """
        print("\n" + "="*70)
        print("4. DATA PIPELINE SPEED TEST")
        print("="*70)
        print(f"Testing with {num_samples} samples...")
        
        try:
            # Create a simple test dataset
            batch_size = 32
            dataset = tf.data.Dataset.range(num_samples)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            # Warmup
            for _ in dataset.take(2):
                pass
            
            # Measure
            start_time = time.time()
            count = 0
            for batch in dataset:
                count += batch_size
            elapsed = time.time() - start_time
            
            samples_per_sec = count / elapsed
            avg_batch_time = (elapsed / (count / batch_size)) * 1000  # ms
            
            print(f"✅ Throughput: {samples_per_sec:.1f} samples/sec")
            print(f"   Average batch time: {avg_batch_time:.2f}ms")
            
            # Check if pipeline is fast enough
            min_required_throughput = 200  # samples/sec for RTX 4090
            if samples_per_sec < min_required_throughput:
                self.issues.append("SLOW_DATA_PIPELINE")
                self.recommendations.append(
                    f"Data pipeline too slow: {samples_per_sec:.1f} samples/sec. "
                    f"Should be >{min_required_throughput}"
                )
                print(f"⚠️  Pipeline may be too slow for RTX 4090")
            
            return samples_per_sec, avg_batch_time
            
        except Exception as e:
            print(f"❌ Speed test failed: {e}")
            return 0.0, 0.0
    
    def check_gpu_utilization_realtime(self, duration: int = 10) -> float:
        """
        Monitor GPU utilization in real-time.
        
        Args:
            duration: Monitoring duration in seconds
            
        Returns:
            Average GPU utilization percentage
        """
        print("\n" + "="*70)
        print("5. REAL-TIME GPU UTILIZATION CHECK")
        print("="*70)
        
        if not GPUTIL_AVAILABLE:
            print("⚠️  GPUtil not available, skipping real-time monitoring")
            print("   Install GPUtil: pip install gputil")
            return 0.0
        
        print(f"Monitoring GPU utilization for {duration} seconds...")
        print("(Run a training script in another terminal)")
        
        samples = []
        try:
            for i in range(duration):
                gpus = GPUtil.getGPUs()
                if gpus:
                    util = gpus[0].load * 100
                    samples.append(util)
                    print(f"  [{i+1}/{duration}] GPU:0 = {util:.1f}%", end='\r')
                time.sleep(1)
            print()  # New line
            
            avg_util = sum(samples) / len(samples) if samples else 0.0
            print(f"\n✅ Average GPU Utilization: {avg_util:.1f}%")
            
            if avg_util < 50:
                self.issues.append("LOW_GPU_UTILIZATION")
                self.recommendations.append(
                    f"GPU utilization {avg_util:.1f}% is very low. "
                    f"This indicates a data pipeline bottleneck."
                )
                print(f"❌ GPU is severely underutilized ({avg_util:.1f}%)")
            elif avg_util < 80:
                self.issues.append("SUBOPTIMAL_GPU_UTILIZATION")
                self.recommendations.append(
                    f"GPU utilization {avg_util:.1f}% could be improved. "
                    f"Target: >80%"
                )
                print(f"⚠️  GPU could be better utilized ({avg_util:.1f}%)")
            else:
                print(f"✅ GPU utilization is good ({avg_util:.1f}%)")
            
            return avg_util
            
        except KeyboardInterrupt:
            print("\n⚠️  Monitoring interrupted")
            return 0.0
    
    def generate_report(self):
        """Generate and print diagnostic report."""
        print("\n" + "="*70)
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        
        if not self.issues:
            print("\n✅ NO ISSUES DETECTED - CONFIGURATION IS OPTIMAL")
            print("\nIf you still have low GPU utilization:")
            print("  1. Check that training is actually running")
            print("  2. Verify batch size is large enough (128+ for RTX 4090)")
            print("  3. Monitor nvidia-smi during training")
            print("  4. Run: python utilities/dual_gpu_bottleneck_profiler.py")
        else:
            print(f"\n❌ {len(self.issues)} ISSUES DETECTED:")
            for i, issue in enumerate(self.issues, 1):
                print(f"\n{i}. {issue}")
                if i <= len(self.recommendations):
                    print(f"   → {self.recommendations[i-1]}")
        
        print("\n" + "="*70)
        print("RECOMMENDED ACTIONS (IN ORDER OF PRIORITY)")
        print("="*70)
        
        priority_actions = [
            ("CRITICAL", [
                "Enable TF-native loading (use_tf_native_loading: true)",
                "Enable static shapes (pad_to_fixed_length: true)",
                "Increase batch size to 128+ for RTX 4090",
                "Increase num_workers to 32+",
            ]),
            ("HIGH", [
                "Enable XLA JIT (tf.config.optimizer.set_jit(True))",
                "Enable mixed precision (tf.keras.mixed_precision.set_global_policy('mixed_float16'))",
                "Increase prefetch_buffer_size to 100+",
                "Enable prefetch_to_gpu",
            ]),
            ("MEDIUM", [
                "Configure TensorFlow threading (inter_op, intra_op)",
                "Apply dataset optimizations (experimental_optimization)",
                "Use memory-isolated dual-GPU trainer",
            ]),
        ]
        
        for priority, actions in priority_actions:
            print(f"\n{priority} PRIORITY:")
            for action in actions:
                # Only show if relevant to detected issues
                if any(keyword in action.lower() for issue in self.issues for keyword in issue.lower().split('_')):
                    print(f"  • {action}")
        
        print("\n" + "="*70)
        print("NEXT STEPS")
        print("="*70)
        print("\n1. Apply the recommended configuration changes")
        print("2. Run: python utilities/configure_max_gpu_utilization.py")
        print("3. Start training with optimized settings:")
        print("   python train_main.py --batch-size 128 --num-workers 32 \\")
        print("       --enable-memory-isolation --data-gpu 0 --model-gpu 1 \\")
        print("       --enable-static-shapes")
        print("4. Monitor GPU utilization:")
        print("   watch -n 1 nvidia-smi")
        print("5. If still low, run profiler:")
        print("   python utilities/dual_gpu_bottleneck_profiler.py")
        print("\n" + "="*70 + "\n")
    
    def run_full_diagnostic(self, skip_realtime: bool = False):
        """Run complete diagnostic suite."""
        print("\n" + "="*80)
        print(" "*20 + "GPU UTILIZATION DIAGNOSTIC TOOL")
        print(" "*15 + "ابزار تشخیص مشکل استفاده پایین از GPU")
        print("="*80)
        
        # Run all checks
        self.check_gpu_availability()
        self.check_tensorflow_configuration()
        self.check_dataset_configuration()
        self.measure_data_pipeline_speed()
        
        if not skip_realtime:
            self.check_gpu_utilization_realtime(duration=10)
        
        # Generate report
        self.generate_report()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Diagnose GPU utilization issues"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file"
    )
    parser.add_argument(
        "--skip-realtime",
        action="store_true",
        help="Skip real-time GPU monitoring"
    )
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = GPUUtilizationDiagnostic(config_path=args.config)
    diagnostic.run_full_diagnostic(skip_realtime=args.skip_realtime)


if __name__ == "__main__":
    main()
