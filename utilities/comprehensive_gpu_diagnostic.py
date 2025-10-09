#!/usr/bin/env python3
"""
Comprehensive GPU Diagnostic and Benchmark Tool

This is the master diagnostic tool that runs all checks and provides
complete analysis and recommendations for GPU oscillation issues.

Usage:
    python utilities/comprehensive_gpu_diagnostic.py
    python utilities/comprehensive_gpu_diagnostic.py --full-benchmark
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("ERROR: TensorFlow not installed")
    print("Please install: pip install tensorflow>=2.12.0")
    sys.exit(1)

from myxtts.config.config import XTTSConfig


class ComprehensiveGPUDiagnostic:
    """Master diagnostic tool for GPU oscillation analysis."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize diagnostic."""
        self.config_path = config_path
        if Path(config_path).exists():
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            print(f"⚠️  Config not found: {config_path}, using defaults")
            self.config = XTTSConfig()
        
        self.issues = []
        self.recommendations = []
        self.config_changes = {}
    
    def run_all_diagnostics(self, data_path: str = None) -> Dict[str, any]:
        """Run all diagnostic checks."""
        print("\n" + "="*70)
        print("COMPREHENSIVE GPU DIAGNOSTIC")
        print("="*70)
        print("This tool will analyze your setup for GPU oscillation issues")
        print()
        
        results = {}
        
        # 1. Hardware check
        print("\n[1/8] Hardware Check...")
        results['hardware'] = self.check_hardware()
        
        # 2. Configuration check
        print("\n[2/8] Configuration Check...")
        results['config'] = self.check_configuration()
        
        # 3. Code analysis
        print("\n[3/8] Code Analysis...")
        results['code'] = self.analyze_code()
        
        # 4. TF-native loader check
        print("\n[4/8] TF-Native Loader Verification...")
        results['tf_native'] = self.verify_tf_native_loader()
        
        # 5. Graph mode check
        print("\n[5/8] Graph Mode & XLA Check...")
        results['graph_mode'] = self.verify_graph_mode()
        
        # 6. Memory check
        print("\n[6/8] Memory Configuration Check...")
        results['memory'] = self.check_memory_config()
        
        # 7. Storage check
        print("\n[7/8] Storage Performance Check...")
        results['storage'] = self.check_storage_performance(data_path)
        
        # 8. Runtime test
        if data_path and Path(data_path).exists():
            print("\n[8/8] Runtime Data Pipeline Test...")
            results['runtime'] = self.test_data_pipeline(data_path)
        else:
            print("\n[8/8] Runtime Data Pipeline Test... SKIPPED (no data path)")
            results['runtime'] = {'skipped': True}
        
        return results
    
    def check_hardware(self) -> Dict[str, any]:
        """Check GPU hardware."""
        print("  Checking GPU availability...")
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            self.issues.append("No GPU detected")
            self.recommendations.append("Install GPU drivers and CUDA toolkit")
            return {'available': False}
        
        print(f"  ✅ Found {len(gpus)} GPU(s)")
        
        # Try to get GPU info
        gpu_info = []
        try:
            import GPUtil
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                info = {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'driver': gpu.driver
                }
                gpu_info.append(info)
                print(f"  ✅ {gpu.name}: {gpu.memoryTotal}MB total")
        except ImportError:
            print("  ⚠️  Install GPUtil for detailed info: pip install gputil")
        
        # Test GPU functionality
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0]])
                b = tf.matmul(a, a)
                _ = b.numpy()
            print("  ✅ GPU is functional")
            functional = True
        except Exception as e:
            print(f"  ❌ GPU test failed: {e}")
            self.issues.append(f"GPU not functional: {e}")
            functional = False
        
        return {
            'available': True,
            'count': len(gpus),
            'functional': functional,
            'info': gpu_info
        }
    
    def check_configuration(self) -> Dict[str, any]:
        """Check configuration settings."""
        print("  Checking configuration settings...")
        
        issues = []
        required_settings = {
            'use_tf_native_loading': True,
            'prefetch_to_gpu': True,
            'enhanced_gpu_prefetch': True,
            'optimize_cpu_gpu_overlap': True,
        }
        
        for setting, expected_value in required_settings.items():
            actual_value = getattr(self.config.data, setting, None)
            if actual_value != expected_value:
                print(f"  ❌ {setting} = {actual_value} (should be {expected_value})")
                issues.append(f"{setting} is not set correctly")
                self.config_changes[f'data.{setting}'] = expected_value
            else:
                print(f"  ✅ {setting} = {actual_value}")
        
        # Check numeric settings
        if self.config.data.num_workers < 8:
            print(f"  ⚠️  num_workers = {self.config.data.num_workers} (recommended: 8-16)")
            self.recommendations.append("Increase num_workers to 8-16")
            self.config_changes['data.num_workers'] = 16
        else:
            print(f"  ✅ num_workers = {self.config.data.num_workers}")
        
        prefetch_size = getattr(self.config.data, 'prefetch_buffer_size', 0)
        if prefetch_size < 8:
            print(f"  ⚠️  prefetch_buffer_size = {prefetch_size} (recommended: 8-16)")
            self.recommendations.append("Increase prefetch_buffer_size to 8-16")
            self.config_changes['data.prefetch_buffer_size'] = 16
        else:
            print(f"  ✅ prefetch_buffer_size = {prefetch_size}")
        
        # Check training settings
        enable_graph = getattr(self.config.training, 'enable_graph_mode', False)
        enable_xla = getattr(self.config.training, 'enable_xla_compilation', False)
        
        if not enable_graph:
            print(f"  ❌ enable_graph_mode = {enable_graph} (should be True)")
            issues.append("Graph mode not enabled")
            self.config_changes['training.enable_graph_mode'] = True
        else:
            print(f"  ✅ enable_graph_mode = {enable_graph}")
        
        if not enable_xla:
            print(f"  ⚠️  enable_xla_compilation = {enable_xla} (recommended: True)")
            self.recommendations.append("Enable XLA compilation for better performance")
            self.config_changes['training.enable_xla_compilation'] = True
        else:
            print(f"  ✅ enable_xla_compilation = {enable_xla}")
        
        return {
            'issues': issues,
            'settings': {
                'use_tf_native_loading': getattr(self.config.data, 'use_tf_native_loading', False),
                'prefetch_to_gpu': getattr(self.config.data, 'prefetch_to_gpu', False),
                'num_workers': self.config.data.num_workers,
                'prefetch_buffer_size': prefetch_size,
                'enable_graph_mode': enable_graph,
                'enable_xla_compilation': enable_xla,
            }
        }
    
    def analyze_code(self) -> Dict[str, any]:
        """Analyze code for bottleneck patterns."""
        print("  Analyzing code for known bottlenecks...")
        
        # Check for tf.numpy_function usage
        ljspeech_file = Path(__file__).parent.parent / "myxtts" / "data" / "ljspeech.py"
        
        issues = []
        if ljspeech_file.exists():
            with open(ljspeech_file, 'r') as f:
                content = f.read()
                
            # Check for problematic patterns
            if 'tf.numpy_function' in content:
                # Check if it's in actual code or just comments/fallback
                lines = content.split('\n')
                actual_usage = False
                for i, line in enumerate(lines):
                    if 'tf.numpy_function' in line and not line.strip().startswith('#'):
                        # Check if it's in the fallback path
                        if 'if not use_tf_native' in '\n'.join(lines[max(0, i-10):i]):
                            print("  ⚠️  tf.numpy_function found in fallback path (OK if not used)")
                        else:
                            print("  🔴 tf.numpy_function found in active code path")
                            actual_usage = True
                            issues.append("tf.numpy_function in active code")
                
                if not actual_usage:
                    print("  ✅ tf.numpy_function only in fallback path")
            else:
                print("  ✅ No tf.numpy_function usage detected")
            
            if 'tf.py_function' in content:
                print("  ⚠️  tf.py_function found (similar issues to tf.numpy_function)")
                issues.append("tf.py_function usage detected")
            else:
                print("  ✅ No tf.py_function usage detected")
        else:
            print("  ⚠️  Could not find ljspeech.py for analysis")
        
        return {'issues': issues}
    
    def verify_tf_native_loader(self) -> Dict[str, any]:
        """Verify TF-native loader is available and working."""
        print("  Verifying TF-native loader...")
        
        try:
            from myxtts.data.tf_native_loader import TFNativeDataLoader
            print("  ✅ TF-native loader module available")
            
            # Test initialization
            loader = TFNativeDataLoader(
                sample_rate=22050,
                n_mels=80,
            )
            print("  ✅ TF-native loader initialized successfully")
            
            # Test mel filterbank creation
            mel_matrix = loader.mel_filterbank
            if mel_matrix is not None:
                print(f"  ✅ Mel filterbank created: shape {mel_matrix.shape}")
            
            return {'available': True, 'functional': True}
            
        except ImportError as e:
            print(f"  ❌ TF-native loader import failed: {e}")
            self.issues.append("TF-native loader not available")
            return {'available': False, 'error': str(e)}
        except Exception as e:
            print(f"  ❌ TF-native loader test failed: {e}")
            self.issues.append(f"TF-native loader error: {e}")
            return {'available': True, 'functional': False, 'error': str(e)}
    
    def verify_graph_mode(self) -> Dict[str, any]:
        """Verify graph mode and XLA."""
        print("  Verifying graph mode and XLA...")
        
        # Test graph compilation
        try:
            @tf.function
            def test_func(x):
                return tf.matmul(x, x)
            
            x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            _ = test_func(x)
            print("  ✅ Graph mode compilation works")
            graph_mode = True
        except Exception as e:
            print(f"  ❌ Graph mode test failed: {e}")
            self.issues.append("Graph mode not working")
            graph_mode = False
        
        # Test XLA
        xla_works = False
        try:
            @tf.function(jit_compile=True)
            def test_xla(x):
                return tf.matmul(x, x)
            
            x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            _ = test_xla(x)
            print("  ✅ XLA compilation works")
            xla_works = True
        except Exception as e:
            print(f"  ⚠️  XLA test failed: {e}")
            print("     XLA may not be available on this system")
        
        return {
            'graph_mode': graph_mode,
            'xla': xla_works
        }
    
    def check_memory_config(self) -> Dict[str, any]:
        """Check memory configuration."""
        print("  Checking memory configuration...")
        
        # Check TensorFlow memory growth setting
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Check if memory growth is set
                for gpu in gpus:
                    memory_growth = tf.config.experimental.get_memory_growth(gpu)
                    print(f"  {'✅' if memory_growth else '⚠️ '} Memory growth: {memory_growth}")
                    
                    if not memory_growth:
                        self.recommendations.append(
                            "Enable memory growth: "
                            "tf.config.experimental.set_memory_growth(gpu, True)"
                        )
            except Exception as e:
                print(f"  ⚠️  Could not check memory growth: {e}")
        
        # Check mixed precision
        policy = tf.keras.mixed_precision.global_policy()
        print(f"  Mixed precision policy: {policy.name}")
        if policy.name == 'float32':
            print("  ⚠️  Consider enabling mixed_float16 for better performance")
            self.recommendations.append("Enable mixed precision training")
        else:
            print("  ✅ Mixed precision enabled")
        
        return {
            'policy': policy.name
        }
    
    def check_storage_performance(self, data_path: str = None) -> Dict[str, any]:
        """Check storage I/O performance."""
        print("  Checking storage performance...")
        
        if not data_path or not Path(data_path).exists():
            print("  ⚠️  No data path provided, skipping I/O test")
            return {'skipped': True}
        
        # Simple I/O test
        try:
            test_file = Path(data_path) / "test_io_performance.tmp"
            
            # Write test
            data = b"x" * (1024 * 1024)  # 1MB
            write_times = []
            for _ in range(5):
                start = time.perf_counter()
                with open(test_file, 'wb') as f:
                    f.write(data)
                write_times.append(time.perf_counter() - start)
            
            # Read test
            read_times = []
            for _ in range(5):
                start = time.perf_counter()
                with open(test_file, 'rb') as f:
                    _ = f.read()
                read_times.append(time.perf_counter() - start)
            
            # Clean up
            test_file.unlink()
            
            avg_write = sum(write_times) / len(write_times) * 1000  # ms
            avg_read = sum(read_times) / len(read_times) * 1000  # ms
            
            print(f"  Write speed: {avg_write:.2f}ms per 1MB")
            print(f"  Read speed: {avg_read:.2f}ms per 1MB")
            
            if avg_read > 50:  # Slow read
                print("  ⚠️  Slow storage detected (HDD?)")
                self.recommendations.append(
                    "Use SSD instead of HDD for significant speedup"
                )
            else:
                print("  ✅ Storage speed is good (likely SSD)")
            
            return {
                'write_speed_ms': avg_write,
                'read_speed_ms': avg_read,
                'slow': avg_read > 50
            }
            
        except Exception as e:
            print(f"  ⚠️  Could not test storage: {e}")
            return {'error': str(e)}
    
    def test_data_pipeline(self, data_path: str) -> Dict[str, any]:
        """Quick test of data pipeline."""
        print("  Testing data pipeline with real data...")
        
        try:
            from myxtts.data.ljspeech import LJSpeechDataset
            
            dataset = LJSpeechDataset(
                data_path=data_path,
                config=self.config.data,
                subset="train",
                download=False
            )
            
            tf_dataset = dataset.create_tf_dataset(
                batch_size=8,
                shuffle=False,
                repeat=False,
                prefetch=True,
                memory_cache=False,
                num_parallel_calls=self.config.data.num_workers,
            )
            
            # Load a few batches and time them
            iterator = iter(tf_dataset)
            times = []
            
            for i in range(10):
                start = time.perf_counter()
                try:
                    _ = next(iterator)
                    times.append((time.perf_counter() - start) * 1000)
                except StopIteration:
                    break
            
            if times:
                import numpy as np
                avg_time = np.mean(times)
                std_time = np.std(times)
                var_ratio = std_time / avg_time if avg_time > 0 else 0
                
                print(f"  Loaded {len(times)} batches")
                print(f"  Average: {avg_time:.2f}ms, Variation: {var_ratio:.2%}")
                
                if var_ratio > 0.5:
                    print("  🔴 High variation detected - likely oscillation")
                    self.issues.append("High batch time variation detected")
                else:
                    print("  ✅ Low variation - stable pipeline")
                
                return {
                    'batches': len(times),
                    'avg_time': avg_time,
                    'variation': var_ratio,
                    'oscillation': var_ratio > 0.5
                }
            
        except Exception as e:
            print(f"  ⚠️  Could not test pipeline: {e}")
            return {'error': str(e)}
        
        return {}
    
    def generate_report(self, results: Dict[str, any], output_file: str = "diagnostic_report.txt"):
        """Generate comprehensive diagnostic report."""
        print(f"\n{'='*70}")
        print("GENERATING DIAGNOSTIC REPORT")
        print("="*70)
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("COMPREHENSIVE GPU DIAGNOSTIC REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-"*70 + "\n")
            f.write(f"Issues found: {len(self.issues)}\n")
            f.write(f"Recommendations: {len(self.recommendations)}\n")
            f.write(f"Config changes needed: {len(self.config_changes)}\n\n")
            
            # Issues
            if self.issues:
                f.write("ISSUES DETECTED:\n")
                for i, issue in enumerate(self.issues, 1):
                    f.write(f"  {i}. {issue}\n")
                f.write("\n")
            
            # Recommendations
            if self.recommendations:
                f.write("RECOMMENDATIONS:\n")
                for i, rec in enumerate(self.recommendations, 1):
                    f.write(f"  {i}. {rec}\n")
                f.write("\n")
            
            # Config changes
            if self.config_changes:
                f.write("CONFIGURATION CHANGES NEEDED:\n")
                f.write("Add these to your config.yaml:\n\n")
                for key, value in self.config_changes.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-"*70 + "\n")
            f.write(json.dumps(results, indent=2))
            f.write("\n")
        
        print(f"✅ Report generated: {output_file}")
        
        # Print summary to console
        self.print_summary()
    
    def print_summary(self):
        """Print diagnostic summary."""
        print(f"\n{'='*70}")
        print("DIAGNOSTIC SUMMARY")
        print("="*70)
        
        if not self.issues and not self.recommendations:
            print("✅ NO ISSUES DETECTED")
            print("   Your configuration appears optimal")
            print("   If you still experience GPU oscillation, it may be:")
            print("   - Hardware limitations (slow storage, insufficient RAM)")
            print("   - Dataset-specific issues")
            print("   - Model-specific bottlenecks")
        else:
            print(f"Found {len(self.issues)} issue(s) and {len(self.recommendations)} recommendation(s)")
            
            if self.issues:
                print(f"\n🔴 ISSUES:")
                for issue in self.issues:
                    print(f"   - {issue}")
            
            if self.recommendations:
                print(f"\n💡 RECOMMENDATIONS:")
                for rec in self.recommendations:
                    print(f"   - {rec}")
            
            if self.config_changes:
                print(f"\n⚙️  CONFIGURATION CHANGES:")
                print("   Add these to configs/config.yaml:")
                for key, value in self.config_changes.items():
                    print(f"   {key}: {value}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive GPU diagnostic tool"
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
        default=None,
        help="Path to dataset for runtime tests"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagnostic_report.txt",
        help="Output file for report"
    )
    
    args = parser.parse_args()
    
    # Run diagnostic
    diagnostic = ComprehensiveGPUDiagnostic(args.config)
    results = diagnostic.run_all_diagnostics(args.data_path)
    
    # Generate report
    diagnostic.generate_report(results, args.output)
    
    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)
    print(f"\nFull report saved to: {args.output}")
    print("\nNext steps:")
    print("1. Review the summary above")
    print("2. Apply recommended configuration changes")
    print("3. Re-run this diagnostic to verify fixes")
    print("4. Use enhanced_gpu_profiler.py for detailed profiling")
    print("5. Use training_step_profiler.py to profile actual training")


if __name__ == "__main__":
    main()
