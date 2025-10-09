#!/usr/bin/env python3
"""
Comprehensive GPU Profiler for MyXTTS Model

This tool profiles the entire training pipeline to identify GPU bottlenecks:
- Device placement verification for all operations
- CPU vs GPU operation breakdown
- Data loading performance analysis
- Model forward/backward pass profiling
- Memory usage tracking
- Real-time GPU utilization monitoring
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tensorflow as tf
import numpy as np
from collections import defaultdict
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from myxtts.config.config import XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.data.ljspeech import LJSpeechDataset


class GPUProfiler:
    """Comprehensive GPU profiling tool."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize profiler with config."""
        self.config_path = config_path or "configs/config.yaml"
        self.config = XTTSConfig.from_yaml(self.config_path)
        
        # Profiling results
        self.device_placements: Dict[str, int] = defaultdict(int)
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.gpu_memory_usage: List[float] = []
        self.cpu_utilization: List[float] = []
        
    def check_gpu_availability(self) -> Tuple[bool, str]:
        """Check if GPU is available and functional."""
        print("\n" + "="*60)
        print("GPU AVAILABILITY CHECK")
        print("="*60)
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU detected!")
            print("\n📋 Required Setup:")
            print("  1. NVIDIA GPU with CUDA support")
            print("  2. CUDA toolkit (11.2+)")
            print("  3. cuDNN (8.1+)")
            print("  4. TensorFlow with GPU support")
            return False, "No GPU detected"
        
        print(f"✅ Found {len(gpus)} GPU(s):")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
        # Test GPU functionality
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.matmul(a, a)
                result = b.numpy()
            print("✅ GPU is functional")
            return True, f"{len(gpus)} GPU(s) available"
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
            return False, str(e)
    
    def profile_device_placement(self, model: XTTS) -> Dict[str, Dict]:
        """Profile which operations are placed on GPU vs CPU."""
        print("\n" + "="*60)
        print("DEVICE PLACEMENT ANALYSIS")
        print("="*60)
        
        device_stats = {
            'GPU': {'count': 0, 'operations': []},
            'CPU': {'count': 0, 'operations': []},
            'Unknown': {'count': 0, 'operations': []}
        }
        
        # Enable device placement logging
        tf.debugging.set_log_device_placement(True)
        
        # Create dummy input
        batch_size = 2
        text_len = 50
        mel_len = 100
        n_mels = self.config.model.n_mels
        
        text_input = tf.random.uniform((batch_size, text_len), 0, 100, dtype=tf.int32)
        mel_input = tf.random.normal((batch_size, mel_len, n_mels))
        text_lengths = tf.constant([text_len, text_len], dtype=tf.int32)
        mel_lengths = tf.constant([mel_len, mel_len], dtype=tf.int32)
        
        # Redirect stderr to capture device placement logs
        import io
        from contextlib import redirect_stderr
        
        f = io.StringIO()
        try:
            with redirect_stderr(f):
                _ = model(
                    text_inputs=text_input,
                    mel_inputs=mel_input,
                    text_lengths=text_lengths,
                    mel_lengths=mel_lengths,
                    training=False
                )
        except Exception as e:
            print(f"⚠️  Model forward pass error: {e}")
        
        # Parse device placement from logs
        logs = f.getvalue()
        for line in logs.split('\n'):
            if '/device:GPU' in line or '/GPU:' in line:
                device_stats['GPU']['count'] += 1
                op_name = line.split(':')[0] if ':' in line else line[:50]
                device_stats['GPU']['operations'].append(op_name)
            elif '/device:CPU' in line or '/CPU:' in line:
                device_stats['CPU']['count'] += 1
                op_name = line.split(':')[0] if ':' in line else line[:50]
                device_stats['CPU']['operations'].append(op_name)
        
        tf.debugging.set_log_device_placement(False)
        
        # Print summary
        print(f"\n📊 Operation Placement Summary:")
        print(f"  GPU Operations: {device_stats['GPU']['count']}")
        print(f"  CPU Operations: {device_stats['CPU']['count']}")
        
        if device_stats['CPU']['count'] > 0:
            print(f"\n⚠️  WARNING: {device_stats['CPU']['count']} operations on CPU!")
            print("  This may cause GPU underutilization.")
            print("\n  Sample CPU operations:")
            for op in device_stats['CPU']['operations'][:5]:
                print(f"    - {op}")
        
        gpu_percentage = (device_stats['GPU']['count'] / 
                         (device_stats['GPU']['count'] + device_stats['CPU']['count'] + 0.001)) * 100
        print(f"\n  GPU Operation Percentage: {gpu_percentage:.1f}%")
        
        return device_stats
    
    def profile_data_loading(self, dataset: tf.data.Dataset, num_batches: int = 10) -> Dict:
        """Profile data loading performance."""
        print("\n" + "="*60)
        print("DATA LOADING PERFORMANCE ANALYSIS")
        print("="*60)
        
        load_times = []
        batch_sizes = []
        
        print(f"\nProfiling {num_batches} batches...")
        iterator = iter(dataset)
        
        for i in range(num_batches):
            start_time = time.perf_counter()
            try:
                batch = next(iterator)
                end_time = time.perf_counter()
                
                load_time = (end_time - start_time) * 1000  # ms
                load_times.append(load_time)
                
                # Get batch size
                text_seq = batch[0]
                batch_sizes.append(text_seq.shape[0])
                
                if i < 3:
                    print(f"  Batch {i+1}: {load_time:.2f}ms")
            except StopIteration:
                print(f"  Dataset exhausted after {i} batches")
                break
            except Exception as e:
                print(f"  Error loading batch {i}: {e}")
                break
        
        if not load_times:
            print("❌ No batches loaded successfully")
            return {}
        
        stats = {
            'mean_load_time_ms': np.mean(load_times),
            'std_load_time_ms': np.std(load_times),
            'min_load_time_ms': np.min(load_times),
            'max_load_time_ms': np.max(load_times),
            'mean_batch_size': np.mean(batch_sizes)
        }
        
        print(f"\n📊 Data Loading Statistics:")
        print(f"  Mean load time: {stats['mean_load_time_ms']:.2f}ms ± {stats['std_load_time_ms']:.2f}ms")
        print(f"  Min/Max: {stats['min_load_time_ms']:.2f}ms / {stats['max_load_time_ms']:.2f}ms")
        print(f"  Mean batch size: {stats['mean_batch_size']:.1f}")
        
        # Assess if data loading is a bottleneck
        if stats['mean_load_time_ms'] > 500:
            print("\n⚠️  WARNING: Data loading is SLOW (>500ms per batch)")
            print("  Recommendations:")
            print("    - Increase num_workers in config")
            print("    - Enable prefetch_to_gpu")
            print("    - Use precomputed features (preprocessing_mode='precompute')")
        elif stats['mean_load_time_ms'] > 100:
            print("\n⚠️  Data loading is moderate (>100ms per batch)")
            print("  Consider optimizations if GPU utilization is low")
        else:
            print("\n✅ Data loading is FAST (<100ms per batch)")
        
        return stats
    
    def profile_training_step(self, model: XTTS, optimizer, num_steps: int = 5) -> Dict:
        """Profile training step performance."""
        print("\n" + "="*60)
        print("TRAINING STEP PERFORMANCE ANALYSIS")
        print("="*60)
        
        # Create dummy data
        batch_size = self.config.data.batch_size
        text_len = 50
        mel_len = 100
        n_mels = self.config.model.n_mels
        
        timings = {
            'forward_pass': [],
            'loss_computation': [],
            'backward_pass': [],
            'optimizer_step': [],
            'total': []
        }
        
        print(f"\nProfiling {num_steps} training steps...")
        
        for step in range(num_steps):
            text_input = tf.random.uniform((batch_size, text_len), 0, 100, dtype=tf.int32)
            mel_input = tf.random.normal((batch_size, mel_len, n_mels))
            text_lengths = tf.constant([text_len] * batch_size, dtype=tf.int32)
            mel_lengths = tf.constant([mel_len] * batch_size, dtype=tf.int32)
            
            # Move to GPU if available
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                with tf.device('/GPU:0'):
                    text_input = tf.identity(text_input)
                    mel_input = tf.identity(mel_input)
            
            step_start = time.perf_counter()
            
            with tf.GradientTape() as tape:
                # Forward pass
                forward_start = time.perf_counter()
                outputs = model(
                    text_inputs=text_input,
                    mel_inputs=mel_input,
                    text_lengths=text_lengths,
                    mel_lengths=mel_lengths,
                    training=True
                )
                forward_time = (time.perf_counter() - forward_start) * 1000
                
                # Loss computation
                loss_start = time.perf_counter()
                mel_loss = tf.reduce_mean(tf.abs(outputs['mel_output'] - mel_input))
                stop_loss = tf.reduce_mean(tf.abs(outputs['stop_tokens']))
                total_loss = mel_loss + stop_loss
                loss_time = (time.perf_counter() - loss_start) * 1000
            
            # Backward pass
            backward_start = time.perf_counter()
            gradients = tape.gradient(total_loss, model.trainable_variables)
            backward_time = (time.perf_counter() - backward_start) * 1000
            
            # Optimizer step
            opt_start = time.perf_counter()
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            opt_time = (time.perf_counter() - opt_start) * 1000
            
            total_time = (time.perf_counter() - step_start) * 1000
            
            timings['forward_pass'].append(forward_time)
            timings['loss_computation'].append(loss_time)
            timings['backward_pass'].append(backward_time)
            timings['optimizer_step'].append(opt_time)
            timings['total'].append(total_time)
            
            if step < 3:
                print(f"\n  Step {step+1}:")
                print(f"    Forward:   {forward_time:6.2f}ms")
                print(f"    Loss:      {loss_time:6.2f}ms")
                print(f"    Backward:  {backward_time:6.2f}ms")
                print(f"    Optimizer: {opt_time:6.2f}ms")
                print(f"    Total:     {total_time:6.2f}ms")
        
        # Compute statistics
        stats = {}
        for key, values in timings.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        print(f"\n📊 Training Step Statistics (mean ± std):")
        print(f"  Forward pass:    {stats['forward_pass']['mean']:6.2f}ms ± {stats['forward_pass']['std']:.2f}ms")
        print(f"  Loss computation:{stats['loss_computation']['mean']:6.2f}ms ± {stats['loss_computation']['std']:.2f}ms")
        print(f"  Backward pass:   {stats['backward_pass']['mean']:6.2f}ms ± {stats['backward_pass']['std']:.2f}ms")
        print(f"  Optimizer step:  {stats['optimizer_step']['mean']:6.2f}ms ± {stats['optimizer_step']['std']:.2f}ms")
        print(f"  Total per step:  {stats['total']['mean']:6.2f}ms ± {stats['total']['std']:.2f}ms")
        
        # Calculate throughput
        steps_per_second = 1000 / stats['total']['mean']
        samples_per_second = steps_per_second * batch_size
        print(f"\n📈 Throughput:")
        print(f"  Steps/second:   {steps_per_second:.2f}")
        print(f"  Samples/second: {samples_per_second:.2f}")
        
        return stats
    
    def monitor_gpu_utilization(self, duration_seconds: int = 10) -> List[float]:
        """Monitor GPU utilization using nvidia-smi."""
        print("\n" + "="*60)
        print(f"GPU UTILIZATION MONITORING ({duration_seconds}s)")
        print("="*60)
        
        utilizations = []
        memories = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                     '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) >= 3:
                            util = float(parts[0].strip())
                            mem_used = float(parts[1].strip())
                            mem_total = float(parts[2].strip())
                            utilizations.append(util)
                            memories.append((mem_used, mem_total))
                
                time.sleep(0.5)
            except Exception as e:
                print(f"Warning: Could not query GPU: {e}")
                break
        
        if utilizations:
            print(f"\n📊 GPU Utilization Statistics:")
            print(f"  Mean: {np.mean(utilizations):.1f}%")
            print(f"  Min:  {np.min(utilizations):.1f}%")
            print(f"  Max:  {np.max(utilizations):.1f}%")
            print(f"  Std:  {np.std(utilizations):.1f}%")
            
            if memories:
                mean_mem_used = np.mean([m[0] for m in memories])
                mem_total = memories[0][1]
                mem_percent = (mean_mem_used / mem_total) * 100
                print(f"\n📊 GPU Memory Usage:")
                print(f"  Used:  {mean_mem_used:.0f}MB / {mem_total:.0f}MB ({mem_percent:.1f}%)")
            
            # Assessment
            mean_util = np.mean(utilizations)
            if mean_util < 30:
                print(f"\n❌ CRITICAL: GPU utilization is VERY LOW ({mean_util:.1f}%)")
                print("  This indicates severe bottlenecks in the training pipeline.")
            elif mean_util < 50:
                print(f"\n⚠️  WARNING: GPU utilization is LOW ({mean_util:.1f}%)")
                print("  Significant performance improvements are possible.")
            elif mean_util < 70:
                print(f"\n⚠️  GPU utilization is MODERATE ({mean_util:.1f}%)")
                print("  Some optimization opportunities remain.")
            else:
                print(f"\n✅ GPU utilization is GOOD ({mean_util:.1f}%)")
        else:
            print("\n⚠️  Could not measure GPU utilization (nvidia-smi not available)")
        
        return utilizations
    
    def generate_report(self) -> str:
        """Generate comprehensive profiling report."""
        print("\n" + "="*60)
        print("COMPREHENSIVE GPU PROFILING REPORT")
        print("="*60)
        
        report = []
        report.append("\n🔍 MyXTTS GPU Profiling Report")
        report.append("="*60)
        report.append(f"Configuration: {self.config_path}")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n")
        
        # GPU check
        gpu_available, gpu_msg = self.check_gpu_availability()
        report.append(f"GPU Status: {'✅ Available' if gpu_available else '❌ Not Available'}")
        report.append(f"Details: {gpu_msg}")
        
        if not gpu_available:
            report.append("\n❌ Cannot proceed with profiling - no GPU available")
            return "\n".join(report)
        
        # Load model
        print("\n📦 Loading model...")
        try:
            model = XTTS(self.config.model)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
            report.append("\n✅ Model loaded successfully")
        except Exception as e:
            report.append(f"\n❌ Failed to load model: {e}")
            return "\n".join(report)
        
        # Device placement analysis
        device_stats = self.profile_device_placement(model)
        report.append(f"\n📍 Device Placement:")
        report.append(f"  GPU operations: {device_stats['GPU']['count']}")
        report.append(f"  CPU operations: {device_stats['CPU']['count']}")
        
        # Training step profiling
        training_stats = self.profile_training_step(model, optimizer)
        report.append(f"\n⚡ Training Performance:")
        report.append(f"  Forward pass:  {training_stats['forward_pass']['mean']:.2f}ms")
        report.append(f"  Backward pass: {training_stats['backward_pass']['mean']:.2f}ms")
        report.append(f"  Total step:    {training_stats['total']['mean']:.2f}ms")
        
        # Data loading profiling (if dataset available)
        try:
            print("\n📦 Loading dataset for profiling...")
            dataset_loader = LJSpeechDataset(
                data_path=self.config.data.data_path,
                config=self.config.data,
                subset="train"
            )
            dataset = dataset_loader.create_tf_dataset(
                batch_size=self.config.data.batch_size,
                shuffle=False,
                repeat=False
            )
            data_stats = self.profile_data_loading(dataset)
            if data_stats:
                report.append(f"\n📊 Data Loading:")
                report.append(f"  Mean time: {data_stats['mean_load_time_ms']:.2f}ms")
                report.append(f"  Batch size: {data_stats['mean_batch_size']:.0f}")
        except Exception as e:
            report.append(f"\n⚠️  Could not profile data loading: {e}")
        
        # GPU utilization monitoring
        utilizations = self.monitor_gpu_utilization(duration_seconds=10)
        if utilizations:
            report.append(f"\n🎯 GPU Utilization:")
            report.append(f"  Mean: {np.mean(utilizations):.1f}%")
            report.append(f"  Range: {np.min(utilizations):.1f}% - {np.max(utilizations):.1f}%")
        
        # Recommendations
        report.append("\n\n💡 RECOMMENDATIONS:")
        report.append("="*60)
        
        if device_stats['CPU']['count'] > 10:
            report.append("\n⚠️  High number of CPU operations detected:")
            report.append("  → Add @tf.function decorator to training step")
            report.append("  → Enable XLA JIT compilation")
            report.append("  → Check for tf.numpy_function usage in data pipeline")
        
        if training_stats['total']['mean'] > 1000:
            report.append("\n⚠️  Training step is slow (>1000ms):")
            report.append("  → Enable graph mode with @tf.function")
            report.append("  → Use mixed precision training")
            report.append("  → Reduce model size or batch size")
        
        if utilizations and np.mean(utilizations) < 50:
            report.append("\n❌ CRITICAL: Low GPU utilization detected:")
            report.append("  → Primary issue: Training not using GPU efficiently")
            report.append("  → Apply @tf.function to training loop")
            report.append("  → Enable XLA compilation")
            report.append("  → Check data loading bottlenecks")
            report.append("  → Verify all tensors are on GPU")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report to file
        report_file = Path("gpu_profiling_report.txt")
        with open(report_file, 'w') as f:
            f.write(report_text)
        print(f"\n📄 Report saved to: {report_file}")
        
        return report_text


def main():
    """Main profiling entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MyXTTS GPU Profiler")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🔍 MyXTTS Comprehensive GPU Profiler")
    print("="*60)
    print("\nThis tool will analyze your training pipeline to identify")
    print("GPU utilization bottlenecks and provide recommendations.")
    print("\nProfiling will take approximately 30-60 seconds...")
    
    profiler = GPUProfiler(config_path=args.config)
    profiler.generate_report()
    
    print("\n✅ Profiling complete!")
    print("\nReview the recommendations above to improve GPU utilization.")


if __name__ == "__main__":
    main()
