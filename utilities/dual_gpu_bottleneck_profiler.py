#!/usr/bin/env python3
"""
Dual-GPU Pipeline Bottleneck Profiler

این ابزار برای شناسایی دقیق bottleneckها در dual-GPU pipeline طراحی شده است.
This tool is designed for precise bottleneck identification in dual-GPU pipeline.

Profiling Areas:
1. Data loading on GPU:0
2. Data preprocessing on GPU:0  
3. GPU-to-GPU transfer (GPU:0 → GPU:1)
4. Model forward pass on GPU:1
5. Loss computation on GPU:1
6. Backward pass on GPU:1
7. Optimizer step on GPU:1
8. Overall pipeline throughput
9. GPU utilization monitoring

Usage:
    python utilities/dual_gpu_bottleneck_profiler.py --batch-size 16 --num-steps 100
"""

import sys
import os
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict, deque
import threading
import queue

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

from myxtts.config.config import XTTSConfig


class GPUMonitor:
    """Monitor GPU utilization and memory in background thread."""
    
    def __init__(self, data_gpu_id: int = 0, model_gpu_id: int = 1):
        self.data_gpu_id = data_gpu_id
        self.model_gpu_id = model_gpu_id
        self.running = False
        self.thread = None
        self.samples = {
            'data_gpu_util': [],
            'model_gpu_util': [],
            'data_gpu_memory': [],
            'model_gpu_memory': [],
            'timestamps': []
        }
        self.lock = threading.Lock()
    
    def start(self):
        """Start monitoring in background thread."""
        if not GPUTIL_AVAILABLE:
            print("⚠️  GPUtil not available, GPU monitoring disabled")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.running:
            try:
                gpus = GPUtil.getGPUs()
                if len(gpus) >= max(self.data_gpu_id + 1, self.model_gpu_id + 1):
                    data_gpu = gpus[self.data_gpu_id]
                    model_gpu = gpus[self.model_gpu_id]
                    
                    with self.lock:
                        self.samples['data_gpu_util'].append(data_gpu.load * 100)
                        self.samples['model_gpu_util'].append(model_gpu.load * 100)
                        self.samples['data_gpu_memory'].append(data_gpu.memoryUsed)
                        self.samples['model_gpu_memory'].append(model_gpu.memoryUsed)
                        self.samples['timestamps'].append(time.time())
                
                time.sleep(0.1)  # Sample every 100ms
                
            except Exception as e:
                print(f"GPU monitoring error: {e}")
                break
    
    def get_statistics(self) -> Dict[str, any]:
        """Get monitoring statistics."""
        with self.lock:
            if not self.samples['data_gpu_util']:
                return {}
            
            return {
                'data_gpu': {
                    'util_avg': np.mean(self.samples['data_gpu_util']),
                    'util_std': np.std(self.samples['data_gpu_util']),
                    'util_min': np.min(self.samples['data_gpu_util']),
                    'util_max': np.max(self.samples['data_gpu_util']),
                    'memory_avg': np.mean(self.samples['data_gpu_memory']),
                    'memory_max': np.max(self.samples['data_gpu_memory']),
                },
                'model_gpu': {
                    'util_avg': np.mean(self.samples['model_gpu_util']),
                    'util_std': np.std(self.samples['model_gpu_util']),
                    'util_min': np.min(self.samples['model_gpu_util']),
                    'util_max': np.max(self.samples['model_gpu_util']),
                    'memory_avg': np.mean(self.samples['model_gpu_memory']),
                    'memory_max': np.max(self.samples['model_gpu_memory']),
                },
                'samples': len(self.samples['data_gpu_util'])
            }


class DualGPUBottleneckProfiler:
    """
    Comprehensive profiler for dual-GPU training pipeline.
    
    Identifies bottlenecks in:
    - Data loading and preprocessing
    - GPU-to-GPU transfers
    - Model training operations
    - Pipeline synchronization
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize profiler."""
        self.config_path = config_path
        if Path(config_path).exists():
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            print(f"⚠️  Config not found: {config_path}, using defaults")
            self.config = XTTSConfig()
        
        # Timing storage
        self.timings = {
            'data_load': [],
            'data_preprocess': [],
            'gpu_transfer': [],
            'model_forward': [],
            'loss_compute': [],
            'backward': [],
            'optimizer': [],
            'total_step': [],
            'data_to_train_gap': [],  # Time between data ready and training start
            'train_to_data_gap': [],  # Time between training end and next data ready
        }
        
        self.gpu_monitor = None
    
    def setup_gpus(self, data_gpu_id: int, model_gpu_id: int) -> bool:
        """
        Setup dual-GPU configuration.
        
        Args:
            data_gpu_id: Physical GPU ID for data processing
            model_gpu_id: Physical GPU ID for model training
            
        Returns:
            True if setup successful
        """
        print("\n" + "="*70)
        print("DUAL-GPU SETUP")
        print("="*70)
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        print(f"Available GPUs: {len(gpus)}")
        
        if len(gpus) < 2:
            print("❌ At least 2 GPUs required for dual-GPU profiling")
            return False
        
        if data_gpu_id >= len(gpus) or model_gpu_id >= len(gpus):
            print(f"❌ Invalid GPU IDs: data={data_gpu_id}, model={model_gpu_id}")
            return False
        
        try:
            # Set visible devices to only the GPUs we want to use
            selected_gpus = [gpus[data_gpu_id], gpus[model_gpu_id]]
            tf.config.set_visible_devices(selected_gpus, 'GPU')
            
            # Enable memory growth
            for gpu in selected_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"✅ GPU {data_gpu_id} → Logical GPU:0 (Data Processing)")
            print(f"✅ GPU {model_gpu_id} → Logical GPU:1 (Model Training)")
            
            return True
            
        except Exception as e:
            print(f"❌ GPU setup failed: {e}")
            return False
    
    def create_dummy_model(self) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer]:
        """
        Create a dummy model that simulates XTTS workload.
        
        Returns:
            Tuple of (model, optimizer)
        """
        print("\n" + "="*70)
        print("CREATING DUMMY MODEL")
        print("="*70)
        
        # Model parameters
        vocab_size = 256
        text_dim = getattr(self.config.model, 'text_encoder_dim', 512)
        decoder_dim = getattr(self.config.model, 'decoder_dim', 512)
        n_mels = getattr(self.config.model, 'n_mels', 80)
        
        # Build model on model GPU
        with tf.device('/GPU:1'):
            # Input layers
            text_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='text')
            mel_target = tf.keras.layers.Input(shape=(None, n_mels), dtype=tf.float32, name='mel')
            
            # Text encoder
            x = tf.keras.layers.Embedding(vocab_size, text_dim)(text_input)
            
            # Transformer layers (simulate XTTS encoder)
            for i in range(4):
                attn = tf.keras.layers.MultiHeadAttention(
                    num_heads=8, key_dim=text_dim // 8
                )(x, x)
                x = tf.keras.layers.Add()([x, attn])
                x = tf.keras.layers.LayerNormalization()(x)
                
                ff = tf.keras.layers.Dense(text_dim * 4, activation='relu')(x)
                ff = tf.keras.layers.Dense(text_dim)(ff)
                x = tf.keras.layers.Add()([x, ff])
                x = tf.keras.layers.LayerNormalization()(x)
            
            # Decoder
            decoder = tf.keras.layers.Dense(decoder_dim, activation='relu')(x)
            for i in range(3):
                decoder = tf.keras.layers.Dense(decoder_dim, activation='relu')(decoder)
            
            # Output
            output = tf.keras.layers.Dense(n_mels)(decoder)
            
            model = tf.keras.Model(inputs=[text_input, mel_target], outputs=output)
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        print(f"✅ Model created on GPU:1")
        print(f"   Text encoder dim: {text_dim}")
        print(f"   Decoder dim: {decoder_dim}")
        print(f"   Mel channels: {n_mels}")
        print(f"   Parameters: {model.count_params():,}")
        
        return model, optimizer
    
    def create_dummy_dataset(self, batch_size: int, num_batches: int) -> tf.data.Dataset:
        """
        Create a dummy dataset for profiling.
        
        This dataset simulates the data loading pipeline without needing actual data.
        
        Args:
            batch_size: Batch size
            num_batches: Number of batches
            
        Returns:
            TensorFlow dataset
        """
        print("\n" + "="*70)
        print("CREATING DUMMY DATASET")
        print("="*70)
        print(f"Batch size: {batch_size}")
        print(f"Number of batches: {num_batches}")
        
        n_mels = getattr(self.config.model, 'n_mels', 80)
        
        def data_generator():
            """Generator for dummy data."""
            for i in range(num_batches):
                # Simulate variable-length sequences
                seq_len = np.random.randint(50, 200)
                mel_len = np.random.randint(100, 500)
                
                # Generate dummy data
                text_tokens = np.random.randint(0, 256, size=(batch_size, seq_len), dtype=np.int32)
                mel_spec = np.random.randn(batch_size, mel_len, n_mels).astype(np.float32)
                
                yield text_tokens, mel_spec
        
        dataset = tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, None), dtype=tf.int32),
                tf.TensorSpec(shape=(batch_size, None, n_mels), dtype=tf.float32)
            )
        )
        
        # Apply prefetching
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        print("✅ Dummy dataset created")
        
        return dataset
    
    @tf.function
    def preprocess_on_data_gpu(self, text_tokens: tf.Tensor, mel_spec: tf.Tensor):
        """
        Simulate preprocessing on data GPU.
        
        Args:
            text_tokens: Text tokens
            mel_spec: Mel spectrogram
            
        Returns:
            Preprocessed tensors
        """
        with tf.device('/GPU:0'):
            # Ensure data is on GPU:0
            text_tokens = tf.identity(text_tokens)
            mel_spec = tf.identity(mel_spec)
            
            # Simulate some preprocessing work
            # This simulates operations like normalization, augmentation, etc.
            mel_spec = tf.nn.l2_normalize(mel_spec, axis=-1)
            
            return text_tokens, mel_spec
    
    @tf.function
    def transfer_to_model_gpu(self, text_tokens: tf.Tensor, mel_spec: tf.Tensor):
        """
        Transfer data from GPU:0 to GPU:1.
        
        Args:
            text_tokens: Text tokens on GPU:0
            mel_spec: Mel spec on GPU:0
            
        Returns:
            Tensors on GPU:1
        """
        with tf.device('/GPU:1'):
            text_tokens = tf.identity(text_tokens)
            mel_spec = tf.identity(mel_spec)
            return text_tokens, mel_spec
    
    @tf.function
    def training_step(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        text_tokens: tf.Tensor,
        mel_spec: tf.Tensor
    ):
        """
        Execute one training step on model GPU.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            text_tokens: Text tokens on GPU:1
            mel_spec: Mel spectrogram on GPU:1
            
        Returns:
            Loss value
        """
        with tf.device('/GPU:1'):
            with tf.GradientTape() as tape:
                # Forward pass
                predictions = model([text_tokens, mel_spec], training=True)
                
                # Loss computation
                loss = tf.reduce_mean(tf.square(predictions - mel_spec))
            
            # Backward pass
            gradients = tape.gradient(loss, model.trainable_variables)
            
            # Optimizer step
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return loss
    
    def profile_pipeline(
        self,
        batch_size: int = 16,
        num_steps: int = 100,
        data_gpu_id: int = 0,
        model_gpu_id: int = 1,
        enable_gpu_monitoring: bool = True
    ) -> Dict[str, any]:
        """
        Profile the complete dual-GPU pipeline.
        
        Args:
            batch_size: Batch size
            num_steps: Number of steps to profile
            data_gpu_id: Physical GPU ID for data
            model_gpu_id: Physical GPU ID for model
            enable_gpu_monitoring: Enable background GPU monitoring
            
        Returns:
            Dictionary with profiling results
        """
        print("\n" + "="*70)
        print("DUAL-GPU PIPELINE PROFILING")
        print("="*70)
        print(f"Batch size: {batch_size}")
        print(f"Number of steps: {num_steps}")
        print(f"Data GPU: {data_gpu_id}")
        print(f"Model GPU: {model_gpu_id}")
        
        # Setup GPUs
        if not self.setup_gpus(data_gpu_id, model_gpu_id):
            return {}
        
        # Start GPU monitoring
        if enable_gpu_monitoring:
            self.gpu_monitor = GPUMonitor(data_gpu_id, model_gpu_id)
            self.gpu_monitor.start()
            print("✅ GPU monitoring started")
        
        # Create model and dataset
        model, optimizer = self.create_dummy_model()
        dataset = self.create_dummy_dataset(batch_size, num_steps + 10)  # Extra batches for warmup
        
        # Warm-up phase
        print("\n" + "="*70)
        print("WARM-UP PHASE")
        print("="*70)
        print("Warming up (5 steps)...")
        
        iterator = iter(dataset)
        for i in range(5):
            try:
                text_tokens, mel_spec = next(iterator)
                text_tokens, mel_spec = self.preprocess_on_data_gpu(text_tokens, mel_spec)
                text_tokens, mel_spec = self.transfer_to_model_gpu(text_tokens, mel_spec)
                loss = self.training_step(model, optimizer, text_tokens, mel_spec)
                _ = loss.numpy()  # Force synchronization
            except Exception as e:
                print(f"⚠️  Warmup error: {e}")
                break
        
        print("✅ Warm-up complete")
        
        # Profiling phase
        print("\n" + "="*70)
        print("PROFILING PHASE")
        print("="*70)
        
        last_data_ready_time = None
        last_train_end_time = None
        
        for step in range(num_steps):
            step_start = time.perf_counter()
            
            try:
                # Phase 1: Data loading (from iterator)
                data_load_start = time.perf_counter()
                text_tokens, mel_spec = next(iterator)
                # Force data to be loaded
                _ = text_tokens.numpy()
                data_load_time = (time.perf_counter() - data_load_start) * 1000
                
                # Phase 2: Data preprocessing on GPU:0
                preprocess_start = time.perf_counter()
                text_tokens, mel_spec = self.preprocess_on_data_gpu(text_tokens, mel_spec)
                # Sync to measure preprocessing time
                _ = mel_spec.numpy()
                preprocess_time = (time.perf_counter() - preprocess_start) * 1000
                
                data_ready_time = time.perf_counter()
                
                # Measure gap between data ready and training start
                if last_train_end_time is not None:
                    train_to_data_gap = (data_ready_time - last_train_end_time) * 1000
                    self.timings['train_to_data_gap'].append(train_to_data_gap)
                
                # Phase 3: Transfer to GPU:1
                transfer_start = time.perf_counter()
                text_tokens, mel_spec = self.transfer_to_model_gpu(text_tokens, mel_spec)
                _ = mel_spec.numpy()
                transfer_time = (time.perf_counter() - transfer_start) * 1000
                
                train_start = time.perf_counter()
                
                # Measure gap between data ready and training start
                if last_data_ready_time is not None:
                    data_to_train_gap = (train_start - last_data_ready_time) * 1000
                    self.timings['data_to_train_gap'].append(data_to_train_gap)
                
                # Phase 4-7: Training on GPU:1 (forward, loss, backward, optimizer)
                train_total_start = time.perf_counter()
                loss = self.training_step(model, optimizer, text_tokens, mel_spec)
                _ = loss.numpy()  # Force synchronization
                train_total_time = (time.perf_counter() - train_total_start) * 1000
                
                train_end_time = time.perf_counter()
                step_time = (train_end_time - step_start) * 1000
                
                # Store timings
                self.timings['data_load'].append(data_load_time)
                self.timings['data_preprocess'].append(preprocess_time)
                self.timings['gpu_transfer'].append(transfer_time)
                self.timings['model_forward'].append(train_total_time)  # Combined training time
                self.timings['total_step'].append(step_time)
                
                last_data_ready_time = data_ready_time
                last_train_end_time = train_end_time
                
                # Progress reporting
                if step < 5 or step % 20 == 0 or step == num_steps - 1:
                    print(f"Step {step+1:3d}: "
                          f"Total={step_time:6.1f}ms | "
                          f"Load={data_load_time:5.1f}ms | "
                          f"Prep={preprocess_time:5.1f}ms | "
                          f"Xfer={transfer_time:5.1f}ms | "
                          f"Train={train_total_time:6.1f}ms")
                
            except StopIteration:
                print(f"⚠️  Dataset exhausted after {step} steps")
                break
            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Stop GPU monitoring
        if self.gpu_monitor:
            self.gpu_monitor.stop()
        
        # Analyze results
        return self._analyze_results()
    
    def _analyze_results(self) -> Dict[str, any]:
        """Analyze profiling results and identify bottlenecks."""
        print("\n" + "="*70)
        print("PROFILING RESULTS ANALYSIS")
        print("="*70)
        
        if not self.timings['total_step']:
            print("❌ No timing data collected")
            return {}
        
        # Calculate statistics for each phase
        results = {}
        phases = ['data_load', 'data_preprocess', 'gpu_transfer', 'model_forward', 'total_step']
        
        print("\nTIMING BREAKDOWN:")
        print("-" * 70)
        
        total_avg = np.mean(self.timings['total_step'])
        
        for phase in phases:
            if self.timings[phase]:
                avg = np.mean(self.timings[phase])
                std = np.std(self.timings[phase])
                min_val = np.min(self.timings[phase])
                max_val = np.max(self.timings[phase])
                p95 = np.percentile(self.timings[phase], 95)
                
                pct = (avg / total_avg * 100) if total_avg > 0 and phase != 'total_step' else 100
                
                results[phase] = {
                    'avg': avg,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'p95': p95,
                    'percentage': pct
                }
                
                phase_name = phase.replace('_', ' ').title()
                print(f"{phase_name:20s}: {avg:6.1f}ms ± {std:5.1f}ms "
                      f"(min={min_val:5.1f}, max={max_val:6.1f}, p95={p95:6.1f}) "
                      f"[{pct:5.1f}%]")
        
        # Analyze gaps
        print("\nPIPELINE GAPS:")
        print("-" * 70)
        
        if self.timings['data_to_train_gap']:
            gap_avg = np.mean(self.timings['data_to_train_gap'])
            gap_std = np.std(self.timings['data_to_train_gap'])
            print(f"Data→Train Gap:      {gap_avg:6.1f}ms ± {gap_std:5.1f}ms")
            results['data_to_train_gap'] = gap_avg
        
        if self.timings['train_to_data_gap']:
            gap_avg = np.mean(self.timings['train_to_data_gap'])
            gap_std = np.std(self.timings['train_to_data_gap'])
            print(f"Train→Data Gap:      {gap_avg:6.1f}ms ± {gap_std:5.1f}ms")
            results['train_to_data_gap'] = gap_avg
        
        # GPU monitoring statistics
        if self.gpu_monitor:
            gpu_stats = self.gpu_monitor.get_statistics()
            if gpu_stats:
                print("\nGPU UTILIZATION:")
                print("-" * 70)
                print(f"Data GPU:  {gpu_stats['data_gpu']['util_avg']:5.1f}% ± {gpu_stats['data_gpu']['util_std']:4.1f}% "
                      f"(min={gpu_stats['data_gpu']['util_min']:5.1f}%, max={gpu_stats['data_gpu']['util_max']:5.1f}%)")
                print(f"Model GPU: {gpu_stats['model_gpu']['util_avg']:5.1f}% ± {gpu_stats['model_gpu']['util_std']:4.1f}% "
                      f"(min={gpu_stats['model_gpu']['util_min']:5.1f}%, max={gpu_stats['model_gpu']['util_max']:5.1f}%)")
                print(f"\nGPU MEMORY:")
                print(f"Data GPU:  {gpu_stats['data_gpu']['memory_avg']:.0f}MB (max={gpu_stats['data_gpu']['memory_max']:.0f}MB)")
                print(f"Model GPU: {gpu_stats['model_gpu']['memory_avg']:.0f}MB (max={gpu_stats['model_gpu']['memory_max']:.0f}MB)")
                
                results['gpu_stats'] = gpu_stats
        
        # Bottleneck identification
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS")
        print("="*70)
        
        bottlenecks = []
        
        # Check each phase
        data_load_pct = results.get('data_load', {}).get('percentage', 0)
        transfer_pct = results.get('gpu_transfer', {}).get('percentage', 0)
        train_pct = results.get('model_forward', {}).get('percentage', 0)
        
        if data_load_pct > 20:
            bottlenecks.append('DATA_LOADING')
            print(f"🔴 DATA LOADING BOTTLENECK: {data_load_pct:.1f}% of time")
            print("   Recommendations:")
            print("   - Increase prefetch buffer size")
            print("   - Use more parallel data loading workers")
            print("   - Consider caching preprocessed data")
        
        if transfer_pct > 15:
            bottlenecks.append('GPU_TRANSFER')
            print(f"🔴 GPU TRANSFER BOTTLENECK: {transfer_pct:.1f}% of time")
            print("   Recommendations:")
            print("   - Implement async transfer with double/triple buffering")
            print("   - Overlap transfer with computation")
            print("   - Consider keeping data on same GPU if possible")
        
        if train_pct < 50:
            bottlenecks.append('UNDERUTILIZED_GPU')
            print(f"🔴 GPU UNDERUTILIZED: Training only {train_pct:.1f}% of time")
            print("   GPU is waiting for data!")
            print("   Recommendations:")
            print("   - Increase async pipeline depth")
            print("   - Improve data loading and preprocessing speed")
            print("   - Add more buffering")
        
        # Check for high variation (indicates oscillation)
        total_var = results.get('total_step', {}).get('std', 0) / results.get('total_step', {}).get('avg', 1)
        if total_var > 0.3:
            bottlenecks.append('OSCILLATION')
            print(f"🔴 HIGH TIMING VARIATION: {total_var:.1%}")
            print("   This indicates cyclic bottleneck pattern")
            print("   Recommendations:")
            print("   - Increase pipeline depth")
            print("   - Balance workload between GPUs")
        
        # Check GPU utilization
        if self.gpu_monitor and results.get('gpu_stats'):
            data_util = results['gpu_stats']['data_gpu']['util_avg']
            model_util = results['gpu_stats']['model_gpu']['util_avg']
            
            if model_util < 70:
                bottlenecks.append('LOW_MODEL_GPU_UTIL')
                print(f"🔴 LOW MODEL GPU UTILIZATION: {model_util:.1f}%")
                print("   Model GPU is idle too much!")
                print("   Recommendations:")
                print("   - Increase batch size if memory allows")
                print("   - Improve data pipeline to keep GPU busy")
                print("   - Add async execution and buffering")
            
            if data_util > 80:
                bottlenecks.append('DATA_GPU_OVERLOADED')
                print(f"⚠️  DATA GPU OVERLOADED: {data_util:.1f}%")
                print("   Data GPU may be bottleneck")
                print("   Recommendations:")
                print("   - Reduce preprocessing on data GPU")
                print("   - Move some preprocessing to CPU")
                print("   - Use simpler data augmentation")
        
        if not bottlenecks:
            print("✅ NO MAJOR BOTTLENECKS DETECTED")
            print("   Pipeline appears well-balanced")
            print(f"   Training utilization: {train_pct:.1f}%")
        
        results['bottlenecks'] = bottlenecks
        
        # Recommendations summary
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        if 'DATA_LOADING' in bottlenecks or 'UNDERUTILIZED_GPU' in bottlenecks:
            print("\n1. IMPROVE DATA PIPELINE:")
            print("   - Increase prefetch: dataset.prefetch(AUTOTUNE)")
            print("   - Add more workers: num_parallel_calls=16 or higher")
            print("   - Use dataset.cache() for small datasets")
            print("   - Prefetch to GPU: prefetch_to_device('/GPU:0', buffer_size=4)")
        
        if 'GPU_TRANSFER' in bottlenecks or 'OSCILLATION' in bottlenecks:
            print("\n2. OPTIMIZE GPU-TO-GPU TRANSFER:")
            print("   - Implement async transfer pipeline")
            print("   - Use double or triple buffering")
            print("   - Overlap transfer with computation")
            print("   - Pipeline pattern: prep_n, transfer_n, train_n-1")
        
        if 'UNDERUTILIZED_GPU' in bottlenecks or 'LOW_MODEL_GPU_UTIL' in bottlenecks:
            print("\n3. INCREASE PIPELINE PARALLELISM:")
            print("   - Use tf.data.experimental.prefetch_to_device()")
            print("   - Implement producer-consumer pattern with queues")
            print("   - Increase buffer sizes (4-8 batches)")
            print("   - Use async execution for all stages")
        
        print("\n4. CONFIGURATION TUNING:")
        print("   Current batch size: Consider increasing if memory allows")
        print("   Target GPU utilization: >80% for both GPUs")
        print("   Target throughput: Minimize gaps between operations")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Dual-GPU pipeline bottleneck profiler"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for profiling"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Number of steps to profile"
    )
    parser.add_argument(
        "--data-gpu",
        type=int,
        default=0,
        help="Physical GPU ID for data processing"
    )
    parser.add_argument(
        "--model-gpu",
        type=int,
        default=1,
        help="Physical GPU ID for model training"
    )
    parser.add_argument(
        "--no-gpu-monitoring",
        action="store_true",
        help="Disable background GPU monitoring"
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DUAL-GPU PIPELINE BOTTLENECK PROFILER")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Batch size: {args.batch_size}")
    print(f"Steps: {args.num_steps}")
    print(f"Data GPU: {args.data_gpu}")
    print(f"Model GPU: {args.model_gpu}")
    
    # Create profiler
    profiler = DualGPUBottleneckProfiler(args.config)
    
    # Run profiling
    results = profiler.profile_pipeline(
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        data_gpu_id=args.data_gpu,
        model_gpu_id=args.model_gpu,
        enable_gpu_monitoring=not args.no_gpu_monitoring
    )
    
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    if results:
        if results.get('bottlenecks'):
            print(f"  Bottlenecks identified: {', '.join(results['bottlenecks'])}")
        else:
            print("  No major bottlenecks found")
        
        if results.get('total_step'):
            print(f"  Average step time: {results['total_step']['avg']:.1f}ms")
            throughput = 1000.0 / results['total_step']['avg']
            print(f"  Throughput: {throughput:.1f} steps/second")
    
    print("\nNext steps:")
    print("1. Review bottleneck analysis and recommendations above")
    print("2. Apply suggested optimizations to memory_isolated_trainer.py")
    print("3. Re-run profiling to verify improvements")
    print("4. Test with actual training: python train_main.py --enable-memory-isolation ...")


if __name__ == "__main__":
    main()
