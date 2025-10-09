#!/usr/bin/env python3
"""
Training Step Profiler - Profile complete training step including data loading and model execution

This tool profiles the entire training loop to identify if bottleneck is in:
1. Data loading
2. Model forward pass
3. Loss computation
4. Backward pass (gradient computation)
5. Optimizer step
6. GPU memory transfers

Usage:
    python utilities/training_step_profiler.py --data-path ./data --num-steps 100
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

from myxtts.config.config import XTTSConfig


class TrainingStepProfiler:
    """Profile complete training steps to identify bottlenecks."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize profiler."""
        self.config_path = config_path
        if Path(config_path).exists():
            self.config = XTTSConfig.from_yaml(config_path)
        else:
            print(f"⚠️  Config not found: {config_path}, using defaults")
            self.config = XTTSConfig()
        
        # Timing storage
        self.data_load_times: List[float] = []
        self.forward_pass_times: List[float] = []
        self.loss_compute_times: List[float] = []
        self.backward_pass_times: List[float] = []
        self.optimizer_step_times: List[float] = []
        self.total_step_times: List[float] = []
    
    def create_dummy_model(self, batch_size: int = 8) -> Tuple[tf.keras.Model, tf.keras.optimizers.Optimizer]:
        """
        Create a dummy model for profiling.
        
        This simulates the computational load of a real TTS model
        without needing actual trained weights.
        """
        print("\n" + "="*70)
        print("CREATING DUMMY MODEL FOR PROFILING")
        print("="*70)
        
        # Create a model that simulates TTS workload
        # Text encoder -> Decoder -> Loss
        
        text_dim = self.config.model.text_encoder_dim
        decoder_dim = self.config.model.decoder_dim
        n_mels = self.config.model.n_mels
        
        # Input layers
        text_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='text_tokens')
        text_len = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='text_length')
        mel_target = tf.keras.layers.Input(shape=(None, n_mels), dtype=tf.float32, name='mel_target')
        mel_len = tf.keras.layers.Input(shape=(), dtype=tf.int32, name='mel_length')
        
        # Text encoder (embedding + transformer layers)
        x = tf.keras.layers.Embedding(256, text_dim, name='text_embedding')(text_input)
        
        for i in range(3):  # Reduced layers for dummy model
            # Multi-head attention
            attn_output = tf.keras.layers.MultiHeadAttention(
                num_heads=8,
                key_dim=text_dim // 8,
                name=f'text_attn_{i}'
            )(x, x)
            x = tf.keras.layers.Add()([x, attn_output])
            x = tf.keras.layers.LayerNormalization()(x)
            
            # Feed-forward
            ff = tf.keras.layers.Dense(text_dim * 4, activation='relu')(x)
            ff = tf.keras.layers.Dense(text_dim)(ff)
            x = tf.keras.layers.Add()([x, ff])
            x = tf.keras.layers.LayerNormalization()(x)
        
        # Decoder (predicts mel spectrogram)
        decoder = tf.keras.layers.Dense(decoder_dim, activation='relu', name='decoder_proj')(x)
        
        for i in range(3):  # Reduced layers
            decoder = tf.keras.layers.Dense(decoder_dim, activation='relu', name=f'decoder_{i}')(decoder)
        
        # Output projection
        mel_output = tf.keras.layers.Dense(n_mels, name='mel_output')(decoder)
        
        # Create model
        model = tf.keras.Model(
            inputs=[text_input, text_len, mel_target, mel_len],
            outputs=mel_output,
            name='dummy_tts_model'
        )
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        
        print(f"✅ Dummy model created")
        print(f"   Text encoder dim: {text_dim}")
        print(f"   Decoder dim: {decoder_dim}")
        print(f"   Mel channels: {n_mels}")
        print(f"   Total parameters: {model.count_params():,}")
        
        return model, optimizer
    
    def profile_training_steps(
        self,
        data_path: str,
        num_steps: int = 100,
        batch_size: int = 8,
        use_xla: bool = True,
        use_mixed_precision: bool = True
    ) -> Dict[str, any]:
        """
        Profile complete training steps.
        
        Args:
            data_path: Path to dataset
            num_steps: Number of training steps to profile
            batch_size: Batch size
            use_xla: Enable XLA compilation
            use_mixed_precision: Enable mixed precision
            
        Returns:
            Dictionary with profiling results
        """
        print("\n" + "="*70)
        print("TRAINING STEP PROFILING")
        print("="*70)
        print(f"Number of steps: {num_steps}")
        print(f"Batch size: {batch_size}")
        print(f"XLA compilation: {use_xla}")
        print(f"Mixed precision: {use_mixed_precision}")
        
        # Setup mixed precision
        if use_mixed_precision:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ Mixed precision enabled")
        
        # Create model and optimizer
        model, optimizer = self.create_dummy_model(batch_size)
        
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
            shuffle=True,
            repeat=True,
            prefetch=True,
            memory_cache=False,
            num_parallel_calls=self.config.data.num_workers,
        )
        
        # Loss function
        loss_fn = tf.keras.losses.MeanSquaredError()
        
        # Training step function
        if use_xla:
            @tf.function(jit_compile=True)
            def train_step(text_tokens, mel_spec, text_len, mel_len):
                with tf.GradientTape() as tape:
                    # Forward pass
                    predictions = model([text_tokens, text_len, mel_spec, mel_len], training=True)
                    
                    # Compute loss
                    loss = loss_fn(mel_spec, predictions)
                    
                    # Scale loss for mixed precision
                    if use_mixed_precision:
                        loss = optimizer.get_scaled_loss(loss)
                
                # Backward pass
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # Unscale gradients for mixed precision
                if use_mixed_precision:
                    gradients = optimizer.get_unscaled_gradients(gradients)
                
                # Optimizer step
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                return loss
        else:
            @tf.function
            def train_step(text_tokens, mel_spec, text_len, mel_len):
                with tf.GradientTape() as tape:
                    predictions = model([text_tokens, text_len, mel_spec, mel_len], training=True)
                    loss = loss_fn(mel_spec, predictions)
                
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                return loss
        
        # Profiling loop
        print(f"\n{'='*70}")
        print("PROFILING IN PROGRESS...")
        print("="*70)
        
        iterator = iter(tf_dataset)
        
        # Warm-up (compile graphs)
        print("Warm-up phase (compiling graphs)...")
        warmup_start = time.perf_counter()
        for i in range(5):
            try:
                batch = next(iterator)
                text_tokens, mel_spec, text_len, mel_len = batch
                _ = train_step(text_tokens, mel_spec, text_len, mel_len)
            except Exception as e:
                print(f"⚠️  Error during warmup: {e}")
                break
        warmup_time = time.perf_counter() - warmup_start
        print(f"Warm-up completed in {warmup_time:.2f}s")
        
        # Actual profiling
        print(f"\nProfiling {num_steps} steps...")
        
        for step in range(num_steps):
            step_start = time.perf_counter()
            
            try:
                # Data loading
                data_start = time.perf_counter()
                batch = next(iterator)
                text_tokens, mel_spec, text_len, mel_len = batch
                data_time = (time.perf_counter() - data_start) * 1000
                
                # Training step (includes forward, loss, backward, optimizer)
                train_start = time.perf_counter()
                loss = train_step(text_tokens, mel_spec, text_len, mel_len)
                
                # Force synchronization
                _ = loss.numpy()
                train_time = (time.perf_counter() - train_start) * 1000
                
                step_time = (time.perf_counter() - step_start) * 1000
                
                # Store timings
                self.data_load_times.append(data_time)
                self.total_step_times.append(step_time)
                # train_time includes forward + backward + optimizer
                self.forward_pass_times.append(train_time)
                
                if step < 5 or step % 20 == 0:
                    print(f"Step {step+1:3d}: Total={step_time:6.2f}ms, "
                          f"Data={data_time:5.2f}ms, Train={train_time:6.2f}ms, "
                          f"Loss={float(loss):.4f}")
                
            except StopIteration:
                print(f"⚠️  Dataset exhausted after {step} steps")
                break
            except Exception as e:
                print(f"❌ Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Analyze results
        return self._analyze_results()
    
    def _analyze_results(self) -> Dict[str, any]:
        """Analyze profiling results."""
        print(f"\n{'='*70}")
        print("PROFILING RESULTS")
        print("="*70)
        
        if not self.total_step_times:
            print("❌ No timing data collected")
            return {}
        
        # Calculate statistics
        total_avg = np.mean(self.total_step_times)
        total_std = np.std(self.total_step_times)
        
        data_avg = np.mean(self.data_load_times)
        data_std = np.std(self.data_load_times)
        
        train_avg = np.mean(self.forward_pass_times)
        train_std = np.std(self.forward_pass_times)
        
        # Calculate percentages
        data_pct = (data_avg / total_avg) * 100 if total_avg > 0 else 0
        train_pct = (train_avg / total_avg) * 100 if total_avg > 0 else 0
        
        print(f"\nTIMING BREAKDOWN:")
        print(f"  Total step:        {total_avg:6.2f}ms ± {total_std:5.2f}ms")
        print(f"  Data loading:      {data_avg:6.2f}ms ± {data_std:5.2f}ms ({data_pct:5.1f}%)")
        print(f"  Training (F+B+O):  {train_avg:6.2f}ms ± {train_std:5.2f}ms ({train_pct:5.1f}%)")
        
        # Throughput
        samples_per_sec = 1000.0 / total_avg if total_avg > 0 else 0
        print(f"\nTHROUGHPUT:")
        print(f"  Steps per second:  {samples_per_sec:.2f}")
        
        # Variation analysis
        total_var = total_std / total_avg if total_avg > 0 else 0
        data_var = data_std / data_avg if data_avg > 0 else 0
        train_var = train_std / train_avg if train_avg > 0 else 0
        
        print(f"\nVARIATION ANALYSIS:")
        print(f"  Total variation:   {total_var:5.2%}")
        print(f"  Data variation:    {data_var:5.2%}")
        print(f"  Train variation:   {train_var:5.2%}")
        
        # Bottleneck identification
        print(f"\n{'='*70}")
        print("BOTTLENECK ANALYSIS")
        print("="*70)
        
        bottlenecks = []
        
        if data_pct > 30:
            bottlenecks.append("DATA_LOADING")
            print("🔴 DATA LOADING BOTTLENECK")
            print(f"   Data loading takes {data_pct:.1f}% of total time")
            print("   Recommendations:")
            print("   - Ensure use_tf_native_loading: true")
            print("   - Increase num_workers (current: {})".format(self.config.data.num_workers))
            print("   - Increase prefetch_buffer_size")
            print("   - Use faster storage (SSD instead of HDD)")
        
        if data_var > 0.5:
            bottlenecks.append("DATA_VARIATION")
            print("🔴 HIGH DATA LOADING VARIATION")
            print(f"   Variation: {data_var:.2%}")
            print("   This indicates unstable data pipeline")
            print("   Recommendations:")
            print("   - Check for tf.numpy_function usage")
            print("   - Verify TF-native loading is working")
            print("   - Check storage I/O performance")
        
        if train_pct > 70:
            print("✅ MODEL TRAINING IS DOMINANT")
            print(f"   Training takes {train_pct:.1f}% of total time")
            print("   This is expected and indicates GPU is well-utilized")
        
        if total_var > 0.5:
            bottlenecks.append("OSCILLATION")
            print("🔴 HIGH OVERALL VARIATION DETECTED")
            print(f"   Total variation: {total_var:.2%}")
            print("   This indicates cyclic GPU utilization pattern")
        
        if not bottlenecks:
            print("✅ NO MAJOR BOTTLENECKS DETECTED")
            print("   Pipeline appears well-balanced")
        
        return {
            'total_avg': total_avg,
            'total_std': total_std,
            'data_avg': data_avg,
            'data_std': data_std,
            'train_avg': train_avg,
            'train_std': train_std,
            'data_percentage': data_pct,
            'train_percentage': train_pct,
            'total_variation': total_var,
            'data_variation': data_var,
            'train_variation': train_var,
            'bottlenecks': bottlenecks,
            'throughput': samples_per_sec
        }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile complete training steps"
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
        "--num-steps",
        type=int,
        default=100,
        help="Number of training steps to profile"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--no-xla",
        action="store_true",
        help="Disable XLA compilation"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision"
    )
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = TrainingStepProfiler(args.config)
    
    # Run profiling
    results = profiler.profile_training_steps(
        data_path=args.data_path,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        use_xla=not args.no_xla,
        use_mixed_precision=not args.no_mixed_precision
    )
    
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Review bottleneck analysis above")
    print("2. Apply recommended fixes")
    print("3. Re-run profiling to verify improvements")
    print("4. Use enhanced_gpu_profiler.py for data pipeline deep dive")


if __name__ == "__main__":
    main()
