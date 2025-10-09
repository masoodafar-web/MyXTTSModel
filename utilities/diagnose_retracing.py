#!/usr/bin/env python3
"""
Diagnostic script to detect tf.function retracing issues before training.

This script runs a few training steps and monitors for retracing warnings,
helping identify configuration or data pipeline issues that could cause
GPU utilization problems.

Usage:
    python utilities/diagnose_retracing.py --config configs/config.yaml
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import tensorflow as tf
import numpy as np

from myxtts.config.config import load_config
from myxtts.data.ljspeech import LJSpeechDataset
from myxtts.models.xtts import XTTS
from myxtts.training.trainer import XTTSTrainer


def diagnose_retracing(config_path: str, num_steps: int = 10):
    """
    Run diagnostic test to detect retracing issues.
    
    Args:
        config_path: Path to config file
        num_steps: Number of training steps to test
    """
    print("=" * 80)
    print("TF.FUNCTION RETRACING DIAGNOSTIC TOOL")
    print("=" * 80)
    print()
    
    # Load config
    print(f"📝 Loading config from: {config_path}")
    config = load_config(config_path)
    
    # Check critical settings
    print()
    print("⚙️  CONFIG VALIDATION:")
    print("-" * 80)
    
    pad_to_fixed = getattr(config.data, 'pad_to_fixed_length', False)
    max_text_len = getattr(config.data, 'max_text_length', None)
    max_mel_frames = getattr(config.data, 'max_mel_frames', None)
    batch_size = getattr(config.data, 'batch_size', None)
    
    print(f"  pad_to_fixed_length: {pad_to_fixed}")
    print(f"  max_text_length: {max_text_len}")
    print(f"  max_mel_frames: {max_mel_frames}")
    print(f"  batch_size: {batch_size}")
    
    if not pad_to_fixed:
        print()
        print("❌ ERROR: pad_to_fixed_length is not enabled!")
        print("   This will cause retracing and poor GPU utilization.")
        print()
        print("   Fix: Add to your config.yaml:")
        print("   data:")
        print("     pad_to_fixed_length: true")
        print("     max_text_length: 200")
        print("     max_mel_frames: 800")
        return False
    
    if not max_text_len or not max_mel_frames:
        print()
        print("❌ ERROR: max_text_length or max_mel_frames not set!")
        print("   These are required when pad_to_fixed_length is enabled.")
        return False
    
    if not batch_size:
        print()
        print("⚠️  WARNING: batch_size not fixed in config!")
        print("   For optimal performance, set a fixed batch_size.")
    
    print()
    print("✅ Config validation passed!")
    
    # Create dataset
    print()
    print("📊 CREATING DATASET:")
    print("-" * 80)
    
    dataset = LJSpeechDataset(
        data_path=config.data.dataset_path,
        config=config.data,
        subset="train"
    )
    
    train_ds = dataset.create_tf_dataset(
        batch_size=batch_size or 4,
        shuffle=False,  # Don't shuffle for diagnostic
        repeat=False,
        prefetch=True,
        drop_remainder=True  # Important for consistent shapes
    )
    
    print(f"✅ Dataset created")
    
    # Get a few batches and check shapes
    print()
    print("🔍 CHECKING BATCH SHAPES:")
    print("-" * 80)
    
    batch_shapes = []
    for i, batch in enumerate(train_ds.take(5)):
        text_seq, mel_spec, text_len, mel_len = batch
        text_shape = text_seq.shape
        mel_shape = mel_spec.shape
        
        batch_shapes.append((text_shape, mel_shape))
        print(f"  Batch {i+1}: text={text_shape}, mel={mel_shape}")
        
        # Check if shapes are fully defined (static)
        if None in text_shape or None in mel_shape:
            print(f"    ⚠️  WARNING: Batch {i+1} has dynamic shapes!")
    
    # Check if all batches have the same shape
    print()
    if len(set(batch_shapes)) == 1:
        print("✅ All batches have consistent shapes (good!)")
    else:
        print("❌ ERROR: Batches have different shapes!")
        print("   This will cause retracing on every batch.")
        print()
        print("   Unique shapes found:")
        for shape in set(batch_shapes):
            print(f"     {shape}")
        return False
    
    # Create model and trainer
    print()
    print("🤖 CREATING MODEL AND TRAINER:")
    print("-" * 80)
    
    model = XTTS(config)
    trainer = XTTSTrainer(config, model=model)
    
    # Prepare dataset for training
    train_ds_prepared = dataset.create_tf_dataset(
        batch_size=batch_size or 4,
        shuffle=False,
        repeat=True,
        prefetch=True,
        drop_remainder=True
    )
    
    print("✅ Model and trainer created")
    
    # Monitor retracing
    print()
    print(f"🏃 RUNNING {num_steps} TRAINING STEPS:")
    print("-" * 80)
    
    retracing_count = 0
    step_times = []
    
    # Get initial retrace count
    if hasattr(trainer, '_retrace_count'):
        initial_retrace_count = trainer._retrace_count
    else:
        initial_retrace_count = 0
    
    dataset_iter = iter(train_ds_prepared)
    
    for step in range(num_steps):
        batch = next(dataset_iter)
        text_seq, mel_spec, text_len, mel_len = batch
        
        start_time = time.time()
        
        try:
            losses = trainer.train_step(text_seq, mel_spec, text_len, mel_len)
            
            step_time = time.time() - start_time
            step_times.append(step_time)
            
            # Check if retracing occurred
            current_retrace_count = getattr(trainer, '_retrace_count', 0)
            if current_retrace_count > initial_retrace_count + retracing_count:
                retracing_count = current_retrace_count - initial_retrace_count
                print(f"  ⚠️  Step {step+1}: RETRACING DETECTED (total: {retracing_count}) - time: {step_time:.2f}s")
            else:
                loss_val = float(losses.get('total_loss', 0))
                print(f"  ✅ Step {step+1}: loss={loss_val:.4f}, time={step_time:.2f}s")
        
        except Exception as e:
            print(f"  ❌ Step {step+1}: ERROR - {e}")
            return False
    
    # Summary
    print()
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY:")
    print("=" * 80)
    
    avg_step_time = np.mean(step_times)
    std_step_time = np.std(step_times)
    
    print(f"  Total steps: {num_steps}")
    print(f"  Retracing events: {retracing_count}")
    print(f"  Average step time: {avg_step_time:.3f}s ± {std_step_time:.3f}s")
    
    if retracing_count == 0:
        print()
        print("✅ SUCCESS: No retracing detected!")
        print("   Your configuration is optimized for stable GPU utilization.")
        return True
    elif retracing_count <= 1:
        print()
        print("⚠️  WARNING: Minimal retracing detected (1 event)")
        print("   This is normal for the first batch (compilation).")
        print("   Your configuration should work well.")
        return True
    else:
        print()
        print(f"❌ FAILURE: {retracing_count} retracing events detected!")
        print("   This will cause severe GPU utilization issues.")
        print()
        print("   Possible causes:")
        print("   1. Dynamic shapes in data pipeline")
        print("   2. pad_to_fixed_length not properly applied")
        print("   3. drop_remainder not set in dataset")
        print("   4. Python objects passed to tf.function")
        return False


def main():
    parser = argparse.ArgumentParser(description='Diagnose tf.function retracing issues')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of training steps to test')
    
    args = parser.parse_args()
    
    success = diagnose_retracing(args.config, args.steps)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
