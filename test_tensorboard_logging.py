#!/usr/bin/env python3
"""
Test script to verify TensorBoard logging of training samples.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
from myxtts.training.trainer import XTTSTrainer


def create_sample_images():
    """Create sample images in the training_samples directory."""
    samples_dir = Path("training_samples")
    samples_dir.mkdir(exist_ok=True)
    
    # Create a few sample images
    for i in range(3):
        # Create a random image
        fig, ax = plt.subplots(figsize=(8, 6))
        data = np.random.rand(100, 100)
        im = ax.imshow(data, cmap='viridis')
        ax.set_title(f"Sample Spectrogram {i+1}")
        plt.colorbar(im, ax=ax)
        
        # Save as PNG
        image_path = samples_dir / f"sample_image_{i+1}.png"
        plt.savefig(image_path)
        plt.close(fig)
        print(f"Created: {image_path}")


def create_sample_audio():
    """Create sample audio files in the training_samples directory."""
    samples_dir = Path("training_samples")
    samples_dir.mkdir(exist_ok=True)
    
    # Create a few sample audio files
    sample_rate = 22050
    duration = 2  # seconds
    
    for i in range(2):
        # Generate a simple sine wave
        t = np.linspace(0, duration, int(sample_rate * duration))
        frequency = 440 * (i + 1)  # A4 note and higher
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Add some harmonics for richer sound
        audio += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
        audio += 0.2 * np.sin(2 * np.pi * frequency * 3 * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Save as WAV
        audio_path = samples_dir / f"sample_audio_{i+1}.wav"
        audio_tensor = tf.constant(audio, dtype=tf.float32)
        audio_tensor = tf.reshape(audio_tensor, [-1, 1])
        wav_bytes = tf.audio.encode_wav(audio_tensor, sample_rate=sample_rate)
        tf.io.write_file(str(audio_path), wav_bytes)
        print(f"Created: {audio_path}")


def test_training_samples_logging():
    """Test the training samples logging functionality."""
    print("\n" + "="*60)
    print("Testing TensorBoard Training Samples Logging")
    print("="*60 + "\n")
    
    # Create sample files
    print("1. Creating sample images...")
    create_sample_images()
    
    print("\n2. Creating sample audio...")
    create_sample_audio()
    
    # Create minimal config
    print("\n3. Creating trainer configuration...")
    model_config = ModelConfig(
        text_encoder_dim=128,
        text_encoder_layers=2,
        text_encoder_heads=4,
        audio_encoder_dim=128,
        audio_encoder_layers=2,
        audio_encoder_heads=4,
        decoder_dim=256,
        decoder_layers=2,
        decoder_heads=4,
        n_mels=80,
        max_attention_sequence_length=100,
    )
    
    data_config = DataConfig(
        batch_size=2,
        num_workers=2,
        dataset_path=".",
    )
    
    training_config = TrainingConfig(
        epochs=1,
        learning_rate=1e-4,
        checkpoint_dir="./test_checkpoints",
        log_step=10,
        training_samples_dir="training_samples",
        training_samples_log_interval=1,  # Log every step for testing
    )
    
    config = XTTSConfig(
        model=model_config,
        data=data_config,
        training=training_config
    )
    
    # Create trainer
    print("\n4. Creating trainer...")
    trainer = XTTSTrainer(config=config)
    
    # Test the logging method directly
    print("\n5. Testing training samples logging...")
    if trainer.summary_writer:
        trainer._log_training_samples_to_tensorboard(step=1)
        print("\n✅ Training samples logged successfully!")
        
        # Get TensorBoard directory
        tb_dir = getattr(config.training, 'tensorboard_log_dir', None)
        if tb_dir:
            tb_path = Path(tb_dir)
        else:
            tb_path = Path(config.training.checkpoint_dir) / "tensorboard"
        
        print(f"\nTensorBoard logs saved to: {tb_path}")
        print("\nTo view in TensorBoard, run:")
        print(f"  tensorboard --logdir={tb_path}")
        
        # List files in TensorBoard directory
        if tb_path.exists():
            print(f"\nFiles in TensorBoard directory:")
            for f in tb_path.rglob("*"):
                if f.is_file():
                    print(f"  - {f.relative_to(tb_path)}")
    else:
        print("❌ TensorBoard writer not available")
        return False
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60 + "\n")
    return True


if __name__ == "__main__":
    try:
        success = test_training_samples_logging()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
