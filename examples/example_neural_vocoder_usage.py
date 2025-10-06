#!/usr/bin/env python3
"""
Example usage of MyXTTS with neural vocoder and modern decoding strategies.

This script demonstrates:
1. Basic usage with neural vocoder
2. Non-autoregressive decoding for faster inference
3. Two-stage training approach
4. Audio quality comparison
"""

import os
import sys
import yaml
import tensorflow as tf
import numpy as np
import soundfile as sf
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from myxtts.config.config import ModelConfig
from myxtts.models.xtts import XTTS
from myxtts.models.vocoder import Vocoder
from myxtts.utils.audio import AudioProcessor
from myxtts.training.two_stage_trainer import TwoStageTrainer, TwoStageTrainingConfig


def load_config_from_yaml(config_path: str) -> ModelConfig:
    """Load model configuration from YAML file."""
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Extract model configuration
    model_dict = yaml_config.get('model', {})
    return ModelConfig(**model_dict)


def example_neural_vocoder_usage():
    """Demonstrate basic usage with HiFi-GAN vocoder."""
    print("🎵 Example 1: HiFi-GAN Vocoder Usage")
    print("=" * 50)
    
    # Load high-quality configuration
    config = load_config_from_yaml('config_high_quality.yaml')
    
    # Create model (uses HiFi-GAN vocoder by default)
    model = XTTS(config)
    
    # Sample text input (dummy token IDs)
    text_inputs = tf.constant([[1, 15, 23, 45, 67, 89, 12, 34, 56, 78]], dtype=tf.int32)
    
    print(f"Input text shape: {text_inputs.shape}")
    
    # Generate with neural vocoder
    outputs = model.generate(
        text_inputs,
        max_length=100,
        generate_audio=True  # Enable direct audio generation
    )
    
    print(f"Generated mel shape: {outputs['mel_output'].shape}")
    if 'audio_output' in outputs:
        print(f"Generated audio shape: {outputs['audio_output'].shape}")
        print("✅ HiFi-GAN vocoder successfully generated audio!")
    
    return outputs


def example_fast_inference():
    """Demonstrate non-autoregressive decoding for faster inference."""
    print("\n⚡ Example 2: Fast Non-Autoregressive Inference")
    print("=" * 50)
    
    # Load fast inference configuration
    config = load_config_from_yaml('config_fast_inference.yaml')
    
    # Create model with non-autoregressive decoder
    model = XTTS(config)
    
    # Sample inputs
    text_inputs = tf.constant([[1, 15, 23, 45, 67, 89, 12, 34]], dtype=tf.int32)
    
    print(f"Using decoder strategy: {config.decoder_strategy}")
    
    # Fast generation
    import time
    start_time = time.time()
    
    outputs = model.generate(
        text_inputs,
        max_length=80,
        generate_audio=True
    )
    
    end_time = time.time()
    
    print(f"Generation time: {end_time - start_time:.3f} seconds")
    print(f"Generated mel shape: {outputs['mel_output'].shape}")
    print("✅ Fast non-autoregressive generation completed!")
    
    return outputs


def example_direct_vocoder_usage():
    """Demonstrate direct HiFi-GAN vocoder usage."""
    print("\n🔄 Example 3: Direct Vocoder Usage")
    print("=" * 50)
    
    # Create vocoder directly
    config = ModelConfig()
    vocoder = Vocoder(config)
    
    # Create sample mel spectrogram
    sample_mel = tf.random.normal([1, 100, 80])
    
    print(f"Input mel shape: {sample_mel.shape}")
    
    # Generate audio with vocoder
    audio = vocoder(sample_mel, training=False)
    
    print(f"Generated audio shape: {audio.shape}")
    print("✅ Direct vocoder usage completed!")
    
    return audio


def example_vocoder_audio_generation():
    """Demonstrate HiFi-GAN vocoder audio generation."""
    print("\n📊 Example 4: HiFi-GAN Audio Generation")
    print("=" * 50)
    
    # Create sample mel spectrogram
    mel_length = 100
    sample_mel = tf.random.normal([1, mel_length, 80])
    
    print(f"Sample mel shape: {sample_mel.shape}")
    
    # Create HiFi-GAN vocoder
    config = ModelConfig()
    vocoder = Vocoder(config)
    
    # Generate audio
    audio = vocoder(sample_mel, training=False)
    audio_np = audio.numpy()[0, :, 0]
    
    print(f"Generated audio length: {len(audio_np)} samples")
    
    # Calculate basic metrics
    energy = np.mean(audio_np ** 2)
    max_amplitude = np.max(np.abs(audio_np))
    
    print(f"Audio energy: {energy:.6f}")
    print(f"Max amplitude: {max_amplitude:.6f}")
    print("✅ Audio generation completed!")
    
    return audio_np


def example_two_stage_training_setup():
    """Demonstrate two-stage training setup."""
    print("\n🏋️ Example 5: Two-Stage Training Setup")
    print("=" * 50)
    
    # Load configuration
    model_config = load_config_from_yaml('config_high_quality.yaml')
    
    # Create dummy training config (in practice, load from YAML)
    class DummyTrainingConfig:
        pass
    
    training_config = DummyTrainingConfig()
    
    # Two-stage training configuration
    two_stage_config = TwoStageTrainingConfig(
        stage1_epochs=5,  # Reduced for demo
        stage2_epochs=5,  # Reduced for demo
        stage1_learning_rate=1e-4,
        stage2_learning_rate=2e-4,
        stage1_checkpoint_path="checkpoints/demo_stage1",
        stage2_checkpoint_path="checkpoints/demo_stage2"
    )
    
    # Initialize trainer
    trainer = TwoStageTrainer(model_config, training_config, two_stage_config)
    
    print("Two-stage trainer initialized successfully!")
    print(f"Stage 1 model: {type(trainer.stage1_model).__name__}")
    print(f"Stage 2 vocoder: {type(trainer.vocoder).__name__}")
    print("✅ Two-stage training setup completed!")
    
    # Note: Actual training would require real datasets
    print("\nNote: To run actual training, provide train_dataset and val_dataset:")
    print("  stage1_history = trainer.train_stage1(train_dataset, val_dataset)")
    print("  stage2_history = trainer.train_stage2(vocoder_dataset, vocoder_val_dataset)")
    print("  combined_model = trainer.create_combined_model()")


def save_audio_examples(outputs_dir: str = "examples/audio_outputs"):
    """Save audio examples for comparison."""
    print(f"\n💾 Saving audio examples to {outputs_dir}")
    print("=" * 50)
    
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Generate and save examples from different configurations
    configs = {
        'high_quality': 'config_high_quality.yaml',
        'fast_inference': 'config_fast_inference.yaml',
        'compatibility': 'config_compatibility.yaml'
    }
    
    for name, config_path in configs.items():
        if os.path.exists(config_path):
            print(f"Generating {name} example...")
            config = load_config_from_yaml(config_path)
            model = XTTS(config)
            
            # Sample input
            text_inputs = tf.constant([[1, 15, 23, 45, 67, 89, 12, 34, 56]], dtype=tf.int32)
            
            # Generate
            outputs = model.generate(text_inputs, max_length=50, generate_audio=True)
            
            # Save mel spectrogram info
            mel_shape = outputs['mel_output'].shape
            print(f"  - Mel shape: {mel_shape}")
            
            # If audio generated, save it
            if 'audio_output' in outputs:
                audio = outputs['audio_output'].numpy()[0, :, 0]
                audio_path = os.path.join(outputs_dir, f"{name}_example.wav")
                sf.write(audio_path, audio, config.sample_rate)
                print(f"  - Saved audio: {audio_path}")
    
    print("✅ Audio examples saved!")


def main():
    """Run all examples."""
    print("🚀 MyXTTS Neural Vocoder Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_neural_vocoder_usage()
        example_fast_inference()
        example_direct_vocoder_usage()
        example_vocoder_audio_generation()
        example_two_stage_training_setup()
        
        # Save examples if requested
        if len(sys.argv) > 1 and sys.argv[1] == "--save-audio":
            save_audio_examples()
        
        print("\n🎉 All examples completed successfully!")
        print("\nKey features demonstrated:")
        print("✅ HiFi-GAN vocoder provides high-quality audio synthesis")
        print("✅ Non-autoregressive decoding enables faster inference")
        print("✅ Two-stage training allows optimized component training")
        print("✅ Simple and unified vocoder interface")
        
        print("\nTo save audio examples, run:")
        print("  python example_neural_vocoder_usage.py --save-audio")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())