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
from myxtts.models.vocoder import HiFiGANGenerator
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
    """Demonstrate basic usage with neural vocoder."""
    print("🎵 Example 1: Neural Vocoder Usage")
    print("=" * 50)
    
    # Load high-quality configuration
    config = load_config_from_yaml('config_high_quality.yaml')
    
    # Create model with HiFi-GAN vocoder
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
        print("✅ Neural vocoder successfully generated audio!")
    
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


def example_compatibility_mode():
    """Demonstrate compatibility mode with Griffin-Lim."""
    print("\n🔄 Example 3: Compatibility Mode")
    print("=" * 50)
    
    # Load compatibility configuration
    config = load_config_from_yaml('config_compatibility.yaml')
    
    # Create model with Griffin-Lim vocoder
    model = XTTS(config)
    
    # Sample inputs
    text_inputs = tf.constant([[1, 15, 23, 45, 67, 89]], dtype=tf.int32)
    
    print(f"Using vocoder: {config.vocoder_type}")
    print(f"Using decoder strategy: {config.decoder_strategy}")
    
    # Generate (will use Griffin-Lim in post-processing)
    outputs = model.generate(
        text_inputs,
        max_length=60
    )
    
    print(f"Generated mel shape: {outputs['mel_output'].shape}")
    print("✅ Compatibility mode generation completed!")
    
    # Convert mel to audio using Griffin-Lim (post-processing)
    audio_processor = AudioProcessor(
        sample_rate=config.sample_rate,
        n_mels=config.n_mels,
        hop_length=config.hop_length
    )
    
    mel_np = outputs['mel_output'].numpy()[0].T  # [n_mels, time]
    audio = audio_processor.mel_to_wav(mel_np)
    print(f"Griffin-Lim audio shape: {audio.shape}")
    
    return outputs, audio


def example_vocoder_comparison():
    """Compare Griffin-Lim vs Neural Vocoder quality."""
    print("\n📊 Example 4: Vocoder Quality Comparison")
    print("=" * 50)
    
    # Create sample mel spectrogram
    mel_length = 100
    sample_mel = tf.random.normal([1, mel_length, 80])
    
    print(f"Sample mel shape: {sample_mel.shape}")
    
    # Initialize audio processor
    audio_processor = AudioProcessor(sample_rate=22050, n_mels=80, hop_length=256)
    
    # Griffin-Lim conversion
    mel_np = sample_mel.numpy()[0].T  # [n_mels, time]
    gl_audio = audio_processor.mel_to_wav(mel_np, n_iter=60)
    
    # Neural vocoder conversion
    config = ModelConfig()
    hifigan = HiFiGANGenerator(config)
    neural_audio = hifigan(sample_mel, training=False)
    neural_audio_np = neural_audio.numpy()[0, :, 0]
    
    print(f"Griffin-Lim audio length: {len(gl_audio)} samples")
    print(f"Neural vocoder audio length: {len(neural_audio_np)} samples")
    
    # Calculate basic metrics
    gl_energy = np.mean(gl_audio ** 2)
    neural_energy = np.mean(neural_audio_np ** 2)
    
    print(f"Griffin-Lim energy: {gl_energy:.6f}")
    print(f"Neural vocoder energy: {neural_energy:.6f}")
    print("✅ Vocoder comparison completed!")
    
    return gl_audio, neural_audio_np


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
        example_compatibility_mode()
        example_vocoder_comparison()
        example_two_stage_training_setup()
        
        # Save examples if requested
        if len(sys.argv) > 1 and sys.argv[1] == "--save-audio":
            save_audio_examples()
        
        print("\n🎉 All examples completed successfully!")
        print("\nKey improvements achieved:")
        print("✅ Neural vocoder provides significantly better audio quality")
        print("✅ Non-autoregressive decoding enables faster inference")
        print("✅ Two-stage training allows optimized component training")
        print("✅ Backward compatibility maintained with Griffin-Lim fallback")
        
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