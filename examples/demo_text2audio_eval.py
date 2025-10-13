#!/usr/bin/env python3
"""
Demo script showing the text-to-audio evaluation callback feature.

This demonstrates how to configure and use the automatic audio generation
during training for quality monitoring.
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from myxtts.config.config import XTTSConfig, TrainingConfig


def demo_basic_usage():
    """Demonstrate basic usage of text-to-audio evaluation."""
    print("=" * 80)
    print("Demo 1: Basic Configuration")
    print("=" * 80)
    
    # Create config with default text-to-audio evaluation
    config = XTTSConfig()
    
    print("\n✅ Default Configuration:")
    print(f"  • Enabled: {config.training.enable_text2audio_eval}")
    print(f"  • Interval: Every {config.training.text2audio_interval_steps} steps")
    print(f"  • Output directory: {config.training.text2audio_output_dir}")
    print(f"  • Number of test texts: {len(config.training.text2audio_texts)}")
    print(f"  • Test texts:")
    for i, text in enumerate(config.training.text2audio_texts):
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"    {i+1}. {preview}")
    print(f"  • Log to TensorBoard: {config.training.text2audio_log_tensorboard}")
    
    print("\n📝 During training, audio will be generated at steps:")
    print("    200, 400, 600, 800, 1000, ...")
    print("    Files will be saved to: eval_samples/step_N/eval_00.wav, eval_01.wav, ...")


def demo_custom_configuration():
    """Demonstrate custom configuration."""
    print("\n" + "=" * 80)
    print("Demo 2: Custom Configuration")
    print("=" * 80)
    
    # Create config with custom settings
    config = XTTSConfig()
    config.training.enable_text2audio_eval = True
    config.training.text2audio_interval_steps = 500  # Every 500 steps
    config.training.text2audio_output_dir = "./custom_eval_samples"
    config.training.text2audio_texts = [
        "This is my first custom evaluation text.",
        "این متن ارزیابی دوم من است.",  # Persian
        "Este es mi tercer texto de evaluación.",  # Spanish
    ]
    config.training.text2audio_log_tensorboard = True
    
    print("\n✅ Custom Configuration:")
    print(f"  • Enabled: {config.training.enable_text2audio_eval}")
    print(f"  • Interval: Every {config.training.text2audio_interval_steps} steps")
    print(f"  • Output directory: {config.training.text2audio_output_dir}")
    print(f"  • Number of test texts: {len(config.training.text2audio_texts)}")
    print(f"  • Test texts:")
    for i, text in enumerate(config.training.text2audio_texts):
        preview = text[:60] + "..." if len(text) > 60 else text
        print(f"    {i+1}. {preview}")
    
    print("\n📝 During training, audio will be generated at steps:")
    print("    500, 1000, 1500, 2000, ...")


def demo_disable_feature():
    """Demonstrate disabling the feature."""
    print("\n" + "=" * 80)
    print("Demo 3: Disabling Text-to-Audio Evaluation")
    print("=" * 80)
    
    # Create config with feature disabled
    config = XTTSConfig()
    config.training.enable_text2audio_eval = False
    
    print("\n✅ Feature Disabled:")
    print(f"  • Enabled: {config.training.enable_text2audio_eval}")
    print("\n📝 No audio will be generated during training.")


def demo_yaml_configuration():
    """Demonstrate YAML configuration."""
    print("\n" + "=" * 80)
    print("Demo 4: YAML Configuration")
    print("=" * 80)
    
    yaml_content = """
training:
  # ... other training parameters ...
  
  # Text-to-Audio Evaluation Settings
  enable_text2audio_eval: true
  text2audio_interval_steps: 200
  text2audio_output_dir: "./eval_samples"
  text2audio_texts:
    - "The quick brown fox jumps over the lazy dog."
    - "سلام! این یک نمونهی ارزیابی است."
    - "Custom evaluation text here."
  text2audio_speaker_id: null
  text2audio_log_tensorboard: true
"""
    
    print("\n✅ Example YAML Configuration:")
    print(yaml_content)
    
    print("\n📝 To use this configuration:")
    print("  1. Save to a YAML file (e.g., config_with_eval.yaml)")
    print("  2. Load in your training script:")
    print("     config = XTTSConfig.from_yaml('config_with_eval.yaml')")


def demo_training_integration():
    """Demonstrate how it integrates with training."""
    print("\n" + "=" * 80)
    print("Demo 5: Training Integration")
    print("=" * 80)
    
    print("""
✅ The text-to-audio evaluation is automatically integrated into the training loop.

During training, the following happens:

1. After each training step, the trainer checks if current_step % interval == 0
2. If true, it calls _maybe_eval_text2audio()
3. The method:
   - Creates output directory (eval_samples/step_N/)
   - Temporarily sets model to eval mode
   - Generates audio for each test text
   - Saves WAV files (eval_00.wav, eval_01.wav, ...)
   - Saves corresponding text files (eval_00.txt, eval_01.txt, ...)
   - Logs to TensorBoard (if enabled)
   - Restores model to training mode

Example output structure:
  eval_samples/
    step_200/
      eval_00.wav
      eval_00.txt
      eval_01.wav
      eval_01.txt
    step_400/
      eval_00.wav
      eval_00.txt
      eval_01.wav
      eval_01.txt
    ...

This allows you to:
  • Listen to generated audio at different training stages
  • Compare quality improvements over time
  • Detect training issues early (e.g., mode collapse, degradation)
  • Monitor multilingual capabilities
    """)


def demo_cli_usage():
    """Demonstrate CLI usage."""
    print("\n" + "=" * 80)
    print("Demo 6: Using with train_main.py")
    print("=" * 80)
    
    print("""
✅ The feature works automatically with the default train_main.py script.

Basic usage (uses defaults):
  python train_main.py --train-data ./data/train --val-data ./data/val

Custom interval:
  # Use a config file with custom settings
  python train_main.py --config config_custom.yaml --train-data ./data/train

Disable evaluation:
  # Modify config to set enable_text2audio_eval: false
  # Or override in code before training

The evaluation audio will be generated during training and saved to the
configured output directory (default: ./eval_samples/).
    """)


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("Text-to-Audio Evaluation Callback Feature Demo")
    print("=" * 80)
    print("\nThis demo shows how to use the automatic audio generation feature")
    print("that monitors training quality by synthesizing test texts at regular intervals.")
    
    demo_basic_usage()
    demo_custom_configuration()
    demo_disable_feature()
    demo_yaml_configuration()
    demo_training_integration()
    demo_cli_usage()
    
    print("\n" + "=" * 80)
    print("✅ Demo Complete!")
    print("=" * 80)
    print("\nFor more information, see:")
    print("  • myxtts/config/config.py - Configuration options")
    print("  • myxtts/training/trainer.py - Implementation details")
    print("  • tests/test_text2audio_eval.py - Unit tests")
    print()


if __name__ == "__main__":
    main()
