#!/usr/bin/env python3
"""
Usage Examples for Enhanced MyXTTS with Multi-Speaker Support and Audio Normalization

This script demonstrates how to use the new multi-speaker dataset support,
enhanced audio normalization, and gradient warning fixes in MyXTTS.
"""

import sys
sys.path.append('.')

from myxtts.config.config import DataConfig, ModelConfig, TrainingConfig, XTTSConfig

def example_single_speaker_with_enhanced_normalization():
    """Example: Single-speaker setup with enhanced audio normalization."""
    print("🎯 Example 1: Single-speaker with enhanced normalization")
    print("-" * 60)
    
    config = DataConfig(
        # Standard single-speaker setup
        enable_multispeaker=False,
        
        # Enhanced audio normalization (NEW)
        enable_loudness_normalization=True,  # Enable LUFS-based normalization
        target_loudness_lufs=-23.0,         # Broadcast standard loudness
        enable_vad=True,                     # Silero VAD for silence removal
        
        # Standard audio processing
        sample_rate=22050,
        normalize_audio=True,
        trim_silence=True,
        
        # Dataset paths
        dataset_path="./data/LJSpeech-1.1"
    )
    
    print("✓ Enhanced single-speaker configuration created")
    print(f"  Loudness normalization: {config.enable_loudness_normalization}")
    print(f"  Target LUFS: {config.target_loudness_lufs}")
    print(f"  VAD enabled: {config.enable_vad}")
    return config

def example_multispeaker_vctk():
    """Example: Multi-speaker setup for VCTK dataset."""
    print("\n🎯 Example 2: Multi-speaker VCTK dataset")
    print("-" * 60)
    
    config = DataConfig(
        # Multi-speaker support (NEW)
        enable_multispeaker=True,
        speaker_id_pattern=r'(p\d+)',  # VCTK pattern: p225, p226, etc.
        max_speakers=1000,
        
        # Enhanced audio normalization
        enable_loudness_normalization=True,
        target_loudness_lufs=-23.0,
        enable_vad=True,
        
        # Audio processing
        sample_rate=22050,
        normalize_audio=True,
        trim_silence=True,
        
        # Dataset paths
        dataset_path="./data/VCTK-Corpus",
        metadata_train_file="./data/VCTK-Corpus/metadata_train.csv",
        metadata_eval_file="./data/VCTK-Corpus/metadata_eval.csv"
    )
    
    print("✓ Multi-speaker VCTK configuration created")
    print(f"  Multi-speaker enabled: {config.enable_multispeaker}")
    print(f"  Speaker ID pattern: {config.speaker_id_pattern}")
    print(f"  Max speakers: {config.max_speakers}")
    return config

def example_multispeaker_librispeech():
    """Example: Multi-speaker setup for LibriSpeech-style dataset."""
    print("\n🎯 Example 3: Multi-speaker LibriSpeech-style dataset")
    print("-" * 60)
    
    config = DataConfig(
        # Multi-speaker support with different pattern
        enable_multispeaker=True,
        speaker_id_pattern=r'(\d+)-\d+-\d+',  # LibriSpeech pattern: 1001-134707-0000
        max_speakers=5000,
        
        # Enhanced audio normalization  
        enable_loudness_normalization=True,
        target_loudness_lufs=-23.0,
        enable_vad=True,
        
        # Audio processing
        sample_rate=16000,  # LibriSpeech uses 16kHz
        normalize_audio=True,
        trim_silence=True,
        
        # Dataset paths
        dataset_path="./data/LibriSpeech",
    )
    
    print("✓ Multi-speaker LibriSpeech configuration created")
    print(f"  Speaker ID pattern: {config.speaker_id_pattern}")
    print(f"  Sample rate: {config.sample_rate}")
    return config

def example_model_config_with_duration_predictor():
    """Example: Model configuration with duration predictor options."""
    print("\n🎯 Example 4: Model configuration with duration predictor control")
    print("-" * 60)
    
    # Configuration with duration predictor enabled (default)
    config_with_duration = ModelConfig(
        # Standard model settings
        text_encoder_dim=512,
        text_encoder_layers=8,
        audio_encoder_dim=768,
        decoder_dim=1024,
        
        # Duration predictor control (NEW)
        use_duration_predictor=True,  # Enable duration prediction
        
        # Voice conditioning
        use_voice_conditioning=True,
        speaker_embedding_dim=512
    )
    
    # Configuration with duration predictor disabled (to avoid gradient warning)
    config_without_duration = ModelConfig(
        # Standard model settings
        text_encoder_dim=512,
        text_encoder_layers=8,
        audio_encoder_dim=768,
        decoder_dim=1024,
        
        # Duration predictor control (NEW)
        use_duration_predictor=False,  # Disable to avoid gradient warning
        
        # Voice conditioning
        use_voice_conditioning=True,
        speaker_embedding_dim=512
    )
    
    print("✓ Model configurations created")
    print(f"  With duration predictor: {config_with_duration.use_duration_predictor}")
    print(f"  Without duration predictor: {config_without_duration.use_duration_predictor}")
    return config_with_duration, config_without_duration

def example_training_commands():
    """Example: Training commands for different scenarios."""
    print("\n🎯 Example 5: Training commands")
    print("-" * 60)
    
    commands = {
        "Single-speaker with enhanced normalization": [
            "python train_main.py",
            "  --train-data ./data/LJSpeech-1.1",
            "  --epochs 100",
            "  --batch-size 32"
        ],
        
        "Multi-speaker VCTK": [
            "# First, prepare VCTK metadata with speaker IDs",
            "python train_main.py",
            "  --train-data ./data/VCTK-Corpus", 
            "  --epochs 200",
            "  --batch-size 24  # Smaller batch for multi-speaker"
        ],
        
        "Disable duration predictor (if gradient warning occurs)": [
            "# Edit train_main.py to set use_duration_predictor=False",
            "# In ModelConfig section, change:",
            "# use_duration_predictor=False",
            "python train_main.py --train-data ./data/LJSpeech-1.1"
        ]
    }
    
    for scenario, cmd_lines in commands.items():
        print(f"\n{scenario}:")
        for line in cmd_lines:
            print(f"  {line}")

def example_inference_commands():
    """Example: Inference commands for different scenarios."""
    print("\n🎯 Example 6: Inference commands")
    print("-" * 60)
    
    commands = {
        "Single-speaker inference": [
            "python inference_main.py",
            "  --text 'Hello, this is a test.'",
            "  --output output.wav",
            "  --reference-audio reference.wav"
        ],
        
        "Multi-speaker inference with speaker ID": [
            "# List available speakers first",
            "python inference_main.py --list-speakers",
            "",
            "# Then generate with specific speaker",
            "python inference_main.py",
            "  --text 'Hello from speaker p225.'", 
            "  --speaker-id p225",
            "  --output output_p225.wav"
        ],
        
        "Voice cloning with multiple references": [
            "python inference_main.py",
            "  --text 'Clone this voice style.'",
            "  --multiple-reference-audios ref1.wav ref2.wav ref3.wav",
            "  --clone-voice",
            "  --output cloned_voice.wav"
        ]
    }
    
    for scenario, cmd_lines in commands.items():
        print(f"\n{scenario}:")
        for line in cmd_lines:
            if line.strip():
                print(f"  {line}")
            else:
                print()

def example_metadata_format():
    """Example: Metadata file formats for different dataset types."""
    print("\n🎯 Example 7: Metadata file formats")
    print("-" * 60)
    
    print("Single-speaker (LJSpeech format):")
    print("  LJ001-0001|This is the transcription.|This is the normalized transcription.")
    print("  LJ001-0002|Another sample text.|Another sample text.")
    
    print("\nMulti-speaker with filename-based speaker extraction:")
    print("  p225_001|Hello world.|Hello world.")  
    print("  p225_002|How are you?|How are you?")
    print("  p226_001|I am fine.|I am fine.")
    
    print("\nMulti-speaker with explicit speaker column (4-column format):")
    print("  001|p225|Hello world.|Hello world.")
    print("  002|p225|How are you?|How are you?")  
    print("  003|p226|I am fine.|I am fine.")

if __name__ == "__main__":
    print("MyXTTS Enhanced Features Usage Examples")
    print("=" * 60)
    
    # Run examples
    example_single_speaker_with_enhanced_normalization()
    example_multispeaker_vctk() 
    example_multispeaker_librispeech()
    example_model_config_with_duration_predictor()
    example_training_commands()
    example_inference_commands()
    example_metadata_format()
    
    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("\nKey Benefits:")
    print("✓ Multi-speaker support with minimal configuration")
    print("✓ Enhanced audio normalization for real-world robustness")
    print("✓ Loudness matching using industry standard LUFS")
    print("✓ Automatic silence removal with Silero VAD")
    print("✓ Gradient warning fix for duration predictor")
    print("✓ Flexible speaker ID extraction patterns")
    print("✓ Backward compatibility with existing datasets")