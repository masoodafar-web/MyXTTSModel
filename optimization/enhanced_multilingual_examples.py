#!/usr/bin/env python3
"""
Enhanced MyXTTS Usage Examples - Multi-Language and Multi-Speaker Features

This script demonstrates the new multi-language and multi-speaker capabilities
added to MyXTTS, including NLLB embedding optimization and audio augmentations.
"""

import os
import sys

def example_1_multilingual_training():
    """Example 1: Training with multi-language support"""
    print("=" * 60)
    print("📚 Example 1: Multi-Language Training Configuration")
    print("=" * 60)
    
    config_example = """
# Enhanced Multi-Language Training Configuration
python train_main.py \\
    --train-data /path/to/multilingual/dataset \\
    --val-data /path/to/multilingual/val \\
    --config config_multilingual.yaml

# config_multilingual.yaml content:
data:
  enable_multilingual: true
  supported_languages: ["en", "es", "fr", "de", "it", "pt", "zh", "ja"]
  language_detection_method: "metadata"  # or "filename" or "auto"
  enable_phone_normalization: true
  
  # Audio augmentations for robustness
  enable_pitch_shift: true
  pitch_shift_range: [-2.0, 2.0]
  enable_noise_mixing: true
  noise_mixing_probability: 0.3
  noise_mixing_snr_range: [15.0, 25.0]

model:
  # NLLB optimization for memory efficiency
"""
    print(config_example)

def example_2_multispeaker_training():
    """Example 2: Multi-speaker training setup"""
    print("=" * 60)
    print("👥 Example 2: Multi-Speaker Training Configuration")
    print("=" * 60)
    
    config_example = """
# Multi-Speaker Training with LibriTTS/VCTK-style datasets
python train_main.py \\
    --train-data /path/to/libri_tts/train \\
    --val-data /path/to/libri_tts/val \\
    --config config_multispeaker.yaml

# config_multispeaker.yaml content:
data:
  enable_multispeaker: true
  speaker_id_pattern: "(p\\\\d+)"  # Extract speaker IDs like p001, p002, etc.
  max_speakers: 1000
  
  # Enhanced audio processing
  enable_loudness_normalization: true
  target_loudness_lufs: -23.0
  enable_vad: true  # Silero VAD for silence removal
  
# Dataset structure should be:
# /path/to/libri_tts/
# ├── metadata_train.csv  # id|transcription|normalized_transcription|speaker_id
# ├── metadata_val.csv
# └── wavs/
#     ├── p001_001.wav
#     ├── p001_002.wav
#     ├── p002_001.wav
#     └── ...
"""
    print(config_example)

def example_3_multilingual_inference():
    """Example 3: Multi-language inference"""
    print("=" * 60)
    print("🌍 Example 3: Multi-Language Inference")
    print("=" * 60)
    
    inference_examples = """
# Automatic language detection
python inference_main.py \\
    --text "Bonjour, comment allez-vous?" \\
    --detect-language \\
    --enable-phone-normalization \\
    --output output_french.wav

# Specify language explicitly
python inference_main.py \\
    --text "Hola, ¿cómo estás?" \\
    --language es \\
    --enable-phone-normalization \\
    --output output_spanish.wav

# Multi-language with voice cloning
python inference_main.py \\
    --text "Guten Tag, wie geht es Ihnen?" \\
    --language de \\
    --reference-audio german_speaker.wav \\
    --clone-voice \\
    --output output_german_cloned.wav

# Chinese text synthesis
python inference_main.py \\
    --text "你好世界" \\
    --detect-language \\
    --output output_chinese.wav
"""
    print(inference_examples)

def example_4_multispeaker_inference():
    """Example 4: Multi-speaker inference"""
    print("=" * 60)
    print("🎭 Example 4: Multi-Speaker Inference")
    print("=" * 60)
    
    inference_examples = """
# List available speakers
python inference_main.py \\
    --list-speakers \\
    --checkpoint ./checkpoints/multispeaker_model

# Use specific speaker
python inference_main.py \\
    --text "Hello, this is a test with a specific speaker." \\
    --speaker-id "p001" \\
    --output output_speaker_p001.wav

# Combine multi-speaker with multi-language
python inference_main.py \\
    --text "Bonjour, je suis le locuteur français." \\
    --language fr \\
    --speaker-id "p_french_001" \\
    --enable-phone-normalization \\
    --output output_french_speaker.wav

# Voice cloning with speaker adaptation
python inference_main.py \\
    --text "This combines voice cloning with speaker modeling." \\
    --speaker-id "p002" \\
    --reference-audio target_voice.wav \\
    --clone-voice \\
    --output output_hybrid_voice.wav
"""
    print(inference_examples)

def example_5_memory_optimization():
    """Example 5: Memory optimization comparison"""
    print("=" * 60)
    print("💾 Example 5: NLLB Memory Optimization")
    print("=" * 60)
    
    comparison = """
# BEFORE (Original NLLB-200):
# - Vocabulary size: 256,256 tokens
# - Embedding memory: ~256MB (assuming 256 dims × 256,256 tokens × 4 bytes)
# - Training memory: High due to large embedding gradients

# AFTER (Optimized NLLB):
# - Vocabulary size: 32,000 tokens (87% reduction)
# - Embedding memory: ~32MB (87% reduction)
# - Weight tying: Additional memory savings for similar languages
# - Training memory: Significantly reduced

# Configuration for memory optimization:
model:
  text_vocab_size: 32000  # vs 256,256 original

# Memory savings calculation:
# Original: 256,256 × 512 dims × 4 bytes = 524MB
# Optimized: 32,000 × 512 dims × 4 bytes = 65MB
# Savings: 87% reduction in embedding memory
"""
    print(comparison)

def example_6_dataset_formats():
    """Example 6: Supported dataset formats"""
    print("=" * 60)
    print("📁 Example 6: Supported Dataset Formats")
    print("=" * 60)
    
    formats = """
# 1. LibriTTS Format (Multi-speaker, Multi-language)
/dataset/
├── metadata_train.csv
├── metadata_val.csv
└── wavs/
    ├── speaker_001/
    │   ├── chapter_001/
    │   │   ├── 001_001.wav
    │   │   └── 001_002.wav
    └── speaker_002/
        └── chapter_001/
            ├── 002_001.wav
            └── 002_002.wav

# metadata_train.csv format:
# id,transcription,normalized_transcription,speaker_id,language
# "speaker_001_chapter_001_001_001","Hello world","hello world","speaker_001","en"
# "speaker_002_chapter_001_002_001","Hola mundo","hola mundo","speaker_002","es"

# 2. VCTK Format (Multi-speaker, Single language)
/dataset/
├── metadata.csv
└── wav48/
    ├── p001/
    │   ├── p001_001.wav
    │   └── p001_002.wav
    └── p002/
        ├── p002_001.wav
        └── p002_002.wav

# 3. LJSpeech Format (Single speaker, enhanced)
/dataset/
├── metadata.csv
└── wavs/
    ├── LJ001-0001.wav
    ├── LJ001-0002.wav
    └── ...

# Configuration for each format:
# LibriTTS:
data:
  enable_multispeaker: true
  enable_multilingual: true
  speaker_id_pattern: "(speaker_\\\\d+)"
  language_detection_method: "metadata"

# VCTK:
data:
  enable_multispeaker: true
  speaker_id_pattern: "(p\\\\d+)"
  language: "en"

# LJSpeech (with enhancements):
data:
  enable_multispeaker: false
  enable_pitch_shift: true  # For augmentation
  enable_noise_mixing: true
"""
    print(formats)

def example_7_training_command_examples():
    """Example 7: Complete training command examples"""
    print("=" * 60)
    print("🚀 Example 7: Complete Training Commands")
    print("=" * 60)
    
    commands = """
# 1. Basic multi-language training
python train_main.py \\
    --train-data /data/multilingual_tts/train \\
    --val-data /data/multilingual_tts/val \\
    --batch-size 32 \\
    --epochs 1000 \\
    --checkpoint-dir ./checkpoints/multilingual

# 2. Multi-speaker with augmentations
python train_main.py \\
    --train-data /data/libri_tts/train \\
    --val-data /data/libri_tts/val \\
    --batch-size 48 \\
    --grad-accum 2 \\
    --checkpoint-dir ./checkpoints/multispeaker \\
    --config config_multispeaker_augmented.yaml

# 3. Memory-optimized for large datasets
python train_main.py \\
    --train-data /data/huge_multilingual_dataset \\
    --val-data /data/huge_multilingual_dataset_val \\
    --batch-size 64 \\
    --enable-grad-checkpointing \\
    --max-memory-fraction 0.85 \\
    --optimization-level enhanced

# 4. Fine-tuning on specific language
python train_main.py \\
    --train-data /data/spanish_speakers \\
    --val-data /data/spanish_speakers_val \\
    --checkpoint ./checkpoints/multilingual/checkpoint_5000 \\
    --lr 1e-5 \\
    --epochs 100 \\
    --language es
"""
    print(commands)

def main():
    """Display all examples"""
    print("🌟 Enhanced MyXTTS: Multi-Language & Multi-Speaker Examples")
    print("This document demonstrates the new capabilities added to MyXTTS")
    print()
    
    examples = [
        example_1_multilingual_training,
        example_2_multispeaker_training,
        example_3_multilingual_inference,
        example_4_multispeaker_inference,
        example_5_memory_optimization,
        example_6_dataset_formats,
        example_7_training_command_examples,
    ]
    
    for example in examples:
        example()
        print()
    
    print("=" * 60)
    print("✨ Summary of New Features:")
    print("=" * 60)
    print("🌍 Multi-Language Support:")
    print("  • Automatic language detection from text")
    print("  • Phone-level normalization with phonemization")
    print("  • Support for 16+ languages")
    print("  • Language-specific text processing")
    print()
    print("👥 Multi-Speaker Support:")
    print("  • Automatic speaker ID extraction")
    print("  • Support for LibriTTS/VCTK formats")
    print("  • Up to 1000 speakers per model")
    print("  • Speaker-specific inference")
    print()
    print("🎵 Audio Augmentations:")
    print("  • Pitch shifting with configurable range")
    print("  • Noise mixing with SNR control")
    print("  • Silero VAD integration")
    print("  • Loudness normalization")
    print()
    print("💾 NLLB Optimization:")
    print("  • 87% reduction in vocabulary size")
    print("  • Weight tying for memory efficiency")
    print("  • Frequency-based token selection")
    print("  • Configurable optimization methods")
    print()
    print("🔧 Enhanced Inference:")
    print("  • Language detection and phone normalization")
    print("  • Multi-speaker voice selection")
    print("  • Improved voice cloning capabilities")
    print("  • Comprehensive CLI options")

if __name__ == "__main__":
    main()
