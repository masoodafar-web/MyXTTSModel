#!/usr/bin/env python3
"""
Flexible Data Paths Example for MyXTTS

This example demonstrates how to use MyXTTS with different data organization scenarios:
1. Separate train and evaluation data with different paths
2. Single dataset with automatic percentage-based splits
"""

import os
import sys
import tempfile
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from myxtts.config.config import XTTSConfig, DataConfig
from trainTestFile import create_default_config


def example_scenario_a_separate_data():
    """
    Example: Scenario A - Separate train and evaluation data with different paths
    
    This scenario is useful when:
    - You have different datasets for training and evaluation
    - Training and evaluation data are in different directories
    - You want to use different wav directory names/structures
    """
    print("=" * 60)
    print("Scenario A: Separate Train and Evaluation Data")
    print("=" * 60)
    
    # Create configuration for separate train/eval data
    config = create_default_config(
        data_path="./my_project",
        metadata_train_file="./train_data/metadata_train.csv",
        metadata_eval_file="./eval_data/metadata_eval.csv",
        wavs_train_dir="./train_data/wavs",
        wavs_eval_dir="./eval_data/audio",  # Different directory name
        batch_size=32,
        epochs=200
    )
    
    print("Configuration created for separate data scenario:")
    print(f"  Base path: {config.data.dataset_path}")
    print(f"  Train metadata: {config.data.metadata_train_file}")
    print(f"  Eval metadata: {config.data.metadata_eval_file}")
    print(f"  Train wav dir: {config.data.wavs_train_dir}")
    print(f"  Eval wav dir: {config.data.wavs_eval_dir}")
    
    # Save configuration
    output_path = "/tmp/scenario_a_config.yaml"
    config.to_yaml(output_path)
    print(f"\nSaved configuration to: {output_path}")
    
    # Show directory structure this configuration expects
    print("\nExpected directory structure:")
    print("""
    my_project/
    ├── train_data/
    │   ├── metadata_train.csv
    │   └── wavs/
    │       ├── train_001.wav
    │       ├── train_002.wav
    │       └── ...
    └── eval_data/
        ├── metadata_eval.csv
        └── audio/  # Different wav directory name
            ├── eval_001.wav
            ├── eval_002.wav
            └── ...
    """)
    
    print("Command line usage:")
    print("  # Train with this configuration")
    print(f"  python trainTestFile.py --mode train --config {output_path}")
    print()
    print("  # Or create directly via command line:")
    print("  python trainTestFile.py --mode train \\")
    print("      --data-path ./my_project \\")
    print("      --metadata-train-file ./train_data/metadata_train.csv \\") 
    print("      --metadata-eval-file ./eval_data/metadata_eval.csv \\")
    print("      --wavs-train-dir ./train_data/wavs \\")
    print("      --wavs-eval-dir ./eval_data/audio")
    

def example_scenario_b_single_dataset():
    """
    Example: Scenario B - Single dataset with automatic percentage-based splits
    
    This is the traditional approach where you have one dataset and want
    to automatically split it into train/validation/test sets by percentage.
    """
    print("=" * 60)
    print("Scenario B: Single Dataset with Percentage Splits") 
    print("=" * 60)
    
    # Create configuration for single dataset with custom split ratios
    config = create_default_config(
        data_path="./ljspeech_data",
        batch_size=16,
        epochs=100
    )
    
    # Customize split ratios
    config.data.train_split = 0.85  # 85% for training
    config.data.val_split = 0.10    # 10% for validation
    # test split is automatic: 1 - 0.85 - 0.10 = 0.05 (5%)
    
    print("Configuration created for single dataset scenario:")
    print(f"  Dataset path: {config.data.dataset_path}")
    print(f"  Train split: {config.data.train_split} ({config.data.train_split*100}%)")
    print(f"  Val split: {config.data.val_split} ({config.data.val_split*100}%)")
    print(f"  Test split: {1 - config.data.train_split - config.data.val_split} ({(1 - config.data.train_split - config.data.val_split)*100}%)")
    print(f"  Metadata file: Default (metadata.csv)")
    print(f"  Wav directory: Default (wavs/)")
    
    # Save configuration
    output_path = "/tmp/scenario_b_config.yaml"
    config.to_yaml(output_path)
    print(f"\nSaved configuration to: {output_path}")
    
    # Show directory structure this configuration expects
    print("\nExpected directory structure:")
    print("""
    ljspeech_data/
    └── LJSpeech-1.1/
        ├── metadata.csv     # Single metadata file
        └── wavs/
            ├── LJ001-0001.wav
            ├── LJ001-0002.wav
            └── ...
    """)
    
    print("Command line usage:")
    print("  # Train with this configuration")
    print(f"  python trainTestFile.py --mode train --config {output_path}")
    print()
    print("  # Or create directly via command line:")
    print("  python trainTestFile.py --mode train \\")
    print("      --data-path ./ljspeech_data \\")
    print("      --batch-size 16 \\")
    print("      --epochs 100")


def example_mixed_scenarios():
    """
    Example: Different ways to specify paths (absolute vs relative)
    """
    print("=" * 60)
    print("Path Resolution Examples")
    print("=" * 60)
    
    # Example with absolute paths
    print("1. Absolute paths (used as-is):")
    config1 = create_default_config(
        metadata_train_file="/absolute/path/to/train/metadata.csv",
        metadata_eval_file="/absolute/path/to/eval/metadata.csv", 
        wavs_train_dir="/absolute/path/to/train/wavs",
        wavs_eval_dir="/absolute/path/to/eval/wavs"
    )
    print(f"   Train metadata: {config1.data.metadata_train_file}")
    print(f"   Eval metadata: {config1.data.metadata_eval_file}")
    
    print("\n2. Relative paths (resolved relative to dataset_path):")
    config2 = create_default_config(
        data_path="/base/dataset",
        metadata_train_file="train/metadata.csv",
        metadata_eval_file="eval/metadata.csv",
        wavs_train_dir="train/audio",
        wavs_eval_dir="eval/audio"
    )
    print(f"   Dataset path: {config2.data.dataset_path}")
    print(f"   Train metadata: {config2.data.metadata_train_file} -> /base/dataset/train/metadata.csv")
    print(f"   Eval metadata: {config2.data.metadata_eval_file} -> /base/dataset/eval/metadata.csv")
    
    print("\n3. Mixed absolute and relative paths:")
    config3 = create_default_config(
        data_path="/base/dataset",
        metadata_train_file="/absolute/train/metadata.csv",  # Absolute
        metadata_eval_file="relative/eval/metadata.csv",    # Relative
        wavs_train_dir="/absolute/train/wavs",              # Absolute
        wavs_eval_dir="relative/eval/wavs"                  # Relative
    )
    print(f"   Train metadata: {config3.data.metadata_train_file} (absolute)")
    print(f"   Eval metadata: {config3.data.metadata_eval_file} -> /base/dataset/relative/eval/metadata.csv")


def main():
    """Run all examples."""
    print("MyXTTS Flexible Data Paths Examples")
    print("This script demonstrates different ways to configure data paths for training.")
    print()
    
    example_scenario_a_separate_data()
    print()
    
    example_scenario_b_single_dataset()
    print()
    
    example_mixed_scenarios()
    print()
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print("Choose Scenario A when:")
    print("  • You have separate training and evaluation datasets")
    print("  • Different directory structures for train/eval data")
    print("  • Want explicit control over data splits")
    print()
    print("Choose Scenario B when:")
    print("  • You have a single, unified dataset")
    print("  • Want automatic train/val/test splits by percentage")
    print("  • Following standard dataset formats like LJSpeech")
    print()
    print("Generated configuration files:")
    print("  • /tmp/scenario_a_config.yaml - Separate data configuration")
    print("  • /tmp/scenario_b_config.yaml - Single dataset configuration")


if __name__ == "__main__":
    main()