#!/usr/bin/env python3
"""
Test Script for NaN Loss Issue - Small Dataset

این اسکریپت با یک dataset کوچک تست می‌کنه که آیا واقعاً NaN loss داریم یا نه
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

def create_small_dataset(source_train_dir, source_val_dir, target_dir, num_samples=100):
    """
    Create a small subset of the dataset for quick testing.
    
    Args:
        source_train_dir: Path to original training dataset
        source_val_dir: Path to original validation dataset  
        target_dir: Path to create small dataset
        num_samples: Number of samples to include
    """
    target_train = Path(target_dir) / "dataset_train_small"
    target_val = Path(target_dir) / "dataset_val_small"
    
    # Create directories
    target_train.mkdir(parents=True, exist_ok=True)
    target_val.mkdir(parents=True, exist_ok=True)
    (target_train / "wavs").mkdir(exist_ok=True)
    (target_val / "wavs").mkdir(exist_ok=True)
    
    print(f"🔄 Creating small dataset with {num_samples} samples...")
    
    # Process training data
    train_metadata_path = Path(source_train_dir) / "metadata_train.csv"
    if train_metadata_path.exists():
        with open(train_metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Take random subset
        if len(lines) > num_samples:
            selected_lines = random.sample(lines, num_samples)
        else:
            selected_lines = lines
            
        # Copy files and create new metadata
        new_metadata = []
        for line in selected_lines:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_file = parts[0] + ".wav"
                source_audio = Path(source_train_dir) / "wavs" / audio_file
                target_audio = target_train / "wavs" / audio_file
                
                if source_audio.exists():
                    shutil.copy2(source_audio, target_audio)
                    new_metadata.append(line)
        
        # Write new metadata
        with open(target_train / "metadata_train.csv", 'w', encoding='utf-8') as f:
            f.writelines(new_metadata)
            
        print(f"✅ Training samples: {len(new_metadata)}")
    
    # Process validation data (smaller subset)
    val_samples = max(1, num_samples // 10)  # 10% of training samples
    val_metadata_path = Path(source_val_dir) / "metadata_eval.csv"
    if val_metadata_path.exists():
        with open(val_metadata_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Take random subset
        if len(lines) > val_samples:
            selected_lines = random.sample(lines, val_samples)
        else:
            selected_lines = lines
            
        # Copy files and create new metadata
        new_metadata = []
        for line in selected_lines:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                audio_file = parts[0] + ".wav"
                source_audio = Path(source_val_dir) / "wavs" / audio_file
                target_audio = target_val / "wavs" / audio_file
                
                if source_audio.exists():
                    shutil.copy2(source_audio, target_audio)
                    new_metadata.append(line)
        
        # Write new metadata
        with open(target_val / "metadata_eval.csv", 'w', encoding='utf-8') as f:
            f.writelines(new_metadata)
            
        print(f"✅ Validation samples: {len(new_metadata)}")
    
    return str(target_train), str(target_val)


def test_nan_loss_with_original_settings():
    """Test with the original problematic settings that cause NaN loss."""
    
    print("\n" + "="*60)
    print("🧪 TESTING NaN LOSS WITH ORIGINAL SETTINGS")
    print("="*60)
    
    # Create small dataset first
    train_path, val_path = create_small_dataset(
        "../dataset/dataset_train",
        "../dataset/dataset_eval", 
        "./small_dataset_test",
        num_samples=50  # Very small for quick epochs
    )
    
    print(f"\n📁 Small dataset created:")
    print(f"   Train: {train_path}")
    print(f"   Val: {val_path}")
    
    # Test command with problematic settings
    test_command = f"""
python3 train_main.py \\
    --model-size tiny \\
    --optimization-level enhanced \\
    --epochs 15 \\
    --batch-size 32 \\
    --lr 8e-5 \\
    --train-data {train_path} \\
    --val-data {val_path} \\
    --checkpoint-dir ./checkpoints_nan_test
"""
    
    print(f"\n🚀 Testing with PROBLEMATIC settings:")
    print("   Learning rate: 8e-5 (HIGH)")
    print("   Mel loss weight: 2.5 (HIGH)") 
    print("   Optimization: enhanced (COMPLEX)")
    print("   Batch size: 32 (LARGE)")
    print("\n📋 Command to run:")
    print(test_command)
    
    return test_command.strip()


def test_nan_loss_with_fixed_settings():
    """Test with the fixed settings that should prevent NaN loss."""
    
    print("\n" + "="*60)
    print("✅ TESTING WITH FIXED SETTINGS")
    print("="*60)
    
    # Use the same small dataset
    train_path = "./small_dataset_test/dataset_train_small"
    val_path = "./small_dataset_test/dataset_val_small"
    
    # Test command with fixed settings
    test_command = f"""
python3 train_main.py \\
    --model-size tiny \\
    --optimization-level basic \\
    --epochs 15 \\
    --batch-size 8 \\
    --train-data {train_path} \\
    --val-data {val_path} \\
    --checkpoint-dir ./checkpoints_fixed_test
"""
    
    print(f"🛡️ Testing with FIXED settings:")
    print("   Learning rate: 1e-5 (LOW)")
    print("   Mel loss weight: 1.0 (BALANCED)")
    print("   Optimization: basic (SIMPLE)")
    print("   Batch size: 8 (SMALL)")
    print("\n📋 Command to run:")
    print(test_command)
    
    return test_command.strip()


def main():
    parser = argparse.ArgumentParser(description="Test NaN loss issue with small dataset")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples in small dataset")
    parser.add_argument("--test-problematic", action="store_true", help="Test with problematic settings")
    parser.add_argument("--test-fixed", action="store_true", help="Test with fixed settings")
    parser.add_argument("--create-dataset-only", action="store_true", help="Only create small dataset")
    
    args = parser.parse_args()
    
    print("🔬 NaN Loss Issue Tester")
    print("="*40)
    
    if args.create_dataset_only:
        train_path, val_path = create_small_dataset(
            "../dataset/dataset_train",
            "../dataset/dataset_eval", 
            "./small_dataset_test",
            num_samples=args.samples
        )
        print(f"\n✅ Small dataset created successfully!")
        print(f"   Use --train-data {train_path} --val-data {val_path}")
        return
    
    # Create small dataset
    train_path, val_path = create_small_dataset(
        "../dataset/dataset_train",
        "../dataset/dataset_eval", 
        "./small_dataset_test",
        num_samples=args.samples
    )
    
    if args.test_problematic:
        print("\n" + "⚠️ " * 20)
        print("WARNING: This will likely cause NaN loss!")
        print("⚠️ " * 20)
        cmd = test_nan_loss_with_original_settings()
        
        print(f"\n💡 To run the test:")
        print(f"   {cmd}")
        
        response = input("\n❓ Do you want to run this test now? (y/N): ")
        if response.lower() == 'y':
            print("🚀 Starting problematic test...")
            os.system(cmd.replace('\\\n', ''))
    
    elif args.test_fixed:
        cmd = test_nan_loss_with_fixed_settings()
        
        print(f"\n💡 To run the test:")
        print(f"   {cmd}")
        
        response = input("\n❓ Do you want to run this test now? (y/N): ")
        if response.lower() == 'y':
            print("🚀 Starting fixed test...")
            os.system(cmd.replace('\\\n', ''))
    
    else:
        print("\n📋 Available tests:")
        print("   --test-problematic : Test settings that cause NaN")
        print("   --test-fixed       : Test settings that prevent NaN")
        print("   --create-dataset-only : Just create small dataset")
        print("\nExample:")
        print("   python3 test_nan_issue.py --test-problematic")
        print("   python3 test_nan_issue.py --test-fixed")


if __name__ == "__main__":
    main()