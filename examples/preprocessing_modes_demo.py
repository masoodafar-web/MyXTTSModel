#!/usr/bin/env python3
"""
Example script demonstrating the new dataset preprocessing modes in MyXTTS.

This script shows how to use the different preprocessing modes to optimize
GPU utilization during training.
"""

import os
import tempfile
from pathlib import Path

# Add parent directory to path for imports
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from myxtts.config.config import XTTSConfig
from trainTestFile import create_default_config


def demonstrate_preprocessing_modes():
    """Demonstrate all three preprocessing modes."""
    
    print("=== MyXTTS Dataset Preprocessing Modes Demo ===\n")
    
    # 1. AUTO mode (default behavior)
    print("1. AUTO Mode (Default)")
    print("   - Attempts to precompute dataset before training")
    print("   - Falls back gracefully if preprocessing fails")
    print("   - Best for most use cases")
    
    config_auto = create_default_config(
        data_path="./data/ljspeech",
        preprocessing_mode="auto",
        batch_size=32,
        epochs=100
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_auto.yaml', delete=False) as f:
        config_auto.to_yaml(f.name)
        print(f"   Example config saved to: {f.name}")
    
    print(f"   Preprocessing mode: {config_auto.data.preprocessing_mode}")
    print()
    
    # 2. PRECOMPUTE mode (optimized for GPU utilization)
    print("2. PRECOMPUTE Mode (Recommended for GPU optimization)")
    print("   - Forces complete dataset preprocessing before training starts")
    print("   - Eliminates CPU preprocessing bottlenecks during training")
    print("   - Maximizes GPU utilization by ensuring data is ready")
    print("   - Fails if preprocessing cannot be completed")
    
    config_precompute = create_default_config(
        data_path="./data/ljspeech",
        preprocessing_mode="precompute",
        batch_size=48,  # Can use larger batch size with precomputed data
        epochs=1000
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_precompute.yaml', delete=False) as f:
        config_precompute.to_yaml(f.name)
        print(f"   Example config saved to: {f.name}")
    
    print(f"   Preprocessing mode: {config_precompute.data.preprocessing_mode}")
    print(f"   Optimized batch size: {config_precompute.data.batch_size}")
    print()
    
    # 3. RUNTIME mode (for limited disk space scenarios)
    print("3. RUNTIME Mode (For limited disk space)")
    print("   - Processes data on-the-fly during training")
    print("   - No disk cache files created")
    print("   - May impact GPU utilization due to CPU preprocessing")
    print("   - Useful when disk space is limited")
    
    config_runtime = create_default_config(
        data_path="./data/ljspeech",
        preprocessing_mode="runtime",
        batch_size=16,  # Smaller batch size to account for runtime processing
        epochs=100
    )
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='_runtime.yaml', delete=False) as f:
        config_runtime.to_yaml(f.name)
        print(f"   Example config saved to: {f.name}")
    
    print(f"   Preprocessing mode: {config_runtime.data.preprocessing_mode}")
    print(f"   Conservative batch size: {config_runtime.data.batch_size}")
    print()


def show_command_line_examples():
    """Show command line usage examples."""
    
    print("=== Command Line Usage Examples ===\n")
    
    print("1. Training with AUTO mode (default):")
    print("   python trainTestFile.py --mode train --data-path ./data/ljspeech")
    print("   python trainTestFile.py --mode train --config config.yaml")
    print()
    
    print("2. Training with PRECOMPUTE mode (for maximum GPU utilization):")
    print("   python trainTestFile.py --mode train --preprocessing-mode precompute --batch-size 48")
    print("   python trainTestFile.py --mode train --config config.yaml --preprocessing-mode precompute")
    print()
    
    print("3. Training with RUNTIME mode (for limited disk space):")
    print("   python trainTestFile.py --mode train --preprocessing-mode runtime --batch-size 16")
    print()
    
    print("4. Creating configuration files with specific preprocessing modes:")
    print("   python trainTestFile.py --mode create-config --output gpu_optimized.yaml --preprocessing-mode precompute")
    print("   python trainTestFile.py --mode create-config --output low_storage.yaml --preprocessing-mode runtime")
    print()


def show_performance_recommendations():
    """Show performance recommendations for each mode."""
    
    print("=== Performance Recommendations ===\n")
    
    print("🚀 For MAXIMUM GPU UTILIZATION:")
    print("   - Use preprocessing_mode: precompute")
    print("   - Increase batch_size (32-64 depending on GPU memory)")
    print("   - Ensure sufficient disk space for cache files")
    print("   - Monitor GPU utilization with: python gpu_monitor.py")
    print()
    
    print("💾 For LIMITED DISK SPACE:")
    print("   - Use preprocessing_mode: runtime")
    print("   - Use smaller batch_size (16-32)")
    print("   - Accept some GPU underutilization")
    print("   - Monitor CPU usage during training")
    print()
    
    print("⚖️ For BALANCED APPROACH:")
    print("   - Use preprocessing_mode: auto (default)")
    print("   - Let the system decide based on available resources")
    print("   - Good starting point for most users")
    print()
    
    print("📊 MONITORING:")
    print("   - Use: python test_gpu_utilization.py")
    print("   - Target: 70-90% GPU utilization during training")
    print("   - If GPU util < 30%, try precompute mode")
    print("   - If disk space low, try runtime mode")
    print()


def show_yaml_configuration_examples():
    """Show YAML configuration examples."""
    
    print("=== YAML Configuration Examples ===\n")
    
    print("GPU-Optimized Configuration (precompute mode):")
    print("```yaml")
    print("data:")
    print("  dataset_path: ./data/ljspeech")
    print("  batch_size: 48")
    print("  preprocessing_mode: precompute  # Maximize GPU utilization")
    print("  num_workers: 12")
    print("  prefetch_buffer_size: 8")
    print("")
    print("training:")
    print("  epochs: 1000")
    print("  learning_rate: 1e-4")
    print("```")
    print()
    
    print("Low-Storage Configuration (runtime mode):")
    print("```yaml") 
    print("data:")
    print("  dataset_path: ./data/ljspeech")
    print("  batch_size: 16")
    print("  preprocessing_mode: runtime  # No disk caching")
    print("  num_workers: 8")
    print("")
    print("training:")
    print("  epochs: 100")
    print("  learning_rate: 1e-4")
    print("```")
    print()


if __name__ == "__main__":
    try:
        demonstrate_preprocessing_modes()
        show_command_line_examples()
        show_performance_recommendations()
        show_yaml_configuration_examples()
        
        print("✅ Demo completed successfully!")
        print("\nNext steps:")
        print("1. Choose the preprocessing mode that fits your setup")
        print("2. Create a config with: python trainTestFile.py --mode create-config")
        print("3. Start training with: python trainTestFile.py --mode train")
        print("4. Monitor performance with: python gpu_monitor.py")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)