#!/usr/bin/env python3
"""
Quick GPU Fix - One-command solution for GPU utilization issues

This script:
1. Checks current GPU status
2. Verifies configuration has optimizations enabled
3. Provides actionable recommendations
4. Optionally runs validation tests

Usage:
    python quick_gpu_fix.py              # Check status
    python quick_gpu_fix.py --test       # Run full validation
    python quick_gpu_fix.py --fix-config # Auto-fix config file
"""

import sys
import os
import subprocess
from pathlib import Path
import argparse

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def check_gpu_available():
    """Check if GPU is available and working."""
    print_header("Step 1: GPU Availability Check")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            print("❌ No GPU detected by TensorFlow")
            print("\n📋 GPU Setup Required:")
            print("  1. Install NVIDIA GPU drivers")
            print("  2. Install CUDA toolkit (11.2+)")
            print("  3. Install cuDNN (8.1+)")
            print("  4. Install TensorFlow with GPU: pip install tensorflow[and-cuda]")
            return False
        
        print(f"✅ GPU detected: {len(gpus)} device(s)")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
        
        # Test GPU
        with tf.device('/GPU:0'):
            a = tf.constant([1.0])
            _ = a + 1
        print("✅ GPU is functional")
        return True
        
    except Exception as e:
        print(f"❌ GPU check failed: {e}")
        return False


def check_config_optimization():
    """Check if config has GPU optimizations enabled."""
    print_header("Step 2: Configuration Check")
    
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print("   Using default configuration")
        return True
    
    try:
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        checks = {
            'enable_graph_mode': False,
            'enable_xla_compilation': False,
            'preprocessing_mode': False,
            'prefetch_to_gpu': False
        }
        
        # Check for critical settings
        if 'enable_graph_mode: true' in config_content.lower() or \
           'enable_graph_mode:true' in config_content.lower():
            checks['enable_graph_mode'] = True
        
        if 'enable_xla_compilation: true' in config_content.lower() or \
           'enable_xla_compilation:true' in config_content.lower():
            checks['enable_xla_compilation'] = True
        
        if 'preprocessing_mode: precompute' in config_content.lower() or \
           'preprocessing_mode:"precompute"' in config_content.lower():
            checks['preprocessing_mode'] = True
        
        if 'prefetch_to_gpu: true' in config_content.lower() or \
           'prefetch_to_gpu:true' in config_content.lower():
            checks['prefetch_to_gpu'] = True
        
        # Report results
        all_enabled = all(checks.values())
        
        if checks['enable_graph_mode']:
            print("✅ Graph compilation: ENABLED")
        else:
            print("❌ Graph compilation: DISABLED (critical for GPU performance)")
        
        if checks['enable_xla_compilation']:
            print("✅ XLA JIT compilation: ENABLED")
        else:
            print("⚠️  XLA JIT compilation: DISABLED (recommended for best performance)")
        
        if checks['preprocessing_mode']:
            print("✅ Precompute mode: ENABLED")
        else:
            print("⚠️  Precompute mode: NOT SET (recommended for fast data loading)")
        
        if checks['prefetch_to_gpu']:
            print("✅ GPU prefetch: ENABLED")
        else:
            print("⚠️  GPU prefetch: DISABLED (recommended)")
        
        if not all_enabled:
            print("\n⚠️  Some optimizations are not enabled in your config")
            print("   Run with --fix-config to automatically enable them")
        
        return checks['enable_graph_mode']  # Most critical
        
    except Exception as e:
        print(f"⚠️  Could not parse config: {e}")
        return True  # Assume defaults are used


def fix_config():
    """Automatically fix config file to enable optimizations."""
    print_header("Fixing Configuration")
    
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Backup original
        backup_path = config_path.with_suffix('.yaml.backup')
        with open(backup_path, 'w') as f:
            f.writelines(lines)
        print(f"✅ Backup created: {backup_path}")
        
        # Find and fix settings
        fixed = []
        in_training = False
        in_data = False
        
        for i, line in enumerate(lines):
            # Track sections
            if line.strip().startswith('training:'):
                in_training = True
                in_data = False
            elif line.strip().startswith('data:'):
                in_data = True
                in_training = False
            elif line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                in_training = False
                in_data = False
            
            # Fix training settings
            if in_training:
                if 'enable_graph_mode:' in line and 'false' in line.lower():
                    line = line.replace('false', 'true').replace('False', 'true')
                    fixed.append('enable_graph_mode')
                elif 'enable_xla_compilation:' in line and 'false' in line.lower():
                    line = line.replace('false', 'true').replace('False', 'true')
                    fixed.append('enable_xla_compilation')
            
            # Fix data settings
            if in_data:
                if 'preprocessing_mode:' in line and 'precompute' not in line.lower():
                    line = '  preprocessing_mode: "precompute"  # Fixed for optimal performance\n'
                    fixed.append('preprocessing_mode')
                elif 'prefetch_to_gpu:' in line and 'false' in line.lower():
                    line = line.replace('false', 'true').replace('False', 'true')
                    fixed.append('prefetch_to_gpu')
            
            lines[i] = line
        
        # Write fixed config
        with open(config_path, 'w') as f:
            f.writelines(lines)
        
        if fixed:
            print(f"✅ Fixed {len(fixed)} settings:")
            for setting in fixed:
                print(f"   - {setting}")
        else:
            print("✅ Configuration already optimal")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to fix config: {e}")
        return False


def run_tests():
    """Run validation tests."""
    print_header("Step 3: Running Validation Tests")
    
    print("Running GPU optimization test suite...")
    print("This will take 1-2 minutes...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'test_gpu_optimization.py'],
            timeout=300,
            capture_output=False
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Tests timed out")
        return False
    except Exception as e:
        print(f"❌ Tests failed: {e}")
        return False


def check_nvidia_smi():
    """Check GPU status with nvidia-smi."""
    print_header("Current GPU Status")
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            print("\nCurrent GPU state:")
            print("  GPU | Name | Utilization | Memory")
            print("  " + "-"*60)
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    print(f"  {parts[0]:3s} | {parts[1]:20s} | {parts[2]:11s} | {parts[3]:>6s}/{parts[4]:>6s}")
        else:
            print("⚠️  Could not query GPU status (nvidia-smi failed)")
            
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - cannot check current GPU status")
    except Exception as e:
        print(f"⚠️  Could not check GPU: {e}")


def provide_recommendations(gpu_ok, config_ok):
    """Provide actionable recommendations."""
    print_header("Recommendations")
    
    if not gpu_ok:
        print("\n🔴 CRITICAL: No GPU detected")
        print("\nAction required:")
        print("  1. Install NVIDIA drivers: https://www.nvidia.com/drivers")
        print("  2. Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads")
        print("  3. Install TensorFlow with GPU:")
        print("     pip install tensorflow[and-cuda]")
        print("  4. Verify with: python -c 'import tensorflow as tf; print(tf.config.list_physical_devices(\"GPU\"))'")
        return
    
    if not config_ok:
        print("\n🟡 WARNING: GPU optimizations not fully enabled")
        print("\nAction required:")
        print("  Run: python quick_gpu_fix.py --fix-config")
        print("\nOr manually add to configs/config.yaml:")
        print("  training:")
        print("    enable_graph_mode: true")
        print("    enable_xla_compilation: true")
        print("  data:")
        print("    preprocessing_mode: \"precompute\"")
        print("    prefetch_to_gpu: true")
        return
    
    print("\n🟢 GREAT: GPU and configuration are ready!")
    print("\nNext steps:")
    print("  1. Run validation: python quick_gpu_fix.py --test")
    print("  2. Start training: python train_main.py")
    print("  3. Monitor GPU: watch -n 1 nvidia-smi")
    print("\n Expected results:")
    print("  ✅ GPU utilization: 70-90%")
    print("  ✅ Training speed: 5-10x faster")
    print("  ✅ Step time: 100-300ms")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Quick GPU optimization fix and validation"
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run full validation tests'
    )
    parser.add_argument(
        '--fix-config',
        action='store_true',
        help='Automatically fix config file'
    )
    parser.add_argument(
        '--no-recommendations',
        action='store_true',
        help='Skip recommendations'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(" 🚀 MyXTTS GPU Optimization Quick Fix")
    print("="*70)
    print("\nThis tool helps diagnose and fix GPU utilization issues.")
    
    # Step 1: Check GPU
    gpu_ok = check_gpu_available()
    
    if not gpu_ok:
        if not args.no_recommendations:
            provide_recommendations(gpu_ok, False)
        return 1
    
    # Step 2: Check current GPU status
    check_nvidia_smi()
    
    # Step 3: Fix config if requested
    if args.fix_config:
        config_fixed = fix_config()
        if config_fixed:
            print("\n✅ Configuration fixed!")
            print("   You can now start training with optimized settings.")
    
    # Step 4: Check config
    config_ok = check_config_optimization()
    
    # Step 5: Run tests if requested
    if args.test:
        tests_ok = run_tests()
        if tests_ok:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed - review output above")
    
    # Step 6: Provide recommendations
    if not args.no_recommendations:
        provide_recommendations(gpu_ok, config_ok)
    
    print("\n" + "="*70)
    print(" For detailed documentation, see:")
    print("   docs/GPU_UTILIZATION_CRITICAL_FIX.md")
    print("="*70 + "\n")
    
    return 0 if (gpu_ok and config_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
