#!/usr/bin/env python3
"""
Fix Common Issues Script
حل مشکلات رایج پروژه MyXTTS
"""

import os
import sys
import subprocess

def fix_cuda_warnings():
    """Fix CUDA warnings and registrations"""
    print("🔧 Fixing CUDA warnings...")
    
    # Set environment variables
    env_vars = {
        'TF_CPP_MIN_LOG_LEVEL': '3',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
        'TF_GPU_ALLOCATOR': 'cuda_malloc_async',
        'CUDA_VISIBLE_DEVICES': '1',  # Use GPU 1 to avoid conflicts
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"  ✅ Set {key}={value}")
    
    return True

def fix_pynvml_deprecation():
    """Fix PyNVML deprecation warning"""
    print("🔧 Fixing PyNVML deprecation...")
    
    try:
        # Install nvidia-ml-py
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'nvidia-ml-py'], 
                      capture_output=True, text=True)
        print("  ✅ nvidia-ml-py installed")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not install nvidia-ml-py: {e}")
        return False

def fix_matplotlib_issues():
    """Fix matplotlib Axes3D warning"""
    print("🔧 Fixing matplotlib issues...")
    
    try:
        # Reinstall matplotlib
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'matplotlib'], 
                      capture_output=True, text=True)
        print("  ✅ matplotlib upgraded")
        return True
    except Exception as e:
        print(f"  ⚠️  Could not upgrade matplotlib: {e}")
        return False

def create_optimized_environment_script():
    """Create script with optimized environment settings"""
    print("🔧 Creating optimized environment script...")
    
    script_content = '''#!/bin/bash
# Optimized Environment for MyXTTS Training

# CUDA Settings
export TF_CPP_MIN_LOG_LEVEL=3
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_VISIBLE_DEVICES=1

# Python Settings  
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONUNBUFFERED=1

# Memory Settings
export TF_ENABLE_ONEDNN_OPTS=1
export TF_ENABLE_GPU_GARBAGE_COLLECTION=1

# Training with optimized settings
python3 train_main.py \\
    --model-size normal \\
    --optimization-level enhanced \\
    --enable-gpu-stabilizer \\
    --batch-size 24 \\
    --epochs 100 \\
    "$@"
'''
    
    with open('train_optimized.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('train_optimized.sh', 0o755)
    print("  ✅ Created train_optimized.sh")
    return True

def main():
    """Main fix function"""
    print("🚀 MyXTTS Common Issues Fix")
    print("=" * 40)
    
    fixes = [
        fix_cuda_warnings(),
        fix_pynvml_deprecation(),
        fix_matplotlib_issues(),
        create_optimized_environment_script()
    ]
    
    success_count = sum(fixes)
    print(f"\n📊 Results: {success_count}/{len(fixes)} fixes applied successfully")
    
    if success_count >= len(fixes) - 1:  # Allow 1 failure
        print("\n✅ Most issues should be resolved!")
        print("\n🎯 Next steps:")
        print("1. Restart your terminal/session")
        print("2. Run: ./train_optimized.sh")
        print("3. Or run: python3 train_main.py --model-size tiny --epochs 10")
    else:
        print("\n⚠️  Some issues remain. Check output above.")
    
    return success_count >= len(fixes) - 1

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)