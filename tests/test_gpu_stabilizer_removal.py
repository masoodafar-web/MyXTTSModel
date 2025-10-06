#!/usr/bin/env python3
"""
Test script to verify GPU Stabilizer has been completely removed.
This ensures no references remain in the codebase.
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_no_gpu_stabilizer_module():
    """Verify the GPU stabilizer module is deleted"""
    gpu_stabilizer_path = project_root / "optimization" / "advanced_gpu_stabilizer.py"
    assert not gpu_stabilizer_path.exists(), "GPU stabilizer module still exists!"
    print("✅ GPU stabilizer module successfully deleted")

def test_no_gpu_stabilizer_references():
    """Verify no GPU stabilizer references in active code"""
    # Search for GPU stabilizer references (excluding archive and test files)
    excluded_dirs = [".git", "tmp_runtime", "archive", "runs", "tests"]
    search_patterns = [
        "gpu_stabilizer",
        "GPU_STABILIZER", 
        "AdvancedGPUStabilizer",
        "enable-gpu-stabilizer",
        "disable-gpu-stabilizer"
    ]
    
    for pattern in search_patterns:
        cmd = ["grep", "-r", pattern, str(project_root)]
        for exclude in excluded_dirs:
            cmd.extend(["--exclude-dir", exclude])
        cmd.extend(["--include=*.py", "--include=*.sh"])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout:
            print(f"❌ Found GPU stabilizer references for pattern '{pattern}':")
            print(result.stdout[:500])
            return False
    
    print("✅ No GPU stabilizer references found in active code")
    return True

def test_trainer_no_gpu_stabilizer_param():
    """Verify trainer __init__ doesn't have gpu_stabilizer_enabled parameter"""
    trainer_file = project_root / "myxtts" / "training" / "trainer.py"
    
    with open(trainer_file, 'r') as f:
        content = f.read()
        
    assert "gpu_stabilizer_enabled" not in content, \
        "Found gpu_stabilizer_enabled parameter in trainer.py"
    
    print("✅ Trainer doesn't have gpu_stabilizer_enabled parameter")

def test_train_main_no_gpu_stabilizer_args():
    """Verify train_main.py doesn't have GPU stabilizer arguments"""
    train_main_file = project_root / "train_main.py"
    
    with open(train_main_file, 'r') as f:
        content = f.read()
    
    assert '--enable-gpu-stabilizer' not in content, \
        "Found --enable-gpu-stabilizer in train_main.py"
    assert '--disable-gpu-stabilizer' not in content, \
        "Found --disable-gpu-stabilizer in train_main.py"
    
    print("✅ train_main.py doesn't have GPU stabilizer arguments")

def test_shell_scripts_valid():
    """Verify all shell scripts have valid syntax"""
    scripts_dir = project_root / "scripts"
    script_files = list(scripts_dir.glob("*.sh"))
    script_files.extend(project_root.glob("*.sh"))
    
    all_valid = True
    for script in script_files:
        result = subprocess.run(
            ["bash", "-n", str(script)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"❌ {script.name} has syntax errors:")
            print(result.stderr)
            all_valid = False
    
    if all_valid:
        print(f"✅ All {len(script_files)} shell scripts have valid syntax")
    
    return all_valid

def test_python_files_valid():
    """Verify key Python files have valid syntax"""
    key_files = [
        "train_main.py",
        "myxtts/training/trainer.py",
        "utilities/benchmark_hyperparameters.py",
        "scripts/hparam_grid_search.py",
        "scripts/quick_param_test.py",
    ]
    
    all_valid = True
    for file_path in key_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue
            
        result = subprocess.run(
            ["python3", "-m", "py_compile", str(full_path)],
            capture_output=True,
            text=True,
            cwd=str(project_root)
        )
        if result.returncode != 0:
            print(f"❌ {file_path} has syntax errors:")
            print(result.stderr)
            all_valid = False
    
    if all_valid:
        print(f"✅ All {len(key_files)} key Python files have valid syntax")
    
    return all_valid

def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing GPU Stabilizer Removal")
    print("=" * 60)
    print()
    
    tests = [
        ("Module Deletion", test_no_gpu_stabilizer_module),
        ("No References", test_no_gpu_stabilizer_references),
        ("Trainer Parameter", test_trainer_no_gpu_stabilizer_param),
        ("CLI Arguments", test_train_main_no_gpu_stabilizer_args),
        ("Shell Scripts", test_shell_scripts_valid),
        ("Python Syntax", test_python_files_valid),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            if result is None or result is True:
                passed += 1
            else:
                failed += 1
        except AssertionError as e:
            print(f"❌ {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        print("\n✨ All tests passed! GPU Stabilizer successfully removed.")
        sys.exit(0)

if __name__ == "__main__":
    main()
