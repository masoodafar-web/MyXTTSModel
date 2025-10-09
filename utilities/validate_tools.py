#!/usr/bin/env python3
"""
Simple validation script to check that all tools exist and are properly structured.

This doesn't require TensorFlow to be installed.
"""

import sys
import os
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def check_tool_structure(filepath):
    """Check if tool has proper structure."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        checks = {
            'has_shebang': content.startswith('#!/usr/bin/env python3'),
            'has_docstring': '"""' in content[:500],
            'has_main': 'def main()' in content,
            'has_main_guard': 'if __name__ == "__main__"' in content,
        }
        
        for check, result in checks.items():
            status = "✅" if result else "❌"
            print(f"  {status} {check}")
        
        return all(checks.values())
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False

def main():
    """Main validation."""
    print("="*70)
    print("VALIDATION: GPU PROFILING TOOLS")
    print("="*70)
    
    base_path = Path(__file__).parent.parent
    all_ok = True
    
    # Check tool files
    print("\n1. Checking Tool Files:")
    print("-"*70)
    
    tools = [
        ('utilities/comprehensive_gpu_diagnostic.py', 'Comprehensive Diagnostic'),
        ('utilities/enhanced_gpu_profiler.py', 'Enhanced Profiler'),
        ('utilities/training_step_profiler.py', 'Training Step Profiler'),
    ]
    
    for tool_path, description in tools:
        full_path = base_path / tool_path
        exists = check_file_exists(full_path, description)
        all_ok = all_ok and exists
        
        if exists:
            structure_ok = check_tool_structure(full_path)
            all_ok = all_ok and structure_ok
        print()
    
    # Check documentation
    print("\n2. Checking Documentation:")
    print("-"*70)
    
    docs = [
        ('docs/COMPREHENSIVE_GPU_BOTTLENECK_ANALYSIS.md', 'Comprehensive Guide'),
        ('docs/GPU_OSCILLATION_FIX.md', 'GPU Oscillation Fix'),
        ('GPU_OSCILLATION_SOLUTION_SUMMARY.md', 'Solution Summary'),
        ('QUICK_START_GPU_OSCILLATION_FIX.md', 'Quick Start Guide'),
    ]
    
    for doc_path, description in docs:
        full_path = base_path / doc_path
        exists = check_file_exists(full_path, description)
        all_ok = all_ok and exists
        print()
    
    # Check test file
    print("\n3. Checking Test Files:")
    print("-"*70)
    
    tests = [
        ('tests/test_new_profiling_tools.py', 'New Tools Test'),
        ('tests/test_gpu_oscillation_fix.py', 'GPU Oscillation Fix Test'),
    ]
    
    for test_path, description in tests:
        full_path = base_path / test_path
        exists = check_file_exists(full_path, description)
        all_ok = all_ok and exists
        print()
    
    # Check original diagnostic tool still exists
    print("\n4. Checking Original Tools:")
    print("-"*70)
    
    original = [
        ('utilities/diagnose_gpu_bottleneck.py', 'Original Diagnostic'),
        ('myxtts/data/tf_native_loader.py', 'TF Native Loader'),
    ]
    
    for tool_path, description in original:
        full_path = base_path / tool_path
        exists = check_file_exists(full_path, description)
        all_ok = all_ok and exists
        print()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    if all_ok:
        print("✅ ALL CHECKS PASSED")
        print("\nNew profiling tools are properly installed:")
        print("  1. comprehensive_gpu_diagnostic.py - Master diagnostic tool")
        print("  2. enhanced_gpu_profiler.py - Deep data pipeline profiling")
        print("  3. training_step_profiler.py - Complete training loop profiling")
        print("\nDocumentation:")
        print("  - docs/COMPREHENSIVE_GPU_BOTTLENECK_ANALYSIS.md")
        print("\nNext steps:")
        print("  1. Install TensorFlow: pip install tensorflow>=2.12.0")
        print("  2. Run: python utilities/comprehensive_gpu_diagnostic.py --data-path ./data")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("Please review the output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
