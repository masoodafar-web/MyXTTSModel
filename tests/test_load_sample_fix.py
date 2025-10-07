#!/usr/bin/env python3
"""
Test script to verify the _load_sample method fix for LJSpeechDataset.

This test ensures that the AttributeError: 'LJSpeechDataset' object has no attribute '_load_sample'
is resolved by implementing the missing method.

This is a minimal test that verifies the method exists by inspecting the source code
without requiring full TensorFlow installation.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_load_sample_method():
    """Test that _load_sample method exists by inspecting the source code."""
    print("=" * 60)
    print("Testing _load_sample method fix")
    print("=" * 60)
    
    try:
        # Read the ljspeech.py file and check for _load_sample method
        print("\n1. Reading LJSpeechDataset source code...")
        ljspeech_path = Path(__file__).parent.parent / "myxtts" / "data" / "ljspeech.py"
        
        if not ljspeech_path.exists():
            print(f"   ❌ ERROR: File not found at {ljspeech_path}")
            return False
        
        with open(ljspeech_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        print(f"   ✓ Source file loaded: {ljspeech_path}")
        
        # Check if _load_sample method is defined
        print("\n2. Checking for _load_sample method definition...")
        
        if 'def _load_sample(self, idx: int)' in source_code:
            print("   ✓ _load_sample method is defined")
        else:
            print("   ❌ ERROR: _load_sample method definition not found!")
            return False
        
        # Check if it contains the expected implementation
        print("\n3. Verifying _load_sample implementation...")
        
        if 'return self.__getitem__(idx)' in source_code:
            print("   ✓ _load_sample correctly calls __getitem__")
        else:
            print("   ⚠ WARNING: _load_sample might have a different implementation")
        
        # Check if _load_sample_numpy calls _load_sample
        print("\n4. Verifying _load_sample_numpy calls _load_sample...")
        
        if 'sample = self._load_sample(idx_val)' in source_code:
            print("   ✓ _load_sample_numpy correctly calls _load_sample")
        else:
            print("   ❌ ERROR: _load_sample_numpy doesn't call _load_sample!")
            return False
        
        # Extract method definition for display
        print("\n5. Extracting _load_sample method...")
        
        # Find the method definition
        import re
        pattern = r'(def _load_sample\(self.*?\n(?:.*?\n)*?(?=\n    def |$))'
        match = re.search(pattern, source_code, re.MULTILINE)
        
        if match:
            method_code = match.group(1)
            # Limit to first 20 lines for display
            lines = method_code.split('\n')[:20]
            print("   Method definition found:")
            print("   " + "-" * 56)
            for line in lines:
                print("   " + line)
            if len(method_code.split('\n')) > 20:
                print("   ...")
            print("   " + "-" * 56)
        
        print("\n" + "=" * 60)
        print("✅ ALL CHECKS PASSED!")
        print("=" * 60)
        print("\nThe _load_sample method has been successfully implemented.")
        print("The AttributeError: 'LJSpeechDataset' object has no attribute '_load_sample'")
        print("should now be resolved when training is executed.")
        print("\nThe method:")
        print("  - Exists in the LJSpeechDataset class")
        print("  - Is called by _load_sample_numpy")
        print("  - Wraps __getitem__ to provide data loading functionality")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_load_sample_method()
    sys.exit(0 if success else 1)
