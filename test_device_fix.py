#!/usr/bin/env python3
"""
Test script to verify the device placement fix works correctly.
Run this script to confirm the fix resolves the InvalidArgumentError.
"""

import tensorflow as tf
import sys
import os

def test_device_placement_fix():
    """Test that the device placement fix resolves the original error."""
    
    print("MyXTTS Device Placement Fix Verification")
    print("=" * 50)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    
    try:
        # Test 1: Import and create the problematic layer
        print("\n[TEST 1] Testing MultiHeadAttention layer creation...")
        
        from myxtts.models.layers import MultiHeadAttention
        
        # This was the exact layer that failed in the original error
        attention = MultiHeadAttention(
            d_model=512,
            num_heads=8,
            dropout=0.1,
            name="test_attention"
        )
        print("✓ MultiHeadAttention layer created successfully!")
        
        # Test 2: Test other layers that use dropout
        print("\n[TEST 2] Testing other layers with dropout...")
        
        from myxtts.models.layers import FeedForward, TransformerBlock
        
        ff_layer = FeedForward(d_model=512, d_ff=2048, dropout=0.1)
        print("✓ FeedForward layer created successfully!")
        
        transformer = TransformerBlock(
            d_model=512,
            num_heads=8,
            d_ff=2048,
            dropout=0.1
        )
        print("✓ TransformerBlock layer created successfully!")
        
        # Test 3: Test positional encoding with Variable creation
        print("\n[TEST 3] Testing PositionalEncoding with Variable creation...")
        
        from myxtts.models.layers import PositionalEncoding
        
        pos_enc = PositionalEncoding(d_model=512, max_length=1000)
        print("✓ PositionalEncoding layer created successfully!")
        
        # Test 4: Test actual forward pass
        print("\n[TEST 4] Testing forward pass...")
        
        # Create test input
        test_input = tf.random.normal((2, 10, 512))  # batch=2, seq_len=10, d_model=512
        
        # Test attention forward pass
        attention_output = attention(test_input)
        print(f"✓ Attention forward pass: {test_input.shape} -> {attention_output.shape}")
        
        # Test positional encoding
        pos_output = pos_enc(test_input)
        print(f"✓ Positional encoding: {test_input.shape} -> {pos_output.shape}")
        
        print("\n" + "=" * 50)
        print("🎉 SUCCESS: All tests passed!")
        print("The device placement fix is working correctly.")
        print("The original InvalidArgumentError has been resolved.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ FAILURE: {e}")
        print("\nError details:")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_device_placement_fix()
    sys.exit(0 if success else 1)