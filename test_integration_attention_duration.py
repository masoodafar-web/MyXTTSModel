#!/usr/bin/env python3
"""
Integration test to verify that the XTTS model with duration and attention outputs
works correctly in a simulated training step.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the myxtts package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from myxtts.models.xtts import XTTS
from myxtts.config.config import XTTSConfig
from myxtts.training.losses import XTTSLoss


def test_training_step():
    """Test a simulated training step with the enhanced model."""
    print("🧪 Testing simulated training step with duration and attention...")
    
    # Create config with enabled attention and duration losses
    config = XTTSConfig()
    config.training.attention_loss_weight = 1.0
    config.training.duration_loss_weight = 0.1
    
    # Create model and loss function
    model = XTTS(config.model)
    loss_fn = XTTSLoss(
        mel_loss_weight=config.training.mel_loss_weight,
        stop_loss_weight=1.0,
        attention_loss_weight=config.training.attention_loss_weight,
        duration_loss_weight=config.training.duration_loss_weight
    )
    
    # Create optimizer 
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    
    # Create dummy training data
    batch_size = 2
    text_len = 15
    mel_len = 30
    n_mels = config.model.n_mels
    
    text_inputs = tf.random.uniform([batch_size, text_len], 0, 100, dtype=tf.int32)
    mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
    text_lengths = tf.constant([text_len, text_len-2], dtype=tf.int32)
    mel_lengths = tf.constant([mel_len, mel_len-5], dtype=tf.int32)
    
    # Create target values
    y_true = {
        "mel_target": tf.random.normal([batch_size, mel_len, n_mels]),
        "stop_target": tf.concat([
            tf.zeros([batch_size, mel_len-1, 1]),
            tf.ones([batch_size, 1, 1])
        ], axis=1),
        "duration_target": tf.random.uniform([batch_size, text_len], 0.5, 1.5),
        "text_lengths": text_lengths,
        "mel_lengths": mel_lengths
    }
    
    # Simulate training step
    print("  📋 Running forward pass...")
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(
            text_inputs=text_inputs,
            mel_inputs=mel_inputs,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        # Compute loss
        loss = loss_fn(y_true, y_pred)
    
    print(f"    ✅ Forward pass successful, loss: {loss.numpy():.4f}")
    
    # Check that we have the expected outputs
    expected_outputs = ["mel_output", "stop_tokens", "text_encoded", "duration_pred", "attention_weights"]
    for output in expected_outputs:
        if output in y_pred:
            print(f"    ✅ {output}: shape {y_pred[output].shape}")
        else:
            print(f"    ❌ Missing output: {output}")
    
    # Compute gradients
    print("  📋 Computing gradients...")
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Check that gradients are computed
    grad_norms = []
    for grad in gradients:
        if grad is not None:
            grad_norms.append(tf.norm(grad).numpy())
    
    print(f"    ✅ Gradients computed for {len(grad_norms)}/{len(gradients)} variables")
    print(f"    📊 Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"    📊 Max gradient norm: {np.max(grad_norms):.6f}")
    
    # Apply gradients
    print("  📋 Applying gradients...")
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    print("    ✅ Gradient update successful")
    
    # Verify individual loss components
    individual_losses = loss_fn.losses
    print("\n  📊 Individual loss components:")
    for loss_name, loss_value in individual_losses.items():
        print(f"    • {loss_name}: {loss_value.numpy():.4f}")
    
    # Verify attention and duration losses are computed
    assert "attention_loss" in individual_losses, "Attention loss should be computed"
    assert "duration_loss" in individual_losses, "Duration loss should be computed"
    print("    ✅ Both attention and duration losses are active")
    
    return loss.numpy()


def test_inference_mode():
    """Test that inference mode works correctly."""
    print("\n🧪 Testing inference mode...")
    
    config = XTTSConfig()
    model = XTTS(config.model)
    
    # Create dummy inputs
    batch_size = 1
    text_len = 10
    mel_len = 20
    n_mels = config.model.n_mels
    
    text_inputs = tf.random.uniform([batch_size, text_len], 0, 100, dtype=tf.int32)
    mel_inputs = tf.random.normal([batch_size, mel_len, n_mels])
    
    # Inference mode should not return duration/attention outputs
    outputs = model(
        text_inputs=text_inputs,
        mel_inputs=mel_inputs,
        training=False
    )
    
    print(f"  📋 Inference outputs:")
    for key, value in outputs.items():
        print(f"    ✅ {key}: shape {value.shape}")
    
    # Verify no training-specific outputs
    assert "duration_pred" not in outputs, "Duration predictions should not be in inference mode"
    assert "attention_weights" not in outputs, "Attention weights should not be in inference mode"
    print("    ✅ Inference mode correctly excludes training-specific outputs")
    
    return outputs


def main():
    """Run integration tests."""
    print("🚀 Starting XTTS Integration Tests with Duration and Attention")
    print("=" * 70)
    
    try:
        # Test training step
        training_loss = test_training_step()
        
        # Test inference mode
        inference_outputs = test_inference_mode()
        
        print("\n" + "=" * 70)
        print("🎉 Integration tests passed!")
        print(f"   ✅ Training with attention/duration loss: {training_loss:.4f}")
        print(f"   ✅ Inference mode works correctly")
        print(f"   ✅ Model ready for training with improved alignment stability")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)