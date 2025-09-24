#!/usr/bin/env python3
"""
Test to simulate the original warning scenario and show it's fixed.

This specifically tests the scenario that would generate:
"Gradients do not exist for variables ['xtts/text_encoder/duration_predictor/kernel', ...]"
"""

import sys
import os
sys.path.append('.')

import tensorflow as tf
import numpy as np
import warnings
from myxtts.config.config import ModelConfig
from myxtts.models.xtts import XTTS
from myxtts.training.losses import XTTSLoss

def test_original_warning_simulation():
    """Simulate the exact scenario that would cause the original gradient warning."""
    print("🎯 Simulating original gradient warning scenario...")
    
    # Suppress TensorFlow info logs but keep warnings
    tf.get_logger().setLevel('WARNING')
    
    # Create model config identical to what might cause the warning
    model_config = ModelConfig(
        text_encoder_dim=256,
        text_encoder_layers=4,
        text_encoder_heads=8,
        text_vocab_size=1000,
        audio_encoder_dim=512,
        audio_encoder_layers=4,
        audio_encoder_heads=8,
        decoder_dim=512,
        decoder_layers=4,
        decoder_heads=8,
        speaker_embedding_dim=256,
        use_voice_conditioning=True,
        use_duration_predictor=True,  # This creates the variables mentioned in the warning
        
        # Enable GST features that create prosody predictor variables
        use_gst=True,
        gst_num_style_tokens=10,
        gst_style_token_dim=256,
        gst_style_embedding_dim=256,
        gst_num_heads=8,
        gst_reference_encoder_dim=256,
        
        n_mels=80,
        sample_rate=22050
    )
    
    model = XTTS(model_config)
    
    # Create loss function as it might be used in training
    criterion = XTTSLoss(
        mel_loss_weight=45.0,
        stop_loss_weight=1.0,
        duration_loss_weight=0.8,  # Non-zero weight but no targets provided
        pitch_loss_weight=0.1,
        energy_loss_weight=0.1,
        prosody_pitch_loss_weight=0.05,
        prosody_energy_loss_weight=0.05,
        speaking_rate_loss_weight=0.05
    )
    
    # Create realistic training batch
    batch_size = 8
    text_len = 100
    mel_len = 400
    
    text_inputs = tf.random.uniform((batch_size, text_len), maxval=1000, dtype=tf.int32)
    mel_inputs = tf.random.normal((batch_size, mel_len, 80))
    audio_conditioning = tf.random.normal((batch_size, 200, 80))
    text_lengths = tf.random.uniform((batch_size,), minval=50, maxval=text_len, dtype=tf.int32)
    mel_lengths = tf.random.uniform((batch_size,), minval=200, maxval=mel_len, dtype=tf.int32)
    
    print(f"   Model components:")
    print(f"     Text encoder with duration predictor: {model.text_encoder.use_duration_predictor}")
    print(f"     GST enabled: {model.gst is not None}")  
    print(f"     Prosody predictor: {model.prosody_predictor is not None}")
    
    # Capture any warnings during training step
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        
        with tf.GradientTape() as tape:
            # Forward pass - this creates all the prosody outputs
            outputs = model(
                text_inputs=text_inputs,
                mel_inputs=mel_inputs,
                audio_conditioning=audio_conditioning,
                reference_mel=audio_conditioning,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=True
            )
            
            print(f"   Generated outputs: {list(outputs.keys())}")
            
            # Create targets as they might appear in real training
            # (without prosody targets - the problematic scenario)
            y_true = {
                "mel_target": mel_inputs,
                "stop_target": tf.ones((batch_size, mel_len, 1), dtype=tf.int32),
                "text_lengths": text_lengths,
                "mel_lengths": mel_lengths,
                # Note: No duration_target, pitch_target, etc. provided
                # This is what would cause the gradient warning in the original code
            }
            
            # Include all outputs to ensure gradient participation (the fix)
            y_pred = {
                "mel_output": outputs["mel_output"],
                "stop_tokens": outputs["stop_tokens"],
            }
            
            # Key fix: Include ALL prosody outputs even without targets
            prosody_keys = [
                "duration_pred", "pitch_output", "energy_output",
                "prosody_pitch", "prosody_energy", "prosody_speaking_rate"
            ]
            
            for key in prosody_keys:
                if key in outputs:
                    y_pred[key] = outputs[key]
            
            print(f"   Prediction keys: {list(y_pred.keys())}")
            
            # Compute loss - this now includes regularization for missing targets
            loss = criterion(y_true, y_pred)
            
        # Check for gradient warnings
        print(f"   Warnings captured: {len(w)}")
        gradient_warnings = [warning for warning in w 
                           if "Gradients do not exist" in str(warning.message)]
        
        if gradient_warnings:
            print("   ❌ Gradient warnings detected:")
            for warning in gradient_warnings:
                print(f"     {warning.message}")
        else:
            print("   ✅ No gradient warnings detected!")
        
        # Compute gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Count missing gradients
        missing_count = sum(1 for grad in gradients if grad is None)
        total_vars = len(model.trainable_variables)
        
        print(f"   Total variables: {total_vars}")
        print(f"   Variables with gradients: {total_vars - missing_count}")
        print(f"   Variables without gradients: {missing_count}")
        print(f"   Loss value: {float(loss):.4f}")
        
        # Check if any individual loss components indicate problems
        if hasattr(criterion, 'losses'):
            print(f"   Loss components: {list(criterion.losses.keys())}")
        
        return missing_count == 0 and len(gradient_warnings) == 0

if __name__ == "__main__":
    print("Original Gradient Warning Simulation Test")
    print("=" * 50)
    
    success = test_original_warning_simulation()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Original gradient warning issue is FIXED!")
        print("   ✅ No gradient warnings detected")
        print("   ✅ All model variables participate in gradient computation")
        print("\nThe fix ensures that:")
        print("• All prosody outputs are included in y_pred during training")
        print("• Regularization loss is added for outputs without targets")
        print("• All model components participate in gradient computation")
    else:
        print("💥 FAILURE: Gradient warnings may still occur")
        sys.exit(1)