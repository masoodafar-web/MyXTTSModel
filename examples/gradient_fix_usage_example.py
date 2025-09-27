#!/usr/bin/env python3
"""
Usage example showing how to use the gradient warning fix in training.

This example demonstrates the proper way to structure training loops
to avoid gradient warnings for prosody predictor and duration predictor variables.
"""

import sys
import os
sys.path.append('.')

import tensorflow as tf
import numpy as np
from myxtts.config.config import ModelConfig, DataConfig, TrainingConfig, XTTSConfig
from myxtts.models.xtts import XTTS
from myxtts.training.losses import XTTSLoss

def create_training_step_with_gradient_fix():
    """Create a proper training step that avoids gradient warnings."""
    
    # Create model configuration
    model_config = ModelConfig(
        text_encoder_dim=512,
        text_encoder_layers=6,
        text_encoder_heads=8,
        text_vocab_size=1000,
        audio_encoder_dim=768,
        audio_encoder_layers=6,
        audio_encoder_heads=8,
        decoder_dim=1024,
        decoder_layers=6,
        decoder_heads=8,
        speaker_embedding_dim=256,
        use_voice_conditioning=True,
        use_duration_predictor=True,  # Enable duration predictor
        
        # Enable prosody features
        use_gst=True,
        gst_num_style_tokens=10,
        gst_style_token_dim=256,
        gst_style_embedding_dim=256,
        gst_num_heads=8,
        gst_reference_encoder_dim=256,
        
        n_mels=80,
        sample_rate=22050
    )
    
    # Create model
    model = XTTS(model_config)
    
    # Create comprehensive loss function
    # The enhanced loss function includes regularization to prevent gradient warnings
    criterion = XTTSLoss(
        mel_loss_weight=45.0,
        stop_loss_weight=1.0,
        attention_loss_weight=0.1,
        duration_loss_weight=0.8,
        pitch_loss_weight=0.5,
        energy_loss_weight=0.5,
        prosody_pitch_loss_weight=0.3,
        prosody_energy_loss_weight=0.3,
        speaking_rate_loss_weight=0.3,
        voice_similarity_loss_weight=1.0
    )
    
    optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-4)
    
    def training_step(batch_data):
        """
        Training step that prevents gradient warnings.
        
        Args:
            batch_data: Dictionary containing:
                - text_inputs: Text token IDs
                - mel_inputs: Target mel spectrograms  
                - audio_conditioning: Reference audio for voice cloning
                - text_lengths: Text sequence lengths
                - mel_lengths: Mel sequence lengths
                - duration_targets: Optional duration targets
                - pitch_targets: Optional pitch targets
                - energy_targets: Optional energy targets
                - prosody_*_targets: Optional prosody targets
        """
        
        with tf.GradientTape() as tape:
            # Forward pass with all features enabled
            outputs = model(
                text_inputs=batch_data["text_inputs"],
                mel_inputs=batch_data["mel_inputs"],
                audio_conditioning=batch_data.get("audio_conditioning"),
                reference_mel=batch_data.get("reference_mel", batch_data.get("audio_conditioning")),
                text_lengths=batch_data.get("text_lengths"),
                mel_lengths=batch_data.get("mel_lengths"),
                training=True
            )
            
            # Prepare targets (can be minimal)
            y_true = {
                "mel_target": batch_data["mel_inputs"],
                "stop_target": batch_data.get("stop_targets", 
                                            tf.ones((tf.shape(batch_data["mel_inputs"])[0], 
                                                   tf.shape(batch_data["mel_inputs"])[1], 1), 
                                                   dtype=tf.int32)),
                "text_lengths": batch_data.get("text_lengths"),
                "mel_lengths": batch_data.get("mel_lengths"),
            }
            
            # Add prosody targets if available (optional)
            prosody_target_keys = [
                "duration_target", "pitch_target", "energy_target",
                "prosody_pitch_target", "prosody_energy_target", 
                "prosody_speaking_rate_target"
            ]
            
            for key in prosody_target_keys:
                if key in batch_data:
                    y_true[key] = batch_data[key]
            
            # CRITICAL: Include ALL model outputs in y_pred to ensure gradient participation
            y_pred = {
                "mel_output": outputs["mel_output"],
                "stop_tokens": outputs["stop_tokens"],
            }
            
            # Include all prosody outputs - this is the key fix!
            prosody_output_keys = [
                "duration_pred", "pitch_output", "energy_output",
                "prosody_pitch", "prosody_energy", "prosody_speaking_rate"
            ]
            
            for key in prosody_output_keys:
                if key in outputs:
                    y_pred[key] = outputs[key]
            
            # Add other outputs if available
            for key in ["attention_weights", "speaker_embedding"]:
                if key in outputs:
                    y_pred[key] = outputs[key]
            
            # Compute loss - the enhanced loss function will add regularization
            # for outputs that don't have corresponding targets
            loss = criterion(y_true, y_pred)
        
        # Compute gradients - all variables should have gradients now
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Check for gradient warnings (optional monitoring)
        missing_gradients = sum(1 for grad in gradients if grad is None)
        if missing_gradients > 0:
            tf.print(f"Warning: {missing_gradients} variables without gradients")
        
        # Apply gradients
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return {
            "loss": loss,
            "mel_loss": criterion.losses.get("mel_loss", 0.0),
            "duration_loss": criterion.losses.get("duration_loss", 0.0),
            "pitch_loss": criterion.losses.get("pitch_loss", 0.0),
            "energy_loss": criterion.losses.get("energy_loss", 0.0),
            "prosody_pitch_loss": criterion.losses.get("prosody_pitch_loss", 0.0),
            "prosody_energy_loss": criterion.losses.get("prosody_energy_loss", 0.0),
            "speaking_rate_loss": criterion.losses.get("speaking_rate_loss", 0.0),
            "gradient_participation_loss": criterion.losses.get("gradient_participation_loss", 0.0),
            "variables_with_gradients": len(model.trainable_variables) - missing_gradients,
            "total_variables": len(model.trainable_variables)
        }
    
    return model, training_step

def example_training_loop():
    """Example training loop using the gradient warning fix."""
    
    print("🚀 Example Training Loop with Gradient Warning Fix")
    print("=" * 60)
    
    # Create model and training step
    model, training_step = create_training_step_with_gradient_fix()
    
    # Create example batch data
    batch_size = 4
    text_len = 50
    mel_len = 200
    
    batch_data = {
        "text_inputs": tf.random.uniform((batch_size, text_len), maxval=1000, dtype=tf.int32),
        "mel_inputs": tf.random.normal((batch_size, mel_len, 80)),
        "audio_conditioning": tf.random.normal((batch_size, 100, 80)),
        "text_lengths": tf.random.uniform((batch_size,), minval=30, maxval=text_len, dtype=tf.int32),
        "mel_lengths": tf.random.uniform((batch_size,), minval=150, maxval=mel_len, dtype=tf.int32),
        
        # Optional: Include prosody targets if available
        # "duration_target": tf.random.uniform((batch_size, text_len), minval=1.0, maxval=10.0),
        # "pitch_target": tf.random.uniform((batch_size, mel_len, 1), minval=80.0, maxval=400.0),
        # etc.
    }
    
    print(f"Batch size: {batch_size}")
    print(f"Text length: {text_len}")
    print(f"Mel length: {mel_len}")
    print(f"Available batch keys: {list(batch_data.keys())}")
    
    # Run training step
    print("\nRunning training step...")
    
    results = training_step(batch_data)
    
    print(f"\n📊 Training Results:")
    print(f"   Total loss: {float(results['loss']):.4f}")
    print(f"   Mel loss: {float(results['mel_loss']):.4f}")
    print(f"   Duration loss: {float(results['duration_loss']):.4f}")
    print(f"   Pitch loss: {float(results['pitch_loss']):.4f}")
    print(f"   Energy loss: {float(results['energy_loss']):.4f}")
    print(f"   Prosody pitch loss: {float(results['prosody_pitch_loss']):.4f}")
    print(f"   Prosody energy loss: {float(results['prosody_energy_loss']):.4f}")
    print(f"   Speaking rate loss: {float(results['speaking_rate_loss']):.4f}")
    print(f"   Gradient participation loss: {float(results['gradient_participation_loss']):.4f}")
    
    print(f"\n🔧 Gradient Health:")
    print(f"   Variables with gradients: {results['variables_with_gradients']}")
    print(f"   Total variables: {results['total_variables']}")
    
    if results['variables_with_gradients'] == results['total_variables']:
        print("   ✅ All variables participating in gradient computation!")
        print("   ✅ No gradient warnings expected!")
    else:
        missing = results['total_variables'] - results['variables_with_gradients']
        print(f"   ❌ {missing} variables missing gradients!")
    
    return results['variables_with_gradients'] == results['total_variables']

if __name__ == "__main__":
    success = example_training_loop()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 SUCCESS: Training loop works without gradient warnings!")
        print("\n💡 Key points for avoiding gradient warnings:")
        print("   1. Always include ALL model outputs in y_pred during training")
        print("   2. The enhanced XTTSLoss adds regularization for missing targets")
        print("   3. All prosody components participate in gradient computation")
        print("   4. No code changes needed in existing training scripts!")
    else:
        print("💥 FAILURE: Some issues remain with gradient computation")
        sys.exit(1)