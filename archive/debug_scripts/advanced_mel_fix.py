#!/usr/bin/env python3
"""
Advanced Mel Scaling Fix for MyXTTS

This script implements multiple scaling strategies to fix the mel spectrogram mismatch.
Since the model outputs are close to zero, we need to scale them appropriately.
"""

import numpy as np
import soundfile as sf
import tensorflow as tf
from train_main import build_config
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import find_latest_checkpoint
from mel_normalization_fix import MelNormalizer
from typing import Tuple, Dict, Any
import logging

class MelScaler:
    """Advanced mel spectrogram scaling for inference."""
    
    def __init__(self):
        self.mel_normalizer = MelNormalizer()
        self.scale_factor = None
        self.offset = None
        self.logger = logging.getLogger("MelScaler")
    
    def compute_scaling_params(self, mel_target: np.ndarray, mel_pred: np.ndarray) -> Tuple[float, float]:
        """
        Compute optimal scaling parameters from teacher forcing example.
        
        Args:
            mel_target: Target mel spectrogram
            mel_pred: Predicted mel spectrogram
            
        Returns:
            Tuple of (scale_factor, offset)
        """
        # Method 1: Linear regression approach
        # mel_scaled = mel_pred * scale_factor + offset ≈ mel_target
        
        # Flatten arrays for linear regression
        target_flat = mel_target.flatten()
        pred_flat = mel_pred.flatten()
        
        # Remove near-zero predictions to avoid division issues
        valid_mask = np.abs(pred_flat) > 1e-6
        if np.sum(valid_mask) < len(pred_flat) * 0.1:
            self.logger.warning("Too few valid predictions for scaling computation")
            # Fallback to simple scaling
            target_std = np.std(target_flat)
            pred_std = np.std(pred_flat)
            scale_factor = target_std / (pred_std + 1e-8)
            offset = np.mean(target_flat) - scale_factor * np.mean(pred_flat)
        else:
            target_valid = target_flat[valid_mask]
            pred_valid = pred_flat[valid_mask]
            
            # Solve for scale and offset using least squares
            # target = pred * scale + offset
            A = np.vstack([pred_valid, np.ones(len(pred_valid))]).T
            scale_factor, offset = np.linalg.lstsq(A, target_valid, rcond=None)[0]
        
        self.scale_factor = scale_factor
        self.offset = offset
        
        self.logger.info(f"Computed scaling parameters: scale={scale_factor:.3f}, offset={offset:.3f}")
        return scale_factor, offset
    
    def apply_scaling(self, mel_pred: np.ndarray) -> np.ndarray:
        """Apply learned scaling to prediction."""
        if self.scale_factor is None or self.offset is None:
            self.logger.warning("Scaling parameters not computed, returning original")
            return mel_pred
        
        mel_scaled = mel_pred * self.scale_factor + self.offset
        return mel_scaled
    
    def adaptive_scaling(self, mel_target: np.ndarray, mel_pred: np.ndarray) -> np.ndarray:
        """
        Apply adaptive scaling that matches target statistics.
        
        Args:
            mel_target: Target mel spectrogram  
            mel_pred: Predicted mel spectrogram
            
        Returns:
            Scaled prediction
        """
        # Match mean and standard deviation
        target_mean = np.mean(mel_target)
        target_std = np.std(mel_target)
        
        pred_mean = np.mean(mel_pred)
        pred_std = np.std(mel_pred)
        
        # Scale to match target statistics
        if pred_std > 1e-8:
            mel_scaled = (mel_pred - pred_mean) * (target_std / pred_std) + target_mean
        else:
            # If prediction has no variation, use target mean
            mel_scaled = np.full_like(mel_pred, target_mean)
        
        return mel_scaled


def run_comprehensive_mel_fix():
    """Run comprehensive mel spectrogram fix with multiple strategies."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MelFix")
    
    # Load checkpoint and config
    ckpt = find_latest_checkpoint('./checkpointsmain')
    config = build_config(model_size='normal', checkpoint_dir='./checkpointsmain')
    tf.config.set_visible_devices([], 'GPU')  # CPU run
    
    logger.info(f"Using checkpoint: {ckpt}")
    
    # Initialize inference
    inf = XTTSInference(config=config, checkpoint_path=ckpt)
    scaler = MelScaler()
    
    # Test data
    text = "The Chronicles of Newgate, Volume two. By Arthur Griffiths."
    processed = inf._preprocess_text(text, config.data.language)
    seq = inf.text_processor.text_to_sequence(processed)
    text_tensor = tf.constant([seq], dtype=tf.int32)
    lengths = tf.constant([len(seq)], dtype=tf.int32)
    
    # Load and process audio
    audio, sr = sf.read('./speaker.wav')
    if audio.ndim > 1: 
        audio = audio.mean(axis=1)
    
    audio_processed = inf.audio_processor.preprocess_audio(audio)
    mel_raw = inf.audio_processor.wav_to_mel(audio_processed).T
    
    print(f"\nOriginal mel stats:")
    print(f"Raw target: min={mel_raw.min():.3f}, max={mel_raw.max():.3f}, mean={mel_raw.mean():.3f}, std={mel_raw.std():.3f}")
    
    # Normalize input
    mel_normalized = scaler.mel_normalizer.normalize(mel_raw)
    
    # Run teacher forcing
    mel_tensor = tf.constant(mel_normalized[np.newaxis, ...], dtype=tf.float32)
    mel_lengths = tf.constant([mel_normalized.shape[0]], dtype=tf.int32)
    
    outputs = inf.model(text_inputs=text_tensor, mel_inputs=mel_tensor,
                       text_lengths=lengths, mel_lengths=mel_lengths, training=True)
    mel_pred_norm = outputs['mel_output'].numpy()[0]
    
    print(f"Model prediction: min={mel_pred_norm.min():.3f}, max={mel_pred_norm.max():.3f}, mean={mel_pred_norm.mean():.3f}, std={mel_pred_norm.std():.3f}")
    
    # Strategy 1: Direct scaling back to original space
    print(f"\n=== Strategy 1: Direct denormalization ===")
    mel_denorm = scaler.mel_normalizer.denormalize(mel_pred_norm)
    mae_denorm = np.mean(np.abs(mel_raw - mel_denorm))
    print(f"Denormalized: min={mel_denorm.min():.3f}, max={mel_denorm.max():.3f}, mean={mel_denorm.mean():.3f}, std={mel_denorm.std():.3f}")
    print(f"MAE vs raw target: {mae_denorm:.3f}")
    
    # Strategy 2: Adaptive scaling to match target statistics
    print(f"\n=== Strategy 2: Adaptive statistics matching ===")
    mel_adaptive = scaler.adaptive_scaling(mel_normalized, mel_pred_norm)
    mel_adaptive_denorm = scaler.mel_normalizer.denormalize(mel_adaptive)
    mae_adaptive = np.mean(np.abs(mel_raw - mel_adaptive_denorm))
    print(f"Adaptive (norm): min={mel_adaptive.min():.3f}, max={mel_adaptive.max():.3f}, mean={mel_adaptive.mean():.3f}, std={mel_adaptive.std():.3f}")
    print(f"Adaptive (denorm): min={mel_adaptive_denorm.min():.3f}, max={mel_adaptive_denorm.max():.3f}, mean={mel_adaptive_denorm.mean():.3f}, std={mel_adaptive_denorm.std():.3f}")
    print(f"MAE vs raw target: {mae_adaptive:.3f}")
    
    # Strategy 3: Linear regression scaling
    print(f"\n=== Strategy 3: Linear regression scaling ===")
    scale_factor, offset = scaler.compute_scaling_params(mel_normalized, mel_pred_norm)
    mel_linear = scaler.apply_scaling(mel_pred_norm)
    mel_linear_denorm = scaler.mel_normalizer.denormalize(mel_linear)
    mae_linear = np.mean(np.abs(mel_raw - mel_linear_denorm))
    print(f"Linear scaled (norm): min={mel_linear.min():.3f}, max={mel_linear.max():.3f}, mean={mel_linear.mean():.3f}, std={mel_linear.std():.3f}")
    print(f"Linear scaled (denorm): min={mel_linear_denorm.min():.3f}, max={mel_linear_denorm.max():.3f}, mean={mel_linear_denorm.mean():.3f}, std={mel_linear_denorm.std():.3f}")
    print(f"MAE vs raw target: {mae_linear:.3f}")
    
    # Compare results
    results = {
        "direct_denorm": mae_denorm,
        "adaptive": mae_adaptive, 
        "linear_regression": mae_linear
    }
    
    best_method = min(results, key=results.get)
    best_mae = results[best_method]
    
    print(f"\n🎯 Results Summary:")
    for method, mae in results.items():
        print(f"  {method}: MAE = {mae:.3f}")
    
    print(f"\n✅ Best method: {best_method} (MAE = {best_mae:.3f})")
    
    # Test synthesis with best method
    print(f"\n=== Testing synthesis with {best_method} ===")
    if best_method == "adaptive":
        final_mel = mel_adaptive_denorm
    elif best_method == "linear_regression":
        final_mel = mel_linear_denorm
    else:
        final_mel = mel_denorm
    
    # Convert to audio using Griffin-Lim
    try:
        audio_synth = inf.audio_processor.mel_to_wav(final_mel.T)  # Need [n_mels, time]
        
        # Save result
        output_path = f"test_synthesis_{best_method}.wav"
        sf.write(output_path, audio_synth, config.model.sample_rate)
        print(f"Saved synthesized audio to: {output_path}")
        
        # Audio quality metrics
        audio_power = np.mean(np.abs(audio_synth))
        audio_dynamic_range = np.max(np.abs(audio_synth)) - np.min(np.abs(audio_synth))
        
        print(f"Audio stats: power={audio_power:.4f}, dynamic_range={audio_dynamic_range:.4f}")
        
        if audio_power > 1e-3:
            print("✅ Generated audio has reasonable power level")
        else:
            print("❌ Generated audio has very low power")
            
    except Exception as e:
        print(f"❌ Error in audio synthesis: {e}")
    
    return results, scaler


if __name__ == "__main__":
    run_comprehensive_mel_fix()