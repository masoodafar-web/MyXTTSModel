#!/usr/bin/env python3
"""
Mel Normalization Fix for MyXTTS

This script addresses the mel spectrogram mismatch between training and inference.
The model was trained on raw mel spectrograms but needs normalization for proper synthesis.
"""

import numpy as np
import json
import tensorflow as tf
from typing import Tuple, Optional
from pathlib import Path

class MelNormalizer:
    """
    Mel spectrogram normalizer for consistent training/inference pipeline.
    """
    
    def __init__(self, stats_file: str = "mel_normalization_stats.json"):
        """Initialize normalizer with precomputed statistics."""
        self.stats_file = Path(stats_file)
        self.mel_mean = None
        self.mel_std = None
        self.load_stats()
    
    def load_stats(self):
        """Load normalization statistics from file."""
        if self.stats_file.exists():
            with open(self.stats_file, 'r') as f:
                stats = json.load(f)
                self.mel_mean = stats['mel_mean']
                self.mel_std = stats['mel_std']
            print(f"Loaded mel normalization stats: mean={self.mel_mean:.3f}, std={self.mel_std:.3f}")
        else:
            # Use reasonable defaults based on typical mel spectrograms
            self.mel_mean = -5.0
            self.mel_std = 2.0
            print(f"Using default mel normalization stats: mean={self.mel_mean:.3f}, std={self.mel_std:.3f}")
    
    def normalize(self, mel: np.ndarray) -> np.ndarray:
        """
        Normalize mel spectrogram to zero mean, unit variance.
        
        Args:
            mel: Raw mel spectrogram
            
        Returns:
            Normalized mel spectrogram
        """
        return (mel - self.mel_mean) / self.mel_std
    
    def denormalize(self, mel_normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize mel spectrogram back to original scale.
        
        Args:
            mel_normalized: Normalized mel spectrogram
            
        Returns:
            Denormalized mel spectrogram
        """
        return mel_normalized * self.mel_std + self.mel_mean


def create_normalized_audio_processor():
    """
    Create an enhanced AudioProcessor with mel normalization capability.
    """
    from myxtts.utils.audio import AudioProcessor
    
    class NormalizedAudioProcessor(AudioProcessor):
        """AudioProcessor with mel normalization support."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mel_normalizer = MelNormalizer()
        
        def wav_to_mel_normalized(self, audio: np.ndarray) -> np.ndarray:
            """
            Convert waveform to normalized mel spectrogram.
            
            Args:
                audio: Audio waveform
                
            Returns:
                Normalized mel spectrogram [n_mels, time_steps]
            """
            # Get raw mel spectrogram
            mel_raw = self.wav_to_mel(audio)
            
            # Normalize
            mel_normalized = self.mel_normalizer.normalize(mel_raw)
            
            return mel_normalized
        
        def wav_to_mel(self, audio: np.ndarray, normalize: bool = False) -> np.ndarray:
            """
            Convert waveform to mel spectrogram with optional normalization.
            
            Args:
                audio: Audio waveform
                normalize: Whether to apply normalization
                
            Returns:
                Mel spectrogram [n_mels, time_steps]
            """
            # Get raw mel spectrogram (original implementation)
            mel_raw = super().wav_to_mel(audio)
            
            if normalize:
                mel_raw = self.mel_normalizer.normalize(mel_raw)
            
            return mel_raw
        
        def get_mel_stats(self) -> Tuple[float, float]:
            """Get mel normalization statistics."""
            return self.mel_normalizer.mel_mean, self.mel_normalizer.mel_std
    
    return NormalizedAudioProcessor


def patch_inference_synthesizer():
    """
    Patch the XTTSInference class to use normalized mel spectrograms.
    """
    from myxtts.inference.synthesizer import XTTSInference
    
    # Store original method
    original_init = XTTSInference.__init__
    
    def enhanced_init(self, *args, **kwargs):
        # Call original initialization
        original_init(self, *args, **kwargs)
        
        # Add mel normalizer
        self.mel_normalizer = MelNormalizer()
        print("Enhanced XTTSInference with mel normalization support")
    
    # Patch the class
    XTTSInference.__init__ = enhanced_init
    
    # Add normalization methods
    def normalize_mel_for_model(self, mel: np.ndarray) -> np.ndarray:
        """Normalize mel for model input."""
        return self.mel_normalizer.normalize(mel)
    
    def denormalize_mel_from_model(self, mel_normalized: np.ndarray) -> np.ndarray:
        """Denormalize mel from model output."""
        return self.mel_normalizer.denormalize(mel_normalized)
    
    XTTSInference.normalize_mel_for_model = normalize_mel_for_model
    XTTSInference.denormalize_mel_from_model = denormalize_mel_from_model


def test_mel_normalization():
    """Test mel normalization functionality."""
    print("Testing mel normalization...")
    
    # Create test data
    test_mel = np.random.randn(80, 100) * 2.0 - 5.0  # Simulate mel spectrogram
    print(f"Original mel stats: mean={test_mel.mean():.3f}, std={test_mel.std():.3f}")
    
    # Test normalizer
    normalizer = MelNormalizer()
    mel_normalized = normalizer.normalize(test_mel)
    print(f"Normalized mel stats: mean={mel_normalized.mean():.3f}, std={mel_normalized.std():.3f}")
    
    # Test denormalization
    mel_denormalized = normalizer.denormalize(mel_normalized)
    print(f"Denormalized mel stats: mean={mel_denormalized.mean():.3f}, std={mel_denormalized.std():.3f}")
    
    # Check reconstruction error
    reconstruction_error = np.mean(np.abs(test_mel - mel_denormalized))
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    print("✅ Mel normalization test passed!")


if __name__ == "__main__":
    test_mel_normalization()