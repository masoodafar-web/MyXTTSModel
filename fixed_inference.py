#!/usr/bin/env python3
"""
Fixed Inference Script for MyXTTS

This script addresses the mel spectrogram normalization issue that was causing
the model to produce only noise. It includes:

1. Mel normalization based on training data statistics
2. Adaptive scaling to match target mel ranges
3. Proper post-processing for audio synthesis

Usage:
    python3 fixed_inference.py --text "Your text here" --speaker-audio speaker.wav --output output.wav
"""

import argparse
import numpy as np
import soundfile as sf
import tensorflow as tf
from pathlib import Path
import logging
import json
from typing import Optional

from train_main import build_config
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import find_latest_checkpoint
from mel_normalization_fix import MelNormalizer


class FixedXTTSSynthesizer:
    """
    Enhanced XTTS synthesizer with mel normalization fixes.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpointsmain", use_gpu: bool = False):
        """
        Initialize the fixed synthesizer.
        
        Args:
            checkpoint_dir: Directory containing model checkpoints
            use_gpu: Whether to use GPU for inference
        """
        self.logger = logging.getLogger("FixedXTTS")
        
        # Setup device
        if not use_gpu:
            tf.config.set_visible_devices([], 'GPU')
            self.logger.info("Using CPU for inference")
        
        # Load config and checkpoint
        self.checkpoint_path = find_latest_checkpoint(checkpoint_dir)
        if not self.checkpoint_path:
            raise ValueError(f"No checkpoint found in {checkpoint_dir}")
        
        self.config = build_config(model_size='normal', checkpoint_dir=checkpoint_dir)
        
        # Initialize inference engine
        self.inference = XTTSInference(config=self.config, checkpoint_path=self.checkpoint_path)
        
        # Initialize mel normalizer and scaler
        self.mel_normalizer = MelNormalizer()
        self.scaling_params = self._load_or_compute_scaling_params()
        
        self.logger.info(f"✅ FixedXTTSSynthesizer initialized with checkpoint: {self.checkpoint_path}")
    
    def _load_or_compute_scaling_params(self) -> dict:
        """Load or compute mel scaling parameters."""
        params_file = "mel_scaling_params.json"
        
        if Path(params_file).exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
            self.logger.info(f"Loaded scaling params: scale={params['scale_factor']:.3f}, offset={params['offset']:.3f}")
            return params
        else:
            self.logger.info("Computing scaling parameters from teacher forcing...")
            return self._compute_scaling_params()
    
    def _compute_scaling_params(self) -> dict:
        """Compute scaling parameters using teacher forcing on speaker audio."""
        try:
            # Use speaker.wav as reference
            audio, sr = sf.read('./speaker.wav')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Simple test text
            text = "The Chronicles of Newgate, Volume two."
            
            # Process audio and text
            audio_processed = self.inference.audio_processor.preprocess_audio(audio)
            mel_raw = self.inference.audio_processor.wav_to_mel(audio_processed).T
            mel_normalized = self.mel_normalizer.normalize(mel_raw)
            
            # Prepare tensors
            processed = self.inference._preprocess_text(text, self.config.data.language)
            seq = self.inference.text_processor.text_to_sequence(processed)
            text_tensor = tf.constant([seq], dtype=tf.int32)
            lengths = tf.constant([len(seq)], dtype=tf.int32)
            mel_tensor = tf.constant(mel_normalized[np.newaxis, ...], dtype=tf.float32)
            mel_lengths = tf.constant([mel_normalized.shape[0]], dtype=tf.int32)
            
            # Teacher forcing
            outputs = self.inference.model(
                text_inputs=text_tensor, 
                mel_inputs=mel_tensor,
                text_lengths=lengths, 
                mel_lengths=mel_lengths, 
                training=True
            )
            mel_pred = outputs['mel_output'].numpy()[0]
            
            # Compute scaling parameters
            target_flat = mel_normalized.flatten()
            pred_flat = mel_pred.flatten()
            
            # Linear regression: target = pred * scale + offset
            valid_mask = np.abs(pred_flat) > 1e-6
            if np.sum(valid_mask) > len(pred_flat) * 0.1:
                target_valid = target_flat[valid_mask]
                pred_valid = pred_flat[valid_mask]
                A = np.vstack([pred_valid, np.ones(len(pred_valid))]).T
                scale_factor, offset = np.linalg.lstsq(A, target_valid, rcond=None)[0]
            else:
                # Fallback
                scale_factor = np.std(target_flat) / (np.std(pred_flat) + 1e-8)
                offset = np.mean(target_flat) - scale_factor * np.mean(pred_flat)
            
            params = {
                'scale_factor': float(scale_factor),
                'offset': float(offset)
            }
            
            # Save parameters
            with open("mel_scaling_params.json", 'w') as f:
                json.dump(params, f, indent=2)
            
            self.logger.info(f"Computed and saved scaling params: scale={scale_factor:.3f}, offset={offset:.3f}")
            return params
            
        except Exception as e:
            self.logger.error(f"Failed to compute scaling parameters: {e}")
            # Use conservative defaults
            return {'scale_factor': 1.0, 'offset': 0.0}
    
    def _apply_mel_fixes(self, mel_pred: np.ndarray) -> np.ndarray:
        """Apply all mel spectrogram fixes."""
        # Apply learned scaling
        scale_factor = self.scaling_params['scale_factor'] 
        offset = self.scaling_params['offset']
        mel_scaled = mel_pred * scale_factor + offset
        
        # Denormalize back to raw mel space
        mel_fixed = self.mel_normalizer.denormalize(mel_scaled)
        
        return mel_fixed
    
    def synthesize(
        self, 
        text: str, 
        speaker_audio_path: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> np.ndarray:
        """
        Synthesize speech from text with proper mel fixes.
        
        Args:
            text: Text to synthesize
            speaker_audio_path: Path to speaker reference audio (optional)
            temperature: Sampling temperature  
            top_p: Top-p sampling parameter
            
        Returns:
            Synthesized audio waveform
        """
        self.logger.info(f"Synthesizing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        try:
            # If speaker audio provided, use it for conditioning
            if speaker_audio_path and Path(speaker_audio_path).exists():
                speaker_audio, sr = sf.read(speaker_audio_path)
                if speaker_audio.ndim > 1:
                    speaker_audio = speaker_audio.mean(axis=1)
                self.logger.info(f"Using speaker conditioning from: {speaker_audio_path}")
            else:
                speaker_audio = None
                self.logger.info("Using default speaker conditioning")
            
            # Process text
            processed_text = self.inference._preprocess_text(text, self.config.data.language)
            text_sequence = self.inference.text_processor.text_to_sequence(processed_text)
            
            # Create tensors
            text_tensor = tf.constant([text_sequence], dtype=tf.int32)
            text_lengths = tf.constant([len(text_sequence)], dtype=tf.int32)
            
            # Synthesize mel spectrogram (autoregressive)
            if speaker_audio is not None:
                # Use teacher forcing approach with speaker conditioning
                speaker_processed = self.inference.audio_processor.preprocess_audio(speaker_audio)
                speaker_mel = self.inference.audio_processor.wav_to_mel(speaker_processed).T
                speaker_mel_norm = self.mel_normalizer.normalize(speaker_mel)
                
                # Use first few frames as conditioning
                conditioning_frames = min(10, speaker_mel_norm.shape[0])
                mel_conditioning = speaker_mel_norm[:conditioning_frames]
                
                # Extend for synthesis length estimate
                estimated_length = len(text_sequence) * 3  # Rough estimate
                mel_input = np.zeros((estimated_length, self.config.model.n_mels), dtype=np.float32)
                mel_input[:conditioning_frames] = mel_conditioning
                
                mel_tensor = tf.constant(mel_input[np.newaxis, ...], dtype=tf.float32)
                mel_lengths = tf.constant([estimated_length], dtype=tf.int32)
            else:
                # Use zero conditioning
                estimated_length = len(text_sequence) * 3
                mel_tensor = tf.zeros((1, estimated_length, self.config.model.n_mels), dtype=tf.float32)
                mel_lengths = tf.constant([estimated_length], dtype=tf.int32)
            
            # Run model
            self.logger.info("Running model inference...")
            outputs = self.inference.model(
                text_inputs=text_tensor,
                mel_inputs=mel_tensor,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=False
            )
            
            # Extract mel prediction
            mel_pred_raw = outputs['mel_output'].numpy()[0]
            
            # Apply fixes
            self.logger.info("Applying mel spectrogram fixes...")
            mel_fixed = self._apply_mel_fixes(mel_pred_raw)
            
            # Convert to audio
            self.logger.info("Converting mel to audio...")
            audio_output = self.inference.audio_processor.mel_to_wav(mel_fixed.T)  # Need [n_mels, time]
            
            # Post-process audio
            audio_output = self.inference._postprocess_audio(audio_output)
            
            self.logger.info(f"✅ Synthesis complete. Audio length: {len(audio_output)/self.config.model.sample_rate:.2f}s")
            return audio_output
            
        except Exception as e:
            self.logger.error(f"❌ Synthesis failed: {e}")
            raise
    
    def synthesize_to_file(
        self, 
        text: str, 
        output_path: str,
        speaker_audio_path: Optional[str] = None,
        **kwargs
    ):
        """Synthesize and save to file."""
        audio = self.synthesize(text, speaker_audio_path, **kwargs)
        sf.write(output_path, audio, self.config.model.sample_rate)
        self.logger.info(f"✅ Saved audio to: {output_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Fixed MyXTTS Inference")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    parser.add_argument("--speaker-audio", help="Speaker reference audio file")
    parser.add_argument("--checkpoint-dir", default="./checkpointsmain", help="Checkpoint directory")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU for inference")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize synthesizer
        synthesizer = FixedXTTSSynthesizer(
            checkpoint_dir=args.checkpoint_dir,
            use_gpu=args.use_gpu
        )
        
        # Synthesize
        synthesizer.synthesize_to_file(
            text=args.text,
            output_path=args.output,
            speaker_audio_path=args.speaker_audio,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        print(f"🎉 Success! Generated audio saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())