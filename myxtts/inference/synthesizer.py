"""
XTTS Inference and Synthesis Module.

This module provides inference capabilities for the MyXTTS model,
including text-to-speech synthesis and voice cloning.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Union, Tuple
import librosa
import soundfile as sf

from ..models.xtts import XTTS
from ..utils.audio import AudioProcessor
from ..utils.text import TextProcessor
from ..utils.commons import load_checkpoint, setup_logging
from ..config.config import XTTSConfig


class XTTSInference:
    """
    XTTS inference engine for text-to-speech synthesis with voice cloning.
    
    Supports multilingual synthesis, voice conditioning from reference audio,
    and various generation parameters for controlling output quality.
    """
    
    def __init__(
        self,
        config: XTTSConfig,
        checkpoint_path: Optional[str] = None,
        model: Optional[XTTS] = None
    ):
        """
        Initialize XTTS inference engine.
        
        Args:
            config: Model configuration
            checkpoint_path: Path to trained model checkpoint
            model: Pre-loaded model (loads from checkpoint if None)
        """
        self.config = config
        self.logger = setup_logging()
        
        # Initialize processors
        self.audio_processor = AudioProcessor(
            sample_rate=config.model.sample_rate,
            n_fft=config.model.n_fft,
            hop_length=config.model.hop_length,
            win_length=config.model.win_length,
            n_mels=config.model.n_mels,
            normalize=config.data.normalize_audio,
            trim_silence=config.data.trim_silence
        )
        
        self.text_processor = TextProcessor(
            language=config.data.language,
            cleaner_names=config.data.text_cleaners,
            add_blank=config.data.add_blank,
            use_phonemes=True
        )
        
        # Initialize model
        if model is not None:
            self.model = model
        else:
            self.model = XTTS(config.model)
            
            if checkpoint_path:
                self._load_checkpoint(checkpoint_path)
            else:
                self.logger.warning("No checkpoint provided. Using untrained model.")
        
        # Set model to evaluation mode
        self.model.trainable = False
        
        self.logger.info("XTTS inference engine initialized")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        try:
            # Create dummy optimizer for loading (not used in inference)
            dummy_optimizer = tf.keras.optimizers.Adam()
            
            # Load checkpoint
            metadata = load_checkpoint(self.model, dummy_optimizer, checkpoint_path)
            
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            self.logger.info(f"Checkpoint step: {metadata.get('step', 'unknown')}")
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        reference_audio: Optional[Union[str, np.ndarray]] = None,
        language: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        Synthesize speech from text.
        
        Args:
            text: Input text to synthesize
            reference_audio: Reference audio for voice cloning (path or array)
            language: Language code (uses config default if None)
            max_length: Maximum generation length
            temperature: Sampling temperature for generation
            repetition_penalty: Penalty for repetition
            length_penalty: Penalty for length
            
        Returns:
            Dictionary containing:
                - audio: Generated audio waveform
                - mel_spectrogram: Generated mel spectrogram
                - text_processed: Processed input text
        """
        self.logger.info(f"Synthesizing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Process text
        language = language or self.config.data.language
        text_processed = self._preprocess_text(text, language)
        text_sequence = self.text_processor.text_to_sequence(text_processed)
        
        # Convert to tensor
        text_tensor = tf.constant([text_sequence], dtype=tf.int32)
        
        # Process reference audio for voice conditioning
        audio_conditioning = None
        if reference_audio is not None and self.config.model.use_voice_conditioning:
            audio_conditioning = self._preprocess_reference_audio(reference_audio)
        
        # Generate mel spectrogram
        outputs = self.model.generate(
            text_inputs=text_tensor,
            audio_conditioning=audio_conditioning,
            max_length=max_length,
            temperature=temperature
        )
        
        # Extract generated mel spectrogram
        mel_spectrogram = outputs["mel_output"].numpy()[0]  # Remove batch dimension
        
        # Convert mel to waveform
        audio_waveform = self.audio_processor.mel_to_wav(mel_spectrogram.T)
        
        # Post-process audio
        audio_waveform = self._postprocess_audio(audio_waveform)
        
        self.logger.info(f"Generated audio: {len(audio_waveform)} samples, "
                        f"{len(audio_waveform) / self.config.model.sample_rate:.2f}s")
        
        return {
            "audio": audio_waveform,
            "mel_spectrogram": mel_spectrogram,
            "text_processed": text_processed,
            "sample_rate": self.config.model.sample_rate
        }
    
    def synthesize_batch(
        self,
        texts: List[str],
        reference_audios: Optional[List[Union[str, np.ndarray]]] = None,
        language: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 1.0
    ) -> List[Dict[str, np.ndarray]]:
        """
        Synthesize speech from multiple texts.
        
        Args:
            texts: List of input texts
            reference_audios: List of reference audios (or single for all)
            language: Language code
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            List of synthesis results
        """
        results = []
        
        for i, text in enumerate(texts):
            # Get reference audio for this text
            ref_audio = None
            if reference_audios:
                if len(reference_audios) == 1:
                    ref_audio = reference_audios[0]
                else:
                    ref_audio = reference_audios[i] if i < len(reference_audios) else None
            
            # Synthesize
            result = self.synthesize(
                text=text,
                reference_audio=ref_audio,
                language=language,
                max_length=max_length,
                temperature=temperature
            )
            
            results.append(result)
        
        return results
    
    def clone_voice(
        self,
        text: str,
        reference_audio: Union[str, np.ndarray],
        language: Optional[str] = None,
        max_length: int = 1000,
        temperature: float = 0.7
    ) -> Dict[str, np.ndarray]:
        """
        Clone voice from reference audio.
        
        Args:
            text: Text to synthesize
            reference_audio: Reference audio for voice cloning
            language: Language code
            max_length: Maximum generation length
            temperature: Sampling temperature (lower for more consistent voice)
            
        Returns:
            Synthesis result with cloned voice
        """
        if not self.config.model.use_voice_conditioning:
            raise ValueError("Voice conditioning not enabled in model configuration")
        
        return self.synthesize(
            text=text,
            reference_audio=reference_audio,
            language=language,
            max_length=max_length,
            temperature=temperature
        )
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text for synthesis."""
        # Update text processor language if different
        if language != self.text_processor.language:
            self.text_processor.language = language
            # Reinitialize phonemizer if needed
            if self.text_processor.use_phonemes:
                try:
                    from phonemizer.backend import EspeakBackend
                    self.text_processor.phonemizer = EspeakBackend(
                        language=language,
                        preserve_punctuation=True,
                        with_stress=True
                    )
                except Exception as e:
                    self.logger.warning(f"Could not initialize phonemizer for {language}: {e}")
                    self.text_processor.use_phonemes = False
        
        # Clean and process text
        text_processed = self.text_processor.clean_text(text)
        
        # Limit text length
        if len(text_processed) > self.config.model.max_text_length:
            self.logger.warning(f"Text too long ({len(text_processed)} chars), truncating")
            text_processed = text_processed[:self.config.model.max_text_length]
        
        return text_processed
    
    def _preprocess_reference_audio(
        self,
        reference_audio: Union[str, np.ndarray]
    ) -> tf.Tensor:
        """Preprocess reference audio for voice conditioning."""
        # Load audio if path provided
        if isinstance(reference_audio, str):
            audio = self.audio_processor.load_audio(reference_audio)
        else:
            audio = reference_audio.astype(np.float32)
        
        # Preprocess audio
        audio = self.audio_processor.preprocess_audio(audio)
        
        # Ensure minimum length for conditioning
        min_samples = int(self.config.data.reference_audio_length * self.config.model.sample_rate)
        if len(audio) < min_samples:
            # Pad with zeros or repeat
            if len(audio) < min_samples // 2:
                # Repeat if too short
                repeats = (min_samples // len(audio)) + 1
                audio = np.tile(audio, repeats)
            
            # Pad to exact length
            audio = self.audio_processor.pad_or_trim(audio, min_samples)
        
        # Limit maximum length
        max_samples = int(self.config.data.max_audio_length * self.config.model.sample_rate)
        if len(audio) > max_samples:
            audio = audio[:max_samples]
        
        # Extract mel spectrogram
        mel_spec = self.audio_processor.wav_to_mel(audio)
        
        # Convert to tensor and add batch dimension
        mel_tensor = tf.constant(mel_spec.T[np.newaxis, :, :], dtype=tf.float32)
        
        return mel_tensor
    
    def _postprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Post-process generated audio."""
        # Trim silence
        if self.config.data.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Normalize
        if self.config.data.normalize_audio:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        # Ensure proper data type
        audio = audio.astype(np.float32)
        
        return audio
    
    def save_audio(
        self,
        audio: np.ndarray,
        output_path: str,
        sample_rate: Optional[int] = None
    ):
        """
        Save audio to file.
        
        Args:
            audio: Audio waveform
            output_path: Output file path
            sample_rate: Sample rate (uses model default if None)
        """
        sample_rate = sample_rate or self.config.model.sample_rate
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save audio
        sf.write(output_path, audio, sample_rate)
        
        self.logger.info(f"Saved audio to: {output_path}")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.config.model.languages
    
    def get_model_info(self) -> Dict[str, any]:
        """Get model information."""
        return {
            "model_name": "MyXTTS",
            "version": "1.0.0",
            "sample_rate": self.config.model.sample_rate,
            "n_mels": self.config.model.n_mels,
            "languages": self.config.model.languages,
            "voice_conditioning": self.config.model.use_voice_conditioning,
            "vocab_size": self.text_processor.get_vocab_size(),
            "max_text_length": self.config.model.max_text_length
        }
    
    def benchmark(
        self,
        texts: List[str],
        reference_audio: Optional[Union[str, np.ndarray]] = None,
        num_runs: int = 5
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            texts: Test texts
            reference_audio: Reference audio for conditioning
            num_runs: Number of runs for averaging
            
        Returns:
            Benchmark results
        """
        import time
        
        self.logger.info(f"Benchmarking with {len(texts)} texts, {num_runs} runs each")
        
        times = []
        audio_lengths = []
        
        for run in range(num_runs):
            for text in texts:
                start_time = time.time()
                
                result = self.synthesize(
                    text=text,
                    reference_audio=reference_audio,
                    temperature=0.7
                )
                
                end_time = time.time()
                
                synthesis_time = end_time - start_time
                audio_length = len(result["audio"]) / result["sample_rate"]
                
                times.append(synthesis_time)
                audio_lengths.append(audio_length)
        
        # Calculate statistics
        avg_synthesis_time = np.mean(times)
        avg_audio_length = np.mean(audio_lengths)
        rtf = avg_synthesis_time / avg_audio_length  # Real-time factor
        
        results = {
            "average_synthesis_time": avg_synthesis_time,
            "average_audio_length": avg_audio_length,
            "real_time_factor": rtf,
            "total_runs": len(times),
            "throughput_samples_per_second": 1.0 / avg_synthesis_time if avg_synthesis_time > 0 else 0
        }
        
        self.logger.info(f"Benchmark results: RTF={rtf:.2f}, "
                        f"Avg time={avg_synthesis_time:.2f}s")
        
        return results


def create_inference_engine(
    config_path: str,
    checkpoint_path: str
) -> XTTSInference:
    """
    Create XTTS inference engine from configuration and checkpoint.
    
    Args:
        config_path: Path to configuration file
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Configured inference engine
    """
    # Load configuration
    config = XTTSConfig.from_yaml(config_path)
    
    # Create inference engine
    inference_engine = XTTSInference(
        config=config,
        checkpoint_path=checkpoint_path
    )
    
    return inference_engine