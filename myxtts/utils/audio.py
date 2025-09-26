"""
Audio processing utilities for MyXTTS.

This module provides audio preprocessing functions including mel spectrogram
extraction, audio normalization, loudness matching, VAD processing and other 
audio-related operations for enhanced real-world robustness.
"""

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy import signal
from typing import Optional, Tuple, Union
import warnings
from urllib.request import urlretrieve
import os

# Optional torch import for Silero VAD
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class AudioProcessor:
    """
    Audio processing class with mel spectrogram extraction and audio utilities.
    Compatible with XTTS audio processing pipeline.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 1.0,
        normalize: bool = True,
        trim_silence: bool = True,
        trim_threshold: float = 0.01,
        enable_loudness_normalization: bool = True,
        target_loudness_lufs: float = -23.0,
        enable_vad: bool = True
    ):
        """
        Initialize AudioProcessor.
        
        Args:
            sample_rate: Target sample rate for audio
            n_fft: FFT window size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            n_mels: Number of mel filter banks
            fmin: Minimum frequency for mel scale
            fmax: Maximum frequency for mel scale (None = sample_rate/2)
            power: Power for mel spectrogram (1.0 for energy, 2.0 for power)
            normalize: Whether to normalize audio
            trim_silence: Whether to trim silence from audio
            trim_threshold: Threshold for silence trimming
            enable_loudness_normalization: Enable loudness-based normalization
            target_loudness_lufs: Target loudness in LUFS
            enable_vad: Enable voice activity detection using Silero VAD
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.power = power
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.trim_threshold = trim_threshold
        
        # Enhanced normalization features
        self.enable_loudness_normalization = enable_loudness_normalization
        self.target_loudness_lufs = target_loudness_lufs
        self.enable_vad = enable_vad
        
        # Initialize Silero VAD model
        self.vad_model = None
        if self.enable_vad:
            self._init_vad_model()
        
        # Initialize mel filter bank
        self.mel_filters = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=self.fmax
        )
        
        # Also initialize mel_basis for backward compatibility
        self.mel_basis = self.mel_filters
    
    def _init_vad_model(self):
        """Initialize Silero VAD model."""
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available - VAD features will be disabled")
            self.enable_vad = False
            self.vad_model = None
            return
            
        try:
            # Load Silero VAD model
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.get_speech_timestamps = utils[0]
            print("Silero VAD model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load Silero VAD model: {e}")
            print("VAD features will be disabled")
            self.enable_vad = False
            self.vad_model = None
    
    def apply_vad(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply voice activity detection to remove silence segments.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Audio with silence segments removed
        """
        if not self.enable_vad or self.vad_model is None or not TORCH_AVAILABLE:
            return audio
        
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor, 
                self.vad_model,
                sampling_rate=self.sample_rate
            )
            
            if not speech_timestamps:
                # No speech detected, return original audio
                return audio
            
            # Concatenate speech segments
            speech_segments = []
            for segment in speech_timestamps:
                start = segment['start']
                end = segment['end']
                speech_segments.append(audio[start:end])
            
            if speech_segments:
                return np.concatenate(speech_segments)
            else:
                return audio
                
        except Exception as e:
            print(f"Warning: VAD processing failed: {e}")
            return audio
    
    def normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio loudness to target LUFS.
        
        Args:
            audio: Input audio waveform
            
        Returns:
            Loudness-normalized audio
        """
        if not self.enable_loudness_normalization:
            return audio
        
        try:
            # Simple RMS-based loudness normalization as fallback
            # This is a simplified version - for production, consider using pyloudnorm
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                # Target RMS level for -23 LUFS (approximate)
                target_rms = 0.1  # Empirical value for -23 LUFS
                scaling_factor = target_rms / rms
                # Apply gentle limiting to avoid clipping
                scaling_factor = min(scaling_factor, 1.0 / np.max(np.abs(audio)))
                audio = audio * scaling_factor
            
            return audio
            
        except Exception as e:
            print(f"Warning: Loudness normalization failed: {e}")
            return audio
    
    def load_audio(
        self, 
        path: str, 
        target_sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Load audio file and resample if necessary.
        
        Args:
            path: Path to audio file
            target_sr: Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Audio waveform as numpy array
        """
        target_sr = target_sr or self.sample_rate
        
        try:
            audio, sr = sf.read(path)
        except Exception as e:
            raise ValueError(f"Error loading audio file {path}: {e}")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if necessary
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        
        return audio.astype(np.float32)
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Preprocess audio waveform with enhanced normalization.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Preprocessed audio waveform
        """
        # Apply VAD first to remove silence segments
        if self.enable_vad:
            audio = self.apply_vad(audio)
        
        # Trim silence (traditional method as backup/complement to VAD)
        if self.trim_silence:
            audio, _ = librosa.effects.trim(
                audio, 
                top_db=20,
                frame_length=self.win_length,
                hop_length=self.hop_length
            )
        
        # Apply loudness normalization
        if self.enable_loudness_normalization:
            audio = self.normalize_loudness(audio)
        
        # Traditional peak normalization (applied after loudness normalization as safety)
        if self.normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
        
        return audio
    
    def wav_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert waveform to mel spectrogram.
        
        Args:
            audio: Audio waveform
            
        Returns:
            Mel spectrogram [n_mels, time_steps]
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            pad_mode='constant'
        )
        
        # Convert to magnitude spectrogram
        magnitude = np.abs(stft) ** self.power
        
        # Apply mel filter bank
        mel_spec = np.dot(self.mel_basis, magnitude)
        
        # Convert to log scale
        mel_spec = np.log(np.maximum(mel_spec, 1e-5))
        
        return mel_spec
    
    def mel_to_wav(
        self, 
        mel: np.ndarray, 
        n_iter: int = 60
    ) -> np.ndarray:
        """
        Convert mel spectrogram to waveform using Griffin-Lim algorithm.
        
        Args:
            mel: Mel spectrogram [n_mels, time_steps]
            n_iter: Number of Griffin-Lim iterations
            
        Returns:
            Audio waveform
        """
        # Convert log mel to linear mel
        mel_linear = np.exp(mel)
        
        # Convert mel to magnitude spectrogram
        magnitude = np.dot(np.linalg.pinv(self.mel_basis), mel_linear)
        
        # Griffin-Lim algorithm
        stft = magnitude * np.exp(1j * np.random.random(magnitude.shape) * 2 * np.pi)
        
        for _ in range(n_iter):
            # ISTFT
            audio = librosa.istft(
                stft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window='hann',
                center=True
            )
            
            # STFT
            stft = librosa.stft(
                audio,
                n_fft=self.n_fft,
                hop_length=self.hop_length, 
                win_length=self.win_length,
                window='hann',
                center=True,
                pad_mode='constant'
            )
            
            # Keep magnitude, update phase
            stft = magnitude * np.exp(1j * np.angle(stft))
        
        # Final ISTFT
        audio = librosa.istft(
            stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann', 
            center=True
        )
        
        return audio
    
    def mel_to_wav_neural(
        self,
        mel: Union[np.ndarray, tf.Tensor],
        vocoder_model: Optional[tf.keras.Model] = None
    ) -> np.ndarray:
        """
        Convert mel spectrogram to waveform using neural vocoder.
        
        Args:
            mel: Mel spectrogram [n_mels, time_steps] or [time_steps, n_mels]
            vocoder_model: Trained neural vocoder model (HiFi-GAN, etc.)
            
        Returns:
            Audio waveform
        """
        if vocoder_model is None:
            # Fallback to Griffin-Lim if no neural vocoder provided
            warnings.warn("No neural vocoder provided, falling back to Griffin-Lim")
            return self.mel_to_wav(mel)
        
        # Ensure mel is TensorFlow tensor
        if isinstance(mel, np.ndarray):
            mel = tf.constant(mel, dtype=tf.float32)
        
        # Ensure proper shape [batch, time, n_mels]
        if len(mel.shape) == 2:
            if mel.shape[0] == self.n_mels:  # [n_mels, time]
                mel = tf.transpose(mel, [1, 0])  # [time, n_mels]
            mel = tf.expand_dims(mel, 0)  # [1, time, n_mels]
        
        # Generate audio using neural vocoder
        with tf.device('/CPU:0'):  # Use CPU for inference if GPU not available
            audio_tensor = vocoder_model(mel, training=False)
        
        # Convert to numpy and remove batch dimension
        audio = audio_tensor.numpy()
        if len(audio.shape) == 3:
            audio = audio[0, :, 0]  # [batch, time, 1] -> [time]
        elif len(audio.shape) == 2:
            audio = audio[0, :]  # [batch, time] -> [time]
        
        return audio
    
    def compute_f0(
        self, 
        audio: np.ndarray,
        method: str = "harvest"
    ) -> np.ndarray:
        """
        Compute fundamental frequency (F0).
        
        Args:
            audio: Audio waveform
            method: F0 extraction method ("harvest", "dio", "yin")
            
        Returns:
            F0 contour
        """
        if method == "yin":
            f0 = librosa.yin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
        else:
            # Use pYIN as alternative
            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'), 
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
            
        # Fill NaN values
        f0 = np.nan_to_num(f0)
        
        return f0
    
    def extract_features(
        self, 
        audio_path: str
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Extract complete audio features for training.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (mel_spectrogram, f0, duration)
        """
        # Load and preprocess audio
        audio = self.load_audio(audio_path)
        audio = self.preprocess_audio(audio)
        
        # Extract mel spectrogram
        mel = self.wav_to_mel(audio)
        
        # Extract F0
        f0 = self.compute_f0(audio)
        
        # Calculate duration
        duration = len(audio) / self.sample_rate
        
        return mel, f0, duration
    
    def save_audio(
        self, 
        audio: np.ndarray, 
        path: str,
        sample_rate: Optional[int] = None
    ) -> None:
        """
        Save audio waveform to file.
        
        Args:
            audio: Audio waveform
            path: Output file path
            sample_rate: Sample rate (uses self.sample_rate if None)
        """
        sample_rate = sample_rate or self.sample_rate
        sf.write(path, audio, sample_rate)
    
    @staticmethod
    def pad_or_trim(
        audio: np.ndarray, 
        target_length: int
    ) -> np.ndarray:
        """
        Pad or trim audio to target length.
        
        Args:
            audio: Audio waveform
            target_length: Target length in samples
            
        Returns:
            Padded or trimmed audio
        """
        if len(audio) > target_length:
            # Trim from center
            start = (len(audio) - target_length) // 2
            return audio[start:start + target_length]
        elif len(audio) < target_length:
            # Pad with zeros
            pad_length = target_length - len(audio)
            return np.pad(audio, (0, pad_length), mode='constant')
        else:
            return audio