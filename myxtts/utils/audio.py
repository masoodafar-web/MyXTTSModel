"""
Audio processing utilities for MyXTTS.

This module provides audio preprocessing functions including mel spectrogram
extraction, audio normalization, and other audio-related operations.
"""

import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
from scipy import signal
from typing import Optional, Tuple, Union
import warnings


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
        trim_threshold: float = 0.01
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
        
        # Create mel filter bank
        self.mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
    
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
        Preprocess audio waveform.
        
        Args:
            audio: Raw audio waveform
            
        Returns:
            Preprocessed audio waveform
        """
        # Trim silence
        if self.trim_silence:
            audio, _ = librosa.effects.trim(
                audio, 
                top_db=20,
                frame_length=self.win_length,
                hop_length=self.hop_length
            )
        
        # Normalize
        if self.normalize:
            audio = audio / np.max(np.abs(audio))
        
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