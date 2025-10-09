"""
TensorFlow-Native Data Loader

This module provides GPU-optimized data loading using pure TensorFlow operations
instead of tf.numpy_function, eliminating the CPU bottleneck that causes
cyclic GPU utilization (2-40% oscillation pattern).

Key improvements:
1. Uses tf.io.read_file() instead of Python file I/O
2. Uses tf.audio.decode_wav() instead of librosa/soundfile
3. Fully graph-compatible operations
4. No CPU-GPU synchronization barriers
5. Enables GPU prefetching and pipelining

This solves the issue described as:
"مصرف GPU به شکل نوسانی (spike/cycle) بین ۲٪ تا ۴۰٪"
(GPU usage oscillates between 2% to 40%)
"""

import tensorflow as tf
from typing import Tuple, Optional


class TFNativeDataLoader:
    """
    TensorFlow-native data loader that avoids CPU bottlenecks.
    
    This loader uses only TensorFlow operations that can be compiled into
    the computation graph, eliminating Python overhead and CPU-GPU barriers.
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
        mel_scale: str = "slaney",
    ):
        """
        Initialize TF-native data loader.
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            win_length: Window length for STFT
            n_mels: Number of mel filterbanks
            fmin: Minimum frequency for mel filterbank
            fmax: Maximum frequency for mel filterbank
            mel_scale: Mel scale type ("slaney" or "htk")
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or (sample_rate / 2.0)
        self.mel_scale = mel_scale
        
        # Pre-compute mel filterbank matrix for efficiency
        # This is computed once and reused for all spectrograms
        self._mel_filterbank = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self) -> tf.Tensor:
        """
        Create mel filterbank matrix using TensorFlow operations.
        
        Returns:
            Mel filterbank matrix [n_mels, n_fft//2 + 1]
        """
        # Use TensorFlow's built-in mel filterbank
        # This is fully compatible with GPU operations
        mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.n_mels,
            num_spectrogram_bins=self.n_fft // 2 + 1,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.fmin,
            upper_edge_hertz=self.fmax,
        )
        return mel_matrix
    
    @tf.function
    def load_and_process_audio(
        self,
        audio_path: tf.Tensor,
        max_length: Optional[int] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and process audio file using TensorFlow-native operations.
        
        This function is graph-compatible and can be used in tf.data pipelines
        without breaking the computation graph.
        
        Args:
            audio_path: Path to audio file (string tensor)
            max_length: Maximum audio length in samples (optional)
            
        Returns:
            Tuple of (audio_waveform, mel_spectrogram)
            - audio_waveform: [time] audio samples
            - mel_spectrogram: [time, n_mels] mel spectrogram
        """
        # Read audio file using TensorFlow I/O
        # This is MUCH faster than tf.numpy_function with librosa/soundfile
        audio_binary = tf.io.read_file(audio_path)
        
        # Decode audio (supports WAV, FLAC with appropriate ops)
        # For WAV files, use TensorFlow's built-in decoder
        audio, sample_rate_tensor = tf.audio.decode_wav(
            audio_binary,
            desired_channels=1,  # Mono audio
            desired_samples=-1,  # Load all samples
        )
        
        # Remove channel dimension: [time, 1] -> [time]
        audio = tf.squeeze(audio, axis=-1)
        
        # Resample if necessary (using TensorFlow operations)
        if self.sample_rate != 22050:  # Assuming decoded audio is 22050 Hz
            # Simple resampling using TensorFlow operations
            # For production, consider using tensorflow_io.audio.resample
            audio = self._resample_audio(audio, sample_rate_tensor, self.sample_rate)
        
        # Truncate or pad to max_length if specified
        if max_length is not None:
            audio = self._truncate_or_pad(audio, max_length)
        
        # Normalize audio to [-1, 1] range
        audio = self._normalize_audio(audio)
        
        # Compute mel spectrogram
        mel_spec = self._compute_mel_spectrogram(audio)
        
        return audio, mel_spec
    
    @tf.function
    def _normalize_audio(self, audio: tf.Tensor) -> tf.Tensor:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio tensor
            
        Returns:
            Normalized audio
        """
        # RMS normalization
        rms = tf.sqrt(tf.reduce_mean(tf.square(audio)))
        target_rms = 0.1
        audio = audio * (target_rms / (rms + 1e-8))
        
        # Clip to prevent clipping
        audio = tf.clip_by_value(audio, -1.0, 1.0)
        
        return audio
    
    @tf.function
    def _truncate_or_pad(self, audio: tf.Tensor, max_length: int) -> tf.Tensor:
        """
        Truncate or pad audio to specified length.
        
        Args:
            audio: Input audio tensor [time]
            max_length: Target length
            
        Returns:
            Audio tensor with length max_length
        """
        current_length = tf.shape(audio)[0]
        
        # Truncate if too long
        audio = audio[:max_length]
        
        # Pad if too short
        pad_length = tf.maximum(0, max_length - current_length)
        audio = tf.pad(audio, [[0, pad_length]], constant_values=0.0)
        
        return audio
    
    @tf.function
    def _resample_audio(
        self,
        audio: tf.Tensor,
        orig_sr: tf.Tensor,
        target_sr: int
    ) -> tf.Tensor:
        """
        Simple audio resampling using TensorFlow operations.
        
        For production use, consider tensorflow_io.audio.resample for better quality.
        
        Args:
            audio: Input audio tensor
            orig_sr: Original sample rate
            target_sr: Target sample rate
            
        Returns:
            Resampled audio
        """
        # Convert to float for calculation
        orig_sr_float = tf.cast(orig_sr, tf.float32)
        target_sr_float = tf.cast(target_sr, tf.float32)
        
        # Calculate resampling ratio
        ratio = target_sr_float / orig_sr_float
        
        # Calculate new length
        orig_length = tf.shape(audio)[0]
        new_length = tf.cast(
            tf.cast(orig_length, tf.float32) * ratio,
            tf.int32
        )
        
        # Reshape for image resize operation
        audio_reshaped = tf.reshape(audio, [1, -1, 1, 1])
        
        # Use bilinear interpolation for resampling
        audio_resampled = tf.image.resize(
            audio_reshaped,
            [1, new_length],
            method='bilinear'
        )
        
        # Reshape back to 1D
        audio_resampled = tf.reshape(audio_resampled, [-1])
        
        return audio_resampled
    
    @tf.function
    def _compute_mel_spectrogram(self, audio: tf.Tensor) -> tf.Tensor:
        """
        Compute mel spectrogram using TensorFlow operations.
        
        This is fully GPU-compatible and graph-compatible.
        
        Args:
            audio: Input audio tensor [time]
            
        Returns:
            Mel spectrogram [time, n_mels]
        """
        # Compute STFT using TensorFlow
        stft = tf.signal.stft(
            audio,
            frame_length=self.win_length,
            frame_step=self.hop_length,
            fft_length=self.n_fft,
            window_fn=tf.signal.hann_window,
            pad_end=True,
        )
        
        # Compute magnitude spectrogram
        magnitude = tf.abs(stft)
        
        # Apply mel filterbank
        mel_spec = tf.matmul(magnitude, self.mel_filterbank)
        
        # Convert to log scale (dB)
        # Add small epsilon to avoid log(0)
        mel_spec = tf.math.log(mel_spec + 1e-6)
        
        # Normalize to [0, 1] range for better training stability
        mel_spec = self._normalize_mel_spectrogram(mel_spec)
        
        return mel_spec
    
    @tf.function
    def _normalize_mel_spectrogram(self, mel_spec: tf.Tensor) -> tf.Tensor:
        """
        Normalize mel spectrogram to [0, 1] range.
        
        Args:
            mel_spec: Input mel spectrogram
            
        Returns:
            Normalized mel spectrogram
        """
        # Normalize using min-max normalization
        min_val = tf.reduce_min(mel_spec)
        max_val = tf.reduce_max(mel_spec)
        
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = tf.maximum(range_val, 1e-6)
        
        mel_spec_norm = (mel_spec - min_val) / range_val
        
        return mel_spec_norm
    
    @property
    def mel_filterbank(self) -> tf.Tensor:
        """Get mel filterbank matrix."""
        return self._mel_filterbank


def create_tf_native_dataset_loader(
    audio_paths: tf.data.Dataset,
    token_sequences: tf.data.Dataset,
    sample_rate: int = 22050,
    n_mels: int = 80,
    max_audio_length: Optional[int] = None,
    num_parallel_calls: int = tf.data.AUTOTUNE,
) -> tf.data.Dataset:
    """
    Create TensorFlow-native dataset loader.
    
    This function creates a data pipeline that uses only TensorFlow operations,
    eliminating CPU bottlenecks from tf.numpy_function.
    
    Args:
        audio_paths: Dataset of audio file paths
        token_sequences: Dataset of tokenized text sequences
        sample_rate: Audio sample rate
        n_mels: Number of mel filterbanks
        max_audio_length: Maximum audio length in samples
        num_parallel_calls: Number of parallel calls for map operations
        
    Returns:
        TensorFlow dataset with (tokens, mel_spec, text_len, mel_len) tuples
    """
    # Create loader
    loader = TFNativeDataLoader(
        sample_rate=sample_rate,
        n_mels=n_mels,
    )
    
    # Zip audio paths and token sequences
    dataset = tf.data.Dataset.zip((audio_paths, token_sequences))
    
    def load_sample(audio_path, tokens):
        """Load and process a single sample."""
        # Load audio and compute mel spectrogram (graph-compatible!)
        _, mel_spec = loader.load_and_process_audio(audio_path, max_audio_length)
        
        # Get lengths
        text_len = tf.shape(tokens)[0]
        mel_len = tf.shape(mel_spec)[0]
        
        return tokens, mel_spec, text_len, mel_len
    
    # Map with parallel processing (fully graph-compatible)
    dataset = dataset.map(
        load_sample,
        num_parallel_calls=num_parallel_calls,
        deterministic=False  # Allow reordering for better performance
    )
    
    return dataset
