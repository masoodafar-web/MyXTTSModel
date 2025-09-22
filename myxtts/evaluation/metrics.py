"""Individual evaluation metrics for TTS quality assessment."""

import os
import logging
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

try:
    import tensorflow as tf
    import transformers
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    metric_name: str
    score: float
    details: Optional[Dict] = None
    error: Optional[str] = None


class BaseEvaluator(ABC):
    """Base class for all TTS evaluators."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def evaluate(self, audio_path: str, reference_text: str = None, **kwargs) -> EvaluationResult:
        """Evaluate TTS output.
        
        Args:
            audio_path: Path to generated audio
            reference_text: Original text (if available)
            **kwargs: Additional evaluation parameters
            
        Returns:
            EvaluationResult with score and details
        """
        pass


class MOSNetEvaluator(BaseEvaluator):
    """MOSNet-based perceptual quality evaluator.
    
    Implements a simplified version of MOSNet for perceptual quality assessment.
    Uses spectral features to predict Mean Opinion Score (MOS).
    """
    
    def __init__(self):
        super().__init__("MOSNet")
        
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features for MOS prediction."""
        # Compute spectral features
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        
        # Spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=sr))
        
        # Spectral bandwidth
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=sr))
        
        # Spectral rolloff
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=sr))
        
        # Zero crossing rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(S=magnitude, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        
        return {
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff,
            'zcr': zcr,
            'mfcc_mean': mfcc_mean,
            'mfcc_std': mfcc_std,
            'chroma_mean': chroma_mean
        }
    
    def _predict_mos(self, features: Dict) -> float:
        """Predict MOS score from spectral features using heuristic model."""
        # Simplified heuristic model for MOS prediction
        # In a real implementation, this would use a trained neural network
        
        # Normalize features to reasonable ranges
        centroid_norm = min(max(features['spectral_centroid'] / 2000, 0), 1)
        bandwidth_norm = min(max(features['spectral_bandwidth'] / 1000, 0), 1)
        zcr_norm = min(max(features['zcr'] * 10, 0), 1)
        
        # Compute quality indicators
        spectral_quality = (centroid_norm + bandwidth_norm) / 2
        temporal_quality = 1 - zcr_norm  # Lower ZCR often indicates better quality
        
        # MFCC-based quality (variance in MFCC indicates naturalness)
        mfcc_variance = np.mean(features['mfcc_std'])
        mfcc_quality = min(max(mfcc_variance / 50, 0), 1)
        
        # Combine into MOS score (1-5 scale)
        base_score = 2.5  # Neutral score
        quality_bonus = (spectral_quality + temporal_quality + mfcc_quality) / 3 * 2.5
        
        mos_score = base_score + quality_bonus
        return min(max(mos_score, 1.0), 5.0)
    
    def evaluate(self, audio_path: str, reference_text: str = None, **kwargs) -> EvaluationResult:
        """Evaluate TTS output using MOSNet-like approach."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            if len(audio) == 0:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.0,
                    error="Empty audio file"
                )
            
            # Extract features
            features = self._extract_spectral_features(audio, sr)
            
            # Predict MOS
            mos_score = self._predict_mos(features)
            
            details = {
                'features': features,
                'audio_duration': len(audio) / sr,
                'sample_rate': sr
            }
            
            return EvaluationResult(
                metric_name=self.name,
                score=mos_score,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"MOSNet evaluation failed: {e}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                error=str(e)
            )


class ASRWordErrorRateEvaluator(BaseEvaluator):
    """ASR-based Word Error Rate evaluator using Whisper."""
    
    def __init__(self, model_name: str = "openai/whisper-base"):
        super().__init__("ASR-WER")
        self.model_name = model_name
        self._processor = None
        self._model = None
        
    def _load_model(self):
        """Lazy load Whisper model."""
        if not TF_AVAILABLE:
            raise ImportError("Transformers and TensorFlow required for ASR evaluation")
            
        if self._processor is None:
            self.logger.info(f"Loading Whisper model: {self.model_name}")
            self._processor = WhisperProcessor.from_pretrained(self.model_name)
            self._model = WhisperForConditionalGeneration.from_pretrained(self.model_name)
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper."""
        self._load_model()
        
        # Load and preprocess audio
        audio, sr = librosa.load(audio_path, sr=16000)  # Whisper expects 16kHz
        
        # Process audio
        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt")
        
        # Generate transcription
        with tf.device('/CPU:0'):  # Use CPU for Whisper to avoid GPU memory issues
            generated_ids = self._model.generate(inputs["input_features"])
        
        transcription = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    
    def _calculate_wer(self, reference: str, hypothesis: str) -> Tuple[float, Dict]:
        """Calculate Word Error Rate between reference and hypothesis."""
        # Normalize texts
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        # Dynamic programming for edit distance
        dp = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        
        # Initialize
        for i in range(len(ref_words) + 1):
            dp[i][0] = i
        for j in range(len(hyp_words) + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        edit_distance = dp[len(ref_words)][len(hyp_words)]
        wer = edit_distance / max(len(ref_words), 1)
        
        details = {
            'reference_words': len(ref_words),
            'hypothesis_words': len(hyp_words),
            'edit_distance': edit_distance,
            'reference_text': reference,
            'hypothesis_text': hypothesis
        }
        
        return wer, details
    
    def evaluate(self, audio_path: str, reference_text: str = None, **kwargs) -> EvaluationResult:
        """Evaluate TTS output using ASR and WER."""
        if reference_text is None:
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,  # Maximum error
                error="Reference text required for WER calculation"
            )
        
        try:
            # Transcribe audio
            transcription = self._transcribe_audio(audio_path)
            
            # Calculate WER
            wer, details = self._calculate_wer(reference_text, transcription)
            
            return EvaluationResult(
                metric_name=self.name,
                score=wer,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"ASR-WER evaluation failed: {e}")
            return EvaluationResult(
                metric_name=self.name,
                score=1.0,  # Maximum error
                error=str(e)
            )


class CMVNEvaluator(BaseEvaluator):
    """Cepstral Mean and Variance Normalization quality evaluator."""
    
    def __init__(self):
        super().__init__("CMVN")
    
    def _extract_cepstral_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract cepstral features for CMVN analysis."""
        # Compute MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13, n_fft=1024, hop_length=256)
        
        # Compute delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        return {
            'mfcc': mfccs,
            'delta_mfcc': delta_mfccs,
            'delta2_mfcc': delta2_mfccs
        }
    
    def _compute_cmvn_score(self, features: Dict[str, np.ndarray]) -> Tuple[float, Dict]:
        """Compute CMVN-based quality score."""
        # Analyze cepstral mean and variance
        mfcc = features['mfcc']
        
        # Compute statistics
        mean_values = np.mean(mfcc, axis=1)
        var_values = np.var(mfcc, axis=1)
        
        # Quality indicators:
        # 1. Mean should be reasonably distributed
        # 2. Variance should indicate good dynamic range
        # 3. No extreme outliers
        
        # Mean distribution score (penalize extreme means)
        mean_score = 1.0 / (1.0 + np.mean(np.abs(mean_values)))
        
        # Variance score (reward moderate variance)
        var_score = np.mean(np.clip(var_values / 100, 0, 1))
        
        # Stability score (low coefficient of variation is good)
        cv_score = 1.0 / (1.0 + np.mean(var_values / (np.abs(mean_values) + 1e-8)))
        
        # Dynamic range score
        dynamic_range = np.max(mfcc) - np.min(mfcc)
        range_score = min(dynamic_range / 100, 1.0)
        
        # Combine scores
        overall_score = (mean_score + var_score + cv_score + range_score) / 4
        
        details = {
            'mean_values': mean_values.tolist(),
            'var_values': var_values.tolist(),
            'mean_score': mean_score,
            'var_score': var_score,
            'cv_score': cv_score,
            'range_score': range_score,
            'dynamic_range': dynamic_range
        }
        
        return overall_score, details
    
    def evaluate(self, audio_path: str, reference_text: str = None, **kwargs) -> EvaluationResult:
        """Evaluate TTS output using CMVN analysis."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            if len(audio) == 0:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.0,
                    error="Empty audio file"
                )
            
            # Extract cepstral features
            features = self._extract_cepstral_features(audio, sr)
            
            # Compute CMVN score
            score, details = self._compute_cmvn_score(features)
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"CMVN evaluation failed: {e}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                error=str(e)
            )


class SpectralQualityEvaluator(BaseEvaluator):
    """Spectral quality evaluator for TTS output."""
    
    def __init__(self):
        super().__init__("SpectralQuality")
    
    def _analyze_spectral_quality(self, audio: np.ndarray, sr: int) -> Tuple[float, Dict]:
        """Analyze spectral quality of audio."""
        # Compute spectrogram
        stft = librosa.stft(audio, n_fft=1024, hop_length=256)
        magnitude = np.abs(stft)
        
        # Spectral measures
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)
        
        # Quality indicators
        # 1. Spectral consistency (low variation is good)
        centroid_consistency = 1.0 / (1.0 + np.std(spectral_centroid))
        bandwidth_consistency = 1.0 / (1.0 + np.std(spectral_bandwidth))
        
        # 2. Spectral richness (moderate contrast is good)
        contrast_score = np.mean(np.clip(spectral_contrast / 30, 0, 1))
        
        # 3. Spectral balance (not too flat, not too peaked)
        flatness_score = 1.0 - np.mean(spectral_flatness)  # Less flat is better for speech
        
        # 4. Frequency distribution
        freq_bins = magnitude.shape[0]
        energy_distribution = np.sum(magnitude, axis=1)
        energy_distribution = energy_distribution / np.sum(energy_distribution)
        
        # Penalize too much energy in extreme frequencies
        low_freq_energy = np.sum(energy_distribution[:freq_bins//10])
        high_freq_energy = np.sum(energy_distribution[freq_bins*8//10:])
        mid_freq_energy = np.sum(energy_distribution[freq_bins//10:freq_bins*8//10])
        
        freq_balance_score = mid_freq_energy * (1.0 - low_freq_energy) * (1.0 - high_freq_energy)
        
        # Combine scores
        overall_score = (
            centroid_consistency + bandwidth_consistency + 
            contrast_score + flatness_score + freq_balance_score
        ) / 5
        
        details = {
            'centroid_consistency': centroid_consistency,
            'bandwidth_consistency': bandwidth_consistency,
            'contrast_score': contrast_score,
            'flatness_score': flatness_score,
            'freq_balance_score': freq_balance_score,
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'low_freq_energy': low_freq_energy,
            'mid_freq_energy': mid_freq_energy,
            'high_freq_energy': high_freq_energy
        }
        
        return overall_score, details
    
    def evaluate(self, audio_path: str, reference_text: str = None, **kwargs) -> EvaluationResult:
        """Evaluate TTS output using spectral analysis."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=22050)
            
            if len(audio) == 0:
                return EvaluationResult(
                    metric_name=self.name,
                    score=0.0,
                    error="Empty audio file"
                )
            
            # Analyze spectral quality
            score, details = self._analyze_spectral_quality(audio, sr)
            
            return EvaluationResult(
                metric_name=self.name,
                score=score,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Spectral quality evaluation failed: {e}")
            return EvaluationResult(
                metric_name=self.name,
                score=0.0,
                error=str(e)
            )