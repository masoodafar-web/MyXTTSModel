"""Automatic evaluation module for MyXTTS TTS quality assessment.

This module provides comprehensive evaluation capabilities for TTS outputs
including perceptual quality metrics, ASR-based accuracy, and spectral analysis.
"""

from .metrics import (
    MOSNetEvaluator,
    ASRWordErrorRateEvaluator, 
    CMVNEvaluator,
    SpectralQualityEvaluator
)
from .evaluator import TTSEvaluator

__all__ = [
    "MOSNetEvaluator",
    "ASRWordErrorRateEvaluator", 
    "CMVNEvaluator",
    "SpectralQualityEvaluator",
    "TTSEvaluator"
]