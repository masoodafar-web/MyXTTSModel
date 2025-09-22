"""Model optimization module for MyXTTS.

This module provides tools for model compression, quantization, and deployment
optimization to create lighter versions of the XTTS model for real-time inference.
"""

from .compression import ModelCompressor, QuantizationAwareTrainer, CompressionConfig
from .distillation import ModelDistiller, DistillationConfig
from .deployment import OptimizedInference, InferenceConfig

__all__ = [
    "ModelCompressor",
    "QuantizationAwareTrainer",
    "CompressionConfig",
    "ModelDistiller",
    "DistillationConfig",
    "OptimizedInference",
    "InferenceConfig"
]