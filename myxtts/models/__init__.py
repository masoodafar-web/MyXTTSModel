"""Models module for MyXTTS."""

# Configure device placement early to avoid conflicts
import warnings
try:
    from ..utils.commons import configure_gpus
    # Configure GPU settings with device placement fix when module is imported
    configure_gpus()
except Exception as e:
    # Silently handle any configuration errors to avoid breaking imports
    warnings.warn(f"GPU configuration warning: {e}")

from .xtts import XTTS
from .layers import (
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    TransformerBlock
)

__all__ = [
    "XTTS",
    "MultiHeadAttention", 
    "PositionalEncoding",
    "FeedForward",
    "TransformerBlock"
]