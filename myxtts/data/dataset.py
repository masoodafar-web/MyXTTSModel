"""
Generic TTS Dataset utilities for MyXTTS.
"""

import tensorflow as tf
from typing import Optional, Dict, Any
from .ljspeech import LJSpeechDataset
from ..config.config import DataConfig


class TTSDataset:
    """Generic TTS dataset wrapper."""
    
    def __init__(self, dataset_type: str = "ljspeech", **kwargs):
        """
        Initialize TTS dataset.
        
        Args:
            dataset_type: Type of dataset ("ljspeech", etc.)
            **kwargs: Dataset-specific arguments
        """
        self.dataset_type = dataset_type
        
        if dataset_type == "ljspeech":
            self.dataset = LJSpeechDataset(**kwargs)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    def __getattr__(self, name):
        """Delegate attribute access to underlying dataset."""
        return getattr(self.dataset, name)


def create_dataset(
    dataset_type: str,
    data_path: str,
    config: DataConfig,
    subset: str = "train",
    **kwargs
) -> TTSDataset:
    """
    Create TTS dataset.
    
    Args:
        dataset_type: Type of dataset
        data_path: Path to dataset
        config: Data configuration
        subset: Dataset subset
        **kwargs: Additional arguments
        
    Returns:
        TTS dataset instance
    """
    return TTSDataset(
        dataset_type=dataset_type,
        data_path=data_path,
        config=config,
        subset=subset,
        **kwargs
    )