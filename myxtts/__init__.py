"""
MyXTTS: A TensorFlow-based XTTS model implementation
===================================================

A comprehensive text-to-speech system with voice cloning capabilities,
built using TensorFlow and supporting the LJSpeech dataset format.

Features:
- Multilingual text-to-speech synthesis
- Voice cloning from audio samples
- LJSpeech dataset support
- TensorFlow-based architecture
- Configurable training pipeline
- Real-time inference capabilities
- API-based configuration loading
"""

__version__ = "0.1.0"
__author__ = "MyXTTS Development Team"
__email__ = "contact@myxtts.com"

# Import configuration first (no heavy dependencies)
from myxtts.config.config import XTTSConfig

# Import main classes directly for easier access
from myxtts.models.xtts import XTTS
from myxtts.inference.synthesizer import XTTSInference  
from myxtts.training.trainer import XTTSTrainer

# Lazy imports for heavy dependencies
def get_xtts_model():
    """Lazy import for XTTS model."""
    from myxtts.models import XTTS
    return XTTS

def get_inference_engine():
    """Lazy import for inference engine."""
    from myxtts.inference import XTTSInference
    return XTTSInference

def get_trainer():
    """Lazy import for trainer."""
    from myxtts.training import XTTSTrainer
    return XTTSTrainer

def get_ljspeech_dataset():
    """Lazy import for LJSpeech dataset."""
    from myxtts.data import LJSpeechDataset
    return LJSpeechDataset

# Core exports
__all__ = [
    "XTTSConfig",
    "XTTS",
    "XTTSInference", 
    "XTTSTrainer",
    "get_xtts_model",
    "get_inference_engine", 
    "get_trainer",
    "get_ljspeech_dataset",
]