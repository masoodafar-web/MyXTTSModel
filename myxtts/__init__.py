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
"""

__version__ = "0.1.0"
__author__ = "MyXTTS Development Team"
__email__ = "contact@myxtts.com"

from myxtts.models import XTTS
from myxtts.inference import XTTSInference
from myxtts.training import XTTSTrainer
from myxtts.data import LJSpeechDataset, AudioProcessor, TextProcessor

__all__ = [
    "XTTS",
    "XTTSInference", 
    "XTTSTrainer",
    "LJSpeechDataset",
    "AudioProcessor",
    "TextProcessor",
]