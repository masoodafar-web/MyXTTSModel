"""Data processing module for MyXTTS."""

from .ljspeech import LJSpeechDataset
from .dataset import TTSDataset, create_dataset
from .audio_processor import AudioProcessor  
from .text_processor import TextProcessor

# TensorFlow-native data loader (GPU-optimized, eliminates CPU bottleneck)
try:
    from .tf_native_loader import TFNativeDataLoader, create_tf_native_dataset_loader
    _tf_native_available = True
except ImportError:
    # TF-native loader optional if dependencies not available
    _tf_native_available = False

__all__ = [
    "LJSpeechDataset", 
    "TTSDataset", 
    "create_dataset",
    "AudioProcessor",
    "TextProcessor"
]

if _tf_native_available:
    __all__.extend(["TFNativeDataLoader", "create_tf_native_dataset_loader"])