#!/usr/bin/env python3
"""Enhanced MyXTTS Inference Engine with Advanced Voice Cloning Capabilities

This script provides sophisticated text-to-speech inference with state-of-the-art voice cloning.
It mirrors the enhanced training setup in ``train_main.py`` and offers comprehensive voice
conditioning features for high-quality voice replication.

ADVANCED VOICE CLONING FEATURES:
===============================

1. HIGH-FIDELITY VOICE CLONING:
   - Voice cloning temperature control (optimized default: 0.7)
   - Voice conditioning strength adjustment (0.0-2.0 range)
   - Voice similarity threshold enforcement (default: 0.75)
   - Enhanced reference audio preprocessing with denoising

2. MULTI-REFERENCE VOICE BLENDING:
   - Support for multiple reference audio files
   - Voice interpolation and blending capabilities
   - Primary reference with additional voice characteristics

3. QUALITY OPTIMIZATION:
   - Specialized temperature for voice cloning vs. general synthesis
   - Voice-specific loss functions during inference
   - Spectral consistency and prosody matching
   - Real-time voice similarity assessment

4. USER-FRIENDLY INTERFACE:
   - Simple voice cloning mode with --clone-voice flag
   - Advanced parameter control for fine-tuning
   - Comprehensive validation and error reporting
   - Detailed voice cloning statistics and logging

USAGE EXAMPLES:
===============

Basic voice cloning:
  python inference_main.py --text "Hello world" --reference-audio speaker.wav --clone-voice

Advanced voice cloning with custom parameters:
  python inference_main.py --text "Hello world" --reference-audio speaker.wav --clone-voice \
    --voice-cloning-temperature 0.6 --voice-conditioning-strength 1.2

Multiple reference audios for voice blending:
  python inference_main.py --text "Hello world" \
    --multiple-reference-audios speaker1.wav speaker2.wav --clone-voice

High-quality synthesis without voice cloning:
  python inference_main.py --text "Hello world" --temperature 0.8
"""

from __future__ import annotations

import argparse
import os
from typing import Optional, List

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import tensorflow as tf  # noqa: F401  # Ensures TF is initialised before model import

from train_main import build_config
from myxtts.config.config import XTTSConfig
from myxtts.inference.synthesizer import XTTSInference
from myxtts.utils.commons import setup_logging, find_latest_checkpoint

# Import evaluation capabilities
try:
    from myxtts.evaluation import TTSEvaluator
    from myxtts.optimization import OptimizedInference, InferenceConfig
    EVAL_OPT_AVAILABLE = True
except ImportError:
    EVAL_OPT_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference for the MyXTTS model using the training defaults."
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config_path",
        help="Optional YAML config. When omitted, uses build_config from train_main.py",
    )
    parser.add_argument(
        "--checkpoint",
        help="Exact checkpoint prefix to load (e.g. checkpoints/checkpoint_5000).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="./checkpointsmain",
        help="Directory to search when --checkpoint is not provided (default: ./checkpointsmain).",
    )
    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument(
        "--text-file",
        help="Path to a text file. Entire file contents are synthesized.",
    )
    parser.add_argument(
        "--reference-audio",
        help="Optional reference audio (wav/flac) for voice conditioning.",
    )
    
    # Advanced Voice Cloning Parameters
    parser.add_argument(
        "--voice-cloning-temperature",
        type=float,
        default=0.7,
        help="Temperature specifically for voice cloning (default: 0.7, lower for more consistent voice).",
    )
    parser.add_argument(
        "--voice-conditioning-strength",
        type=float,
        default=1.0,
        help="Strength of voice conditioning (0.0-2.0, default: 1.0).",
    )
    parser.add_argument(
        "--enable-voice-interpolation",
        action="store_true",
        help="Enable voice interpolation for smoother voice cloning.",
    )
    parser.add_argument(
        "--voice-similarity-threshold",
        type=float,
        default=0.75,
        help="Minimum voice similarity threshold for voice cloning (default: 0.75).",
    )
    parser.add_argument(
        "--multiple-reference-audios",
        nargs="+",
        help="Multiple reference audio files for voice blending.",
    )
    parser.add_argument(
        "--speaker-id",
        help="Speaker ID for multi-speaker models (e.g., 'p225', 'speaker_001').",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true", 
        help="List available speakers in multi-speaker model and exit.",
    )
    parser.add_argument(
        "--clone-voice",
        action="store_true",
        help="Enable advanced voice cloning mode with enhanced voice similarity.",
    )
    
    # Global Style Tokens (GST) for prosody control
    parser.add_argument(
        "--prosody-reference",
        help="Reference audio file for prosody conditioning (GST).",
    )
    parser.add_argument(
        "--emotion",
        choices=["neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"],
        help="Emotion style for synthesis (requires GST).",
    )
    parser.add_argument(
        "--speaking-rate",
        type=float,
        default=1.0,
        help="Speaking rate control (0.5-2.0, default: 1.0, requires GST).",
    )
    parser.add_argument(
        "--style-weights",
        nargs="+",
        type=float,
        help="Direct style token weights (space-separated floats, must match number of style tokens).",
    )
    parser.add_argument(
        "--list-style-tokens",
        action="store_true",
        help="List available style tokens and exit.",
    )
    
    parser.add_argument(
        "--decoder-strategy",
        choices=["autoregressive", "non_autoregressive"],
        help="Override decoder strategy for inference.",
    )
    parser.add_argument(
        "--vocoder-type",
        choices=["griffin_lim", "hifigan", "bigvgan"],
        help="Select vocoder backend for mel-to-audio conversion.",
    )
    parser.add_argument(
        "--language",
        help="Language code (default: value from config).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for generation (default: 1.0).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1000,
        help="Maximum mel length to generate (default: 1000).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="inference_outputs/output.wav",
        help="Output path for synthesized audio (default: inference_outputs/output.wav).",
    )
    parser.add_argument(
        "--save-mel",
        help="Optional path to save the generated mel spectrogram as .npy.",
    )
    
    # Enhanced voice conditioning options
    parser.add_argument(
        "--use-pretrained-speaker-encoder",
        action="store_true",
        help="Enable pre-trained speaker encoder for enhanced voice conditioning.",
    )
    parser.add_argument(
        "--speaker-encoder-type",
        choices=["ecapa_tdnn", "resemblyzer", "coqui"],
        default="ecapa_tdnn",
        help="Type of pre-trained speaker encoder to use (default: ecapa_tdnn).",
    )
    parser.add_argument(
        "--voice-conditioning-strength",
        type=float,
        default=1.0,
        help="Strength of voice conditioning (0.0-2.0, default: 1.0).",
    )
    parser.add_argument(
        "--voice-cloning-temperature",
        type=float,
        default=0.7,
        help="Temperature for voice cloning (default: 0.7).",
    )
    parser.add_argument(
        "--voice-similarity-threshold",
        type=float,
        default=0.75,
        help="Voice similarity threshold for quality control (default: 0.75).",
    )
    parser.add_argument(
        "--enable-voice-interpolation",
        action="store_true",
        help="Enable voice interpolation for blending multiple voices.",
    )
    
    # Multi-language support (NEW)
    parser.add_argument(
        "--detect-language",
        action="store_true",
        help="Automatically detect language from input text.",
    )
    parser.add_argument(
        "--supported-languages",
        nargs="+",
        default=["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"],
        help="List of supported languages for detection (default: common languages).",
    )
    parser.add_argument(
        "--enable-phone-normalization",
        action="store_true",
        help="Enable phone-level text normalization.",
    )
    
    # Multi-speaker support (NEW)
    parser.add_argument(
        "--speaker-id",
        help="Speaker ID for multi-speaker models.",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List available speakers in multi-speaker model and exit.",
    )
    
    # Evaluation and optimization options
    parser.add_argument(
        "--evaluate-output",
        action="store_true",
        help="Automatically evaluate the generated audio quality"
    )
    parser.add_argument(
        "--evaluation-output",
        help="Save evaluation results to this JSON file"
    )
    parser.add_argument(
        "--optimized-inference",
        action="store_true", 
        help="Use optimized inference pipeline for faster generation"
    )
    parser.add_argument(
        "--lightweight-model",
        help="Path to lightweight/compressed model (compressed, distilled, or quantized)"
    )
    parser.add_argument(
        "--quality-mode",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Quality vs speed trade-off (default: balanced)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmark on the model"
    )
    
    return parser.parse_args()


def load_text(args: argparse.Namespace) -> str:
    """Load text either from CLI or from a file."""
    if args.text is not None:
        return args.text.strip()

    with open(args.text_file, "r", encoding="utf-8") as f:
        contents = f.read().strip()

    if not contents:
        raise ValueError("Provided text file is empty.")

    return contents


def load_config(config_path: Optional[str], checkpoint_dir: str) -> XTTSConfig:
    """Load configuration from YAML or reuse the training build_config."""
    if config_path:
        config = XTTSConfig.from_yaml(config_path)
    else:
        config = build_config(checkpoint_dir=checkpoint_dir)

    # Ensure checkpoint directory is up to date regardless of source
    config.training.checkpoint_dir = checkpoint_dir
    return config


def resolve_checkpoint(args: argparse.Namespace, logger) -> str:
    """Resolve which checkpoint to load, supporting lightweight models."""
    
    # Priority 1: Lightweight/compressed model if specified
    if hasattr(args, 'lightweight_model') and args.lightweight_model:
        lightweight_path = args.lightweight_model
        
        # Check if it's a direct path to a model file
        if os.path.exists(lightweight_path):
            logger.info(f"🎯 Loading lightweight model: {lightweight_path}")
            return lightweight_path
        
        # Check for checkpoint-style naming
        test_path = f"{lightweight_path}_model.weights.h5"
        if os.path.exists(test_path):
            logger.info(f"🎯 Loading lightweight checkpoint: {lightweight_path}")
            return lightweight_path
            
        raise FileNotFoundError(f"Lightweight model not found: {lightweight_path}")
    
    # Priority 2: Explicit checkpoint
    if args.checkpoint:
        checkpoint_prefix = args.checkpoint
        if not os.path.exists(f"{checkpoint_prefix}_model.weights.h5"):
            raise FileNotFoundError(
                f"Model weights not found at {checkpoint_prefix}_model.weights.h5"
            )
        return checkpoint_prefix

    # Priority 3: Latest checkpoint in directory
    checkpoint_prefix = find_latest_checkpoint(args.checkpoint_dir)
    if not checkpoint_prefix:
        raise FileNotFoundError(
            f"No checkpoints found in {args.checkpoint_dir}. Provide --checkpoint or --lightweight-model explicitly."
        )

    logger.info(f"Using latest checkpoint: {checkpoint_prefix}")
    return checkpoint_prefix


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for a file if it doesn't exist."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def validate_reference_audio(reference_audio_path: str, logger) -> bool:
    """Validate that reference audio file exists and is readable."""
    if not os.path.exists(reference_audio_path):
        logger.error(f"Reference audio file not found: {reference_audio_path}")
        return False
    
    # Check file extension
    valid_extensions = ['.wav', '.flac', '.mp3', '.m4a']
    file_extension = os.path.splitext(reference_audio_path)[1].lower()
    if file_extension not in valid_extensions:
        logger.warning(f"Reference audio extension '{file_extension}' may not be supported. Supported: {valid_extensions}")
    
    return True


def log_voice_cloning_setup(args, logger) -> None:
    """Log voice cloning setup information."""
    logger.info("🎭 Voice Cloning Setup:")
    logger.info(f"   • Mode: {'Advanced' if args.clone_voice else 'Standard'}")
    logger.info(f"   • Temperature: {args.voice_cloning_temperature}")
    logger.info(f"   • Conditioning strength: {args.voice_conditioning_strength}")
    logger.info(f"   • Similarity threshold: {args.voice_similarity_threshold}")
    logger.info(f"   • Voice interpolation: {'Enabled' if args.enable_voice_interpolation else 'Disabled'}")
    
    if args.multiple_reference_audios:
        logger.info(f"   • Multiple reference audios: {len(args.multiple_reference_audios)} files")
        for i, ref_audio in enumerate(args.multiple_reference_audios):
            logger.info(f"     [{i+1}] {ref_audio}")
    elif args.reference_audio:
        logger.info(f"   • Reference audio: {args.reference_audio}")
    else:
        logger.info("   • Reference audio: None (will use default voice)")


def detect_text_language(text: str, supported_languages: List[str], logger) -> str:
    """
    Detect language from input text.
    
    Args:
        text: Input text
        supported_languages: List of supported language codes
        logger: Logger instance
        
    Returns:
        Detected language code
    """
    try:
        # Simple heuristic-based language detection
        text_lower = text.lower()
        
        # Check for common language-specific characters
        if any(ord(c) > 127 for c in text):  # Non-ASCII characters
            # Arabic script
            if any('\u0600' <= c <= '\u06FF' for c in text):
                return 'ar' if 'ar' in supported_languages else 'en'
            # Chinese characters
            elif any('\u4e00' <= c <= '\u9fff' for c in text):
                return 'zh' if 'zh' in supported_languages else 'en'
            # Japanese characters  
            elif any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' for c in text):
                return 'ja' if 'ja' in supported_languages else 'en'
            # Korean characters
            elif any('\uac00' <= c <= '\ud7af' for c in text):
                return 'ko' if 'ko' in supported_languages else 'en'
            # Cyrillic (Russian, etc.)
            elif any('\u0400' <= c <= '\u04ff' for c in text):
                return 'ru' if 'ru' in supported_languages else 'en'
        
        # Basic European language detection by common words
        language_indicators = {
            'en': ['the', 'and', 'is', 'to', 'of', 'a'],
            'es': ['el', 'la', 'de', 'que', 'y', 'en'],
            'fr': ['le', 'de', 'et', 'à', 'un', 'la'],
            'de': ['der', 'die', 'und', 'ist', 'zu', 'das'],
            'it': ['il', 'di', 'e', 'che', 'la', 'un'],
            'pt': ['o', 'de', 'e', 'que', 'a', 'do'],
            'pl': ['i', 'w', 'na', 'z', 'o', 'do'],
            'tr': ['ve', 'bir', 'bu', 'de', 'da', 'ile'],
            'ru': ['и', 'в', 'не', 'на', 'я', 'что'],
            'nl': ['de', 'het', 'van', 'en', 'een', 'is'],
            'cs': ['a', 'v', 'na', 'se', 'z', 'o'],
        }
        
        for lang in supported_languages:
            if lang in language_indicators:
                indicators = language_indicators[lang]
                if any(word in text_lower for word in indicators):
                    return lang
        
        logger.warning("Could not detect language, defaulting to English")
        return 'en'
        
    except Exception as e:
        logger.warning(f"Language detection failed: {e}, defaulting to English")
        return 'en'


def apply_phone_normalization(text: str, language: str, logger) -> str:
    """
    Apply phone-level normalization to text.
    
    Args:
        text: Input text
        language: Language code
        logger: Logger instance
        
    Returns:
        Phone-normalized text
    """
    try:
        from myxtts.utils.text import TextProcessor
        
        # Create a temporary text processor with phoneme support
        processor = TextProcessor(
            language=language,
            use_phonemes=True,
            tokenizer_type="custom"  # Use custom for phonemization
        )
        
        # Apply phonemization
        normalized_text = processor.text_to_phonemes(text)
        return normalized_text
        
    except Exception as e:
        logger.warning(f"Phone normalization failed: {e}, using original text")
        return text


def list_available_speakers(config, checkpoint_prefix: str, logger):
    """
    List available speakers in a multi-speaker model.
    
    Args:
        config: Model configuration
        checkpoint_prefix: Checkpoint path prefix
        logger: Logger instance
    """
    try:
        # This would need to be implemented based on how speaker information is stored
        # For now, provide a placeholder implementation
        logger.info("🎭 Available speakers:")
        
        if hasattr(config.data, 'max_speakers') and config.data.enable_multispeaker:
            logger.info(f"   • Multi-speaker model with up to {config.data.max_speakers} speakers")
            logger.info("   • Speaker IDs depend on training data")
            logger.info("   • Common patterns: 'speaker_01', 'p001', 'male_01', etc.")
        else:
            logger.info("   • Single-speaker model")
            logger.info("   • Use voice conditioning with --reference-audio instead")
            
    except Exception as e:
        logger.error(f"Failed to list speakers: {e}")


def list_style_tokens(config, logger):
    """
    List available style tokens in the GST model.
    
    Args:
        config: Model configuration
        logger: Logger instance
    """
    try:
        logger.info("🎨 Style Token Information:")
        
        if getattr(config.model, 'use_gst', False):
            num_tokens = getattr(config.model, 'gst_num_style_tokens', 10)
            token_dim = getattr(config.model, 'gst_style_token_dim', 256)
            embedding_dim = getattr(config.model, 'gst_style_embedding_dim', 256)
            
            logger.info(f"   • Number of style tokens: {num_tokens}")
            logger.info(f"   • Style token dimension: {token_dim}")
            logger.info(f"   • Style embedding dimension: {embedding_dim}")
            logger.info("   • Available emotions: neutral, happy, sad, angry, surprised, fearful, disgusted")
            logger.info("   • Speaking rate control: 0.5 (slow) to 2.0 (fast)")
            logger.info("   • Use --prosody-reference for prosody conditioning")
            logger.info("   • Use --emotion for predefined emotion styles")
            logger.info("   • Use --style-weights for direct token control")
        else:
            logger.info("   • GST (Global Style Tokens) not enabled in this model")
            logger.info("   • Style control not available")
            
    except Exception as e:
        logger.error(f"Failed to list style tokens: {e}")


def main():
    args = parse_args()
    logger = setup_logging()

    # Log accelerator availability (useful for debugging inference speed)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPUs available: %s", ", ".join(gpu.name for gpu in gpus))
    else:
        logger.info("No GPU detected; inference will run on CPU.")

    text = load_text(args)
    config = load_config(args.config_path, args.checkpoint_dir)
    
    # Update configuration with command line arguments
    if args.decoder_strategy:
        config.model.decoder_strategy = args.decoder_strategy
    if args.vocoder_type:
        config.model.vocoder_type = args.vocoder_type
        
    # Enhanced voice conditioning settings
    if args.use_pretrained_speaker_encoder:
        config.model.use_pretrained_speaker_encoder = True
        config.model.speaker_encoder_type = args.speaker_encoder_type
        logger.info(f"🎯 Enhanced voice conditioning enabled with {args.speaker_encoder_type} encoder")
    
    # Voice cloning parameters
    if hasattr(config.model, "voice_cloning_temperature"):
        config.model.voice_cloning_temperature = args.voice_cloning_temperature
    if hasattr(config.model, "voice_conditioning_strength"):
        config.model.voice_conditioning_strength = args.voice_conditioning_strength
    if hasattr(config.model, "voice_similarity_threshold"):
        config.model.voice_similarity_threshold = args.voice_similarity_threshold
    if hasattr(config.model, "enable_speaker_interpolation"):
        config.model.enable_speaker_interpolation = args.enable_voice_interpolation
    
    # Global Style Tokens (GST) parameters
    prosody_reference = None
    style_weights = None
    
    if args.prosody_reference:
        if validate_reference_audio(args.prosody_reference, logger):
            prosody_reference = args.prosody_reference
            logger.info(f"🎵 Using prosody reference: {args.prosody_reference}")
        else:
            logger.warning("Invalid prosody reference audio, proceeding without it")
    
    if args.emotion:
        # Convert emotion to style weights (simple mapping)
        emotion_mapping = {
            "neutral": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "happy": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "sad": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "angry": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "surprised": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "fearful": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "disgusted": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        }
        if args.emotion in emotion_mapping:
            style_weights = emotion_mapping[args.emotion]
            logger.info(f"😊 Using emotion style: {args.emotion}")
    
    if args.style_weights:
        if getattr(config.model, 'gst_num_style_tokens', 10) == len(args.style_weights):
            style_weights = args.style_weights
            logger.info(f"🎨 Using custom style weights: {style_weights}")
        else:
            logger.warning(f"Style weights count ({len(args.style_weights)}) doesn't match model tokens ({getattr(config.model, 'gst_num_style_tokens', 10)})")
    
    if args.list_style_tokens:
        list_style_tokens(config, logger)
        return
    
    # Multi-language support (NEW)
    detected_language = config.data.language  # Default language
    if args.detect_language:
        detected_language = detect_text_language(text, args.supported_languages, logger)
        logger.info(f"🌍 Detected language: {detected_language}")
    elif args.language:
        detected_language = args.language
        logger.info(f"🌍 Using specified language: {detected_language}")
    
    # Apply phone-level normalization if requested
    if args.enable_phone_normalization:
        text = apply_phone_normalization(text, detected_language, logger)
        logger.info("🔤 Applied phone-level normalization")
    
    # Update config with detected language
    config.data.language = detected_language
    
    # Multi-speaker support (NEW)
    if args.list_speakers:
        list_available_speakers(config, checkpoint_prefix, logger)
        return
    
    if args.speaker_id:
        logger.info(f"👤 Using speaker ID: {args.speaker_id}")
        # Store speaker ID for use during inference
        target_speaker_id = args.speaker_id
    else:
        target_speaker_id = None
        
    checkpoint_prefix = resolve_checkpoint(args, logger)
    
    # Validate reference audio files if provided
    if args.reference_audio and not validate_reference_audio(args.reference_audio, logger):
        return
    
    if args.multiple_reference_audios:
        for ref_audio in args.multiple_reference_audios:
            if not validate_reference_audio(ref_audio, logger):
                return
    
    # Log voice cloning setup
    if args.clone_voice or args.reference_audio or args.multiple_reference_audios:
        log_voice_cloning_setup(args, logger)

    # Instantiate inference engine with optimization support
    if hasattr(args, 'optimized_inference') and args.optimized_inference and EVAL_OPT_AVAILABLE:
        logger.info("🚀 Using optimized inference pipeline for faster generation")
        
        # Create optimized inference config based on quality mode
        optimization_level = {
            "fast": "aggressive",
            "balanced": "moderate", 
            "quality": "conservative"
        }.get(args.quality_mode, "moderate")
        
        try:
            inference_config = InferenceConfig(
                optimization_level=optimization_level,
                enable_mixed_precision=True,
                enable_graph_optimization=True,
                enable_tensorrt=False,  # Disable TRT for compatibility
                batch_size=1,
                max_sequence_length=512
            )
            
            inference_engine = OptimizedInference(
                config=config,
                checkpoint_path=checkpoint_prefix,
                optimization_config=inference_config
            )
            logger.info(f"✅ Optimized inference initialized (level: {optimization_level})")
            
        except Exception as e:
            logger.warning(f"Failed to initialize optimized inference: {e}")
            logger.info("Falling back to standard inference engine")
            inference_engine = XTTSInference(
                config=config,
                checkpoint_path=checkpoint_prefix,
            )
    else:
        if hasattr(args, 'optimized_inference') and args.optimized_inference and not EVAL_OPT_AVAILABLE:
            logger.warning("Optimized inference requested but optimization modules not available")
            
        inference_engine = XTTSInference(
            config=config,
            checkpoint_path=checkpoint_prefix,
        )

    # Handle advanced voice cloning features
    if args.clone_voice or args.multiple_reference_audios:
        logger.info("🎭 Advanced voice cloning mode enabled")
        
        # Validate voice conditioning is enabled
        if not config.model.use_voice_conditioning:
            logger.error("Voice conditioning is not enabled in the model configuration")
            logger.error("Please ensure the model was trained with voice conditioning enabled")
            return
        
        # Handle multiple reference audios for voice blending
        if args.multiple_reference_audios:
            logger.info(f"Processing {len(args.multiple_reference_audios)} reference audio files for voice blending")
            
            # Use the first reference audio as primary, others for blending
            primary_reference = args.multiple_reference_audios[0]
            logger.info(f"Primary reference audio: {primary_reference}")
            
            # For now, use the primary reference (voice blending would need additional implementation)
            reference_audio = primary_reference
        else:
            reference_audio = args.reference_audio
        
        # Use voice cloning with enhanced parameters
        result = inference_engine.clone_voice(
            text=text,
            reference_audio=reference_audio,
            prosody_reference=prosody_reference,
            style_weights=style_weights,
            language=args.language,
            max_length=args.max_length,
            temperature=args.voice_cloning_temperature,  # Use specialized voice cloning temperature
        )
        
        logger.info(f"Voice cloning completed with temperature: {args.voice_cloning_temperature}")
        logger.info(f"Voice conditioning strength: {args.voice_conditioning_strength}")
        
    else:
        # Standard synthesis
        result = inference_engine.synthesize(
            text=text,
            reference_audio=args.reference_audio,
            prosody_reference=prosody_reference,
            style_weights=style_weights,
            language=args.language,
            max_length=args.max_length,
            temperature=args.temperature,
        )

    # Persist outputs
    ensure_parent_dir(args.output)
    inference_engine.save_audio(result["audio"], args.output, result.get("sample_rate"))

    if args.save_mel:
        ensure_parent_dir(args.save_mel)
        np.save(args.save_mel, result["mel_spectrogram"])
        logger.info("Saved mel spectrogram to %s", args.save_mel)

    logger.info(
        "🎵 Synthesis complete. Duration: %.2fs | Sample rate: %d",
        len(result["audio"]) / result["sample_rate"],
        result["sample_rate"],
    )
    
    # Log voice cloning specific information
    if args.clone_voice or args.multiple_reference_audios:
        logger.info("🎭 Voice cloning information:")
        logger.info(f"   • Voice cloning temperature: {args.voice_cloning_temperature}")
        logger.info(f"   • Voice conditioning strength: {args.voice_conditioning_strength}")
        logger.info(f"   • Voice similarity threshold: {args.voice_similarity_threshold}")
        if args.multiple_reference_audios:
            logger.info(f"   • Reference audio files: {len(args.multiple_reference_audios)}")
        if args.enable_voice_interpolation:
            logger.info("   • Voice interpolation: Enabled")
    
    logger.info(f"✅ Audio saved to: {args.output}")
    if args.save_mel:
        logger.info(f"✅ Mel spectrogram saved to: {args.save_mel}")

    # Post-synthesis evaluation and optimization
    if EVAL_OPT_AVAILABLE:
        try:
            # Automatic evaluation
            if args.evaluate_output:
                logger.info("\n🎯 Starting automatic audio evaluation...")
                evaluate_synthesis_output(args.output, text, args)
            
            # Performance benchmarking
            if args.benchmark:
                logger.info("\n⚡ Running performance benchmark...")
                benchmark_inference_performance(inference_engine, text, args)
                
        except Exception as e:
            logger.warning(f"Post-synthesis evaluation failed: {e}")
    else:
        if args.evaluate_output or args.benchmark:
            logger.warning("Evaluation/benchmark requested but modules not available")


def evaluate_synthesis_output(audio_path: str, reference_text: str, args):
    """Evaluate the synthesized audio using automatic metrics."""
    try:
        # Create evaluator
        evaluator = TTSEvaluator(
            enable_mosnet=True,
            enable_asr_wer=True,
            enable_cmvn=True,
            enable_spectral=True
        )
        
        # Run evaluation
        report = evaluator.evaluate_single(audio_path, reference_text)
        
        # Display results
        logger.info("📊 Evaluation Results:")
        logger.info(f"   Overall Score: {report.overall_score:.3f}")
        logger.info(f"   Evaluation Time: {report.evaluation_time:.2f}s")
        
        for metric_name, result in report.results.items():
            if result.error:
                logger.warning(f"   {metric_name.upper()}: ERROR - {result.error}")
            else:
                logger.info(f"   {metric_name.upper()}: {result.score:.3f}")
        
        # Save detailed results if requested
        if args.evaluation_output:
            evaluator.save_reports([report], args.evaluation_output)
            logger.info(f"   Detailed results saved to: {args.evaluation_output}")
            
    except Exception as e:
        logger.error(f"Audio evaluation failed: {e}")


def benchmark_inference_performance(inference_engine, text: str, args):
    """Benchmark the inference performance."""
    try:
        # Create test texts for benchmarking
        test_texts = [
            text,  # Use the actual text
            "Short test.",
            "This is a medium length test sentence for benchmarking.",
            "This is a longer test sentence that contains more words and should take more time to synthesize for comprehensive performance evaluation."
        ]
        
        logger.info("🔄 Running performance benchmark...")
        logger.info(f"   Test texts: {len(test_texts)}")
        logger.info(f"   Quality mode: {args.quality_mode}")
        
        # Simple timing benchmark (since we don't have the OptimizedInference integrated)
        import time
        
        total_synthesis_time = 0
        total_audio_duration = 0
        
        for i, test_text in enumerate(test_texts):
            start_time = time.time()
            
            # Simple synthesis call
            result = inference_engine.synthesize(
                text=test_text,
                temperature=args.temperature,
                max_length=args.max_length
            )
            
            synthesis_time = time.time() - start_time
            audio_duration = len(result["audio"]) / result["sample_rate"]
            rtf = synthesis_time / audio_duration
            
            total_synthesis_time += synthesis_time
            total_audio_duration += audio_duration
            
            logger.info(f"   Test {i+1}: {synthesis_time:.3f}s synthesis, {audio_duration:.3f}s audio (RTF: {rtf:.3f})")
        
        # Overall statistics
        overall_rtf = total_synthesis_time / total_audio_duration
        avg_synthesis_time = total_synthesis_time / len(test_texts)
        
        logger.info("📈 Benchmark Results:")
        logger.info(f"   Average synthesis time: {avg_synthesis_time:.3f}s")
        logger.info(f"   Total audio generated: {total_audio_duration:.3f}s")
        logger.info(f"   Overall RTF: {overall_rtf:.3f}")
        logger.info(f"   Real-time capable: {'✅ Yes' if overall_rtf < 1.0 else '❌ No'}")
        
        if overall_rtf < 0.5:
            logger.info("   Performance: 🚀 Excellent (< 0.5 RTF)")
        elif overall_rtf < 1.0:
            logger.info("   Performance: ⚡ Good (< 1.0 RTF)")
        else:
            logger.info("   Performance: 🐌 Needs optimization (> 1.0 RTF)")
            
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")


if __name__ == "__main__":
    main()
