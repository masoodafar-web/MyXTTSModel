"""
Configuration classes for MyXTTS model.

This module contains configuration classes that define model architecture,
training parameters, and data processing settings, similar to XTTS configuration.
"""

import yaml
from dataclasses import dataclass, asdict, fields
from typing import List, Dict, Any, Optional


@dataclass
class ModelConfig:
    """Core model architecture configuration."""

    # Transformer architecture
    text_encoder_dim: int = 512
    text_encoder_layers: int = 8
    text_encoder_heads: int = 8
    text_vocab_size: int = 256_256

    audio_encoder_dim: int = 768
    audio_encoder_layers: int = 8
    audio_encoder_heads: int = 12

    decoder_dim: int = 1536
    decoder_layers: int = 16
    decoder_heads: int = 24

    # Acoustic front-end
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    sample_rate: int = 22050

    # Voice conditioning
    speaker_embedding_dim: int = 512
    use_voice_conditioning: bool = True

    decoder_strategy: str = "autoregressive"
    use_duration_predictor: bool = True

    # Vocoder settings
    vocoder_type: str = "hifigan"
    vocoder_upsample_rates: List[int] = None
    vocoder_upsample_kernel_sizes: List[int] = None
    vocoder_resblock_kernel_sizes: List[int] = None
    vocoder_resblock_dilation_sizes: List[List[int]] = None
    vocoder_initial_channel: int = 512

    # Tokenization / language support
    languages: List[str] = None
    max_text_length: int = 500
    tokenizer_type: str = "nllb"
    tokenizer_model: str = "facebook/nllb-200-distilled-600M"

    # Optional pre-trained speaker encoder
    use_pretrained_speaker_encoder: bool = False
    pretrained_speaker_encoder_path: Optional[str] = None
    freeze_speaker_encoder: bool = True
    speaker_encoder_type: str = "ecapa_tdnn"

    # Global Style Tokens
    use_gst: bool = True
    gst_num_style_tokens: int = 10
    gst_style_token_dim: int = 256
    gst_style_embedding_dim: int = 256
    gst_num_heads: int = 4
    gst_reference_encoder_dim: int = 128

    # Memory optimisation
    enable_gradient_checkpointing: bool = False
    max_attention_sequence_length: int = 512

    def __post_init__(self):
        if self.languages is None:
            self.languages = [
                "en", "es", "fr", "de", "it", "pt", "pl", "tr",
                "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko",
            ]

        if self.vocoder_upsample_rates is None:
            self.vocoder_upsample_rates = [8, 8, 2, 2]

        if self.vocoder_upsample_kernel_sizes is None:
            self.vocoder_upsample_kernel_sizes = [16, 16, 4, 4]

        if self.vocoder_resblock_kernel_sizes is None:
            self.vocoder_resblock_kernel_sizes = [3, 7, 11]

        if self.vocoder_resblock_dilation_sizes is None:
            self.vocoder_resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Dataset paths
    dataset_path: str = ""
    dataset_name: str = "ljspeech"

    # Custom metadata file paths (optional)
    metadata_train_file: Optional[str] = None  # Custom path for train metadata
    metadata_eval_file: Optional[str] = None   # Custom path for eval/val metadata

    # Custom wav directories (optional, used with custom metadata files)
    wavs_train_dir: Optional[str] = None       # Custom path for train wav files
    wavs_eval_dir: Optional[str] = None        # Custom path for eval wav files

    # Audio processing
    sample_rate: int = 22050
    trim_silence: bool = True
    normalize_audio: bool = True

    # Text processing
    text_cleaners: List[str] = None
    language: str = "en"
    add_blank: bool = True

    # Multi-speaker support (NEW)
    enable_multispeaker: bool = False
    speaker_id_pattern: Optional[str] = None  # Regex pattern for extracting speaker ID from filename
    max_speakers: int = 1000  # Maximum number of speakers to support

    # Enhanced audio processing (NEW)
    enable_loudness_normalization: bool = True
    target_loudness_lufs: float = -23.0
    enable_vad: bool = True  # Voice Activity Detection using Silero VAD
    
    # Audio augmentations for robustness (NEW)
    enable_pitch_shift: bool = False  # Enable pitch shifting augmentation
    pitch_shift_range: List[float] = None  # Pitch shift range in semitones
    enable_noise_mixing: bool = False  # Enable noise mixing augmentation
    noise_mixing_probability: float = 0.3  # Probability of applying noise mixing
    noise_mixing_snr_range: List[float] = None  # SNR range for noise mixing in dB
    
    # Phone-level normalization (NEW)
    enable_phone_normalization: bool = False  # Enable phone-level text normalization
    use_phonemes: bool = True  # Use phonemic representation
    phoneme_language: Optional[str] = None  # Language for phonemization (auto-detect if None)
    
    # Multi-language support (NEW)
    enable_multilingual: bool = False  # Enable multi-language support
    supported_languages: List[str] = None  # List of supported languages (auto-detect if None)
    language_detection_method: str = "metadata"  # "metadata", "filename", or "auto"

    # Training splits and optional subsampling
    train_split: float = 0.9
    val_split: float = 0.1
    train_subset_fraction: float = 1.0  # 0-1 fraction of train to use
    eval_subset_fraction: float = 1.0   # 0-1 fraction of val/test to use
    subset_seed: int = 42
    max_text_tokens: int = 1024

    # Batching and workers (optimized for better GPU utilization)
    batch_size: int = 56  # Increased further for better GPU utilization with 80GB+ memory
    num_workers: int = 18  # Increased for better CPU-GPU overlap

    # Data pipeline performance (optimized for GPU)
    prefetch_buffer_size: int = 12  # Increased from 8 for sustained GPU utilization
    shuffle_buffer_multiplier: int = 30  # Increased from 20 for better randomization
    enable_memory_mapping: bool = True
    cache_verification: bool = True
    prefetch_to_gpu: bool = True

    # Sequence length caps to avoid OOM (optimized for GPU)
    # Increase these for longer sentences: max_mel_frames for audio duration, 
    # ModelConfig.max_attention_sequence_length for text length
    max_mel_frames: int = 1024  # Increased from 512 for better GPU utilization (~13 seconds at 22kHz)

    # GPU-specific optimizations
    enable_xla: bool = True
    enable_tensorrt: bool = False
    mixed_precision: bool = True

    # Data loading extra options
    pin_memory: bool = True
    persistent_workers: bool = True

    # Dataset preprocessing control (default to precompute for GPU optimization)
    preprocessing_mode: str = "precompute"  # Changed from "auto" - forces cache files for GPU optimization
    
    # Reference audio controls for voice cloning
    reference_audio_length: float = 3.0  # Seconds of reference audio to use for conditioning
    max_audio_length: float = 11.0       # Maximum seconds of audio to keep after trimming

    # Advanced GPU optimization options
    use_tf_native_loading: bool = True  # Use TensorFlow-native file loading instead of Python functions
    enhanced_gpu_prefetch: bool = True  # Enable advanced GPU prefetching strategies
    optimize_cpu_gpu_overlap: bool = True  # Enable maximum CPU-GPU overlap optimizations
    auto_tune_performance: bool = True  # Automatically adjust performance settings based on hardware

    def __post_init__(self):
        if self.text_cleaners is None:
            self.text_cleaners = ["english_cleaners"]
        if self.pitch_shift_range is None:
            self.pitch_shift_range = [-2.0, 2.0]
        if self.noise_mixing_snr_range is None:
            self.noise_mixing_snr_range = [10.0, 30.0]
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]
        valid_modes = ["auto", "precompute", "runtime"]
        if self.preprocessing_mode not in valid_modes:
            raise ValueError(
                f"preprocessing_mode must be one of {valid_modes}, got '{self.preprocessing_mode}'"
            )


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training parameters
    epochs: int = 1000
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    max_memory_fraction: float = 0.9
    enable_memory_cleanup: bool = True

    # Optimizer
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Scheduler
    scheduler: str = "noam"
    scheduler_params: Dict[str, Any] = None
    
    # Loss weights (adjusted for better balance and voice cloning)
    mel_loss_weight: float = 35.0  # Reduced from 45.0 for better stability
    kl_loss_weight: float = 1.0
    duration_loss_weight: float = 0.1  # Enabled with small weight for stability
    attention_loss_weight: float = 0.02  # Lower weight to prevent attention loss spikes
    pitch_loss_weight: float = 0.1       # Weight for mel-level pitch prediction head
    energy_loss_weight: float = 0.1      # Weight for mel-level energy prediction head
    prosody_pitch_loss_weight: float = 0.05   # Weight for text-level prosody pitch predictor
    prosody_energy_loss_weight: float = 0.05  # Weight for text-level prosody energy predictor
    speaking_rate_loss_weight: float = 0.05   # Weight for speaking-rate predictor regularization
    max_duration_frames: float = 30.0         # Cap per-token duration target to avoid loss spikes
    auxiliary_head_regularization: float = 1e-6  # L2 regularization for auxiliary heads
    
    # Voice Cloning Loss Components - NEW for enhanced voice cloning
    voice_similarity_loss_weight: float = 3.0    # Weight for voice similarity loss
    speaker_classification_loss_weight: float = 1.5  # Weight for speaker classification
    voice_reconstruction_loss_weight: float = 2.0    # Weight for voice reconstruction
    prosody_matching_loss_weight: float = 1.0        # Weight for prosody matching
    spectral_consistency_loss_weight: float = 1.5    # Weight for spectral consistency
    
    # Training stability improvements
    use_adaptive_loss_weights: bool = True      # Enable adaptive loss weight scaling
    loss_smoothing_factor: float = 0.1          # Exponential smoothing factor
    max_loss_spike_threshold: float = 2.0       # Maximum allowed loss spike multiplier
    gradient_norm_threshold: float = 5.0        # Threshold for gradient norm monitoring
    loss_alert_threshold: float = 500.0         # Threshold for triggering loss breakdown logging
    
    # Enhanced loss function options
    use_label_smoothing: bool = True            # Enable label smoothing for regularization
    mel_label_smoothing: float = 0.05          # Label smoothing for mel loss
    stop_label_smoothing: float = 0.1          # Label smoothing for stop token loss
    use_huber_loss: bool = True                # Use Huber loss for mel loss stability
    huber_delta: float = 1.0                   # Delta parameter for Huber loss
    stop_token_positive_weight: float = 5.0    # Weight for positive stop tokens
    
    # Debug/simple-loss switch (for diagnosing training stalls)
    use_simple_loss: bool = False
    
    # Learning rate improvements
    use_warmup_cosine_schedule: bool = True     # Use cosine annealing with warmup
    cosine_restarts: bool = False              # Enable cosine restarts
    min_learning_rate: float = 1e-6           # Minimum learning rate
    
    # Early stopping improvements
    early_stopping_patience: int = 25          # Increased patience for stability
    early_stopping_min_delta: float = 0.0005  # Smaller delta for fine-grained stopping
    
    # Checkpointing
    save_step: int = 25000
    checkpoint_dir: str = "./checkpoints"
    
    # Validation
    val_step: int = 5000
    
    # Logging
    log_step: int = 100
    use_wandb: bool = False
    wandb_project: str = "myxtts"
    
    # Device / distribution
    multi_gpu: bool = False            # Enable MirroredStrategy when True
    visible_gpus: Optional[str] = None # e.g., "0" or "0,1"; None = all visible
    
    # Automatic evaluation parameters for checkpoint quality monitoring
    enable_automatic_evaluation: bool = False  # Enable MOSNet, ASR-WER evaluation during training
    evaluation_interval: int = 10              # Evaluate model every N epochs
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}


@dataclass
class XTTSConfig:
    """Main configuration class combining all settings."""
    
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    def __init__(self, model: ModelConfig = None, data: DataConfig = None, training: TrainingConfig = None, **kwargs):
        """Initialize XTTSConfig with optional direct parameter passing.
        
        Args:
            model: ModelConfig instance
            data: DataConfig instance 
            training: TrainingConfig instance
            **kwargs: Direct parameters that will be distributed to appropriate sub-configs
                     based on parameter names
        """
        # Set defaults if not provided
        if model is None:
            model = ModelConfig()
        if data is None:
            data = DataConfig()
        if training is None:
            training = TrainingConfig()
            
        # If kwargs provided, distribute them to appropriate configs
        if kwargs:
            model_kwargs, data_kwargs, training_kwargs = self._distribute_kwargs(kwargs)
            
            # Update configs with kwargs
            if model_kwargs:
                # Create new ModelConfig with updated parameters
                model_dict = asdict(model)
                model_dict.update(model_kwargs)
                model = ModelConfig(**model_dict)
                
            if data_kwargs:
                # Create new DataConfig with updated parameters
                data_dict = asdict(data)
                data_dict.update(data_kwargs)
                data = DataConfig(**data_dict)
                
            if training_kwargs:
                # Create new TrainingConfig with updated parameters
                training_dict = asdict(training)
                training_dict.update(training_kwargs)
                training = TrainingConfig(**training_dict)
        
        self.model = model
        self.data = data
        self.training = training
    
    @staticmethod
    def _distribute_kwargs(kwargs: Dict[str, Any]) -> tuple:
        """Distribute kwargs to appropriate config classes based on field names.
        
        Args:
            kwargs: Dictionary of parameters to distribute
            
        Returns:
            Tuple of (model_kwargs, data_kwargs, training_kwargs)
        """
        # Get field names for each config class
        model_fields = {f.name for f in fields(ModelConfig)}
        data_fields = {f.name for f in fields(DataConfig)}
        training_fields = {f.name for f in fields(TrainingConfig)}
        
        model_kwargs = {}
        data_kwargs = {}
        training_kwargs = {}
        
        for key, value in kwargs.items():
            # Handle parameters that might belong to multiple configs
            if key in model_fields and key in data_fields:
                # Parameter exists in both configs, set in both
                model_kwargs[key] = value
                data_kwargs[key] = value
            elif key in model_fields and key in training_fields:
                # Parameter exists in both configs, set in both
                model_kwargs[key] = value
                training_kwargs[key] = value
            elif key in data_fields and key in training_fields:
                # Parameter exists in both configs, set in both
                data_kwargs[key] = value
                training_kwargs[key] = value
            elif key in model_fields:
                model_kwargs[key] = value
            elif key in data_fields:
                data_kwargs[key] = value
            elif key in training_fields:
                training_kwargs[key] = value
            else:
                # For convenience, handle some common parameter name variations
                if key == "batch_size":
                    data_kwargs[key] = value
                elif key == "epochs":
                    training_kwargs[key] = value
                elif key == "learning_rate":
                    training_kwargs[key] = value
                elif key == "metadata_train_file":
                    data_kwargs[key] = value
                elif key == "metadata_eval_file":
                    data_kwargs[key] = value
                else:
                    raise ValueError(f"Unknown parameter: {key}. "
                                   f"Valid parameters are: {sorted(model_fields | data_fields | training_fields)}")
        
        return model_kwargs, data_kwargs, training_kwargs
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'XTTSConfig':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        return cls(
            model=model_config,
            data=data_config,
            training=training_config
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': asdict(self.model),
            'data': asdict(self.data), 
            'training': asdict(self.training)
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'model': asdict(self.model),
            'data': asdict(self.data),
            'training': asdict(self.training)
        }
