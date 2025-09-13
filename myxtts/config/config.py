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
    """Model architecture configuration."""
    
    # Text encoder settings
    text_encoder_dim: int = 512
    text_encoder_layers: int = 6
    text_encoder_heads: int = 8
    text_vocab_size: int = 256_256  # Updated for NLLB-200 tokenizer
    
    # Audio encoder settings (for voice conditioning)
    audio_encoder_dim: int = 512
    audio_encoder_layers: int = 6
    audio_encoder_heads: int = 8
    
    # Decoder settings
    decoder_dim: int = 1024
    decoder_layers: int = 12
    decoder_heads: int = 16
    
    # Mel spectrogram settings
    n_mels: int = 80
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    sample_rate: int = 22050
    
    # Voice conditioning
    speaker_embedding_dim: int = 256
    use_voice_conditioning: bool = True
    
    # Language support
    languages: List[str] = None
    max_text_length: int = 500
    
    # Tokenizer settings
    tokenizer_type: str = "nllb"  # "custom" or "nllb"
    tokenizer_model: str = "facebook/nllb-200-distilled-600M"
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]


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
    
    # Training data
    train_split: float = 0.9
    val_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 8  # Increased for better CPU utilization
    
    # Voice conditioning
    reference_audio_length: float = 3.0  # seconds
    min_audio_length: float = 1.0
    max_audio_length: float = 11.0
    
    # Performance optimization settings
    prefetch_buffer_size: int = 8  # Increased for better GPU utilization  
    shuffle_buffer_multiplier: int = 20  # Increased for better shuffling
    enable_memory_mapping: bool = True  # Use memory mapping for cache files
    cache_verification: bool = True  # Verify cache integrity on startup
    prefetch_to_gpu: bool = True  # Prefetch batches directly to GPU (disable for low-memory)
    
    # Sequence length caps to avoid OOM
    max_mel_frames: int = 512  # Truncate mel length during training
    
    # GPU-specific optimizations
    enable_xla: bool = True  # Enable XLA compilation for faster training
    enable_tensorrt: bool = False  # Enable TensorRT optimization (requires TensorRT)
    mixed_precision: bool = True  # Enable mixed precision training
    
    # Data loading optimizations
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    persistent_workers: bool = True  # Keep workers alive between epochs
    
    # Dataset preprocessing control
    preprocessing_mode: str = "auto"  # "auto", "precompute", "runtime"
    
    def __post_init__(self):
        if self.text_cleaners is None:
            self.text_cleaners = ["english_cleaners"]
        
        # Validate preprocessing_mode
        valid_modes = ["auto", "precompute", "runtime"]
        if self.preprocessing_mode not in valid_modes:
            raise ValueError(f"preprocessing_mode must be one of {valid_modes}, got '{self.preprocessing_mode}'")


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Training parameters
    epochs: int = 1000
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    weight_decay: float = 1e-6
    gradient_clip_norm: float = 1.0
    
    # Optimizer
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Scheduler
    scheduler: str = "noam"
    scheduler_params: Dict[str, Any] = None
    
    # Loss weights
    mel_loss_weight: float = 45.0
    kl_loss_weight: float = 1.0
    duration_loss_weight: float = 1.0
    
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
