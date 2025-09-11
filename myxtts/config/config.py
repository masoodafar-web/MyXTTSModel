"""
Configuration classes for MyXTTS model.

This module contains configuration classes that define model architecture,
training parameters, and data processing settings, similar to XTTS configuration.
"""

import yaml
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Text encoder settings
    text_encoder_dim: int = 512
    text_encoder_layers: int = 6
    text_encoder_heads: int = 8
    text_vocab_size: int = 256
    
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
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh", "ja", "hu", "ko"]


@dataclass 
class DataConfig:
    """Data processing configuration."""
    
    # Dataset paths
    dataset_path: str = ""
    dataset_name: str = "ljspeech"
    
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
    num_workers: int = 4
    
    # Voice conditioning
    reference_audio_length: float = 3.0  # seconds
    min_audio_length: float = 1.0
    max_audio_length: float = 11.0
    
    def __post_init__(self):
        if self.text_cleaners is None:
            self.text_cleaners = ["english_cleaners"]


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
    
    def __post_init__(self):
        if self.scheduler_params is None:
            self.scheduler_params = {}


@dataclass
class XTTSConfig:
    """Main configuration class combining all settings."""
    
    model: ModelConfig = None
    data: DataConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.training is None:
            self.training = TrainingConfig()
    
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