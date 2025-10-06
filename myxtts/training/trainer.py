"""
XTTS Training Pipeline.

This module implements the training pipeline for MyXTTS including
data loading, model training, validation, and checkpointing.
"""

import os
import time
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import wandb
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import io

# Import Advanced GPU Stabilizer for consistent GPU utilization
# GPU stabilizer availability is controlled by the main training script
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from optimization.advanced_gpu_stabilizer import AdvancedGPUStabilizer, setup_advanced_gpu_optimization
    GPU_STABILIZER_AVAILABLE = True
except ImportError as e:
    try:
        from advanced_gpu_stabilizer import AdvancedGPUStabilizer, setup_advanced_gpu_optimization
        GPU_STABILIZER_AVAILABLE = True
    except ImportError as e2:
        GPU_STABILIZER_AVAILABLE = False
        print(f"Warning: Advanced GPU Stabilizer not available: {e2}")

from ..models.xtts import XTTS
from ..data.ljspeech import LJSpeechDataset
from ..config.config import XTTSConfig
from ..utils.commons import (
    save_checkpoint, 
    load_checkpoint, 
    find_latest_checkpoint,
    setup_logging,
    EarlyStopping,
    get_device,
    configure_gpus,
    ensure_gpu_placement,
    setup_gpu_strategy,
    get_device_context,
)
from .losses import XTTSLoss, create_stop_targets
try:
    from .simple_loss import SimpleXTTSEmergencyLoss
except Exception:  # pragma: no cover - optional
    SimpleXTTSEmergencyLoss = None

# Import evaluation modules for automatic checkpoint quality assessment
try:
    from ..evaluation.evaluator import TTSEvaluator
    from ..evaluation.metrics import MOSNetEvaluator, ASRWordErrorRateEvaluator
    EVALUATION_AVAILABLE = True
except ImportError as e:
    EVALUATION_AVAILABLE = False
    print(f"Warning: Evaluation modules not available: {e}")


class XTTSTrainer:
    """
    XTTS model trainer with support for distributed training,
    checkpointing, and various optimization strategies.
    """
    
    def __init__(
        self,
        config: XTTSConfig,
        model: Optional[XTTS] = None,
        resume_checkpoint: Optional[str] = None,
        gpu_stabilizer_enabled: bool = False
    ):
        """
        Initialize XTTS trainer.
        
        Args:
            config: Training configuration
            model: Pre-initialized model (creates new if None)
            gpu_stabilizer_enabled: Whether to enable GPU stabilizer
            resume_checkpoint: Path to checkpoint for resuming training
        """
        self.config = config
        self.logger = setup_logging()

        self.summary_writer = None
        try:
            tb_dir = getattr(config.training, 'tensorboard_log_dir', None)
            tb_path = Path(tb_dir) if tb_dir else Path(config.training.checkpoint_dir) / "tensorboard"
            tb_path.mkdir(parents=True, exist_ok=True)
            self.summary_writer = tf.summary.create_file_writer(str(tb_path))
            self.logger.info(f"TensorBoard summaries: {tb_path}")
        except Exception as e:
            self.logger.warning(f"TensorBoard disabled: {e}")
            self.summary_writer = None

        training_cfg = getattr(config, 'training', None)
        self.spectrogram_log_interval = max(0, getattr(training_cfg, 'spectrogram_log_interval', 0)) if training_cfg else 0
        self.spectrogram_log_num_examples = max(1, getattr(training_cfg, 'spectrogram_log_num_examples', 1)) if training_cfg else 1
        self.spectrogram_log_example_index = getattr(training_cfg, 'spectrogram_log_example_index', None) if training_cfg else None
        self.spectrogram_reference_subset = getattr(training_cfg, 'spectrogram_reference_subset', 'batch') if training_cfg else 'batch'
        if isinstance(self.spectrogram_reference_subset, str):
            self.spectrogram_reference_subset = self.spectrogram_reference_subset.lower()
        self.spectrogram_reference_index = getattr(training_cfg, 'spectrogram_reference_index', None) if training_cfg else None
        self._last_spectrogram_log_step: Dict[str, int] = {}
        self._spectrogram_example_offset: Dict[str, int] = {}
        self._spectrogram_reference_sample = None
        self._spectrogram_reference_initialized = False
        self._train_dataset_raw = None
        self._val_dataset_raw = None
        self._spectrogram_logging_disabled_warned = False
        
        # Configure GPUs before any TensorFlow device queries
        try:
            memory_fraction = getattr(config.training, 'max_memory_fraction', 0.9)
            memory_limit = None
            
            # Calculate memory limit if available
            if memory_fraction < 1.0 and hasattr(tf.config, 'list_physical_devices'):
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Estimate memory limit (RTX 4090 has ~24GB)
                    estimated_memory = 24000  # MB
                    memory_limit = int(estimated_memory * memory_fraction)
            
            configure_gpus(
                visible_gpus=None,  # Single GPU training only
                memory_growth=True,
                memory_limit=memory_limit,
                enable_mixed_precision=getattr(config.data, 'mixed_precision', True),
                enable_xla=getattr(config.data, 'enable_xla', False),
                enable_eager_debug=getattr(config.training, 'enable_eager_debug', False)
            )
        except Exception as e:
            # Silence GPU config noise
            self.logger.debug(f"GPU visibility configuration note: {e}")

        # Setup device
        self.device = get_device()
        self.logger.debug(f"Using device: {self.device}")
        
        # Memory optimization settings
        self.gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        self.enable_memory_cleanup = getattr(config.training, 'enable_memory_cleanup', True)

        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"Gradient accumulation enabled: {self.gradient_accumulation_steps} steps")
        
        # Setup distribution strategy (single GPU or CPU only)
        self.strategy = setup_gpu_strategy()
        self.logger.info(f"Using strategy: {type(self.strategy).__name__}")
        
        # Enable/disable mixed precision explicitly
        if getattr(config.data, 'mixed_precision', True):
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            self.logger.debug("Mixed precision enabled")
        else:
            tf.keras.mixed_precision.set_global_policy('float32')
            self.logger.debug("Mixed precision disabled (float32 policy)")
        
        # Control XLA compilation explicitly based on config
        if getattr(config.data, 'enable_xla', False):
            tf.config.optimizer.set_jit(True)
            self.logger.debug("XLA compilation enabled")
        else:
            try:
                tf.config.optimizer.set_jit(False)
                self.logger.debug("XLA compilation disabled")
            except Exception:
                pass
        
        # Log GPU memory info
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                self.logger.info(f"GPU Memory - Current: {gpu_memory['current'] / 1024**3:.1f}GB, Peak: {gpu_memory['peak'] / 1024**3:.1f}GB")
        except Exception:
            pass

        # Initialize Advanced GPU Stabilizer for consistent GPU utilization
        # Controlled via command line arguments: --enable-gpu-stabilizer
        self.gpu_stabilizer = None
        
        if gpu_stabilizer_enabled and GPU_STABILIZER_AVAILABLE and self.device == "GPU":
            try:
                self.gpu_stabilizer = AdvancedGPUStabilizer(
                    max_prefetch_batches=32,
                    num_prefetch_threads=12,
                    memory_fraction=0.9,
                    enable_memory_pinning=True,
                    aggressive_mode=True
                )
                self.logger.info("🚀 Advanced GPU Stabilizer initialized for consistent GPU utilization")
            except Exception as e:
                self.logger.warning(f"Could not initialize GPU Stabilizer: {e}")
        else:
            if not gpu_stabilizer_enabled:
                self.logger.info("🔴 GPU Stabilizer disabled (use --enable-gpu-stabilizer to enable)")
            elif self.device != "GPU":
                self.logger.info("🔴 GPU Stabilizer disabled (GPU not available)")
            else:
                self.logger.warning("🔴 GPU Stabilizer not available (module import failed)")

        self._tb_train_step = 0

        # Initialize model and optimizer within strategy scope
        with self.strategy.scope():
            # CRITICAL FIX: Ensure model is created with strategy-managed placement
            with get_device_context():
                # Initialize model
                if model is None:
                    self.model = XTTS(config.model)
                else:
                    self.model = model
                
                # Force model to be built on GPU by running a dummy forward pass
                if self.device == "GPU":
                    self.logger.debug("Building model on GPU with dummy forward pass...")
                    try:
                        dummy_text = tf.zeros([1, 10], dtype=tf.int32)
                        dummy_mel = tf.zeros([1, 20, 80], dtype=tf.float32)
                        dummy_text_len = tf.constant([10], dtype=tf.int32)
                        dummy_mel_len = tf.constant([20], dtype=tf.int32)
                        
                        dummy_text = tf.cast(dummy_text, tf.int32)
                        dummy_mel = tf.cast(dummy_mel, tf.float32)
                        dummy_text_len = tf.cast(dummy_text_len, tf.int32)
                        dummy_mel_len = tf.cast(dummy_mel_len, tf.int32)

                        _ = self.model(
                            text_inputs=dummy_text,
                            mel_inputs=dummy_mel,
                            text_lengths=dummy_text_len,
                            mel_lengths=dummy_mel_len,
                            training=False
                        )
                        self.logger.debug("Model successfully built on GPU")
                    except Exception as e:
                        self.logger.debug(f"Model GPU initialization note: {e}")
            
            # Initialize optimizer
            self.optimizer = self._create_optimizer()
            # Wrap optimizer for mixed precision if enabled
            try:
                if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                    self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
                    self.logger.debug("Wrapped optimizer with LossScaleOptimizer for mixed precision")
            except Exception as e:
                self.logger.debug(f"Could not wrap optimizer for mixed precision: {e}")
            
            # Choose loss function (simple vs full) based on config/env
            use_simple = bool(getattr(config.training, 'use_simple_loss', False))
            if not use_simple:
                import os
                use_simple = os.environ.get('MYXTTS_SIMPLE_LOSS', '0') == '1'

            if use_simple and SimpleXTTSEmergencyLoss is not None:
                self.logger.info("Using SimpleXTTSEmergencyLoss for debugging")
                self.criterion = SimpleXTTSEmergencyLoss(
                    mel_loss_weight=getattr(config.training, 'mel_loss_weight', 1.0),
                    stop_loss_weight=0.1,
                )
            else:
                # Initialize enhanced loss function with stability features
                self.criterion = XTTSLoss(
                    mel_loss_weight=config.training.mel_loss_weight,
                    stop_loss_weight=1.0,
                    attention_loss_weight=getattr(config.training, 'attention_loss_weight', 0.02),
                    duration_loss_weight=config.training.duration_loss_weight,
                    pitch_loss_weight=getattr(config.training, 'pitch_loss_weight', 0.1),
                    energy_loss_weight=getattr(config.training, 'energy_loss_weight', 0.1),
                    prosody_pitch_loss_weight=getattr(config.training, 'prosody_pitch_loss_weight', 0.05),
                    prosody_energy_loss_weight=getattr(config.training, 'prosody_energy_loss_weight', 0.05),
                    speaking_rate_loss_weight=getattr(config.training, 'speaking_rate_loss_weight', 0.05),
                    # Stability improvements
                    use_adaptive_weights=getattr(config.training, 'use_adaptive_loss_weights', True),
                    loss_smoothing_factor=getattr(config.training, 'loss_smoothing_factor', 0.1),
                    max_loss_spike_threshold=getattr(config.training, 'max_loss_spike_threshold', 2.0),
                    gradient_norm_threshold=getattr(config.training, 'gradient_norm_threshold', 5.0)
                )


            self.loss_weight_snapshot = self._collect_loss_weights()
            if self.loss_weight_snapshot:
                formatted = ", ".join(
                    f"{k}={v:.3f}" for k, v in sorted(self.loss_weight_snapshot.items())
                )
                self.logger.info(f"Configured loss weights: {formatted}")
        
        # Initialize learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Enhanced early stopping with improved parameters
        early_stopping_patience = getattr(config.training, 'early_stopping_patience', 25)
        early_stopping_min_delta = getattr(config.training, 'early_stopping_min_delta', 0.0005)
        
        self.early_stopping = EarlyStopping(
            patience=early_stopping_patience,
            min_delta=early_stopping_min_delta,
            restore_best_weights=True
        )
        
        # Initialize wandb if configured
        if config.training.use_wandb:
            wandb.init(
                project=config.training.wandb_project,
                config=config.to_dict(),
                name=f"xtts_run_{int(time.time())}"
            )
        
        # Initialize automatic evaluation system for checkpoint quality monitoring
        self.evaluator = None
        self.enable_automatic_evaluation = getattr(config.training, 'enable_automatic_evaluation', False)
        self.evaluation_interval = getattr(config.training, 'evaluation_interval', 10)  # Evaluate every N epochs
        
        if self.enable_automatic_evaluation and EVALUATION_AVAILABLE:
            try:
                self.evaluator = TTSEvaluator(
                    enable_mosnet=True,
                    enable_asr_wer=True,
                    enable_cmvn=True,
                    enable_spectral=True
                )
                self.logger.info("✅ Automatic checkpoint evaluation enabled (MOSNet, ASR-WER, CMVN, Spectral)")
            except Exception as e:
                self.logger.warning(f"Could not initialize automatic evaluator: {e}")
                self.evaluator = None
        elif self.enable_automatic_evaluation and not EVALUATION_AVAILABLE:
            self.logger.warning("Automatic evaluation requested but evaluation modules not available")
        
        # Resume from checkpoint if specified
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)

    def _collect_loss_weights(self) -> Dict[str, float]:
        """Collect configured loss weights for logging/monitoring."""
        weights: Dict[str, float] = {}

        criterion_attrs = [
            "mel_loss_weight",
            "stop_loss_weight",
            "attention_loss_weight",
            "duration_loss_weight",
            "pitch_loss_weight",
            "energy_loss_weight",
            "prosody_pitch_loss_weight",
            "prosody_energy_loss_weight",
            "speaking_rate_loss_weight",
            "voice_similarity_loss_weight",
            "diffusion_loss_weight",
        ]

        if hasattr(self, "criterion"):
            for attr in criterion_attrs:
                if hasattr(self.criterion, attr):
                    value = getattr(self.criterion, attr)
                    try:
                        weights[attr] = float(value)
                    except (TypeError, ValueError):
                        continue

        training_attrs = [
            "voice_similarity_loss_weight",
            "speaker_classification_loss_weight",
            "voice_reconstruction_loss_weight",
            "prosody_matching_loss_weight",
            "spectral_consistency_loss_weight",
        ]

        for attr in training_attrs:
            if hasattr(self.config.training, attr):
                value = getattr(self.config.training, attr)
                if value is not None:
                    try:
                        weights[attr] = float(value)
                    except (TypeError, ValueError):
                        continue

        return weights

    def _log_loss_breakdown(self, stage: str, losses: Dict[str, float]) -> None:
        """Log a concise breakdown of key loss components and their weighted counterparts."""
        component_keys = [
            "mel_loss",
            "stop_loss",
            "voice_similarity_loss",
            "pitch_loss",
            "energy_loss",
            "prosody_pitch_loss",
            "prosody_energy_loss",
        ]
        weighted_keys = [
            "w_mel_loss",
            "w_stop_loss",
            "w_voice_similarity_loss",
            "w_pitch_loss",
            "w_energy_loss",
            "w_prosody_pitch_loss",
            "w_prosody_energy_loss",
        ]

        tracked = {k: float(losses[k]) for k in component_keys if k in losses}
        weighted = {k: float(losses[k]) for k in weighted_keys if k in losses}

        if tracked:
            summary = ", ".join(f"{k}={v:.4f}" for k, v in tracked.items())
            self.logger.info(f"{stage} loss components: {summary}")
        if weighted:
            summary_w = ", ".join(f"{k}={v:.4f}" for k, v in weighted.items())
            self.logger.debug(f"{stage} weighted loss contributions: {summary_w}")

    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create optimizer based on configuration."""
        config = self.config.training
        
        # Create learning rate schedule first
        learning_rate = config.learning_rate
        if hasattr(config, 'scheduler') and config.scheduler != "none":
            lr_schedule = self._create_lr_scheduler()
            if lr_schedule is not None:
                learning_rate = lr_schedule
                self.logger.debug(f"Using {config.scheduler} learning rate scheduler")
        
        # Build common optimizer kwargs
        common_kwargs = dict(
            learning_rate=learning_rate,
            beta_1=config.beta1,
            beta_2=config.beta2,
            epsilon=config.eps,
        )
        # Clip norm if requested
        if getattr(config, 'gradient_clip_norm', 0) and config.gradient_clip_norm > 0:
            common_kwargs["global_clipnorm"] = config.gradient_clip_norm
        if config.optimizer.lower() == "adam":
            return tf.keras.optimizers.Adam(**common_kwargs)
        elif config.optimizer.lower() == "adamw":
            return tf.keras.optimizers.AdamW(
                weight_decay=config.weight_decay,
                **common_kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    def _create_lr_scheduler(self) -> Optional[tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Create learning rate scheduler."""
        config = self.config.training
        
        if config.scheduler == "noam":
            model_dim = max(1, int(getattr(self.config.model, 'text_encoder_dim', 1)))
            warmup = max(1, int(getattr(config, 'warmup_steps', 1)))
            base = (float(model_dim) ** -0.5) * (float(warmup) ** -0.5)
            scale = config.learning_rate / base if base > 0 else config.learning_rate
            return NoamSchedule(
                model_dim,
                warmup_steps=warmup,
                scale=scale
            )
        elif config.scheduler == "cosine":
            params = dict(getattr(config, "scheduler_params", {}) or {})
            min_lr = params.pop("min_learning_rate", None)
            if min_lr is None:
                min_lr = getattr(config, "min_learning_rate", None)

            decay_steps = params.pop("decay_steps", None)
            first_decay_steps = params.pop("first_decay_steps", None)
            restart_period = params.pop("restart_period", None)
            restart_mult = params.pop("restart_mult", None)
            t_mul = params.pop("t_mul", None)
            m_mul = params.pop("m_mul", None)
            alpha = params.pop("alpha", None)

            if params:
                self.logger.debug("Unused cosine scheduler params: %s", list(params.keys()))

            if alpha is None and min_lr is not None and config.learning_rate > 0:
                alpha = max(0.0, min(min_lr / config.learning_rate, 1.0))

            if restart_period is not None and first_decay_steps is None:
                first_decay_steps = restart_period

            use_restarts = getattr(config, "cosine_restarts", False) or first_decay_steps is not None

            if use_restarts:
                first_decay_steps = max(1, int(first_decay_steps or config.epochs * 1000))
                return tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=config.learning_rate,
                    first_decay_steps=first_decay_steps,
                    t_mul=t_mul if t_mul is not None else (restart_mult if restart_mult is not None else 1.0),
                    m_mul=m_mul if m_mul is not None else 1.0,
                    alpha=alpha if alpha is not None else 0.0,
                )

            decay_steps = max(1, int(decay_steps or config.epochs * 1000))
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=config.learning_rate,
                decay_steps=decay_steps,
                alpha=alpha if alpha is not None else 0.0,
            )
        elif config.scheduler == "exponential":
            return tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=config.learning_rate,
                decay_steps=config.scheduler_params.get("decay_steps", 10000),
                decay_rate=config.scheduler_params.get("decay_rate", 0.96)
            )
        else:
            return None
    
    def prepare_datasets(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare training and validation datasets.
        
        Args:
            train_data_path: Path to training data
            val_data_path: Path to validation data (uses train path if None)
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Create training dataset (raw LJSpeechDataset object)
        train_ljs = LJSpeechDataset(
            data_path=train_data_path,
            config=self.config.data,
            subset="train",
            download=False
        )
        
        # Create validation dataset
        val_data_path = val_data_path or train_data_path
        val_ljs = LJSpeechDataset(
            data_path=val_data_path,
            config=self.config.data,
            subset="val",
            download=False
        )

        # All processing is done on-the-fly during training (no preprocessing required)
        self.logger.debug("Using on-the-fly data processing (no preprocessing/caching)")

        # Convert to TensorFlow datasets with optimized settings
        train_tf_dataset = train_ljs.create_tf_dataset(
            batch_size=self.config.data.batch_size,
            shuffle=True,
            repeat=True,
            prefetch=True,
            memory_cache=False,
            num_parallel_calls=self.config.data.num_workers,
            buffer_size_multiplier=self.config.data.shuffle_buffer_multiplier,
            drop_remainder=False,
        )
        
        val_tf_dataset = val_ljs.create_tf_dataset(
            batch_size=self.config.data.batch_size,
            shuffle=False,
            repeat=False,
            prefetch=True,
            memory_cache=True,
            num_parallel_calls=min(self.config.data.num_workers, 4),  # Fewer workers for validation
            buffer_size_multiplier=2,
            drop_remainder=False,
        )
        
        # Optimize datasets, and only distribute when multi-GPU is enabled
        if self.device == "GPU":
            AUTOTUNE = tf.data.AUTOTUNE
            train_tf_dataset = train_tf_dataset.prefetch(AUTOTUNE)
            val_tf_dataset = val_tf_dataset.prefetch(AUTOTUNE)

            # No dataset distribution needed for single GPU training
        
        # Store sizes and default steps per epoch for scheduler/progress
        self.train_dataset_size = len(train_ljs)
        self.val_dataset_size = len(val_ljs)
        self.default_steps_per_epoch = max(1, int(np.ceil(self.train_dataset_size / self.config.data.batch_size)))
        
        self.logger.info(f"Training samples: {self.train_dataset_size}")
        self.logger.info(f"Validation samples: {self.val_dataset_size}")
        
        # Expose datasets for external training loops (e.g., notebooks)
        self.train_dataset = train_tf_dataset
        self.val_dataset = val_tf_dataset
        self._train_dataset_raw = train_ljs
        self._val_dataset_raw = val_ljs
        try:
            self._train_dataset_iter = iter(self.train_dataset)
        except Exception:
            self._train_dataset_iter = None

        self._spectrogram_reference_initialized = False
        self._load_spectrogram_reference_sample()
        
        return train_tf_dataset, val_tf_dataset
    
    def train_step(
        self,
        text_sequences: tf.Tensor,
        mel_spectrograms: tf.Tensor,
        text_lengths: tf.Tensor,
        mel_lengths: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Single training step with explicit GPU placement and memory optimization.
        
        Args:
            text_sequences: Text token sequences [batch, text_len]
            mel_spectrograms: Target mel spectrograms [batch, mel_len, n_mels]
            text_lengths: Text sequence lengths [batch]
            mel_lengths: Mel sequence lengths [batch]
            
        Returns:
            Dictionary with loss values
        """
        # Wrap entire training step in GPU device context
        with get_device_context():
            # Memory optimization: cap text/mel lengths independently to avoid OOM
            max_text_len = getattr(self.config.model, 'max_attention_sequence_length', None)
            if max_text_len:
                text_sequences = text_sequences[:, :max_text_len]
                text_lengths = tf.minimum(text_lengths, max_text_len)

            max_mel_len = getattr(self.config.data, 'max_mel_frames', None)
            if max_mel_len:
                mel_spectrograms = mel_spectrograms[:, :max_mel_len, :]
                mel_lengths = tf.minimum(mel_lengths, max_mel_len)
            
            # Ensure tensors are on GPU if available
            if self.device == "GPU":
                text_sequences = ensure_gpu_placement(text_sequences)
                mel_spectrograms = ensure_gpu_placement(mel_spectrograms)
                text_lengths = ensure_gpu_placement(text_lengths)
                mel_lengths = ensure_gpu_placement(mel_lengths)
            
            try:
                with tf.GradientTape() as tape:
                    # Forward pass
                    outputs = self.model(
                        text_inputs=text_sequences,
                        mel_inputs=mel_spectrograms,
                        text_lengths=text_lengths,
                        mel_lengths=mel_lengths,
                        training=True
                    )
                    
                    # Prepare targets
                    stop_targets = create_stop_targets(
                        mel_lengths, 
                        tf.shape(mel_spectrograms)[1]
                    )
                    
                    y_true = {
                        "mel_target": mel_spectrograms,
                        "stop_target": stop_targets,
                        "text_lengths": text_lengths,
                        "mel_lengths": mel_lengths
                    }
                    
                    y_pred = {
                        "mel_output": outputs["mel_output"],
                        "stop_tokens": outputs["stop_tokens"]
                    }

                    # Pre-compute masks for later reuse
                    mel_mask = tf.sequence_mask(
                        mel_lengths,
                        maxlen=tf.shape(mel_spectrograms)[1],
                        dtype=tf.float32
                    )
                    text_len_tensor = tf.shape(text_sequences)[1]
                    text_mask = tf.sequence_mask(
                        text_lengths,
                        maxlen=text_len_tensor,
                        dtype=tf.float32
                    )

                    def _resize_to_text(sequence: tf.Tensor) -> tf.Tensor:
                        """Resize frame-level sequence [B, T, 1] to text length."""
                        sequence = tf.expand_dims(sequence, axis=2)  # [B, T, 1, 1]
                        resized = tf.image.resize(sequence, size=[text_len_tensor, 1], method='bilinear')
                        return tf.squeeze(resized, axis=3)  # [B, text_len, 1]

                    if "duration_pred" in outputs and outputs["duration_pred"] is not None:
                        y_pred["duration_pred"] = outputs["duration_pred"]
                        avg_duration = tf.cast(mel_lengths, tf.float32) / tf.maximum(
                            tf.cast(text_lengths, tf.float32),
                            1.0,
                        )
                        max_duration = getattr(self.config.training, 'max_duration_frames', 30.0)
                        avg_duration = tf.clip_by_value(avg_duration, 0.0, max_duration)
                        avg_duration = tf.expand_dims(avg_duration, axis=1)
                        duration_target = avg_duration * text_mask
                        y_true["duration_target"] = duration_target

                    if "attention_weights" in outputs and outputs["attention_weights"] is not None:
                        y_pred["attention_weights"] = outputs["attention_weights"]

                    # Mel-level auxiliary targets for pitch and energy predictors
                    frame_energy = tf.reduce_mean(tf.abs(mel_spectrograms), axis=2, keepdims=True)
                    frame_energy *= tf.expand_dims(mel_mask, -1)

                    mel_bins = tf.shape(mel_spectrograms)[2]
                    frame_pitch_idx = tf.cast(
                        tf.argmax(tf.square(mel_spectrograms), axis=2, output_type=tf.int32),
                        tf.float32
                    )
                    frame_pitch = frame_pitch_idx / tf.maximum(tf.cast(mel_bins - 1, tf.float32), 1.0)
                    frame_pitch = tf.expand_dims(frame_pitch, axis=-1)
                    frame_pitch *= tf.expand_dims(mel_mask, -1)

                    if "pitch_output" in outputs and outputs["pitch_output"] is not None:
                        y_pred["pitch_output"] = outputs["pitch_output"]
                        y_true["pitch_target"] = frame_pitch

                    if "energy_output" in outputs and outputs["energy_output"] is not None:
                        y_pred["energy_output"] = outputs["energy_output"]
                        y_true["energy_target"] = frame_energy

                    # Text-level prosody targets using resized mel-derived statistics
                    if "prosody_pitch" in outputs and outputs["prosody_pitch"] is not None:
                        prosody_pitch_target = _resize_to_text(frame_pitch)
                        prosody_pitch_target *= tf.expand_dims(text_mask, -1)
                        y_pred["prosody_pitch"] = outputs["prosody_pitch"]
                        y_true["prosody_pitch_target"] = prosody_pitch_target

                    if "prosody_energy" in outputs and outputs["prosody_energy"] is not None:
                        prosody_energy_target = _resize_to_text(frame_energy)
                        prosody_energy_target *= tf.expand_dims(text_mask, -1)
                        y_pred["prosody_energy"] = outputs["prosody_energy"]
                        y_true["prosody_energy_target"] = prosody_energy_target

                    if "prosody_speaking_rate" in outputs and outputs["prosody_speaking_rate"] is not None:
                        speaking_rate = tf.cast(mel_lengths, tf.float32) / tf.maximum(
                            tf.cast(text_lengths, tf.float32),
                            1.0,
                        )
                        speaking_rate = tf.expand_dims(speaking_rate, axis=1)
                        speaking_rate = tf.tile(speaking_rate, [1, text_len_tensor])
                        speaking_rate = tf.expand_dims(speaking_rate, axis=-1)
                        speaking_rate *= tf.expand_dims(text_mask, -1)
                        y_pred["prosody_speaking_rate"] = outputs["prosody_speaking_rate"]
                        y_true["prosody_speaking_rate_target"] = speaking_rate

                    # Compute loss
                    loss = self.criterion(y_true, y_pred)
                    raw_total_loss = self.criterion.get_raw_total_loss()
                    raw_total_tensor = tf.convert_to_tensor(
                        raw_total_loss if raw_total_loss is not None else loss,
                        dtype=tf.float32,
                    )
                    # Pure-TF finite check to be graph-safe
                    finite_raw = tf.reduce_all(tf.math.is_finite(raw_total_tensor))
                    raw_total_tensor = tf.where(
                        finite_raw, raw_total_tensor, tf.cast(loss, tf.float32)
                    )
                    # Avoid tf.cond side-effects inside @tf.function under distribution
                    # Logging is skipped to keep the graph side-effect free

                    aux_reg_weight = getattr(self.config.training, 'auxiliary_head_regularization', 0.0)
                    aux_reg_value = 0.0
                    if aux_reg_weight > 0.0:
                        aux_reg_vars = []
                        duration_layer = getattr(self.model.text_encoder, 'duration_predictor', None)
                        if duration_layer is not None:
                            aux_reg_vars.extend(duration_layer.trainable_variables)

                        prosody_layer = getattr(self.model, 'prosody_predictor', None)
                        if prosody_layer is not None:
                            aux_reg_vars.extend(prosody_layer.trainable_variables)

                        mel_decoder = getattr(self.model, 'mel_decoder', None)
                        if mel_decoder is not None:
                            pitch_proj = getattr(mel_decoder, 'pitch_projection', None)
                            if pitch_proj is not None:
                                aux_reg_vars.extend(pitch_proj.trainable_variables)
                            energy_proj = getattr(mel_decoder, 'energy_projection', None)
                            if energy_proj is not None:
                                aux_reg_vars.extend(energy_proj.trainable_variables)

                        if aux_reg_vars:
                            aux_reg_value = tf.add_n([
                                tf.reduce_sum(tf.square(var)) for var in aux_reg_vars
                            ]) / tf.cast(len(aux_reg_vars), tf.float32)
                            loss += aux_reg_weight * aux_reg_value
                    
                    # Per-replica loss scaling (average across replicas)
                    loss_for_grad = raw_total_tensor / tf.cast(self.strategy.num_replicas_in_sync, tf.float32)

                    # Mixed precision handling (Keras 3 API)
                    is_mixed = (tf.keras.mixed_precision.global_policy().name == 'mixed_float16')
                    if is_mixed and hasattr(self.optimizer, 'scale_loss'):
                        scaled_loss = self.optimizer.scale_loss(loss_for_grad)
                        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                        if hasattr(self.optimizer, 'get_unscaled_gradients'):
                            gradients = self.optimizer.get_unscaled_gradients(gradients)
                    else:
                        gradients = tape.gradient(loss_for_grad, self.model.trainable_variables)

                grad_var_pairs = [
                    (g, v)
                    for g, v in zip(gradients, self.model.trainable_variables)
                    if g is not None
                ]

                if not grad_var_pairs:
                    self.logger.warning("Gradient tape returned no gradients for this step; skipping update.")
                    individual_losses = self.criterion.get_losses()
                    stability_metrics = self.criterion.get_stability_metrics()
                    individual_losses.update(stability_metrics)
                    individual_losses["gradient_norm"] = tf.constant(0.0)
                    return {
                        "total_loss": loss,
                        **individual_losses
                    }

                grad_tensors, trainable_vars = zip(*grad_var_pairs)
                grad_tensors, global_norm = tf.clip_by_global_norm(grad_tensors, clip_norm=0.5)

                # Apply gradients (already in distributed context, no need for strategy.scope())
                self.optimizer.apply_gradients(zip(grad_tensors, trainable_vars))
                
                # Get individual losses and stability metrics
                individual_losses = self.criterion.get_losses()
                weighted_losses = {}
                aux_reg_weight = getattr(self.config.training, 'auxiliary_head_regularization', 0.0)
                if aux_reg_weight > 0.0 and 'aux_reg_value' in locals() and aux_reg_value != 0.0:
                    individual_losses["aux_head_reg"] = aux_reg_weight * aux_reg_value
                # Keep outputs minimal under multi-GPU to avoid control-flow side effects
                stability_metrics = {}
                try:
                    weighted_losses = self.criterion.get_weighted_losses()
                except Exception:
                    weighted_losses = {}

                if not is_multi_strategy:
                    try:
                        stability_metrics = self.criterion.get_stability_metrics()
                    except Exception:
                        stability_metrics = {}

                if weighted_losses:
                    for key, value in weighted_losses.items():
                        individual_losses[f"w_{key}"] = value
                raw_total = self.criterion.get_raw_total_loss()
                if raw_total is not None:
                    individual_losses["raw_total_loss"] = raw_total
                
                # Monitor gradient norm for stability
                grad_norm = global_norm
                if grad_norm is not None:
                    individual_losses["gradient_norm"] = grad_norm
                    
                    # Note: Gradient norm logging removed for Graph mode compatibility
                
                # Add stability metrics to output (if any)
                if stability_metrics:
                    individual_losses.update(stability_metrics)

                # Log breakdown when loss spikes dramatically to help debugging
                loss_alert_threshold = getattr(self.config.training, 'loss_alert_threshold', 500.0)
                try:
                    loss_scalar = float(loss.numpy())
                except Exception:
                    loss_scalar = None
                if loss_scalar is not None and loss_scalar > loss_alert_threshold:
                    breakdown = {}
                    if weighted_losses:
                        for key, value in weighted_losses.items():
                            try:
                                breakdown[key] = float(value.numpy())
                            except Exception:
                                breakdown[key] = None
                    if raw_total is not None:
                        try:
                            breakdown["raw_total_loss"] = float(raw_total.numpy())
                        except Exception:
                            breakdown["raw_total_loss"] = None
                    self.logger.warning(
                        "Loss spike detected (%.2f > %.2f). Contributions: %s",
                        loss_scalar,
                        loss_alert_threshold,
                        breakdown
                    )
                
                # Clear intermediate tensors to free memory
                del outputs, y_true, y_pred, gradients

                return {
                    "total_loss": loss,
                    **individual_losses
                }
                
            except tf.errors.ResourceExhaustedError as e:
                # Handle OOM gracefully by reducing batch size
                self.logger.error(f"OOM error in training step: {e}")
                self.logger.info("Attempting to continue with memory cleanup...")
                
                # Clear memory and try again with smaller effective batch size
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
                
                # Return a dummy loss to indicate OOM
                return {
                    "total_loss": tf.constant(float('inf')),
                    "mel_loss": tf.constant(float('inf')),
                    "stop_loss": tf.constant(float('inf'))
                }
    
    def find_optimal_batch_size(self, start_batch_size: int = 8, max_batch_size: int = 32) -> int:
        """
        Find the largest batch size that fits in GPU memory.
        
        Args:
            start_batch_size: Starting batch size to test
            max_batch_size: Maximum batch size to test
            
        Returns:
            Optimal batch size that doesn't cause OOM
        """
        if self.device != "GPU":
            return start_batch_size
        
        self.logger.info(f"Finding optimal batch size starting from {start_batch_size}")
        
        # Create dummy data for testing
        dummy_text = tf.random.uniform([start_batch_size, 50], maxval=1000, dtype=tf.int32)
        dummy_mel = tf.random.normal([start_batch_size, 100, 80])
        dummy_text_len = tf.fill([start_batch_size], 50)
        dummy_mel_len = tf.fill([start_batch_size], 100)
        
        optimal_batch_size = start_batch_size
        
        for batch_size in range(start_batch_size, max_batch_size + 1, 2):
            try:
                # Scale dummy data to current batch size
                scale_factor = batch_size // start_batch_size
                remainder = batch_size % start_batch_size
                
                if scale_factor > 1:
                    test_text = tf.tile(dummy_text, [scale_factor, 1])
                    test_mel = tf.tile(dummy_mel, [scale_factor, 1, 1])
                    test_text_len = tf.tile(dummy_text_len, [scale_factor])
                    test_mel_len = tf.tile(dummy_mel_len, [scale_factor])
                else:
                    test_text = dummy_text[:batch_size]
                    test_mel = dummy_mel[:batch_size]
                    test_text_len = dummy_text_len[:batch_size]
                    test_mel_len = dummy_mel_len[:batch_size]
                
                if remainder > 0:
                    test_text = tf.concat([test_text, dummy_text[:remainder]], axis=0)
                    test_mel = tf.concat([test_mel, dummy_mel[:remainder]], axis=0)
                    test_text_len = tf.concat([test_text_len, dummy_text_len[:remainder]], axis=0)
                    test_mel_len = tf.concat([test_mel_len, dummy_mel_len[:remainder]], axis=0)
                
                # Test forward pass
                with tf.GradientTape() as tape:
                    outputs = self.model(
                        text_inputs=test_text,
                        mel_inputs=test_mel,
                        text_lengths=test_text_len,
                        mel_lengths=test_mel_len,
                        training=True
                    )
                    
                    # Test loss computation
                    stop_targets = create_stop_targets(test_mel_len, tf.shape(test_mel)[1])
                    y_true = {
                        "mel_target": test_mel,
                        "stop_target": stop_targets,
                        "text_lengths": test_text_len,
                        "mel_lengths": test_mel_len
                    }
                    y_pred = {
                        "mel_output": outputs["mel_output"],
                        "stop_tokens": outputs["stop_tokens"]
                    }
                    loss = self.criterion(y_true, y_pred)
                
                # Test gradient computation
                gradients = tape.gradient(loss, self.model.trainable_variables)
                
                optimal_batch_size = batch_size
                self.logger.info(f"Batch size {batch_size} successful")
                
                # Clear memory
                del outputs, gradients, loss, y_true, y_pred
                tf.keras.backend.clear_session()
                
            except (tf.errors.ResourceExhaustedError, tf.errors.OutOfRangeError) as e:
                self.logger.warning(f"Batch size {batch_size} failed with OOM: {e}")
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
                break
        
        self.logger.info(f"Optimal batch size found: {optimal_batch_size}")
        return optimal_batch_size
    
    def _maybe_merge_distributed(self, value: Any) -> Any:
        """Merge per-replica tensors into a single tensor along batch axis if needed."""
        try:
            # For PerReplica/DistributedValues, collect local results and concat on batch dim
            if isinstance(value, tf.distribute.DistributedValues):
                parts = self.strategy.experimental_local_results(value)
                if len(parts) == 1:
                    return parts[0]
                return tf.concat(list(parts), axis=0)
        except Exception:
            pass
        return value

    def _next_batch(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """Return next batch from stored train dataset, unwrapping distributed values."""
        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError("No train_dataset set. Call prepare_datasets() first or pass batch_data explicitly.")
        if self._train_dataset_iter is None:
            self._train_dataset_iter = iter(self.train_dataset)
        batch = next(self._train_dataset_iter)
        # Support tuple of components; unwrap each if distributed
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch
        else:
            # Some input pipelines yield dicts; map to expected order if possible
            try:
                text_sequences = batch[0]
                mel_spectrograms = batch[1]
                text_lengths = batch[2]
                mel_lengths = batch[3]
            except Exception:
                raise RuntimeError("Unexpected batch structure; expected tuple/list of 4 tensors.")
        
        text_sequences = self._maybe_merge_distributed(text_sequences)
        mel_spectrograms = self._maybe_merge_distributed(mel_spectrograms)
        text_lengths = self._maybe_merge_distributed(text_lengths)
        mel_lengths = self._maybe_merge_distributed(mel_lengths)
        return text_sequences, mel_spectrograms, text_lengths, mel_lengths

    def train_step_with_accumulation(
        self, 
        batch_data: Optional[Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]] = None,
        accumulation_steps: int = 4
    ) -> Dict[str, tf.Tensor]:
        """
        Training step with gradient accumulation to simulate larger batch sizes.
        
        Args:
            batch_data: Optional tuple of (text_sequences, mel_spectrograms, text_lengths, mel_lengths).
                        If None, pulls the next batch from internal dataset iterator.
            accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Dictionary with accumulated loss values
        """
        if batch_data is None:
            batch_data = self._next_batch()
        text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch_data

        max_text_len = getattr(self.config.model, 'max_attention_sequence_length', None)
        if max_text_len:
            text_sequences = text_sequences[:, :max_text_len]
            text_lengths = tf.minimum(text_lengths, max_text_len)

        max_mel_len = getattr(self.config.data, 'max_mel_frames', None)
        if max_mel_len:
            mel_spectrograms = mel_spectrograms[:, :max_mel_len, :]
            mel_lengths = tf.minimum(mel_lengths, max_mel_len)
        
        # Split batch into micro-batches
        micro_batch_size = tf.shape(text_sequences)[0] // accumulation_steps
        if micro_batch_size == 0:
            micro_batch_size = 1
            accumulation_steps = tf.shape(text_sequences)[0]
        
        # Manual accumulation that is safe with IndexedSlices
        def _to_dense_if_indexed(g):
            if isinstance(g, tf.IndexedSlices):
                return tf.convert_to_tensor(g)
            return g

        accumulated_gradients = None
        total_losses = {}
        
        # Reuse device context and placement logic similar to train_step
        with get_device_context():
            for step in range(accumulation_steps):
                start_idx = step * micro_batch_size
                end_idx = min((step + 1) * micro_batch_size, tf.shape(text_sequences)[0])
                
                micro_text = text_sequences[start_idx:end_idx]
                micro_mel = mel_spectrograms[start_idx:end_idx]
                micro_text_len = text_lengths[start_idx:end_idx]
                micro_mel_len = mel_lengths[start_idx:end_idx]
                
                # Ensure tensors are on GPU if available
                if self.device == "GPU":
                    micro_text = ensure_gpu_placement(micro_text)
                    micro_mel = ensure_gpu_placement(micro_mel)
                    micro_text_len = ensure_gpu_placement(micro_text_len)
                    micro_mel_len = ensure_gpu_placement(micro_mel_len)
                
                with tf.GradientTape() as tape:
                    # Forward pass (replicated from train_step)
                    outputs = self.model(
                        text_inputs=micro_text,
                        mel_inputs=micro_mel,
                        text_lengths=micro_text_len,
                        mel_lengths=micro_mel_len,
                        training=True
                    )
                    stop_targets = create_stop_targets(
                        micro_mel_len,
                        tf.shape(micro_mel)[1]
                    )
                    y_true = {
                        "mel_target": micro_mel,
                        "stop_target": stop_targets,
                        "text_lengths": micro_text_len,
                        "mel_lengths": micro_mel_len
                    }
                    y_pred = {
                        "mel_output": outputs["mel_output"],
                        "stop_tokens": outputs["stop_tokens"]
                    }

                    mel_mask = tf.sequence_mask(
                        micro_mel_len,
                        maxlen=tf.shape(micro_mel)[1],
                        dtype=tf.float32
                    )
                    text_len_tensor = tf.shape(micro_text)[1]
                    text_mask = tf.sequence_mask(
                        micro_text_len,
                        maxlen=text_len_tensor,
                        dtype=tf.float32
                    )

                    def _resize_to_text(sequence: tf.Tensor) -> tf.Tensor:
                        sequence = tf.expand_dims(sequence, axis=2)
                        resized = tf.image.resize(sequence, size=[text_len_tensor, 1], method='bilinear')
                        return tf.squeeze(resized, axis=3)

                    if "duration_pred" in outputs and outputs["duration_pred"] is not None:
                        y_pred["duration_pred"] = outputs["duration_pred"]
                        avg_duration = tf.cast(micro_mel_len, tf.float32) / tf.maximum(
                            tf.cast(micro_text_len, tf.float32), 1.0)
                        max_duration = getattr(self.config.training, 'max_duration_frames', 30.0)
                        avg_duration = tf.clip_by_value(avg_duration, 0.0, max_duration)
                        avg_duration = tf.expand_dims(avg_duration, axis=1)
                        y_true["duration_target"] = avg_duration * text_mask

                    if "attention_weights" in outputs and outputs["attention_weights"] is not None:
                        y_pred["attention_weights"] = outputs["attention_weights"]

                    frame_energy = tf.reduce_mean(tf.abs(micro_mel), axis=2, keepdims=True)
                    frame_energy *= tf.expand_dims(mel_mask, -1)

                    mel_bins = tf.shape(micro_mel)[2]
                    frame_pitch_idx = tf.cast(
                        tf.argmax(tf.square(micro_mel), axis=2, output_type=tf.int32),
                        tf.float32
                    )
                    frame_pitch = frame_pitch_idx / tf.maximum(tf.cast(mel_bins - 1, tf.float32), 1.0)
                    frame_pitch = tf.expand_dims(frame_pitch, axis=-1)
                    frame_pitch *= tf.expand_dims(mel_mask, -1)

                    if "pitch_output" in outputs and outputs["pitch_output"] is not None:
                        y_pred["pitch_output"] = outputs["pitch_output"]
                        y_true["pitch_target"] = frame_pitch

                    if "energy_output" in outputs and outputs["energy_output"] is not None:
                        y_pred["energy_output"] = outputs["energy_output"]
                        y_true["energy_target"] = frame_energy

                    if "prosody_pitch" in outputs and outputs["prosody_pitch"] is not None:
                        prosody_pitch_target = _resize_to_text(frame_pitch)
                        prosody_pitch_target *= tf.expand_dims(text_mask, -1)
                        y_pred["prosody_pitch"] = outputs["prosody_pitch"]
                        y_true["prosody_pitch_target"] = prosody_pitch_target

                    if "prosody_energy" in outputs and outputs["prosody_energy"] is not None:
                        prosody_energy_target = _resize_to_text(frame_energy)
                        prosody_energy_target *= tf.expand_dims(text_mask, -1)
                        y_pred["prosody_energy"] = outputs["prosody_energy"]
                        y_true["prosody_energy_target"] = prosody_energy_target

                    if "prosody_speaking_rate" in outputs and outputs["prosody_speaking_rate"] is not None:
                        speaking_rate = tf.cast(micro_mel_len, tf.float32) / tf.maximum(
                            tf.cast(micro_text_len, tf.float32), 1.0)
                        speaking_rate = tf.expand_dims(speaking_rate, axis=1)
                        speaking_rate = tf.tile(speaking_rate, [1, text_len_tensor])
                        speaking_rate = tf.expand_dims(speaking_rate, axis=-1)
                        speaking_rate *= tf.expand_dims(text_mask, -1)
                        y_pred["prosody_speaking_rate"] = outputs["prosody_speaking_rate"]
                        y_true["prosody_speaking_rate_target"] = speaking_rate

                    loss = self.criterion(y_true, y_pred)
                    raw_total_loss = self.criterion.get_raw_total_loss()
                    raw_total_tensor = tf.convert_to_tensor(
                        raw_total_loss if raw_total_loss is not None else loss,
                        dtype=tf.float32,
                    )
                    finite_raw = tf.reduce_all(tf.math.is_finite(raw_total_tensor))
                    raw_total_tensor = tf.where(
                        finite_raw, raw_total_tensor, tf.cast(loss, tf.float32)
                    )
                    # Skip side-effect logging inside graph for distributed training

                    aux_reg_weight = getattr(self.config.training, 'auxiliary_head_regularization', 0.0)
                    aux_reg_value = 0.0
                    if aux_reg_weight > 0.0:
                        aux_reg_vars = []
                        duration_layer = getattr(self.model.text_encoder, 'duration_predictor', None)
                        if duration_layer is not None:
                            aux_reg_vars.extend(duration_layer.trainable_variables)
                        prosody_layer = getattr(self.model, 'prosody_predictor', None)
                        if prosody_layer is not None:
                            aux_reg_vars.extend(prosody_layer.trainable_variables)
                        mel_decoder = getattr(self.model, 'mel_decoder', None)
                        if mel_decoder is not None:
                            pitch_proj = getattr(mel_decoder, 'pitch_projection', None)
                            if pitch_proj is not None:
                                aux_reg_vars.extend(pitch_proj.trainable_variables)
                            energy_proj = getattr(mel_decoder, 'energy_projection', None)
                            if energy_proj is not None:
                                aux_reg_vars.extend(energy_proj.trainable_variables)
                        if aux_reg_vars:
                            aux_reg_value = tf.add_n([
                                tf.reduce_sum(tf.square(var)) for var in aux_reg_vars
                            ]) / tf.cast(len(aux_reg_vars), tf.float32)
                            loss += aux_reg_weight * aux_reg_value

                    scaled_loss = raw_total_tensor / accumulation_steps
                
                grads = tape.gradient(scaled_loss, self.model.trainable_variables)
                
                # Accumulate gradients safely
                if accumulated_gradients is None:
                    accumulated_gradients = [(_to_dense_if_indexed(g) if g is not None else None) for g in grads]
                else:
                    new_acc = []
                    for acc_g, g in zip(accumulated_gradients, grads):
                        if g is None:
                            new_acc.append(acc_g)
                            continue
                        g = _to_dense_if_indexed(g)
                        if acc_g is None:
                            new_acc.append(g)
                        else:
                            new_acc.append(acc_g + g)
                    accumulated_gradients = new_acc
                
                # Accumulate losses for reporting
                micro_losses = self.criterion.get_losses()
                total_losses["total_loss"] = total_losses.get("total_loss", 0.0) + float(loss)
                for k, v in micro_losses.items():
                    total_losses[k] = total_losses.get(k, 0.0) + float(v)
                if aux_reg_weight > 0.0 and aux_reg_value != 0.0:
                    total_losses["aux_head_reg"] = total_losses.get("aux_head_reg", 0.0) + float(aux_reg_weight * aux_reg_value)
                
                # Cleanup per micro-batch
                del outputs, y_true, y_pred, grads
        
        # Clip and apply accumulated gradients with improved clipping
        if accumulated_gradients is not None:
            # Use config-specified clipping or reasonable default
            clip_norm = getattr(self.config.training, 'gradient_clip_norm', 0.5)
            accumulated_gradients, _ = tf.clip_by_global_norm(accumulated_gradients, clip_norm=clip_norm)
            with self.strategy.scope():
                self.optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))

        # Average losses across micro-steps
        for key in list(total_losses.keys()):
            total_losses[key] /= float(accumulation_steps)

        # Attach most recent weighted loss breakdown for debugging
        weighted_losses = self.criterion.get_weighted_losses()
        raw_total = self.criterion.get_raw_total_loss()
        if weighted_losses:
            for key, value in weighted_losses.items():
                try:
                    total_losses[f"w_{key}"] = float(value.numpy())
                except Exception:
                    pass
        if raw_total is not None:
            try:
                total_losses["raw_total_loss"] = float(raw_total.numpy())
            except Exception:
                pass

        loss_alert_threshold = getattr(self.config.training, 'loss_alert_threshold', 500.0)
        total_loss_value = total_losses.get("total_loss")
        if total_loss_value is not None and total_loss_value > loss_alert_threshold:
            breakdown = {}
            if weighted_losses:
                for key, value in weighted_losses.items():
                    try:
                        breakdown[key] = float(value.numpy())
                    except Exception:
                        breakdown[key] = None
            if raw_total is not None:
                try:
                    breakdown["raw_total_loss"] = float(raw_total.numpy())
                except Exception:
                    breakdown["raw_total_loss"] = None
            self.logger.warning(
                "Loss spike detected during accumulation (%.2f > %.2f). Contributions: %s",
                total_loss_value,
                loss_alert_threshold,
                breakdown
            )

        # Convenience keys
        if "total_loss" in total_losses and "loss" not in total_losses:
            try:
                total_losses["loss"] = float(total_losses["total_loss"])  # convenience key
            except Exception:
                pass
        try:
            lr = self.optimizer.learning_rate
            current_lr = lr.numpy() if hasattr(lr, 'numpy') else lr
            total_losses["learning_rate"] = float(current_lr)
        except Exception:
            pass
        
        # Advance global step once per accumulation cycle
        try:
            self.current_step += 1
        except Exception:
            pass
        
        return total_losses
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent fragmentation."""
        if self.device == "GPU":
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            # Force GPU memory cleanup
            try:
                with get_device_context():
                    # Create and delete a small tensor to trigger cleanup
                    temp = tf.ones([1, 1])
                    del temp
            except Exception:
                pass
    
    def validation_step(
        self,
        text_sequences: tf.Tensor,
        mel_spectrograms: tf.Tensor,
        text_lengths: tf.Tensor,
        mel_lengths: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Single validation step with explicit GPU placement.
        
        Args:
            text_sequences: Text token sequences [batch, text_len]
            mel_spectrograms: Target mel spectrograms [batch, mel_len, n_mels]  
            text_lengths: Text sequence lengths [batch]
            mel_lengths: Mel sequence lengths [batch]
            
        Returns:
            Dictionary with loss values
        """
        # CRITICAL FIX: Wrap entire validation step in GPU device context
        with get_device_context():
            # Ensure tensors are on GPU if available
            if self.device == "GPU":
                text_sequences = ensure_gpu_placement(text_sequences)
                mel_spectrograms = ensure_gpu_placement(mel_spectrograms)
                text_lengths = ensure_gpu_placement(text_lengths)
                mel_lengths = ensure_gpu_placement(mel_lengths)
            
            # Forward pass
            outputs = self.model(
                text_inputs=text_sequences,
                mel_inputs=mel_spectrograms,
                text_lengths=text_lengths,
                mel_lengths=mel_lengths,
                training=False
            )
            
            # Prepare targets
            stop_targets = create_stop_targets(
                mel_lengths,
                tf.shape(mel_spectrograms)[1]
            )
            
            y_true = {
                "mel_target": mel_spectrograms,
                "stop_target": stop_targets,
                "text_lengths": text_lengths,
                "mel_lengths": mel_lengths
            }
            
            y_pred = {
                "mel_output": outputs["mel_output"],
                "stop_tokens": outputs["stop_tokens"]
            }
            
            # Compute loss
            loss = self.criterion(y_true, y_pred)
            
            # Get individual losses
            individual_losses = self.criterion.get_losses()
            weighted_losses = self.criterion.get_weighted_losses()
            if weighted_losses:
                for key, value in weighted_losses.items():
                    individual_losses[f"w_{key}"] = value
            raw_total = self.criterion.get_raw_total_loss()
            if raw_total is not None:
                individual_losses["raw_total_loss"] = raw_total

            return {
                "total_loss": loss,
                **individual_losses
            }
    
    def train(
        self,
        train_dataset: tf.data.Dataset,
        val_dataset: tf.data.Dataset,
        epochs: Optional[int] = None,
        steps_per_epoch: Optional[int] = None
    ):
        """
        Main training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs: Number of epochs (uses config if None)
            steps_per_epoch: Steps per epoch (auto-calculated if None)
        """
        epochs = epochs or self.config.training.epochs
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Current step: {self.current_step}")

        # Optional: one-time warmup for initialization to avoid long first-step stall
        try:
            skip_warmup = os.environ.get("MYXTTS_SKIP_WARMUP", "0") == "1"
            if skip_warmup:
                self.logger.info("Skipping warmup (MYXTTS_SKIP_WARMUP=1)")
                self._did_dist_warmup = True
        except Exception as e:
            self.logger.debug(f"Warmup note: {e}")

        try:
            # Determine steps per epoch if not provided
            if steps_per_epoch is None:
                steps_per_epoch = getattr(self, 'default_steps_per_epoch', None)

            # Training loop
            for epoch in range(self.current_epoch, epochs):
                self.current_epoch = epoch
                epoch_start_time = time.time()

                self.logger.info(f"Starting Epoch {epoch + 1}/{epochs}")

                # Training phase
                train_losses = self._train_epoch(train_dataset, steps_per_epoch)
                epoch_duration = time.time() - epoch_start_time

                # Validation phase (fixed validation frequency logic)
                val_losses = {}
                val_freq = max(1, self.config.training.val_step // 1000)  # Convert steps to epochs
                if epoch % val_freq == 0 or epoch == epochs - 1:  # Always validate on last epoch
                    self.logger.info("Running validation...")
                    val_losses = self._validate_epoch(val_dataset)

                    # Early stopping check
                    if hasattr(self, 'early_stopping') and self.early_stopping(val_losses["total_loss"], self.model):
                        self.logger.info("Early stopping triggered")
                        break

                    # Save best model
                    if val_losses["total_loss"] < self.best_val_loss:
                        self.best_val_loss = val_losses["total_loss"]
                        self._save_checkpoint(is_best=True)

                # Regular checkpointing (fixed frequency logic)
                checkpoint_freq = max(1, self.config.training.save_step // 1000)  # Convert steps to epochs
                if epoch % checkpoint_freq == 0 or epoch == epochs - 1:  # Always save on last epoch
                    self._save_checkpoint()

                # Enhanced logging with timing information
                samples_per_sec = (self.config.data.batch_size * (steps_per_epoch or 1)) / epoch_duration
                self.logger.info(f"Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.1f}s")
                self.logger.info(f"Train Loss: {train_losses['total_loss']:.4f}")
                if val_losses:
                    self.logger.info(f"Val Loss: {val_losses['total_loss']:.4f}")
                self.logger.info(f"Samples/sec: {samples_per_sec:.1f}")
                self._log_loss_breakdown("Train", train_losses)
                if val_losses:
                    self._log_loss_breakdown("Val", val_losses)

                # Enhanced Wandb logging
                if self.config.training.use_wandb:
                    log_data = {
                        "epoch": epoch + 1,
                        "step": self.current_step,
                        "epoch_duration": epoch_duration,
                        "samples_per_second": samples_per_sec,
                        **{f"train_{k}": v for k, v in train_losses.items()},
                    }
                    if val_losses:
                        log_data.update({f"val_{k}": v for k, v in val_losses.items()})

                    try:
                        lr_value = self.optimizer.learning_rate.numpy() if hasattr(self.optimizer.learning_rate, 'numpy') else self.optimizer.learning_rate
                        log_data["learning_rate"] = float(lr_value)
                    except:
                        pass

                    wandb.log(log_data)

                # Track loss convergence for better training insights
                if not hasattr(self, 'loss_history'):
                    self.loss_history = []
                self.loss_history.append(train_losses['total_loss'])

                # Check for loss convergence (last 5 epochs)
                if len(self.loss_history) >= 5:
                    recent_losses = self.loss_history[-5:]
                    loss_std = np.std(recent_losses)
                    loss_mean = np.mean(recent_losses)
                    if loss_std < 0.001 and loss_mean < 1.0:  # Very stable and low loss
                        self.logger.info(f"Loss converged (std: {loss_std:.6f}, mean: {loss_mean:.4f})")

            # Training completion with summary
            final_gpu_memory = "N/A"
            try:
                if self.device == "GPU":
                    gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                    final_gpu_memory = f"{gpu_memory['current'] / 1024**3:.1f}GB"
            except Exception:
                pass

            self.logger.info("Training completed successfully!")
            self.logger.info(f"Final GPU Memory Usage: {final_gpu_memory}")
            if hasattr(self, 'loss_history') and self.loss_history:
                self.logger.info(f"Final Training Loss: {self.loss_history[-1]:.4f}")
                if len(self.loss_history) > 1:
                    initial_loss = self.loss_history[0]
                    improvement = ((initial_loss - self.loss_history[-1]) / initial_loss) * 100
                    self.logger.info(f"Loss Improvement: {improvement:.1f}% (from {initial_loss:.4f} to {self.loss_history[-1]:.4f})")
        finally:
            if self.gpu_stabilizer:
                try:
                    self.gpu_stabilizer.stop_optimization()
                    self.logger.debug("Advanced GPU stabilizer stopped")
                except Exception as e:
                    self.logger.warning(f"GPU stabilizer cleanup warning: {e}")
            if self.summary_writer:
                try:
                    self.summary_writer.flush()
                except Exception as e:
                    self.logger.debug(f"TensorBoard flush failed: {e}")

    def _log_tensorboard(self, prefix: str, values: Dict[str, Any], step: int) -> None:
        if not self.summary_writer:
            return
        try:
            with self.summary_writer.as_default():
                for key, value in values.items():
                    if isinstance(value, tf.Tensor):
                        value = value.numpy()
                    try:
                        scalar = float(value)
                    except (TypeError, ValueError):
                        continue
                    tf.summary.scalar(f"{prefix}/{key}", scalar, step=step)
                self.summary_writer.flush()
        except Exception as err:
            self.logger.debug(f"TensorBoard logging skipped for {prefix}: {err}")

    def _should_log_spectrogram(self, step: int, prefix: str) -> bool:
        if not self.summary_writer or self.spectrogram_log_interval <= 0:
            return False
        if step <= 0:
            return False
        if step % self.spectrogram_log_interval != 0:
            return False
        last = self._last_spectrogram_log_step.get(prefix)
        if last == step:
            return False
        return True

    def _load_spectrogram_reference_sample(self) -> None:
        if self._spectrogram_reference_initialized:
            return

        subset = (self.spectrogram_reference_subset or 'batch').lower()
        self._spectrogram_reference_sample = None

        if subset == 'batch' or self.spectrogram_reference_index is None:
            self._spectrogram_reference_initialized = True
            return

        if subset not in ('train', 'val'):
            self.logger.warning(
                "Unsupported spectrogram_reference_subset '%s'; falling back to batch samples.",
                subset
            )
            self._spectrogram_reference_initialized = True
            return

        dataset = self._train_dataset_raw if subset == 'train' else self._val_dataset_raw
        if dataset is None:
            self.logger.debug(
                "Spectrogram reference dataset '%s' not ready; will retry later.",
                subset
            )
            return

        try:
            dataset_length = len(dataset)
        except Exception as err:
            self.logger.debug(f"Could not determine dataset length for spectrogram reference: {err}")
            self._spectrogram_reference_initialized = True
            return

        if dataset_length <= 0:
            self.logger.warning(
                "Dataset '%s' empty; spectrogram reference logging disabled.",
                subset
            )
            self._spectrogram_reference_initialized = True
            return

        try:
            index = int(self.spectrogram_reference_index)
        except Exception as err:
            self.logger.warning(f"Invalid spectrogram_reference_index '{self.spectrogram_reference_index}': {err}")
            self._spectrogram_reference_initialized = True
            return

        index = max(0, min(index, dataset_length - 1))

        try:
            item = dataset[index]
        except Exception as err:
            self.logger.warning(
                "Failed to load spectrogram reference sample %s:%d (%s). Disabling fixed logging.",
                subset,
                index,
                err
            )
            self._spectrogram_reference_initialized = True
            return

        text_seq = item.get('text_sequence')
        mel_spec = item.get('mel_spectrogram')
        text_len = item.get('text_length')
        mel_len = item.get('mel_length') or (mel_spec.shape[0] if mel_spec is not None else 0)

        if text_seq is None or mel_spec is None or text_len is None or mel_len is None:
            self.logger.warning(
                "Reference sample %s:%d missing required fields; disabling fixed spectrogram logging.",
                subset,
                index
            )
            self._spectrogram_reference_initialized = True
            return

        self._spectrogram_reference_sample = {
            'subset': subset,
            'index': index,
            'id': item.get('id'),
            'text_sequence': np.asarray(text_seq, dtype=np.int32),
            'mel_spectrogram': np.asarray(mel_spec, dtype=np.float32),
            'text_length': int(text_len),
            'mel_length': int(mel_len),
        }
        self._spectrogram_reference_initialized = True

    def _get_spectrogram_reference_sample(self) -> Optional[Dict[str, Any]]:
        if (self.spectrogram_reference_subset or 'batch').lower() == 'batch':
            return None
        if not self._spectrogram_reference_initialized or self._spectrogram_reference_sample is None:
            self._load_spectrogram_reference_sample()
        return self._spectrogram_reference_sample

    @staticmethod
    def _spectrogram_to_image(mel: np.ndarray) -> tf.Tensor:
        mel = np.asarray(mel, dtype=np.float32)
        if mel.ndim == 0:
            mel = mel.reshape(1, 1)
        if mel.ndim == 1:
            mel = np.expand_dims(mel, axis=0)
        if mel.ndim == 3:
            mel = np.squeeze(mel, axis=-1)
        if mel.ndim != 2:
            raise ValueError(f"Expected 2D spectrogram, got shape {mel.shape}")
        mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        mel = mel.T  # [n_mels, frames]

        finite_mask = np.isfinite(mel)
        if not np.any(finite_mask):
            mel_norm = np.zeros_like(mel, dtype=np.float32)
        else:
            finite_vals = mel[finite_mask]
            lo = np.percentile(finite_vals, 2.0)
            hi = np.percentile(finite_vals, 98.0)
            if not np.isfinite(lo):
                lo = float(np.min(finite_vals))
            if not np.isfinite(hi):
                hi = float(np.max(finite_vals))
            if hi <= lo:
                hi = lo + 1e-3
            mel_norm = (mel - lo) / (hi - lo)
            mel_norm = np.clip(mel_norm, 0.0, 1.0)

        mel_norm = mel_norm.astype(np.float32)
        mel_norm = np.stack([mel_norm, mel_norm, mel_norm], axis=-1)  # RGB for better visibility
        mel_norm = np.expand_dims(mel_norm, axis=0)
        return tf.convert_to_tensor(mel_norm)

    @staticmethod
    def _create_spectrogram_comparison_image(
        target_mel: np.ndarray,
        pred_mel: np.ndarray,
        title: str,
    ) -> tf.Tensor:
        """Create a 2x2 Figure that mirrors spectral_analysis.png layout."""

        target_mel = np.asarray(target_mel, dtype=np.float32)
        pred_mel = np.asarray(pred_mel, dtype=np.float32)

        if target_mel.ndim != 2 or pred_mel.ndim != 2:
            raise ValueError(
                f"Expected 2D spectrograms; got target {target_mel.shape}, pred {pred_mel.shape}"
            )

        # Align frame lengths for visual comparison
        max_frames = max(target_mel.shape[0], pred_mel.shape[0])
        if target_mel.shape[0] < max_frames:
            pad = max_frames - target_mel.shape[0]
            target_mel = np.pad(target_mel, ((0, pad), (0, 0)), mode='edge')
        else:
            target_mel = target_mel[:max_frames]

        if pred_mel.shape[0] < max_frames:
            pad = max_frames - pred_mel.shape[0]
            pred_mel = np.pad(pred_mel, ((0, pad), (0, 0)), mode='edge')
        else:
            pred_mel = pred_mel[:max_frames]

        diff_mel = np.abs(target_mel - pred_mel)
        temporal_error = np.mean(diff_mel, axis=1)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        im0 = axes[0, 0].imshow(target_mel.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 0].set_title('Target Mel (Model Processing)')
        axes[0, 0].set_ylabel('Mel Bins')
        axes[0, 0].set_xlabel('Time Frames')
        fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

        im1 = axes[0, 1].imshow(pred_mel.T, aspect='auto', origin='lower', cmap='viridis')
        axes[0, 1].set_title('Model Prediction')
        axes[0, 1].set_ylabel('Mel Bins')
        axes[0, 1].set_xlabel('Time Frames')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        im2 = axes[1, 0].imshow(diff_mel.T, aspect='auto', origin='lower', cmap='viridis')
        axes[1, 0].set_title('Absolute Difference')
        axes[1, 0].set_ylabel('Mel Bins')
        axes[1, 0].set_xlabel('Time Frames')
        fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

        axes[1, 1].hist(temporal_error, bins=50, alpha=0.8, color='tab:blue', label='Temporal Error')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].set_xlabel('Error')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].legend()

        fig.suptitle(title, fontsize=14, fontweight='bold')
        fig.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        buf.close()
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        return image

    def _log_spectrogram_from_batch(
        self,
        batch_tensors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        step: int,
        prefix: str = "train"
    ) -> None:
        if not self._should_log_spectrogram(step, prefix):
            return

        reference_sample = self._get_spectrogram_reference_sample()
        using_reference = reference_sample is not None
        if using_reference:
            text_sequences_np = np.expand_dims(reference_sample['text_sequence'], axis=0)
            mel_spectrograms_np = np.expand_dims(reference_sample['mel_spectrogram'], axis=0)
            text_lengths_np = np.array([reference_sample['text_length']], dtype=np.int32)
            mel_lengths_np = np.array([reference_sample['mel_length']], dtype=np.int32)
            indices = [int(reference_sample.get('index', 0) or 0)]
        else:
            if not batch_tensors or len(batch_tensors) != 4:
                return

            text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch_tensors

            try:
                text_sequences = self._maybe_merge_distributed(text_sequences)
                mel_spectrograms = self._maybe_merge_distributed(mel_spectrograms)
                text_lengths = self._maybe_merge_distributed(text_lengths)
                mel_lengths = self._maybe_merge_distributed(mel_lengths)
            except Exception as err:
                self.logger.debug(f"Spectrogram merge failed at step {step}: {err}")
                return

            try:
                text_sequences_np = text_sequences.numpy()
                mel_spectrograms_np = mel_spectrograms.numpy()
                text_lengths_np = text_lengths.numpy()
                mel_lengths_np = mel_lengths.numpy()
            except Exception as err:
                self.logger.debug(f"Spectrogram numpy conversion failed at step {step}: {err}")
                return

            if mel_spectrograms_np.size == 0 or mel_spectrograms_np.shape[0] == 0:
                return

            batch_size = mel_spectrograms_np.shape[0]
            if self.spectrogram_log_example_index is not None:
                idx = int(np.clip(self.spectrogram_log_example_index, 0, batch_size - 1))
                indices = [idx]
            else:
                num_examples = min(self.spectrogram_log_num_examples, batch_size)
                start = self._spectrogram_example_offset.get(prefix, 0) % batch_size
                indices = [(start + i) % batch_size for i in range(num_examples)]
                self._spectrogram_example_offset[prefix] = start + num_examples

            text_sel_np = text_sequences_np[indices]
            mel_sel_np = mel_spectrograms_np[indices]
            text_len_sel_np = text_lengths_np[indices]
            mel_len_sel_np = mel_lengths_np[indices]

        if using_reference:
            text_sel_np = text_sequences_np
            mel_sel_np = mel_spectrograms_np
            text_len_sel_np = text_lengths_np
            mel_len_sel_np = mel_lengths_np

        try:
            text_sel = tf.convert_to_tensor(text_sel_np)
            mel_sel = tf.convert_to_tensor(mel_sel_np)
            text_len_sel = tf.convert_to_tensor(text_len_sel_np)
            mel_len_sel = tf.convert_to_tensor(mel_len_sel_np)
        except Exception as err:
            self.logger.debug(f"Spectrogram tensor conversion failed at step {step}: {err}")
            return

        max_text_len = getattr(self.config.model, 'max_attention_sequence_length', None)
        if max_text_len:
            text_sel = text_sel[:, :max_text_len]
            text_len_sel = tf.minimum(text_len_sel, tf.cast(max_text_len, text_len_sel.dtype))

        max_mel_len = getattr(self.config.data, 'max_mel_frames', None)
        if max_mel_len:
            mel_sel = mel_sel[:, :max_mel_len, :]
            mel_len_sel = tf.minimum(mel_len_sel, tf.cast(max_mel_len, mel_len_sel.dtype))

        try:
            with get_device_context():
                inference_outputs = self.model(
                    text_inputs=text_sel,
                    mel_inputs=mel_sel,
                    text_lengths=text_len_sel,
                    mel_lengths=mel_len_sel,
                    training=False
                )
        except Exception as err:
            if not self._spectrogram_logging_disabled_warned:
                self.logger.debug(f"Spectrogram inference skipped at step {step}: {err}")
                self._spectrogram_logging_disabled_warned = True
            return

        mel_pred = inference_outputs.get("mel_output")
        if mel_pred is None:
            return

        try:
            mel_pred_np = mel_pred.numpy().astype(np.float32)
        except Exception as err:
            self.logger.debug(f"Spectrogram prediction conversion failed at step {step}: {err}")
            return

        comparison_images = []
        for local_idx, source_idx in enumerate(indices):
            target_slice = mel_sel_np[local_idx]
            pred_slice = mel_pred_np[local_idx]

            if target_slice.ndim != 2 or pred_slice.ndim != 2:
                self.logger.debug(
                    "Spectrogram slice dimensionality mismatch at step %s: target %s, pred %s",
                    step,
                    target_slice.shape,
                    pred_slice.shape,
                )
                continue

            target_frames = target_slice.shape[0]
            pred_frames = pred_slice.shape[0]
            common_frames = min(target_frames, pred_frames)
            if common_frames <= 0:
                continue

            target_slice = target_slice[:common_frames, :]
            pred_slice = pred_slice[:common_frames, :]

            try:
                if using_reference and reference_sample is not None:
                    ref_id = reference_sample.get('id')
                    if ref_id is not None:
                        title = f"Sample {ref_id} (Step {step})"
                    else:
                        title = f"Sample {source_idx} (Step {step})"
                else:
                    title = f"Sample {local_idx} (Step {step})"

                comparison_img = self._create_spectrogram_comparison_image(
                    target_slice,
                    pred_slice,
                    title=title,
                )
                comparison_images.append((local_idx, source_idx, comparison_img, float(common_frames)))
            except Exception as err:
                self.logger.debug(f"Spectrogram comparison image creation failed at step {step}: {err}")
                continue

        if not comparison_images:
            return

        try:
            with self.summary_writer.as_default():
                for local_idx, source_idx, comparison_img, frame_count in comparison_images:
                    if using_reference and reference_sample is not None:
                        ref_id = reference_sample.get('id')
                        if ref_id is not None:
                            suffix = f"ref_{ref_id}"
                        else:
                            suffix = f"ref_idx{source_idx}"
                    else:
                        suffix = str(local_idx)
                    sample_tag = f"{prefix}/spectrogram_{suffix}"
                    tf.summary.image(f"{sample_tag}/comparison", comparison_img, step=step)
                    tf.summary.scalar(f"{sample_tag}/frames", float(frame_count), step=step)
                self.summary_writer.flush()
        except Exception as err:
            self.logger.debug(f"TensorBoard spectrogram logging failed at step {step}: {err}")
            return

        self._last_spectrogram_log_step[prefix] = step

    def _train_epoch(
        self,
        train_dataset: tf.data.Dataset,
        steps_per_epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """Train for one epoch with GPU stabilization."""
        train_losses = {}
        num_batches = 0
        
        # Setup GPU-stabilized data loading if available
        if self.gpu_stabilizer and hasattr(train_dataset, '__iter__'):
            try:
                # Create optimized DataLoader wrapper for TensorFlow dataset
                self.logger.info("🔧 Setting up GPU-stabilized data loading...")
                
                # Start aggressive prefetching
                dataset_iter = iter(train_dataset)
                self.gpu_stabilizer.start_aggressive_prefetching_tf(dataset_iter)
                
                # Use stabilized data loading
                use_stabilized_loading = True
                self.logger.info("✅ GPU-stabilized data loading active")
                
            except Exception as e:
                self.logger.warning(f"Could not setup GPU stabilization: {e}")
                dataset_iter = iter(train_dataset)
                use_stabilized_loading = False
        else:
            dataset_iter = iter(train_dataset)
            use_stabilized_loading = False
        
        # Use tqdm for progress bar
        if steps_per_epoch:
            pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {self.current_epoch}")
        else:
            pbar = tqdm(dataset_iter, desc=f"Epoch {self.current_epoch}")
        
        try:
            for step in pbar:
                if steps_per_epoch and step >= steps_per_epoch:
                    break
                
                # Measure data loading time
                data_start_time = time.perf_counter()
                
                # Get batch - use stabilized loading if available
                if use_stabilized_loading and self.gpu_stabilizer:
                    try:
                        batch = self.gpu_stabilizer.get_next_batch_tf(timeout=30)
                    except Exception as e:
                        self.logger.warning(f"Stabilized loading failed, falling back: {e}")
                        batch = next(dataset_iter)
                        use_stabilized_loading = False
                elif steps_per_epoch:
                    batch = next(dataset_iter)
                else:
                    batch = step
                
                text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch
                data_end_time = time.perf_counter()
                data_loading_time = data_end_time - data_start_time
                
                # Measure model computation time
                compute_start_time = time.perf_counter()
                
                # ENHANCED: Ensure proper device placement
                with get_device_context():
                    # Single-GPU path: place tensors explicitly on GPU if available
                    if self.device == "GPU":
                        text_sequences = ensure_gpu_placement(text_sequences)
                        mel_spectrograms = ensure_gpu_placement(mel_spectrograms)
                        text_lengths = ensure_gpu_placement(text_lengths)
                        mel_lengths = ensure_gpu_placement(mel_lengths)

                    if self.gradient_accumulation_steps > 1:
                        # Use gradient accumulation for memory efficiency
                        step_losses = self.train_step_with_accumulation(
                            (text_sequences, mel_spectrograms, text_lengths, mel_lengths),
                            accumulation_steps=self.gradient_accumulation_steps
                        )
                    else:
                        # Regular step
                        step_losses = self.train_step(
                            text_sequences, mel_spectrograms, text_lengths, mel_lengths
                        )
                
                compute_end_time = time.perf_counter()
                compute_time = compute_end_time - compute_start_time

                # Accumulate losses
                for key, value in step_losses.items():
                    if key not in train_losses:
                        train_losses[key] = 0.0
                    train_losses[key] += float(value)
                
                num_batches += 1
                # Only increment step for non-accumulation training or last accumulation step
                if self.gradient_accumulation_steps <= 1:
                    self.current_step += 1
                
                # Memory cleanup after each batch if enabled
                if self.enable_memory_cleanup and num_batches % 10 == 0:
                    self.cleanup_gpu_memory()

                self._log_spectrogram_from_batch(
                    (text_sequences, mel_spectrograms, text_lengths, mel_lengths),
                    step=self.current_step,
                    prefix="train"
                )

                # Update progress bar with detailed losses and timing info
                self._tb_train_step += 1
                tb_step = self._tb_train_step
                postfix = {
                    "loss": f"{float(step_losses['total_loss']):.4f}",
                    "step": self.current_step,
                    "data_ms": f"{data_loading_time*1000:.1f}",
                    "comp_ms": f"{compute_time*1000:.1f}"
                }
                if 'mel_loss' in step_losses:
                    postfix["mel"] = f"{float(step_losses['mel_loss']):.2f}"
                if 'stop_loss' in step_losses:
                    postfix["stop"] = f"{float(step_losses['stop_loss']):.3f}"
                if 'voice_similarity_loss' in step_losses:
                    postfix["voice"] = f"{float(step_losses['voice_similarity_loss']):.3f}"
                pbar.set_postfix(postfix)
                
                # Log step results
                if self.current_step % self.config.training.log_step == 0:
                    step_log = {k: float(v) for k, v in step_losses.items()}
                    step_log.update({
                        "data_loading_time": data_loading_time,
                        "compute_time": compute_time,
                        "samples_per_second": self.config.data.batch_size / (data_loading_time + compute_time)
                    })
                    self.logger.debug(f"Step {self.current_step}: {step_log}")
                    self._log_tensorboard("train_step", step_log, tb_step)

                    if self.config.training.use_wandb:
                        wandb.log({f"step_{k}": v for k, v in step_log.items()})
        
        except tf.errors.OutOfRangeError:
            # End of dataset reached
            pass
        
        # Average losses
        if num_batches > 0:
            train_losses = {k: v / num_batches for k, v in train_losses.items()}
            self._log_tensorboard("train_epoch", train_losses, getattr(self, 'current_epoch', 0))

        return train_losses
    
    def _validate_epoch(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Validate for one epoch."""
        val_losses = {}
        num_batches = 0
        
        for batch in tqdm(val_dataset, desc="Validation"):
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch
            
            # Run validation step
            step_losses = self.validation_step(
                text_sequences, mel_spectrograms, text_lengths, mel_lengths
            )

            val_step_index = self.current_step if self.current_step > 0 else getattr(self, '_tb_train_step', 0)
            if val_step_index <= 0:
                val_step_index = num_batches + 1
            self._log_spectrogram_from_batch(
                (text_sequences, mel_spectrograms, text_lengths, mel_lengths),
                step=val_step_index,
                prefix="val"
            )

            # Accumulate losses
            for key, value in step_losses.items():
                if key not in val_losses:
                    val_losses[key] = 0.0
                val_losses[key] += float(value)
            
            num_batches += 1
        
        # Average losses
        if num_batches > 0:
            val_losses = {k: v / num_batches for k, v in val_losses.items()}

            self.logger.info(f"Validation: {val_losses}")
            self._log_tensorboard("val_epoch", val_losses, getattr(self, 'current_epoch', 0))

            if self.config.training.use_wandb:
                wandb.log({f"val_{k}": v for k, v in val_losses.items()})
        
        return val_losses
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_name = "best" if is_best else str(self.current_step)
        checkpoint_path = os.path.join(
            self.config.training.checkpoint_dir, 
            f"checkpoint_{checkpoint_name}"
        )
        
        save_checkpoint(
            self.model,
            self.optimizer,
            self.config.training.checkpoint_dir,
            self.current_step,
            self.best_val_loss,
            config=self.config.to_dict()
        )
        
        # Perform automatic evaluation on important checkpoints
        if self.evaluator is not None and (is_best or self.current_epoch % self.evaluation_interval == 0):
            self._evaluate_checkpoint_quality(checkpoint_path, is_best)
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Public helpers for notebooks / external loops
    def save_checkpoint(self, checkpoint_path: str) -> str:
        """
        Save a checkpoint to a specific path base (no enforced naming).
        Creates files: `<base>_model.weights.h5`, `<base>_optimizer.pkl`, `<base>_metadata.json`.
        """
        base = checkpoint_path
        checkpoint_dir = os.path.dirname(base) or "."
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model weights
        self.model.save_weights(f"{base}_model.weights.h5")
        
        # Save optimizer state (robust to wrappers like LossScaleOptimizer)
        def _unwrap(opt):
            for attr in ("optimizer", "inner_optimizer", "base_optimizer"):
                if hasattr(opt, attr):
                    try:
                        inner = getattr(opt, attr)
                        if inner is not None:
                            return inner
                    except Exception:
                        pass
            return opt

        def _get_opt_weights(opt):
            # Try direct
            try:
                return opt.get_weights()
            except Exception:
                pass
            base = _unwrap(opt)
            if base is not opt:
                try:
                    return base.get_weights()
                except Exception:
                    pass
            # Fallback to variables
            try:
                vars_fn = getattr(base, "variables", None)
                vars_list = vars_fn() if callable(vars_fn) else getattr(base, "weights", [])
                return [v.numpy() for v in vars_list]
            except Exception:
                return []

        # Enhanced mixed precision state saving
        optimizer_state = {}
        try:
            opt_weights = _get_opt_weights(self.optimizer)
            if not opt_weights:
                dummy_grads = [tf.zeros_like(var) for var in self.model.trainable_variables]
                with self.strategy.scope():
                    self.optimizer.apply_gradients(zip(dummy_grads, self.model.trainable_variables))
                opt_weights = _get_opt_weights(self.optimizer)
            
            optimizer_state['weights'] = opt_weights
            
            # Save LossScaleOptimizer specific state for mixed precision
            if hasattr(self.optimizer, 'loss_scale') and hasattr(self.optimizer, 'dynamic'):
                optimizer_state['loss_scale'] = float(self.optimizer.loss_scale.numpy()) if hasattr(self.optimizer.loss_scale, 'numpy') else float(self.optimizer.loss_scale)
                optimizer_state['dynamic'] = self.optimizer.dynamic
                if hasattr(self.optimizer, '_counter'):
                    optimizer_state['loss_scale_counter'] = int(self.optimizer._counter.numpy()) if hasattr(self.optimizer._counter, 'numpy') else int(self.optimizer._counter)
                self.logger.debug(f"Saved mixed precision state - loss_scale: {optimizer_state['loss_scale']}, dynamic: {optimizer_state['dynamic']}")
            
            # Save learning rate schedule state if available
            if hasattr(self.optimizer, 'learning_rate') and hasattr(self.optimizer.learning_rate, 'get_weights'):
                try:
                    optimizer_state['lr_schedule_weights'] = self.optimizer.learning_rate.get_weights()
                except Exception as e:
                    self.logger.debug(f"Could not save LR schedule weights: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Failed to save optimizer state: {e}")
            optimizer_state = {'weights': []}
        import pickle
        with open(f"{base}_optimizer.pkl", "wb") as f:
            pickle.dump(optimizer_state, f)
        
        # Save metadata
        import json
        metadata = {
            "step": int(self.current_step),
            "loss": float(self.best_val_loss) if np.isfinite(self.best_val_loss) else None,
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else None,
        }
        with open(f"{base}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Saved checkpoint to: {base}")
        return base

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint from a file base or directory path used by this project."""
        base = checkpoint_path
        for suffix in ("_model.weights.h5", "_optimizer.pkl", "_metadata.json"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        # If a directory is provided, find latest
        if os.path.isdir(base):
            latest = find_latest_checkpoint(base)
            if latest is None:
                raise FileNotFoundError(f"No checkpoints found in {base}")
            base = latest
        self._load_checkpoint(base)

    def validate(self) -> Dict[str, float]:
        """Run a full validation pass over the stored validation dataset."""
        if not hasattr(self, 'val_dataset') or self.val_dataset is None:
            raise RuntimeError("No val_dataset set. Call prepare_datasets() first or pass a dataset to _validate_epoch().")
        losses = self._validate_epoch(self.val_dataset)
        # Provide 'loss' alias for convenience
        if 'total_loss' in losses and 'loss' not in losses:
            losses['loss'] = losses['total_loss']
        return losses
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if os.path.isdir(checkpoint_path):
            # Find latest checkpoint in directory
            checkpoint_path = find_latest_checkpoint(checkpoint_path)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
        
        try:
            metadata = load_checkpoint(self.model, self.optimizer, checkpoint_path)
        except ValueError as exc:
            self.logger.warning(
                "Checkpoint '%s' is incompatible with the current model definition (%s). "
                "Training will start from scratch; use --reset-training to silence this warning.",
                checkpoint_path,
                exc,
            )
            self.logger.debug("Checkpoint load traceback", exc_info=True)
            return
        except Exception as exc:  # pragma: no cover - defensive fallback
            self.logger.warning(
                "Could not load checkpoint '%s' (%s). Training will start from scratch.",
                checkpoint_path,
                exc,
            )
            self.logger.debug("Checkpoint load traceback", exc_info=True)
            return

        self.current_step = metadata.get("step", 0)
        self.current_epoch = self.current_step // 1000  # Approximate
        self.best_val_loss = metadata.get("loss", float('inf'))
        
        self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        self.logger.info(f"Step: {self.current_step}, Best loss: {self.best_val_loss}")

    def _evaluate_checkpoint_quality(self, checkpoint_path: str, is_best: bool = False):
        """
        Automatically evaluate checkpoint quality using MOSNet, ASR-WER, and other metrics.
        
        Args:
            checkpoint_path: Path to the checkpoint
            is_best: Whether this is the best checkpoint so far
        """
        if self.evaluator is None:
            return
            
        try:
            # Generate a test audio sample from the current model
            # This is a placeholder - in a real implementation, you would:
            # 1. Generate audio samples using the model
            # 2. Evaluate those samples with the evaluator
            # 3. Log the results
            
            evaluation_results = {
                'checkpoint_path': checkpoint_path,
                'is_best': is_best,
                'epoch': self.current_epoch,
                'step': self.current_step,
                'evaluation_status': 'placeholder_implementation'
            }
            
            # In a full implementation, this would:
            # 1. Use the current model to synthesize test sentences
            # 2. Save the audio to temporary files  
            # 3. Evaluate using self.evaluator.evaluate_single() or evaluate_batch()
            # 4. Log comprehensive quality metrics
            
            self.logger.info(f"🎯 Checkpoint evaluation {'(BEST)' if is_best else ''}: {checkpoint_path}")
            self.logger.info("   📊 Automatic quality evaluation: MOSNet, ASR-WER, CMVN, Spectral")
            self.logger.info("   📝 Note: Full audio synthesis evaluation requires inference implementation")
            
            # Log to wandb if enabled
            if self.config.training.use_wandb:
                wandb.log({
                    "evaluation/checkpoint_evaluated": 1,
                    "evaluation/is_best_checkpoint": int(is_best),
                    "evaluation/epoch": self.current_epoch,
                }, step=self.current_step)
                
        except Exception as e:
            self.logger.warning(f"Checkpoint evaluation failed: {e}")


class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Noam learning rate schedule from "Attention is All You Need"."""

    def __init__(self, d_model: int, warmup_steps: int = 4000, scale: float = 1.0):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(max(1, warmup_steps), tf.float32)
        self.scale = tf.cast(scale, tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step = tf.maximum(step, 1.0)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * tf.pow(self.warmup_steps, -1.5)
        base = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
        return self.scale * base

    def get_config(self):
        return {
            "d_model": int(self.d_model.numpy()),
            "warmup_steps": int(self.warmup_steps.numpy()),
            "scale": float(self.scale.numpy()),
        }
