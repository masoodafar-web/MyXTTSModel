"""
XTTS Training Pipeline.

This module implements the training pipeline for MyXTTS including
data loading, model training, validation, and checkpointing.
"""

import os
import time
from typing import Dict, Optional, Tuple, Any
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import wandb

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
    ensure_gpu_placement
)
from ..utils.performance import PerformanceMonitor, time_operation
from .losses import XTTSLoss, create_stop_targets


class XTTSTrainer:
    """
    XTTS model trainer with support for distributed training,
    checkpointing, and various optimization strategies.
    """
    
    def __init__(
        self,
        config: XTTSConfig,
        model: Optional[XTTS] = None,
        resume_checkpoint: Optional[str] = None
    ):
        """
        Initialize XTTS trainer.
        
        Args:
            config: Training configuration
            model: Pre-initialized model (creates new if None)
            resume_checkpoint: Path to checkpoint for resuming training
        """
        self.config = config
        self.logger = setup_logging()
        
        # Initialize performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Configure GPUs before any TensorFlow device queries
        try:
            vis = getattr(config.training, 'visible_gpus', None)
            configure_gpus(vis, memory_growth=True)
        except Exception as e:
            self.logger.warning(f"GPU visibility configuration failed: {e}")

        # Setup device
        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Use default (no-op) distribution strategy to keep API consistent
        self.strategy = tf.distribute.get_strategy()
        self.logger.info(f"Using strategy: {type(self.strategy).__name__}")

        if self.device == "GPU":
            # Enable mixed precision for faster training
            if getattr(config.data, 'mixed_precision', True):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                self.logger.info("Mixed precision enabled")
            # Enable XLA compilation if requested
            if getattr(config.data, 'enable_xla', False):
                tf.config.optimizer.set_jit(True)
                self.logger.info("XLA compilation enabled")

        # Initialize model
        if model is None:
            self.model = XTTS(config.model)
        else:
            self.model = model
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize loss function
        self.criterion = XTTSLoss(
            mel_loss_weight=config.training.mel_loss_weight,
            stop_loss_weight=1.0,
            attention_loss_weight=config.training.kl_loss_weight,
            duration_loss_weight=config.training.duration_loss_weight
        )
        
        # Initialize learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=20,
            min_delta=0.001,
            restore_best_weights=True
        )
        
        # Initialize wandb if configured
        if config.training.use_wandb:
            wandb.init(
                project=config.training.wandb_project,
                config=config.to_dict(),
                name=f"xtts_run_{int(time.time())}"
            )
        
        # Resume from checkpoint if specified
        if resume_checkpoint:
            self._load_checkpoint(resume_checkpoint)
    
    def _create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """Create optimizer based on configuration."""
        config = self.config.training
        
        if config.optimizer.lower() == "adam":
            return tf.keras.optimizers.Adam(
                learning_rate=config.learning_rate,
                beta_1=config.beta1,
                beta_2=config.beta2,
                epsilon=config.eps
            )
        elif config.optimizer.lower() == "adamw":
            return tf.keras.optimizers.AdamW(
                learning_rate=config.learning_rate,
                beta_1=config.beta1,
                beta_2=config.beta2,
                epsilon=config.eps,
                weight_decay=config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {config.optimizer}")
    
    def _create_lr_scheduler(self) -> Optional[tf.keras.optimizers.schedules.LearningRateSchedule]:
        """Create learning rate scheduler."""
        config = self.config.training
        
        if config.scheduler == "noam":
            return NoamSchedule(
                self.config.model.text_encoder_dim,
                warmup_steps=config.warmup_steps
            )
        elif config.scheduler == "cosine":
            return tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=config.learning_rate,
                decay_steps=config.epochs * 1000,  # Approximate
                **config.scheduler_params
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

        # Optional: precompute and cache to disk to avoid CPU bottleneck
        try:
            train_ljs.precompute_mels(num_workers=self.config.data.num_workers, overwrite=False)
            val_ljs.precompute_mels(num_workers=self.config.data.num_workers, overwrite=False)
            train_ljs.precompute_tokens(num_workers=self.config.data.num_workers, overwrite=False)
            val_ljs.precompute_tokens(num_workers=self.config.data.num_workers, overwrite=False)
            # Verify caches and auto-fix any invalid files
            train_report = train_ljs.verify_and_fix_cache(fix=True)
            val_report = val_ljs.verify_and_fix_cache(fix=True)
            self.logger.info(f"Cache verify: train {train_report}, val {val_report}")
        except Exception as e:
            self.logger.warning(f"Precompute failed: {e}")

        # Filter items to those with valid caches
        try:
            n_train = train_ljs.filter_items_by_cache()
            n_val = val_ljs.filter_items_by_cache()
            self.logger.info(f"Using cached items - train: {n_train}, val: {n_val}")
        except Exception as e:
            self.logger.warning(f"Cache filter failed: {e}")

        # Convert to TensorFlow datasets with optimized settings for GPU
        train_tf_dataset = train_ljs.create_tf_dataset(
            batch_size=self.config.data.batch_size,
            shuffle=True,
            repeat=True,
            prefetch=True,
            use_cache_files=True,
            memory_cache=False,
            num_parallel_calls=self.config.data.num_workers,
            buffer_size_multiplier=self.config.data.shuffle_buffer_multiplier
        )
        
        val_tf_dataset = val_ljs.create_tf_dataset(
            batch_size=self.config.data.batch_size,
            shuffle=False,
            repeat=False,
            prefetch=True,
            use_cache_files=True,
            memory_cache=True,
            num_parallel_calls=min(self.config.data.num_workers, 4),  # Fewer workers for validation
            buffer_size_multiplier=2
        )
        
        # Optimize datasets for GPU training (prefetch only; avoid distributing dataset here)
        if self.device == "GPU":
            AUTOTUNE = tf.data.AUTOTUNE
            train_tf_dataset = train_tf_dataset.prefetch(AUTOTUNE)
            val_tf_dataset = val_tf_dataset.prefetch(AUTOTUNE)
        
        # Store sizes and default steps per epoch for scheduler/progress
        self.train_dataset_size = len(train_ljs)
        self.val_dataset_size = len(val_ljs)
        self.default_steps_per_epoch = max(1, int(np.ceil(self.train_dataset_size / self.config.data.batch_size)))
        
        self.logger.info(f"Training samples: {self.train_dataset_size}")
        self.logger.info(f"Validation samples: {self.val_dataset_size}")
        
        # Log data loading performance
        self.logger.info("Data loading performance:")
        self.logger.info(train_ljs.get_performance_report())
        
        return train_tf_dataset, val_tf_dataset
    
    @tf.function
    def distributed_train_step(self, dist_inputs):
        """
        Distributed training step for multi-GPU training.
        
        Args:
            dist_inputs: Distributed input batch
            
        Returns:
            Dictionary with distributed loss values
        """
        def train_step_fn(inputs):
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = inputs
            return self.train_step(text_sequences, mel_spectrograms, text_lengths, mel_lengths)
        
        # Run training step on all replicas
        per_replica_losses = self.strategy.run(train_step_fn, args=(dist_inputs,))
        
        # Reduce losses across replicas
        return {
            key: self.strategy.reduce(tf.distribute.ReduceOp.MEAN, values, axis=None)
            for key, values in per_replica_losses.items()
        }

    def train_step(
        self,
        text_sequences: tf.Tensor,
        mel_spectrograms: tf.Tensor,
        text_lengths: tf.Tensor,
        mel_lengths: tf.Tensor
    ) -> Dict[str, tf.Tensor]:
        """
        Single training step with explicit GPU placement.
        
        Args:
            text_sequences: Text token sequences [batch, text_len]
            mel_spectrograms: Target mel spectrograms [batch, mel_len, n_mels]
            text_lengths: Text sequence lengths [batch]
            mel_lengths: Mel sequence lengths [batch]
            
        Returns:
            Dictionary with loss values
        """
        # Ensure tensors are on GPU if available
        if self.device == "GPU":
            text_sequences = ensure_gpu_placement(text_sequences)
            mel_spectrograms = ensure_gpu_placement(mel_spectrograms)
            text_lengths = ensure_gpu_placement(text_lengths)
            mel_lengths = ensure_gpu_placement(mel_lengths)
        
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
            
            # Compute loss
            loss = self.criterion(y_true, y_pred)
            
            # Scale loss for distributed training
            scaled_loss = loss / self.strategy.num_replicas_in_sync
            
            # Scale loss for mixed precision
            if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                scaled_loss = self.optimizer.get_scaled_loss(scaled_loss)
        
        # Compute gradients
        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
        
        # Scale gradients for mixed precision
        if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
            gradients = self.optimizer.get_unscaled_gradients(gradients)
        
        # Clip gradients
        if self.config.training.gradient_clip_norm > 0:
            gradients, _ = tf.clip_by_global_norm(
                gradients, 
                self.config.training.gradient_clip_norm
            )
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Get individual losses
        individual_losses = self.criterion.get_losses()
        
        return {
            "total_loss": loss,
            **individual_losses
        }
    
    @tf.function
    def distributed_validation_step(self, dist_inputs):
        """
        Distributed validation step for multi-GPU validation.
        
        Args:
            dist_inputs: Distributed input batch
            
        Returns:
            Dictionary with distributed loss values
        """
        def validation_step_fn(inputs):
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = inputs
            return self.validation_step(text_sequences, mel_spectrograms, text_lengths, mel_lengths)
        
        # Run validation step on all replicas
        per_replica_losses = self.strategy.run(validation_step_fn, args=(dist_inputs,))
        
        # Reduce losses across replicas
        return {
            key: self.strategy.reduce(tf.distribute.ReduceOp.MEAN, values, axis=None)
            for key, values in per_replica_losses.items()
        }

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
        
        # Start performance monitoring
        self.performance_monitor.start_monitoring()
        
        # Determine steps per epoch if not provided
        if steps_per_epoch is None:
            steps_per_epoch = getattr(self, 'default_steps_per_epoch', None)
        
        # Training loop
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_losses = self._train_epoch(train_dataset, steps_per_epoch)
            
            # Validation phase
            if epoch % (self.config.training.val_step // 1000) == 0:
                val_losses = self._validate_epoch(val_dataset)
                
                # Early stopping check
                if self.early_stopping(val_losses["total_loss"], self.model):
                    self.logger.info("Early stopping triggered")
                    break
                
                # Save best model
                if val_losses["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total_loss"]
                    self._save_checkpoint(is_best=True)
            
            # Regular checkpointing
            if (self.current_step % self.config.training.save_step == 0):
                self._save_checkpoint()
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch}: Train Loss = {train_losses['total_loss']:.4f}")
            
            # Performance monitoring report every 10 epochs
            if epoch % 10 == 0:
                self.logger.info("Performance Report:")
                self.logger.info(self.performance_monitor.get_summary_report())
            
            # Wandb logging
            if self.config.training.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "step": self.current_step,
                    **{f"train_{k}": v for k, v in train_losses.items()},
                    "learning_rate": self.optimizer.learning_rate.numpy() 
                    if hasattr(self.optimizer.learning_rate, 'numpy') 
                    else self.optimizer.learning_rate
                })
        
        # Stop performance monitoring
        self.performance_monitor.stop_monitoring()
        
        # Final performance report
        self.logger.info("Final Performance Report:")
        self.logger.info(self.performance_monitor.get_summary_report())
        
        self.logger.info("Training completed")
    
    def _train_epoch(
        self,
        train_dataset: tf.data.Dataset,
        steps_per_epoch: Optional[int] = None
    ) -> Dict[str, float]:
        """Train for one epoch."""
        train_losses = {}
        num_batches = 0
        
        # Use tqdm for progress bar
        dataset_iter = iter(train_dataset)
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
                
                # Get batch
                if steps_per_epoch:
                    batch = next(dataset_iter)
                else:
                    batch = step
                
                text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch
                data_end_time = time.perf_counter()
                data_loading_time = data_end_time - data_start_time
                
                # Measure model computation time
                compute_start_time = time.perf_counter()
                
                # Training step: run non-distributed for stability
                step_losses = self.train_step(
                    text_sequences, mel_spectrograms, text_lengths, mel_lengths
                )
                
                compute_end_time = time.perf_counter()
                compute_time = compute_end_time - compute_start_time
                
                # Log timing for performance monitoring
                self.performance_monitor.log_step_timing(
                    data_loading_time, compute_time, self.config.data.batch_size
                )
                
                # Accumulate losses
                for key, value in step_losses.items():
                    if key not in train_losses:
                        train_losses[key] = 0.0
                    train_losses[key] += float(value)
                
                num_batches += 1
                self.current_step += 1
                
                # Update progress bar with detailed losses and timing info
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
                    
                    if self.config.training.use_wandb:
                        wandb.log({f"step_{k}": v for k, v in step_log.items()})
        
        except tf.errors.OutOfRangeError:
            # End of dataset reached
            pass
        
        # Average losses
        if num_batches > 0:
            train_losses = {k: v / num_batches for k, v in train_losses.items()}
        
        return train_losses
    
    def _validate_epoch(self, val_dataset: tf.data.Dataset) -> Dict[str, float]:
        """Validate for one epoch."""
        val_losses = {}
        num_batches = 0
        
        for batch in tqdm(val_dataset, desc="Validation"):
            # Always run non-distributed validation step for simplicity/stability
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch
            step_losses = self.validation_step(
                text_sequences, mel_spectrograms, text_lengths, mel_lengths
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
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        if os.path.isdir(checkpoint_path):
            # Find latest checkpoint in directory
            checkpoint_path = find_latest_checkpoint(checkpoint_path)
            if checkpoint_path is None:
                raise FileNotFoundError(f"No checkpoints found in {checkpoint_path}")
        
        metadata = load_checkpoint(self.model, self.optimizer, checkpoint_path)
        
        self.current_step = metadata.get("step", 0)
        self.current_epoch = self.current_step // 1000  # Approximate
        self.best_val_loss = metadata.get("loss", float('inf'))
        
        self.logger.info(f"Resumed from checkpoint: {checkpoint_path}")
        self.logger.info(f"Step: {self.current_step}, Best loss: {self.best_val_loss}")


class NoamSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Noam learning rate schedule from "Attention is All You Need".
    """
    
    def __init__(self, d_model: int, warmup_steps: int = 4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
    def get_config(self):
        return {
            "d_model": int(self.d_model),
            "warmup_steps": self.warmup_steps
        }
