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
    ensure_gpu_placement,
    setup_gpu_strategy
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
            memory_fraction = getattr(config.training, 'max_memory_fraction', 0.9)
            memory_limit = None
            
            # Calculate memory limit if available
            if memory_fraction < 1.0 and hasattr(tf.config, 'list_physical_devices'):
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    # Estimate memory limit (RTX 4090 has ~24GB)
                    estimated_memory = 24000  # MB
                    memory_limit = int(estimated_memory * memory_fraction)
            
            configure_gpus(vis, memory_growth=True, memory_limit=memory_limit)
        except Exception as e:
            self.logger.warning(f"GPU visibility configuration failed: {e}")

        # Setup device
        self.device = get_device()
        self.logger.info(f"Using device: {self.device}")
        
        # Memory optimization settings
        self.gradient_accumulation_steps = getattr(config.training, 'gradient_accumulation_steps', 1)
        self.enable_memory_cleanup = getattr(config.training, 'enable_memory_cleanup', True)
        
        if self.gradient_accumulation_steps > 1:
            self.logger.info(f"Gradient accumulation enabled: {self.gradient_accumulation_steps} steps")
        
        # Setup distribution strategy based on configuration
        self.strategy = setup_gpu_strategy(
            enable_multi_gpu=getattr(config.training, 'multi_gpu', False)
        )
        self.logger.info(f"Using strategy: {type(self.strategy).__name__}")

        if self.device == "GPU":
            # Enable mixed precision for faster training
            if getattr(config.data, 'mixed_precision', True):
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                self.logger.info("Mixed precision enabled")
            # Control XLA compilation explicitly based on config
            if getattr(config.data, 'enable_xla', False):
                tf.config.optimizer.set_jit(True)
                self.logger.info("XLA compilation enabled")
            else:
                try:
                    tf.config.optimizer.set_jit(False)
                    self.logger.info("XLA compilation disabled")
                except Exception:
                    pass

        # Initialize model and optimizer within strategy scope
        with self.strategy.scope():
            # CRITICAL FIX: Ensure model is created on GPU
            device_context = tf.device('/GPU:0') if self.device == "GPU" else tf.device('/CPU:0')
            with device_context:
                # Initialize model
                if model is None:
                    self.model = XTTS(config.model)
                else:
                    self.model = model
                
                # Force model to be built on GPU by running a dummy forward pass
                if self.device == "GPU":
                    self.logger.info("Building model on GPU with dummy forward pass...")
                    try:
                        dummy_text = tf.zeros([1, 10], dtype=tf.int32)
                        dummy_mel = tf.zeros([1, 20, 80], dtype=tf.float32)
                        dummy_text_len = tf.constant([10], dtype=tf.int32)
                        dummy_mel_len = tf.constant([20], dtype=tf.int32)
                        
                        with tf.device('/GPU:0'):
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
                            self.logger.info("✓ Model successfully built on GPU")
                    except Exception as e:
                        self.logger.warning(f"Model GPU initialization warning: {e}")
            
            # Initialize optimizer
            self.optimizer = self._create_optimizer()
            # Wrap optimizer for mixed precision if enabled
            try:
                if tf.keras.mixed_precision.global_policy().name == 'mixed_float16':
                    self.optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.optimizer)
                    self.logger.info("Wrapped optimizer with LossScaleOptimizer for mixed precision")
            except Exception as e:
                self.logger.warning(f"Could not wrap optimizer for mixed precision: {e}")
            
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
        
        # Build common optimizer kwargs
        common_kwargs = dict(
            learning_rate=config.learning_rate,
            beta_1=config.beta1,
            beta_2=config.beta2,
            epsilon=config.eps,
        )
        # Clip norm if requested
        if getattr(config, 'gradient_clip_norm', 0) and config.gradient_clip_norm > 0:
            common_kwargs["global_clipnorm"] = config.gradient_clip_norm
        # Only include gradient_accumulation_steps when >= 2 (Keras requirement)
        ga_steps = int(getattr(config, 'gradient_accumulation_steps', 1) or 1)
        if ga_steps >= 2:
            common_kwargs["gradient_accumulation_steps"] = ga_steps

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

        # Handle dataset preprocessing based on mode
        preprocessing_mode = getattr(self.config.data, 'preprocessing_mode', 'auto')
        self.logger.info(f"Dataset preprocessing mode: {preprocessing_mode}")
        
        if preprocessing_mode == "precompute":
            # Force complete preprocessing before training starts
            self.logger.info("Preprocessing mode: PRECOMPUTE - Ensuring all data is fully preprocessed...")
            try:
                train_ljs.precompute_mels(num_workers=self.config.data.num_workers, overwrite=False)
                val_ljs.precompute_mels(num_workers=self.config.data.num_workers, overwrite=False)
                train_ljs.precompute_tokens(num_workers=self.config.data.num_workers, overwrite=False)
                val_ljs.precompute_tokens(num_workers=self.config.data.num_workers, overwrite=False)
                
                # Verify and fix any cache issues
                train_report = train_ljs.verify_and_fix_cache(fix=True)
                val_report = val_ljs.verify_and_fix_cache(fix=True)
                self.logger.info(f"Cache verify: train {train_report}, val {val_report}")
                
                # Filter to only use items with valid caches
                n_train = train_ljs.filter_items_by_cache()
                n_val = val_ljs.filter_items_by_cache()
                
                if n_train == 0 or n_val == 0:
                    raise RuntimeError(f"No valid cached items found after preprocessing. Train: {n_train}, Val: {n_val}")
                
                self.logger.info(f"Using fully cached items - train: {n_train}, val: {n_val}")
                use_cache_files = True
                
            except Exception as e:
                self.logger.error(f"Precompute mode failed: {e}")
                raise RuntimeError(f"Preprocessing failed in precompute mode: {e}")
                
        elif preprocessing_mode == "runtime":
            # Disable preprocessing, do everything on-the-fly during training
            self.logger.info("Preprocessing mode: RUNTIME - Processing data on-the-fly during training")
            self.logger.warning("Runtime mode may impact GPU utilization due to CPU preprocessing during training")
            use_cache_files = False
            
        else:  # preprocessing_mode == "auto" or unrecognized mode
            # Current behavior: try to precompute but fall back gracefully
            self.logger.info("Preprocessing mode: AUTO - Attempting to precompute with graceful fallback")
            try:
                train_ljs.precompute_mels(num_workers=self.config.data.num_workers, overwrite=False)
                val_ljs.precompute_mels(num_workers=self.config.data.num_workers, overwrite=False)
                train_ljs.precompute_tokens(num_workers=self.config.data.num_workers, overwrite=False)
                val_ljs.precompute_tokens(num_workers=self.config.data.num_workers, overwrite=False)
                # Verify caches and auto-fix any invalid files
                train_report = train_ljs.verify_and_fix_cache(fix=True)
                val_report = val_ljs.verify_and_fix_cache(fix=True)
                self.logger.info(f"Cache verify: train {train_report}, val {val_report}")
                use_cache_files = True
            except Exception as e:
                self.logger.warning(f"Precompute failed: {e}")
                use_cache_files = False

            # Filter items to those with valid caches (only if using cache files)
            if use_cache_files:
                try:
                    n_train = train_ljs.filter_items_by_cache()
                    n_val = val_ljs.filter_items_by_cache()
                    self.logger.info(f"Using cached items - train: {n_train}, val: {n_val}")
                except Exception as e:
                    self.logger.warning(f"Cache filter failed: {e}")
                    use_cache_files = False

        # Convert to TensorFlow datasets with optimized settings for GPU
        train_tf_dataset = train_ljs.create_tf_dataset(
            batch_size=self.config.data.batch_size,
            shuffle=True,
            repeat=True,
            prefetch=True,
            use_cache_files=use_cache_files,
            memory_cache=False,
            num_parallel_calls=self.config.data.num_workers,
            buffer_size_multiplier=self.config.data.shuffle_buffer_multiplier
        )
        
        val_tf_dataset = val_ljs.create_tf_dataset(
            batch_size=self.config.data.batch_size,
            shuffle=False,
            repeat=False,
            prefetch=True,
            use_cache_files=use_cache_files,
            memory_cache=True,
            num_parallel_calls=min(self.config.data.num_workers, 4),  # Fewer workers for validation
            buffer_size_multiplier=2
        )
        
        # Optimize datasets for GPU training and distribute
        if self.device == "GPU":
            AUTOTUNE = tf.data.AUTOTUNE
            train_tf_dataset = train_tf_dataset.prefetch(AUTOTUNE)
            val_tf_dataset = val_tf_dataset.prefetch(AUTOTUNE)
            
            # CRITICAL FIX: Properly distribute datasets to GPU strategy
            try:
                train_tf_dataset = self.strategy.experimental_distribute_dataset(train_tf_dataset)
                val_tf_dataset = self.strategy.experimental_distribute_dataset(val_tf_dataset)
                self.logger.info("✓ Datasets distributed to GPU strategy")
            except Exception as e:
                self.logger.warning(f"Dataset distribution failed: {e}")
        
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
        Single training step with explicit GPU placement and memory optimization.
        
        Args:
            text_sequences: Text token sequences [batch, text_len]
            mel_spectrograms: Target mel spectrograms [batch, mel_len, n_mels]
            text_lengths: Text sequence lengths [batch]
            mel_lengths: Mel sequence lengths [batch]
            
        Returns:
            Dictionary with loss values
        """
        # CRITICAL FIX: Wrap entire training step in GPU device context
        device_context = tf.device('/GPU:0') if self.device == "GPU" else tf.device('/CPU:0')
        
        with device_context:
            # Memory optimization: limit sequence lengths to prevent OOM
            max_text_len = getattr(self.config.model, 'max_attention_sequence_length', 512)
            max_mel_len = getattr(self.config.model, 'max_attention_sequence_length', 512)
            
            # Truncate sequences if they're too long
            text_seq_len = tf.shape(text_sequences)[1]
            mel_seq_len = tf.shape(mel_spectrograms)[1]
            
            if text_seq_len > max_text_len:
                text_sequences = text_sequences[:, :max_text_len]
                text_lengths = tf.minimum(text_lengths, max_text_len)
            
            if mel_seq_len > max_mel_len:
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
                    
                    # Compute loss
                    loss = self.criterion(y_true, y_pred)
                    
                    # Per-replica loss scaling (average across replicas)
                    loss_for_grad = loss / self.strategy.num_replicas_in_sync

                    # Mixed precision handling (Keras 3 API)
                    is_mixed = (tf.keras.mixed_precision.global_policy().name == 'mixed_float16')
                    if is_mixed and hasattr(self.optimizer, 'scale_loss'):
                        scaled_loss = self.optimizer.scale_loss(loss_for_grad)
                        gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
                    else:
                        gradients = tape.gradient(loss_for_grad, self.model.trainable_variables)

                # Clip gradients to prevent memory explosion
                if gradients:
                    gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.5)  # Reduced clip norm

                # Apply gradients (LossScaleOptimizer will unscale internally if used)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                # Get individual losses
                individual_losses = self.criterion.get_losses()
                
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
    
    def train_step_with_accumulation(
        self, 
        batch_data: Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor],
        accumulation_steps: int = 4
    ) -> Dict[str, tf.Tensor]:
        """
        Training step with gradient accumulation to simulate larger batch sizes.
        
        Args:
            batch_data: Tuple of (text_sequences, mel_spectrograms, text_lengths, mel_lengths)
            accumulation_steps: Number of steps to accumulate gradients
            
        Returns:
            Dictionary with accumulated loss values
        """
        text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch_data
        
        # Split batch into micro-batches
        micro_batch_size = tf.shape(text_sequences)[0] // accumulation_steps
        if micro_batch_size == 0:
            micro_batch_size = 1
            accumulation_steps = tf.shape(text_sequences)[0]
        
        accumulated_gradients = None
        total_loss = 0.0
        total_losses = {}
        
        for step in range(accumulation_steps):
            start_idx = step * micro_batch_size
            end_idx = min((step + 1) * micro_batch_size, tf.shape(text_sequences)[0])
            
            # Get micro-batch
            micro_text = text_sequences[start_idx:end_idx]
            micro_mel = mel_spectrograms[start_idx:end_idx]
            micro_text_len = text_lengths[start_idx:end_idx]
            micro_mel_len = mel_lengths[start_idx:end_idx]
            
            # Compute gradients for micro-batch
            with tf.GradientTape() as tape:
                step_losses = self.train_step(micro_text, micro_mel, micro_text_len, micro_mel_len)
                scaled_loss = step_losses["total_loss"] / accumulation_steps
            
            gradients = tape.gradient(scaled_loss, self.model.trainable_variables)
            
            # Accumulate gradients
            if accumulated_gradients is None:
                accumulated_gradients = gradients
            else:
                accumulated_gradients = [
                    acc_grad + grad for acc_grad, grad in zip(accumulated_gradients, gradients)
                ]
            
            # Accumulate losses
            total_loss += float(step_losses["total_loss"])
            for key, value in step_losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += float(value)
        
        # Apply accumulated gradients
        if accumulated_gradients:
            # Clip gradients
            accumulated_gradients, _ = tf.clip_by_global_norm(accumulated_gradients, clip_norm=1.0)
            self.optimizer.apply_gradients(zip(accumulated_gradients, self.model.trainable_variables))
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= accumulation_steps
        
        return total_losses
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent fragmentation."""
        if self.device == "GPU":
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            # Force GPU memory cleanup
            try:
                with tf.device('/GPU:0'):
                    # Create and delete a small tensor to trigger cleanup
                    temp = tf.ones([1, 1])
                    del temp
            except Exception:
                pass
    
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
        # CRITICAL FIX: Wrap entire validation step in GPU device context
        device_context = tf.device('/GPU:0') if self.device == "GPU" else tf.device('/CPU:0')
        
        with device_context:
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
                
                # Training step: use gradient accumulation if configured
                if self.gradient_accumulation_steps > 1:
                    # Use gradient accumulation for memory efficiency
                    step_losses = self.train_step_with_accumulation(
                        (text_sequences, mel_spectrograms, text_lengths, mel_lengths),
                        accumulation_steps=self.gradient_accumulation_steps
                    )
                elif isinstance(self.strategy, (tf.distribute.MirroredStrategy, tf.distribute.OneDeviceStrategy)):
                    # Use distributed training step
                    step_losses = self.distributed_train_step((text_sequences, mel_spectrograms, text_lengths, mel_lengths))
                else:
                    # Use regular training step
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
                
                # Memory cleanup after each batch if enabled
                if self.enable_memory_cleanup and num_batches % 10 == 0:
                    self.cleanup_gpu_memory()
                
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
            text_sequences, mel_spectrograms, text_lengths, mel_lengths = batch
            
            # Use strategy-backed validation when strategy provides replica context
            if isinstance(self.strategy, (tf.distribute.MirroredStrategy, tf.distribute.OneDeviceStrategy)):
                step_losses = self.distributed_validation_step((text_sequences, mel_spectrograms, text_lengths, mel_lengths))
            else:
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
