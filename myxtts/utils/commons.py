"""
Common utilities for MyXTTS.

This module provides common functions for checkpointing, device management,
and other shared utilities across the MyXTTS system.
"""

import os
import json
import pickle
import tensorflow as tf
from typing import Dict, Any, Optional, List
import logging


def configure_gpus(visible_gpus: Optional[str] = None, memory_growth: bool = True, memory_limit: Optional[int] = None) -> None:
    """Configure which GPUs are visible and set memory growth.

    Call this before any other TensorFlow GPU operations.

    Args:
        visible_gpus: Comma-separated GPU indices to make visible (e.g., "0" or "0,1").
        memory_growth: Whether to enable memory growth on visible GPUs.
        memory_limit: Optional memory limit in MB for each GPU.
    """
    try:
        if visible_gpus is not None:
            # Map to actual physical devices
            all_gpus = tf.config.list_physical_devices('GPU')
            indices = [int(x.strip()) for x in visible_gpus.split(',') if x.strip() != '']
            selected = []
            for idx in indices:
                if idx < 0 or idx >= len(all_gpus):
                    raise ValueError(f"GPU index {idx} out of range (found {len(all_gpus)} GPUs)")
                selected.append(all_gpus[idx])
            tf.config.set_visible_devices(selected, 'GPU')
        
        # CRITICAL FIX: Better GPU configuration for utilization
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"Found {len(gpus)} GPU(s): {gpus}")
            
            # Configure memory growth and limits
            for gpu in gpus:
                try:
                    if memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"✓ Enabled memory growth for {gpu}")
                    
                    if memory_limit is not None:
                        tf.config.experimental.set_memory_limit(gpu, memory_limit)
                        print(f"✓ Set memory limit to {memory_limit}MB for {gpu}")
                        
                except Exception as e:
                    print(f"GPU memory configuration warning for {gpu}: {e}")
            
            # CRITICAL FIX: Set device policy to handle device placement issues
            try:
                # Use 'silent' policy to allow automatic tensor copying between devices
                # This fixes the InvalidArgumentError with dropout layer seed generators
                tf.config.experimental.set_device_policy('silent')
                print("✓ Set silent device policy to handle automatic device placement")
            except Exception as e:
                print(f"Device policy warning: {e}")
                
            # Set additional memory optimization settings
            try:
                # Enable mixed precision for memory efficiency
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("✓ Mixed precision policy enabled")
                
                # Enable XLA for better GPU utilization
                tf.config.optimizer.set_jit(True)
                print("✓ XLA JIT compilation enabled")
                
                # Force eager execution for better debugging (can disable in production)
                if not tf.executing_eagerly():
                    tf.config.run_functions_eagerly(True)
                    print("✓ Eager execution enabled for debugging")
                    
            except Exception as e:
                print(f"Advanced GPU optimization warning: {e}")
        else:
            print("⚠️ No GPUs detected - falling back to CPU")
                
    except Exception as e:
        print(f"GPU configuration warning: {e}")

def get_device() -> str:
    """
    Get the appropriate device for computation.
    
    Returns:
        Device string ("GPU" or "CPU")
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set up logical GPU devices for optimal utilization
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
            return "GPU"
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
            return "CPU"
    else:
        return "CPU"


def setup_gpu_strategy(enable_multi_gpu: bool = False):
    """
    Set up GPU distribution strategy for optimal GPU utilization.
    
    Args:
        enable_multi_gpu: Whether to enable MirroredStrategy for multi-GPU training.
                         If False, uses OneDeviceStrategy even with multiple GPUs.
    
    Returns:
        tf.distribute.Strategy for training
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) == 0:
        print("No GPUs available, using CPU strategy")
        return tf.distribute.get_strategy()  # Default strategy for CPU
    elif len(gpus) == 1:
        # Single GPU strategy with optimizations
        print("Using single GPU strategy")
        with tf.device('/GPU:0'):
            strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
        return strategy
    else:
        # Multi-GPU available - check if user wants to enable it
        if enable_multi_gpu:
            print(f"Using multi-GPU strategy with {len(gpus)} GPUs")
            strategy = tf.distribute.MirroredStrategy()
            return strategy
        else:
            # Use only first GPU even with multiple GPUs available
            print(f"Multi-GPU disabled, using single GPU strategy (GPU 0 of {len(gpus)} available)")
            with tf.device('/GPU:0'):
                strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
            return strategy


def ensure_gpu_placement(tensor):
    """
    Ensure a tensor is placed on GPU if available.
    
    Args:
        tensor: TensorFlow tensor
        
    Returns:
        Tensor placed on GPU if available, otherwise CPU
    """
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            # Use tf.cast to force tensor movement to GPU memory
            # tf.identity alone might not move the tensor
            if tensor.dtype == tf.float64:
                return tf.cast(tensor, tf.float32)  # Convert to float32 for GPU efficiency
            else:
                return tf.cast(tensor, tensor.dtype)  # Force movement to GPU
    return tensor


def get_device_context():
    """
    Get the appropriate device context for layer creation.
    
    Returns:
        Device context manager
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return tf.device('/GPU:0')
    else:
        return tf.device('/CPU:0')


def create_dropout_layer(rate: float, seed: Optional[int] = None, name: str = "dropout"):
    """
    Create a dropout layer with proper device placement.
    
    Args:
        rate: Dropout rate
        seed: Random seed for reproducibility
        name: Layer name
        
    Returns:
        Dropout layer with proper device context
    """
    # Ensure device placement is handled correctly for dropout layers
    with get_device_context():
        return tf.keras.layers.Dropout(rate, seed=seed, name=name)


def setup_mixed_precision():
    """Set up mixed precision training for better performance."""
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("Mixed precision training enabled")
    except Exception as e:
        print(f"Could not enable mixed precision: {e}")


def save_checkpoint(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    checkpoint_dir: str,
    step: int,
    loss: float,
    config: Optional[Dict[str, Any]] = None,
    max_checkpoints: int = 5
) -> str:
    """
    Save model checkpoint.
    
    Args:
        model: Keras model to save
        optimizer: Optimizer to save
        checkpoint_dir: Directory to save checkpoints
        step: Current training step
        loss: Current loss value
        config: Configuration dictionary
        max_checkpoints: Maximum number of checkpoints to keep
        
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}")
    
    # Save model weights
    model.save_weights(f"{checkpoint_path}_model.h5")
    
    # Save optimizer state
    optimizer_weights = optimizer.get_weights()
    with open(f"{checkpoint_path}_optimizer.pkl", "wb") as f:
        pickle.dump(optimizer_weights, f)
    
    # Save metadata
    metadata = {
        "step": step,
        "loss": loss,
        "model_config": model.get_config() if hasattr(model, 'get_config') else None,
        "optimizer_config": optimizer.get_config(),
    }
    
    if config is not None:
        metadata["config"] = config
    
    with open(f"{checkpoint_path}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Clean up old checkpoints
    cleanup_old_checkpoints(checkpoint_dir, max_checkpoints)
    
    print(f"Checkpoint saved at step {step}: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    checkpoint_path: str
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: Keras model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint (without extension)
        
    Returns:
        Dictionary with checkpoint metadata
    """
    # Load model weights
    model_path = f"{checkpoint_path}_model.h5"
    if os.path.exists(model_path):
        model.load_weights(model_path)
        print(f"Loaded model weights from {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load optimizer state
    optimizer_path = f"{checkpoint_path}_optimizer.pkl"
    if os.path.exists(optimizer_path):
        with open(optimizer_path, "rb") as f:
            optimizer_weights = pickle.load(f)
        
        # Build optimizer first if needed
        dummy_grads = [tf.zeros_like(var) for var in model.trainable_variables]
        optimizer.apply_gradients(zip(dummy_grads, model.trainable_variables))
        
        # Load optimizer weights
        optimizer.set_weights(optimizer_weights)
        print(f"Loaded optimizer state from {optimizer_path}")
    
    # Load metadata
    metadata_path = f"{checkpoint_path}_metadata.json"
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"Loaded metadata from {metadata_path}")
    
    return metadata


def cleanup_old_checkpoints(checkpoint_dir: str, max_checkpoints: int):
    """
    Remove old checkpoints, keeping only the most recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """
    if not os.path.exists(checkpoint_dir):
        return
    
    # Find all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith("_metadata.json"):
            step_str = file.replace("checkpoint_", "").replace("_metadata.json", "")
            try:
                step = int(step_str)
                checkpoint_files.append((step, file.replace("_metadata.json", "")))
            except ValueError:
                continue
    
    # Sort by step number and remove old ones
    checkpoint_files.sort(key=lambda x: x[0], reverse=True)
    
    for i, (step, checkpoint_base) in enumerate(checkpoint_files):
        if i >= max_checkpoints:
            # Remove checkpoint files
            for ext in ["_model.h5", "_optimizer.pkl", "_metadata.json"]:
                file_path = os.path.join(checkpoint_dir, checkpoint_base + ext)
                if os.path.exists(file_path):
                    os.remove(file_path)
            print(f"Removed old checkpoint: {checkpoint_base}")


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in a directory.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        
    Returns:
        Path to latest checkpoint (without extension) or None
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    latest_step = -1
    latest_checkpoint = None
    
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith("_metadata.json"):
            step_str = file.replace("checkpoint_", "").replace("_metadata.json", "")
            try:
                step = int(step_str)
                if step > latest_step:
                    latest_step = step
                    latest_checkpoint = os.path.join(checkpoint_dir, f"checkpoint_{step}")
            except ValueError:
                continue
    
    return latest_checkpoint


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        
    Returns:
        Logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    return logging.getLogger("MyXTTS")


def count_parameters(model: tf.keras.Model) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: Keras model
        
    Returns:
        Number of trainable parameters
    """
    return sum([tf.size(var).numpy() for var in model.trainable_variables])


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    tf.random.set_seed(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs with no improvement to wait
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(
        self, 
        val_loss: float, 
        model: tf.keras.Model
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.get_weights()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.set_weights(self.best_weights)
            return True
        
        return False
