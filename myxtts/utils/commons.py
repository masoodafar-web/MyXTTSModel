"""
Common utilities for MyXTTS.

This module provides common functions for checkpointing, device management,
and other shared utilities across the MyXTTS system.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from contextlib import nullcontext
from typing import Dict, Any, Optional, List
import logging
from logging.handlers import RotatingFileHandler


def configure_gpus(
    visible_gpus: Optional[str] = None,
    memory_growth: bool = True,
    memory_limit: Optional[int] = None,
    enable_mixed_precision: Optional[bool] = None,
    enable_xla: Optional[bool] = None,
    enable_eager_debug: bool = False
) -> None:
    """Configure which GPUs are visible and set memory growth.

    Call this before any other TensorFlow GPU operations.

    Args:
        visible_gpus: Comma-separated GPU indices to make visible (e.g., "0" or "0,1").
        memory_growth: Whether to enable memory growth on visible GPUs.
        memory_limit: Optional memory limit in MB for each GPU.
    """
    logger = logging.getLogger("MyXTTS")
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
            logger.debug(f"Found {len(gpus)} GPU(s): {gpus}")
            
            # Configure memory growth and limits
            for gpu in gpus:
                try:
                    if memory_growth:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.debug(f"Enabled memory growth for {gpu}")
                    
                    if memory_limit is not None:
                        tf.config.experimental.set_memory_limit(gpu, memory_limit)
                        logger.debug(f"Set memory limit to {memory_limit}MB for {gpu}")
                        
                except Exception as e:
                    logger.debug(f"GPU memory configuration note for {gpu}: {e}")
            
            # CRITICAL FIX: Set device policy to handle device placement issues
            try:
                # Use 'silent' policy to allow automatic tensor copying between devices
                # This fixes the InvalidArgumentError with dropout layer seed generators
                tf.config.experimental.set_device_policy('silent')
                logger.debug("Set silent device policy to handle automatic device placement")
            except Exception as e:
                logger.debug(f"Device policy note: {e}")
                
            # Set additional memory optimization settings
            try:
                if enable_mixed_precision is None:
                    env_flag = os.getenv('MYXTTS_ENABLE_MIXED_PRECISION')
                    enable_mixed_precision = (env_flag.lower() not in {"0", "false"}) if env_flag else True

                if enable_xla is None:
                    env_flag = os.getenv('MYXTTS_ENABLE_XLA')
                    enable_xla = (env_flag.lower() not in {"0", "false"}) if env_flag else True

                if enable_mixed_precision:
                    desired_policy = tf.keras.mixed_precision.Policy('mixed_float16')
                    if tf.keras.mixed_precision.global_policy().name != desired_policy.name:
                        tf.keras.mixed_precision.set_global_policy(desired_policy)
                        logger.debug("Mixed precision policy enabled")
                else:
                    if tf.keras.mixed_precision.global_policy().name != 'float32':
                        tf.keras.mixed_precision.set_global_policy('float32')
                        logger.debug("Mixed precision policy disabled (float32)")

                if enable_xla:
                    tf.config.optimizer.set_jit(True)
                    logger.debug("XLA JIT compilation enabled")
                else:
                    try:
                        tf.config.optimizer.set_jit(False)
                    except Exception:
                        pass
                    logger.debug("XLA JIT compilation disabled")

                if enable_eager_debug:
                    if not tf.config.functions_run_eagerly():
                        tf.config.run_functions_eagerly(True)
                    logger.debug("Eager execution enabled for debugging")
                else:
                    if tf.config.functions_run_eagerly() and os.getenv('MYXTTS_FORCE_EAGER', '').lower() not in {"1", "true"}:
                        tf.config.run_functions_eagerly(False)
                        logger.debug("Eager execution disabled for performance")

            except Exception as e:
                logger.debug(f"Advanced GPU optimization note: {e}")
        else:
            logger.debug("No GPUs detected - falling back to CPU")
                
    except Exception as e:
        logger.debug(f"GPU configuration note: {e}")

def check_gpu_setup():
    """
    Comprehensive GPU setup validation with actionable feedback.
    
    Returns:
        Tuple of (success: bool, device: str, recommendations: List[str])
    """
    logger = logging.getLogger("MyXTTS")
    recommendations = []
    
    # Check 1: NVIDIA driver
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            recommendations.append("Install NVIDIA GPU drivers (visit: https://www.nvidia.com/drivers)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        recommendations.append("NVIDIA drivers not found - install from https://www.nvidia.com/drivers")
    
    # Check 2: CUDA availability in TensorFlow
    if not tf.test.is_built_with_cuda():
        recommendations.append("TensorFlow not built with CUDA - install tensorflow[and-cuda] or tensorflow-gpu")
    
    # Check 3: GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        recommendations.append("No GPU devices detected by TensorFlow")
        recommendations.append("Verify: 1) GPU drivers installed, 2) CUDA toolkit installed, 3) TensorFlow-GPU installed")
    
    # Check 4: GPU functionality
    device = "CPU"
    try:
        if gpus:
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
                _ = result.numpy()  # Force execution
            device = "GPU"
            logger.info("✅ GPU setup validation passed")
    except Exception as e:
        recommendations.append(f"GPU computation failed: {str(e)}")
        recommendations.append("Try: nvidia-smi to check GPU status")
    
    success = device == "GPU" and len(recommendations) == 0
    return success, device, recommendations


def get_device() -> str:
    """
    Get the appropriate device for computation with detailed GPU availability checking.
    
    Returns:
        Device string ("GPU" or "CPU")
    """
    logger = logging.getLogger("MyXTTS")
    
    # Check for physical GPU devices
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        logger.warning("❌ No GPU devices detected")
        logger.info("📋 GPU Setup Required:")
        logger.info("   1. Install NVIDIA GPU drivers (version 450.80.02+)")
        logger.info("   2. Install CUDA toolkit (version 11.2+)")
        logger.info("   3. Install cuDNN (version 8.1+)")
        logger.info("   4. Verify with: nvidia-smi")
        logger.info("   5. Install TensorFlow-GPU: pip install tensorflow[and-cuda]")
        logger.info("🔄 Falling back to CPU mode (training will be much slower)")
        return "CPU"
    
    # Try to enable memory growth without treating failures as fatal. TensorFlow raises
    # RuntimeError once GPUs are initialized, but that doesn't mean the device is unusable.
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logger.debug(f"Memory growth already configured for {gpu}: {e}")
        except Exception as e:  # pragma: no cover - keep GPU probing resilient
            logger.warning(f"Could not adjust memory growth for {gpu}: {e}")

    # Test GPU functionality
    try:
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([1.0])
            result = test_tensor + 1
            _ = result.numpy()  # Force execution to verify GPU works

        logical_gpus = tf.config.list_logical_devices('GPU')
        logger.info("✅ GPU device available and functional")
        logger.info(f"   Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        logger.info(f"   Primary GPU: {gpus[0]}")
        return "GPU"

    except Exception as e:
        logger.error(f"❌ GPU device detected but not functional: {e}")
        logger.info("🔧 Common GPU Issues:")
        logger.info("   • CUDA driver version mismatch with TensorFlow")
        logger.info("   • Insufficient GPU memory available")
        logger.info("   • TensorFlow-GPU not properly installed")
        logger.info("   • GPU is being used by another process")
        logger.info("   • Virtual environment missing GPU libraries")
        logger.info("🔄 Falling back to CPU mode")
        return "CPU"


def setup_gpu_strategy():
    """
    Set up GPU strategy for single GPU or CPU training.
    
    Returns:
        tf.distribute.Strategy for training (always default strategy)
    """
    logger = logging.getLogger("MyXTTS")
    gpus = tf.config.list_physical_devices('GPU')
    
    if len(gpus) == 0:
        logger.debug("No GPUs available, using CPU strategy")
    else:
        logger.debug(f"GPU(s) detected, using default strategy for single GPU training")
    
    return tf.distribute.get_strategy()  # Default strategy for single GPU or CPU


def ensure_gpu_placement(tensor):
    """
    Ensure a tensor is placed on GPU when appropriate for single GPU training.
    """
    # Move to GPU:0 if available
    if tf.config.list_physical_devices('GPU'):
        with tf.device('/GPU:0'):
            if tensor.dtype == tf.float64:
                return tf.cast(tensor, tf.float32)
            return tf.cast(tensor, tensor.dtype)
    return tensor


def get_device_context():
    """
    Return an appropriate device context manager for ops/variable creation.
    
    Prefers GPU:0 when available, otherwise CPU:0.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        return tf.device('/GPU:0')
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
    logger = logging.getLogger("MyXTTS")
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.debug("Mixed precision training enabled")
    except Exception as e:
        logger.debug(f"Could not enable mixed precision: {e}")


def _unwrap_optimizer(optimizer):
    """Return the underlying/base optimizer if wrapped (e.g., LossScaleOptimizer)."""
    for attr in ("optimizer", "inner_optimizer", "base_optimizer"):
        if hasattr(optimizer, attr):
            try:
                inner = getattr(optimizer, attr)
                if inner is not None:
                    return inner
            except Exception:
                pass
    return optimizer


def _get_optimizer_weights(optimizer):
    """Best-effort retrieval of optimizer weights across Keras/TF variants.

    Tries get_weights(); if not available, unwraps known wrappers and falls back to variables().
    Returns a Python list (may be empty if unavailable)."""
    # Direct
    try:
        return optimizer.get_weights()
    except Exception:
        pass
    # Unwrap wrappers like LossScaleOptimizer
    base = _unwrap_optimizer(optimizer)
    if base is not optimizer:
        try:
            return base.get_weights()
        except Exception:
            pass
    # Fallback: variables list
    try:
        vars_fn = getattr(base, "variables", None)
        if callable(vars_fn):
            vars_list = vars_fn()
        else:
            vars_list = getattr(base, "weights", [])
        return [v.numpy() for v in vars_list]
    except Exception:
        return []


def _set_optimizer_weights(optimizer, weights) -> bool:
    """Best-effort setter for optimizer weights. Returns True on success."""
    # Direct
    try:
        optimizer.set_weights(weights)
        return True
    except Exception:
        pass
    # Unwrap
    base = _unwrap_optimizer(optimizer)
    if base is not optimizer:
        try:
            base.set_weights(weights)
            return True
        except Exception:
            pass
    # Fallback: assign to variables
    try:
        vars_fn = getattr(base, "variables", None)
        if callable(vars_fn):
            vars_list = vars_fn()
        else:
            vars_list = getattr(base, "weights", [])
        for v, w in zip(vars_list, weights):
            v.assign(w)
        return True
    except Exception:
        return False


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
    model.save_weights(f"{checkpoint_path}_model.weights.h5")
    
    # Save optimizer state (robust to wrappers like LossScaleOptimizer)
    try:
        optimizer_weights = _get_optimizer_weights(optimizer)
        # Ensure optimizer is built if empty
        if not optimizer_weights:
            dummy_grads = [tf.zeros_like(var) for var in model.trainable_variables]
            optimizer.apply_gradients(zip(dummy_grads, model.trainable_variables))
            optimizer_weights = _get_optimizer_weights(optimizer)
    except Exception:
        optimizer_weights = []
    with open(f"{checkpoint_path}_optimizer.pkl", "wb") as f:
        pickle.dump(optimizer_weights, f)
    
    # Save metadata
    metadata = {
        "step": step,
        "loss": loss,
        "model_config": model.get_config() if hasattr(model, 'get_config') else None,
        "optimizer_config": None,
    }
    try:
        metadata["optimizer_config"] = optimizer.get_config()
    except Exception:
        try:
            metadata["optimizer_config"] = _unwrap_optimizer(optimizer).get_config()
        except Exception:
            metadata["optimizer_config"] = None
    
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
    Load model checkpoint with enhanced mixed precision support.
    
    Args:
        model: Keras model to load weights into
        optimizer: Optimizer to load state into
        checkpoint_path: Path to checkpoint (without extension)
        
    Returns:
        Dictionary with checkpoint metadata
    """
    logger = logging.getLogger("MyXTTS.load_checkpoint")
    
    # Load model weights
    model_path = f"{checkpoint_path}_model.weights.h5"
    if os.path.exists(model_path):
        model.load_weights(model_path)
        logger.info(f"Loaded model weights from {model_path}")
        
        # Mark vocoder weights as initialized if model has a vocoder
        if hasattr(model, 'vocoder') and hasattr(model.vocoder, 'mark_weights_loaded'):
            model.vocoder.mark_weights_loaded()
            logger.debug("Marked vocoder weights as loaded from checkpoint")
    else:
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load optimizer state with enhanced mixed precision support
    optimizer_path = f"{checkpoint_path}_optimizer.pkl"
    if os.path.exists(optimizer_path):
        with open(optimizer_path, "rb") as f:
            optimizer_state = pickle.load(f)
        
        # Handle both old format (just weights) and new format (dict with mixed precision state)
        if isinstance(optimizer_state, dict):
            optimizer_weights = optimizer_state.get('weights', [])
        else:
            # Legacy format - just weights
            optimizer_weights = optimizer_state
            optimizer_state = {'weights': optimizer_weights}
        
        # Build optimizer first if needed
        try:
            dummy_grads = [tf.zeros_like(var) for var in model.trainable_variables]
            optimizer.apply_gradients(zip(dummy_grads, model.trainable_variables))
        except Exception:
            pass
        
        # Load optimizer weights (robust to wrappers)
        if optimizer_weights:
            if _set_optimizer_weights(optimizer, optimizer_weights):
                logger.info(f"Loaded optimizer state from {optimizer_path}")
            else:
                logger.warning("Failed to restore optimizer weights; continuing with fresh optimizer state")
        
        # Restore mixed precision state if available
        if hasattr(optimizer, 'loss_scale') and 'loss_scale' in optimizer_state:
            try:
                if hasattr(optimizer.loss_scale, 'assign'):
                    optimizer.loss_scale.assign(optimizer_state['loss_scale'])
                else:
                    optimizer._loss_scale = optimizer_state['loss_scale']
                logger.info(f"Restored mixed precision loss_scale: {optimizer_state['loss_scale']}")
                
                # Restore loss scale counter if available
                if 'loss_scale_counter' in optimizer_state and hasattr(optimizer, '_counter'):
                    if hasattr(optimizer._counter, 'assign'):
                        optimizer._counter.assign(optimizer_state['loss_scale_counter'])
                    else:
                        optimizer._counter = optimizer_state['loss_scale_counter']
                    logger.debug(f"Restored loss scale counter: {optimizer_state['loss_scale_counter']}")
                    
            except Exception as e:
                logger.warning(f"Could not restore mixed precision state: {e}")
        
        # Restore learning rate schedule state if available
        if 'lr_schedule_weights' in optimizer_state and hasattr(optimizer, 'learning_rate'):
            try:
                if hasattr(optimizer.learning_rate, 'set_weights'):
                    optimizer.learning_rate.set_weights(optimizer_state['lr_schedule_weights'])
                    logger.debug("Restored learning rate schedule state")
            except Exception as e:
                logger.debug(f"Could not restore LR schedule state: {e}")
    
    # Load metadata
    metadata_path = f"{checkpoint_path}_metadata.json"
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
    
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
            for ext in ["_model.weights.h5", "_optimizer.pkl", "_metadata.json"]:
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
    logger = logging.getLogger("MyXTTS")
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_dir = Path(os.environ.get("MYXTTS_LOG_DIR", Path.cwd() / "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        log_dir = Path.cwd()

    run_name = os.environ.get("MYXTTS_RUN_NAME") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}.log"

    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    os.environ["MYXTTS_LOG_FILE"] = str(log_file)
    logger.info(f"Logging to {log_file}")

    return logger


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


def auto_tune_performance_settings(config):
    """
    Automatically tune performance settings based on available hardware.
    
    Args:
        config: Configuration object to modify
    
    Returns:
        Modified configuration with optimized settings
    """
    import psutil
    
    logger = logging.getLogger("MyXTTS")
    
    try:
        # Get system information
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Get GPU information if available
        gpu_count = 0
        gpu_memory_gb = 0
        try:
            gpus = tf.config.list_physical_devices('GPU')
            gpu_count = len(gpus)
            if gpu_count > 0:
                # Try to get GPU memory info
                try:
                    gpu_memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    gpu_memory_gb = gpu_memory_info.get('current', 8*1024**3) / (1024**3)
                except:
                    gpu_memory_gb = 8  # Default assumption
        except:
            pass
        
        # Auto-tune settings
        if hasattr(config, 'data'):
            # Optimize batch size based on available memory
            if gpu_memory_gb >= 24:
                config.data.batch_size = min(64, max(32, config.data.batch_size))
            elif gpu_memory_gb >= 12:
                config.data.batch_size = min(48, max(24, config.data.batch_size))
            else:
                config.data.batch_size = min(32, max(16, config.data.batch_size))
            
            # Optimize number of workers based on CPU cores
            optimal_workers = min(20, max(4, cpu_count // 2))
            config.data.num_workers = optimal_workers
            
            # Optimize prefetch buffer based on memory and workers
            config.data.prefetch_buffer_size = min(16, max(4, optimal_workers // 2))
            
            # Enable memory optimizations for limited memory systems
            if memory_gb < 16:
                config.data.enable_memory_mapping = True
                config.data.persistent_workers = False
            
        logger.info(f"Auto-tuned performance settings: batch_size={getattr(config.data, 'batch_size', 'N/A')}, "
                   f"num_workers={getattr(config.data, 'num_workers', 'N/A')}, "
                   f"prefetch_buffer={getattr(config.data, 'prefetch_buffer_size', 'N/A')}")
        
    except Exception as e:
        logger.debug(f"Auto-tuning failed, using default settings: {e}")
    
    return config
