"""
GPU Memory Management Utilities for Memory-Isolated Multi-GPU Training.

This module provides utilities for setting up memory isolation between GPUs,
monitoring GPU memory usage, and managing memory limits for producer-consumer
pipeline patterns.
"""

import logging
from typing import Optional, Dict, List, Tuple
import tensorflow as tf

logger = logging.getLogger("MyXTTS")


def setup_gpu_memory_isolation(
    data_gpu_id: int,
    model_gpu_id: int,
    data_gpu_memory_limit: int = 8192,
    model_gpu_memory_limit: int = 16384
) -> bool:
    """
    Setup memory isolation between data processing GPU and model training GPU.
    
    This function configures TensorFlow to use separate memory limits for each GPU,
    enabling true producer-consumer pipeline pattern without memory conflicts.
    
    The function automatically detects and uses the appropriate TensorFlow API:
    - TensorFlow 2.10+: Uses set_virtual_device_configuration
    - TensorFlow < 2.10: Uses set_logical_device_configuration  
    - Fallback: Uses set_memory_growth if neither API is available
    
    Args:
        data_gpu_id: Physical GPU ID for data processing (0-indexed)
        model_gpu_id: Physical GPU ID for model training (0-indexed)
        data_gpu_memory_limit: Memory limit in MB for data GPU (default: 8192 = 8GB)
        model_gpu_memory_limit: Memory limit in MB for model GPU (default: 16384 = 16GB)
    
    Returns:
        bool: True if memory isolation was successfully configured with memory limits,
              False if fallback to memory growth was used or configuration failed
        
    Example:
        >>> setup_gpu_memory_isolation(data_gpu_id=0, model_gpu_id=1)
        >>> # GPU 0 limited to 8GB for data processing
        >>> # GPU 1 limited to 16GB for model training
    
    Note:
        This function must be called BEFORE any TensorFlow operations that initialize
        the GPUs, otherwise it will fail with a RuntimeError.
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) < 2:
            logger.error(f"❌ Memory isolation requires at least 2 GPUs, found {len(gpus)}")
            return False
        
        if data_gpu_id < 0 or data_gpu_id >= len(gpus):
            logger.error(f"❌ Invalid data_gpu_id={data_gpu_id}, must be 0-{len(gpus)-1}")
            return False
            
        if model_gpu_id < 0 or model_gpu_id >= len(gpus):
            logger.error(f"❌ Invalid model_gpu_id={model_gpu_id}, must be 0-{len(gpus)-1}")
            return False
        
        if data_gpu_id == model_gpu_id:
            logger.error(f"❌ data_gpu_id and model_gpu_id must be different")
            return False
        
        logger.info("🎯 Setting up GPU Memory Isolation...")
        logger.info(f"   Data GPU {data_gpu_id}: {data_gpu_memory_limit}MB limit")
        logger.info(f"   Model GPU {model_gpu_id}: {model_gpu_memory_limit}MB limit")
        
        # Try to use the new API (TensorFlow 2.10+) first, fallback to old API
        use_virtual_device_api = hasattr(tf.config.experimental, 'set_virtual_device_configuration')
        
        if use_virtual_device_api:
            logger.debug("   Using set_virtual_device_configuration (TensorFlow 2.10+)")
        else:
            logger.debug("   Using set_logical_device_configuration (TensorFlow < 2.10)")
        
        # Configure data GPU with memory limit
        try:
            if use_virtual_device_api:
                # New API for TensorFlow 2.10+
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[data_gpu_id],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=data_gpu_memory_limit
                    )]
                )
            else:
                # Old API for backward compatibility
                tf.config.experimental.set_logical_device_configuration(
                    gpus[data_gpu_id],
                    [tf.config.experimental.LogicalDeviceConfiguration(
                        memory_limit=data_gpu_memory_limit
                    )]
                )
            logger.info(f"   ✅ Data GPU memory limit set to {data_gpu_memory_limit}MB")
        except AttributeError as e:
            # API not available, fallback to memory growth only
            logger.warning(f"   ⚠️  Virtual/Logical device configuration API not available")
            logger.warning(f"      Falling back to memory growth only")
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except Exception as growth_error:
                    logger.debug(f"      Memory growth setup note: {growth_error}")
            logger.info(f"   ✅ Enabled memory growth for all GPUs as fallback")
            return False
        except RuntimeError as e:
            if "Virtual devices cannot be modified" in str(e) or "cannot be modified after being initialized" in str(e):
                logger.warning(f"   ⚠️  GPU already initialized, cannot set memory limit")
                logger.warning(f"      Ensure setup_gpu_memory_isolation() is called before any TF operations")
                return False
            raise
        
        # Configure model GPU with memory limit
        try:
            if use_virtual_device_api:
                # New API for TensorFlow 2.10+
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[model_gpu_id],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=model_gpu_memory_limit
                    )]
                )
            else:
                # Old API for backward compatibility
                tf.config.experimental.set_logical_device_configuration(
                    gpus[model_gpu_id],
                    [tf.config.experimental.LogicalDeviceConfiguration(
                        memory_limit=model_gpu_memory_limit
                    )]
                )
            logger.info(f"   ✅ Model GPU memory limit set to {model_gpu_memory_limit}MB")
        except AttributeError as e:
            # API not available, this shouldn't happen if data GPU succeeded
            logger.warning(f"   ⚠️  Unexpected: API became unavailable for model GPU")
            return False
        except RuntimeError as e:
            if "Virtual devices cannot be modified" in str(e) or "cannot be modified after being initialized" in str(e):
                logger.warning(f"   ⚠️  GPU already initialized, cannot set memory limit")
                return False
            raise
        
        # Set visible devices
        selected_gpus = [gpus[data_gpu_id], gpus[model_gpu_id]]
        tf.config.set_visible_devices(selected_gpus, 'GPU')
        logger.info(f"   ✅ Set visible devices: GPU {data_gpu_id} and GPU {model_gpu_id}")
        
        # Enable memory growth for safer allocation
        visible_gpus = tf.config.list_physical_devices('GPU')
        for i, gpu in enumerate(visible_gpus):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"   ✅ Enabled memory growth for visible GPU {i}")
            except Exception as e:
                logger.warning(f"   ⚠️  Could not enable memory growth for GPU {i}: {e}")
        
        logger.info("✅ GPU Memory Isolation configured successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to setup GPU memory isolation: {e}")
        return False


def get_gpu_memory_info(gpu_id: Optional[int] = None) -> Dict[str, any]:
    """
    Get memory information for specified GPU or all GPUs.
    
    Args:
        gpu_id: GPU ID to query (None = all GPUs)
    
    Returns:
        Dict with memory info including total, used, and free memory
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        
        if gpu_id is not None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                'gpu_id': gpu_id,
                'total_mb': mem_info.total // (1024 * 1024),
                'used_mb': mem_info.used // (1024 * 1024),
                'free_mb': mem_info.free // (1024 * 1024),
                'utilization_%': (mem_info.used / mem_info.total) * 100
            }
        else:
            device_count = pynvml.nvmlDeviceGetCount()
            gpu_infos = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_infos.append({
                    'gpu_id': i,
                    'total_mb': mem_info.total // (1024 * 1024),
                    'used_mb': mem_info.used // (1024 * 1024),
                    'free_mb': mem_info.free // (1024 * 1024),
                    'utilization_%': (mem_info.used / mem_info.total) * 100
                })
            return {'gpus': gpu_infos}
            
    except ImportError:
        logger.warning("pynvml not available, cannot get GPU memory info")
        return {}
    except Exception as e:
        logger.warning(f"Failed to get GPU memory info: {e}")
        return {}


def monitor_gpu_memory(data_gpu_id: int, model_gpu_id: int) -> Tuple[Dict, Dict]:
    """
    Monitor memory usage for both data and model GPUs.
    
    Args:
        data_gpu_id: Data processing GPU ID
        model_gpu_id: Model training GPU ID
    
    Returns:
        Tuple of (data_gpu_info, model_gpu_info) dictionaries
    """
    data_info = get_gpu_memory_info(data_gpu_id)
    model_info = get_gpu_memory_info(model_gpu_id)
    return data_info, model_info


def log_memory_stats(data_gpu_id: int, model_gpu_id: int, step: Optional[int] = None):
    """
    Log memory statistics for both GPUs.
    
    Args:
        data_gpu_id: Data processing GPU ID
        model_gpu_id: Model training GPU ID
        step: Optional training step number
    """
    data_info, model_info = monitor_gpu_memory(data_gpu_id, model_gpu_id)
    
    step_str = f"[Step {step}] " if step is not None else ""
    
    if data_info:
        logger.info(
            f"{step_str}Data GPU {data_gpu_id}: "
            f"{data_info['used_mb']}/{data_info['total_mb']}MB "
            f"({data_info['utilization_%']:.1f}%)"
        )
    
    if model_info:
        logger.info(
            f"{step_str}Model GPU {model_gpu_id}: "
            f"{model_info['used_mb']}/{model_info['total_mb']}MB "
            f"({model_info['utilization_%']:.1f}%)"
        )


def detect_memory_leak(
    gpu_id: int,
    baseline_mb: float,
    threshold_mb: float = 100
) -> bool:
    """
    Detect potential memory leak by comparing current memory usage to baseline.
    
    Args:
        gpu_id: GPU ID to monitor
        baseline_mb: Baseline memory usage in MB
        threshold_mb: Threshold for leak detection (default: 100MB)
    
    Returns:
        bool: True if potential memory leak detected
    """
    info = get_gpu_memory_info(gpu_id)
    if not info:
        return False
    
    current_mb = info['used_mb']
    diff = current_mb - baseline_mb
    
    if diff > threshold_mb:
        logger.warning(
            f"⚠️  Potential memory leak detected on GPU {gpu_id}: "
            f"{diff:.1f}MB increase from baseline"
        )
        return True
    
    return False


def get_optimal_memory_limits(
    data_gpu_id: int,
    model_gpu_id: int,
    data_fraction: float = 0.33,
    model_fraction: float = 0.67
) -> Tuple[int, int]:
    """
    Calculate optimal memory limits based on available GPU memory.
    
    Args:
        data_gpu_id: Data processing GPU ID
        model_gpu_id: Model training GPU ID
        data_fraction: Fraction of total memory for data GPU (default: 0.33)
        model_fraction: Fraction of total memory for model GPU (default: 0.67)
    
    Returns:
        Tuple of (data_memory_limit_mb, model_memory_limit_mb)
    """
    data_info = get_gpu_memory_info(data_gpu_id)
    model_info = get_gpu_memory_info(model_gpu_id)
    
    # Default values if detection fails
    data_limit = 8192  # 8GB
    model_limit = 16384  # 16GB
    
    if data_info and 'total_mb' in data_info:
        data_limit = int(data_info['total_mb'] * data_fraction)
        logger.info(f"Data GPU {data_gpu_id}: Optimal limit {data_limit}MB ({data_fraction*100:.0f}% of {data_info['total_mb']}MB)")
    
    if model_info and 'total_mb' in model_info:
        model_limit = int(model_info['total_mb'] * model_fraction)
        logger.info(f"Model GPU {model_gpu_id}: Optimal limit {model_limit}MB ({model_fraction*100:.0f}% of {model_info['total_mb']}MB)")
    
    return data_limit, model_limit
