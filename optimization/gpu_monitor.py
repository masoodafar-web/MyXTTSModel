#!/usr/bin/env python3
"""
Basic GPU Detection for MyXTTS

This script provides basic GPU detection and availability checking
for essential device placement and configuration.
"""

import tensorflow as tf
from typing import Dict, Optional


def get_gpu_info() -> Dict[str, any]:
    """
    Get basic GPU information for device configuration.
    
    Returns:
        Dictionary with basic GPU information
    """
    gpu_info = {
        "available": False,
        "count": 0,
        "names": [],
        "tensorflow_detected": False
    }
    
    # Check TensorFlow GPU detection
    tf_gpus = tf.config.list_physical_devices('GPU')
    gpu_info["tensorflow_detected"] = len(tf_gpus) > 0
    gpu_info["count"] = len(tf_gpus)
    
    if tf_gpus:
        gpu_info["available"] = True
        try:
            for i, gpu in enumerate(tf_gpus):
                gpu_info["names"].append(f"GPU_{i}")
            print(f"✓ {len(tf_gpus)} GPU(s) detected by TensorFlow")
        except Exception as e:
            print(f"Warning: Error getting GPU details: {e}")
    else:
        print("ℹ No GPUs detected")
    
    return gpu_info


def check_gpu_availability() -> bool:
    """
    Simple check if GPU is available for use.
    
    Returns:
        True if GPU is available, False otherwise
    """
    return len(tf.config.list_physical_devices('GPU')) > 0


def main():
    """Main function for basic GPU information."""
    print("=== Basic GPU Detection ===")
    gpu_info = get_gpu_info()
    
    print(f"GPU Available: {gpu_info['available']}")
    print(f"GPU Count: {gpu_info['count']}")
    
    if gpu_info['available']:
        print("TensorFlow can use GPU for computation")
    else:
        print("TensorFlow will use CPU for computation")


if __name__ == "__main__":
    main()