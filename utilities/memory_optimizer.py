#!/usr/bin/env python3
"""
Memory Optimization Utility for MyXTTS

This utility provides functions to automatically detect GPU memory capacity
and adjust training parameters to prevent OOM errors.
"""

import os
import sys
import subprocess
import yaml
from typing import Dict, Any, Optional, Tuple
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    warnings.warn("TensorFlow not available for memory optimization")

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    warnings.warn("pynvml not available for GPU memory detection")


def get_gpu_memory_info() -> Optional[Dict[str, int]]:
    """
    Get GPU memory information.
    
    Returns:
        Dictionary with total_memory, free_memory, used_memory in MB,
        or None if not available
    """
    if not NVML_AVAILABLE:
        return None
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # First GPU
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        return {
            'total_memory': mem_info.total // (1024 * 1024),  # MB
            'free_memory': mem_info.free // (1024 * 1024),    # MB
            'used_memory': mem_info.used // (1024 * 1024)     # MB
        }
    except Exception as e:
        warnings.warn(f"Could not get GPU memory info: {e}")
        return None


def get_recommended_settings(gpu_memory_mb: int) -> Dict[str, Any]:
    """
    Get recommended settings based on GPU memory capacity.
    
    Args:
        gpu_memory_mb: GPU memory in MB
        
    Returns:
        Dictionary with recommended configuration settings
    """
    if gpu_memory_mb >= 20000:  # 20GB+ (RTX 4090, A100, etc.)
        # Plenty of headroom: favour large per-step batches while keeping decoder length manageable.
        return {
            'batch_size': 48,
            'gradient_accumulation_steps': 1,
            'max_attention_sequence_length': 512,
            'max_memory_fraction': 0.85,  # Reduced to prevent OOM during async prefetching
            'enable_gradient_checkpointing': False,
            'text_encoder_dim': 512,
            'decoder_dim': 1536,  # Keep divisible by decoder heads (24)
            'num_workers': 16,
            'prefetch_buffer_size': 48,  # Increased for better GPU utilization
            'shuffle_buffer_multiplier': 64,  # Increased
            'prefetch_factor': 8,  # New parameter for DataLoader
            'persistent_workers': True,
            'description': "High-end GPU (20GB+)"
        }
    elif gpu_memory_mb >= 12000:  # 12-20GB (RTX 3080 Ti, RTX 4070 Ti, etc.)
        return {
            'batch_size': 32,
            'gradient_accumulation_steps': 1,
            'max_attention_sequence_length': 512,
            'max_memory_fraction': 0.80,  # Reduced for async prefetching
            'enable_gradient_checkpointing': True,
            'text_encoder_dim': 384,
            'decoder_dim': 768,   # 24 heads -> 32-dim per head
            'num_workers': 12,
            'prefetch_buffer_size': 32,  # Increased
            'shuffle_buffer_multiplier': 48,  # Increased
            'prefetch_factor': 6,  # New parameter
            'persistent_workers': True,
            'description': "Mid-range GPU (12-20GB)"
        }
    elif gpu_memory_mb >= 8000:   # 8-12GB (RTX 3070, RTX 4060 Ti, etc.)
        return {
            'batch_size': 16,
            'gradient_accumulation_steps': 2,
            'max_attention_sequence_length': 384,
            'max_memory_fraction': 0.70,  # Reduced for stability
            'enable_gradient_checkpointing': True,
            'text_encoder_dim': 256,
            'decoder_dim': 576,   # Still divisible by 24 heads while staying compact
            'num_workers': 8,
            'prefetch_buffer_size': 24,  # Increased
            'shuffle_buffer_multiplier': 36,  # Increased
            'prefetch_factor': 4,  # New parameter
            'persistent_workers': True,
            'description': "Entry-level GPU (8-12GB)"
        }
    else:  # <8GB
        return {
            'batch_size': 8,
            'gradient_accumulation_steps': 4,
            'max_attention_sequence_length': 256,
            'max_memory_fraction': 0.60,  # Reduced for safety
            'enable_gradient_checkpointing': True,
            'text_encoder_dim': 128,
            'decoder_dim': 384,   # Minimum size that respects 24-way attention split
            'num_workers': 4,
            'prefetch_buffer_size': 16,  # Increased slightly
            'shuffle_buffer_multiplier': 28,  # Increased
            'prefetch_factor': 3,  # New parameter
            'persistent_workers': True,
            'description': "Low-memory GPU (<8GB)"
        }


def auto_optimize_config(config_path: str, output_path: Optional[str] = None) -> str:
    """
    Automatically optimize a configuration file based on GPU memory.
    
    Args:
        config_path: Path to input configuration file
        output_path: Path to output optimized configuration (uses input path if None)
        
    Returns:
        Path to optimized configuration file
    """
    # Load original config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get GPU memory info
    gpu_info = get_gpu_memory_info()
    if gpu_info is None:
        print("Warning: Could not detect GPU memory, using conservative settings")
        gpu_memory_mb = 8000  # Conservative default
    else:
        gpu_memory_mb = gpu_info['total_memory']
        print(f"Detected GPU memory: {gpu_memory_mb} MB")
    
    # Get recommended settings
    recommended = get_recommended_settings(gpu_memory_mb)
    print(f"GPU category: {recommended['description']}")
    
    # Update configuration
    if 'data' not in config:
        config['data'] = {}
    if 'training' not in config:
        config['training'] = {}
    if 'model' not in config:
        config['model'] = {}
    
    # Apply data settings
    config['data']['batch_size'] = recommended['batch_size']
    
    # Apply training settings
    config['training']['gradient_accumulation_steps'] = recommended['gradient_accumulation_steps']
    config['training']['max_memory_fraction'] = recommended['max_memory_fraction']
    config['training']['enable_memory_cleanup'] = True
    config['training']['gradient_clip_norm'] = 0.5
    
    # Apply model settings
    config['model']['enable_gradient_checkpointing'] = recommended['enable_gradient_checkpointing']
    config['model']['max_attention_sequence_length'] = recommended['max_attention_sequence_length']
    config['model']['use_memory_efficient_attention'] = True
    config['model']['text_encoder_dim'] = recommended['text_encoder_dim']
    config['model']['decoder_dim'] = recommended['decoder_dim']
    
    # Calculate effective batch size
    effective_batch_size = recommended['batch_size'] * recommended['gradient_accumulation_steps']
    
    # Save optimized config
    output_path = output_path or config_path
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"Optimized configuration saved to: {output_path}")
    print(f"Settings applied:")
    print(f"  - Batch size: {recommended['batch_size']}")
    print(f"  - Gradient accumulation: {recommended['gradient_accumulation_steps']}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Max attention length: {recommended['max_attention_sequence_length']}")
    print(f"  - Memory fraction: {recommended['max_memory_fraction']}")
    print(f"  - Gradient checkpointing: {recommended['enable_gradient_checkpointing']}")
    
    return output_path


def run_memory_test(batch_sizes: list = None) -> Dict[int, bool]:
    """
    Run a memory test to find the maximum usable batch size.
    
    Args:
        batch_sizes: List of batch sizes to test (default: [1, 2, 4, 8])
        
    Returns:
        Dictionary mapping batch size to success status
    """
    if not TF_AVAILABLE:
        print("TensorFlow not available for memory testing")
        return {}
    
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8]
    
    print("Running memory test...")
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        
        try:
            # Create test tensors
            text_seq = tf.random.uniform([batch_size, 100], maxval=1000, dtype=tf.int32)
            mel_spec = tf.random.normal([batch_size, 200, 80])
            
            # Simulate attention computation (the main memory bottleneck)
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                # Simulate transformer attention
                d_model = 512
                seq_len = 100
                
                q = tf.random.normal([batch_size, seq_len, d_model])
                k = tf.random.normal([batch_size, seq_len, d_model])
                v = tf.random.normal([batch_size, seq_len, d_model])
                
                # This is the operation that causes OOM
                scores = tf.matmul(q, k, transpose_b=True)
                weights = tf.nn.softmax(scores, axis=-1)
                output = tf.matmul(weights, v)
                
                # Force execution
                _ = tf.reduce_mean(output)
            
            results[batch_size] = True
            print(f"  ✓ Batch size {batch_size} successful")
            
            # Clean up
            del text_seq, mel_spec, q, k, v, scores, weights, output
            tf.keras.backend.clear_session()
            
        except tf.errors.ResourceExhaustedError:
            results[batch_size] = False
            print(f"  ✗ Batch size {batch_size} failed (OOM)")
            tf.keras.backend.clear_session()
        except Exception as e:
            results[batch_size] = False
            print(f"  ✗ Batch size {batch_size} failed ({e})")
    
    return results


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Memory optimization utility for MyXTTS")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file to optimize')
    parser.add_argument('--output', type=str, 
                       help='Output path for optimized config (default: overwrite input)')
    parser.add_argument('--test-memory', action='store_true',
                       help='Run memory test to find optimal batch size')
    parser.add_argument('--gpu-info', action='store_true',
                       help='Show GPU memory information')
    
    args = parser.parse_args()
    
    if args.gpu_info:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            print(f"GPU Memory Information:")
            print(f"  Total: {gpu_info['total_memory']} MB")
            print(f"  Free: {gpu_info['free_memory']} MB")
            print(f"  Used: {gpu_info['used_memory']} MB")
        else:
            print("Could not retrieve GPU memory information")
    
    if args.test_memory:
        results = run_memory_test()
        successful_sizes = [size for size, success in results.items() if success]
        if successful_sizes:
            max_batch_size = max(successful_sizes)
            print(f"\nRecommended maximum batch size: {max_batch_size}")
        else:
            print("\nNo batch sizes succeeded - consider using CPU or reducing model size")
    
    if args.config:
        auto_optimize_config(args.config, args.output)


if __name__ == "__main__":
    main()
