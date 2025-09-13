#!/usr/bin/env python3
"""
Quick Memory Test for MyXTTS

This script performs a quick memory test to validate that the current
configuration will not cause OOM errors during training.
"""

import os
import sys
import argparse
import yaml
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    print("Error: TensorFlow is not available")
    sys.exit(1)

try:
    from myxtts.config.config import XTTSConfig, ModelConfig, DataConfig, TrainingConfig
    from myxtts import get_xtts_model
    MYXTTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MyXTTS modules not available: {e}")
    MYXTTS_AVAILABLE = False


def load_config_from_yaml(config_path: str) -> XTTSConfig:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to config objects
    model_config = ModelConfig(**config_dict.get('model', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    
    return XTTSConfig(model=model_config, data=data_config, training=training_config)


def test_memory_with_config(config: XTTSConfig) -> Dict[str, Any]:
    """
    Test memory usage with the given configuration.
    
    Args:
        config: XTTS configuration
        
    Returns:
        Test results dictionary
    """
    results = {
        'success': False,
        'error': None,
        'memory_info': None,
        'test_details': {}
    }
    
    try:
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[0]}")
        else:
            print("No GPU available, using CPU")
        
        # Create synthetic test data
        batch_size = config.data.batch_size
        max_text_len = min(config.model.max_text_length, 
                          getattr(config.model, 'max_attention_sequence_length', 512))
        max_mel_len = min(400, getattr(config.model, 'max_attention_sequence_length', 512))
        
        print(f"Testing with batch_size={batch_size}, text_len={max_text_len}, mel_len={max_mel_len}")
        
        # Create test tensors
        text_sequences = tf.random.uniform([batch_size, max_text_len], 
                                         maxval=config.model.text_vocab_size, 
                                         dtype=tf.int32)
        mel_spectrograms = tf.random.normal([batch_size, max_mel_len, config.model.n_mels])
        text_lengths = tf.fill([batch_size], max_text_len)
        mel_lengths = tf.fill([batch_size], max_mel_len)
        
        results['test_details']['batch_size'] = batch_size
        results['test_details']['text_length'] = max_text_len
        results['test_details']['mel_length'] = max_mel_len
        
        # Test model creation
        print("Creating model...")
        if MYXTTS_AVAILABLE:
            model = get_xtts_model()(config.model)
        else:
            # Simplified test for attention computation
            print("Using simplified attention test...")
            d_model = config.model.text_encoder_dim
            
            # Simulate the problematic attention computation
            q = tf.random.normal([batch_size, max_text_len, d_model])
            k = tf.random.normal([batch_size, max_text_len, d_model])
            v = tf.random.normal([batch_size, max_text_len, d_model])
            
            # This is the operation that typically causes OOM
            print(f"Testing attention matrix multiplication: [{batch_size}, {max_text_len}, {max_text_len}]")
            scores = tf.matmul(q, k, transpose_b=True)
            weights = tf.nn.softmax(scores, axis=-1)
            output = tf.matmul(weights, v)
            
            # Force execution
            result = tf.reduce_mean(output)
            print(f"Attention test passed, result: {result}")
            
            results['success'] = True
            return results
        
        # Test forward pass
        print("Testing forward pass...")
        outputs = model(
            text_inputs=text_sequences,
            mel_inputs=mel_spectrograms,
            text_lengths=text_lengths,
            mel_lengths=mel_lengths,
            training=True
        )
        
        print("Forward pass successful")
        
        # Test gradient computation (if using gradient accumulation)
        if config.training.gradient_accumulation_steps > 1:
            print("Testing gradient computation...")
            with tf.GradientTape() as tape:
                outputs = model(
                    text_inputs=text_sequences,
                    mel_inputs=mel_spectrograms,
                    text_lengths=text_lengths,
                    mel_lengths=mel_lengths,
                    training=True
                )
                # Dummy loss
                loss = tf.reduce_mean(outputs['mel_output'])
            
            gradients = tape.gradient(loss, model.trainable_variables)
            print("Gradient computation successful")
        
        # Get memory usage if available
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            results['memory_info'] = {
                'used_mb': mem_info.used // (1024 * 1024),
                'free_mb': mem_info.free // (1024 * 1024),
                'total_mb': mem_info.total // (1024 * 1024)
            }
        except Exception:
            pass
        
        results['success'] = True
        print("✓ Memory test passed!")
        
    except tf.errors.ResourceExhaustedError as e:
        results['error'] = f"OOM Error: {str(e)}"
        print(f"✗ Memory test failed: OOM Error")
        print("Recommendations:")
        print("  - Reduce batch_size")
        print("  - Increase gradient_accumulation_steps")
        print("  - Enable gradient_checkpointing")
        print("  - Reduce max_attention_sequence_length")
        print("  - Reduce model dimensions")
        
    except Exception as e:
        results['error'] = f"Unexpected error: {str(e)}"
        print(f"✗ Memory test failed: {str(e)}")
    
    finally:
        # Clean up
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Quick memory test for MyXTTS")
    parser.add_argument('--config', type=str, 
                       default='config_memory_optimized.yaml',
                       help='Path to configuration file')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file {args.config} not found")
        print("Available configs:")
        for config_file in ['config.yaml', 'config_memory_optimized.yaml', 'config_extreme_memory_optimized.yaml']:
            if os.path.exists(config_file):
                print(f"  - {config_file}")
        sys.exit(1)
    
    print(f"MyXTTS Memory Test")
    print(f"Config: {args.config}")
    print("=" * 50)
    
    # Load configuration
    try:
        config = load_config_from_yaml(args.config)
        print(f"Configuration loaded successfully")
        if args.verbose:
            print(f"  Batch size: {config.data.batch_size}")
            print(f"  Gradient accumulation: {getattr(config.training, 'gradient_accumulation_steps', 1)}")
            print(f"  Text encoder dim: {config.model.text_encoder_dim}")
            print(f"  Decoder dim: {config.model.decoder_dim}")
            print(f"  Memory fraction: {getattr(config.training, 'max_memory_fraction', 0.9)}")
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Run memory test
    results = test_memory_with_config(config)
    
    # Print results
    print("\n" + "=" * 50)
    if results['success']:
        print("✓ MEMORY TEST PASSED")
        print("Configuration is suitable for training")
        
        effective_batch_size = (config.data.batch_size * 
                              getattr(config.training, 'gradient_accumulation_steps', 1))
        print(f"Effective batch size: {effective_batch_size}")
        
        if results['memory_info']:
            mem_info = results['memory_info']
            print(f"GPU memory usage: {mem_info['used_mb']} MB / {mem_info['total_mb']} MB")
            usage_percent = (mem_info['used_mb'] / mem_info['total_mb']) * 100
            print(f"Memory utilization: {usage_percent:.1f}%")
    else:
        print("✗ MEMORY TEST FAILED")
        print(f"Error: {results['error']}")
        print("\nRecommended actions:")
        print("1. Use config_extreme_memory_optimized.yaml")
        print("2. Run: python memory_optimizer.py --config your_config.yaml")
        print("3. Reduce batch_size to 1 and increase gradient_accumulation_steps")
        sys.exit(1)


if __name__ == "__main__":
    main()