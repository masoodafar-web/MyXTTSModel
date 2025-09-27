#!/usr/bin/env python3
"""
Quick GPU utilization test to verify the fixes work.
This script tests the core GPU utilization improvements.
"""

import os
import sys
import time
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    from myxtts.utils.commons import configure_gpus, get_device, ensure_gpu_placement, setup_gpu_strategy
    
    def test_gpu_utilization():
        """Test basic GPU utilization with the fixes."""
        logger.info("=== GPU Utilization Test ===")
        
        # Configure GPUs with our improved function
        logger.info("Configuring GPUs...")
        configure_gpus(memory_growth=True)
        
        # Get device
        device = get_device()
        logger.info(f"Device: {device}")
        
        # Test GPU strategy
        strategy = setup_gpu_strategy()
        logger.info(f"Strategy: {type(strategy).__name__}")
        logger.info(f"Replicas: {strategy.num_replicas_in_sync}")
        
        if device == "GPU":
            logger.info("Testing GPU tensor operations...")
            
            # Test our improved ensure_gpu_placement function
            with tf.device('/GPU:0'):
                # Create test tensors
                test_tensor = tf.random.normal([100, 100])
                logger.info(f"Test tensor device: {test_tensor.device}")
                
                # Test ensure_gpu_placement
                gpu_tensor = ensure_gpu_placement(test_tensor)
                logger.info(f"GPU tensor device: {gpu_tensor.device}")
                
                # Test computation
                start_time = time.time()
                result = tf.matmul(gpu_tensor, gpu_tensor)
                computation_time = time.time() - start_time
                logger.info(f"Matrix multiplication time: {computation_time:.4f}s")
                logger.info(f"Result shape: {result.shape}")
                
            # Test model creation on GPU
            logger.info("Testing model creation on GPU...")
            with strategy.scope():
                with tf.device('/GPU:0'):
                    model = tf.keras.Sequential([
                        tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(64, activation='relu'),
                        tf.keras.layers.Dense(10)
                    ])
                    
                    # Force model to build on GPU
                    dummy_input = tf.random.normal([32, 100])
                    dummy_input = ensure_gpu_placement(dummy_input)
                    
                    start_time = time.time()
                    output = model(dummy_input)
                    forward_time = time.time() - start_time
                    
                    logger.info(f"Model forward pass time: {forward_time:.4f}s")
                    logger.info(f"Output device: {output.device}")
                    logger.info(f"Output shape: {output.shape}")
            
            logger.info("✓ GPU utilization test completed successfully!")
            return True
        else:
            logger.warning("No GPU available for testing")
            return False
    
    def test_training_step():
        """Test a simplified training step to verify GPU usage."""
        logger.info("\n=== Training Step Test ===")
        
        device = get_device()
        if device != "GPU":
            logger.warning("Skipping training step test - no GPU available")
            return False
        
        strategy = setup_gpu_strategy()
        
        with strategy.scope():
            # Create a simple model
            with tf.device('/GPU:0'):
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(128, activation='relu', input_shape=(50,)),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(10)
                ])
                
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                
                # Test training step
                logger.info("Testing training step...")
                
                device_context = tf.device('/GPU:0')
                with device_context:
                    # Create training data
                    batch_size = 32
                    x = tf.random.normal([batch_size, 50])
                    y = tf.random.uniform([batch_size, 10])
                    
                    # Ensure tensors are on GPU
                    x = ensure_gpu_placement(x)
                    y = ensure_gpu_placement(y)
                    
                    logger.info(f"Input device: {x.device}")
                    logger.info(f"Target device: {y.device}")
                    
                    with tf.GradientTape() as tape:
                        predictions = model(x, training=True)
                        loss = tf.reduce_mean(tf.square(predictions - y))
                    
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    
                    logger.info(f"Loss: {loss.numpy():.4f}")
                    logger.info(f"Predictions device: {predictions.device}")
                    logger.info("✓ Training step completed successfully on GPU!")
        
        return True
    
    def main():
        """Main test function."""
        logger.info("MyXTTS GPU Fix Verification Test")
        logger.info("=" * 50)
        
        try:
            # Test basic GPU utilization
            gpu_test_passed = test_gpu_utilization()
            
            # Test training step
            training_test_passed = test_training_step()
            
            # Summary
            logger.info("\n=== Test Summary ===")
            if gpu_test_passed and training_test_passed:
                logger.info("✓ All GPU tests passed! The fixes should improve GPU utilization.")
            elif gpu_test_passed:
                logger.info("✓ Basic GPU test passed, but training test skipped (no GPU).")
            else:
                logger.warning("⚠️ GPU tests failed or no GPU available.")
                
            logger.info("\nRecommendations:")
            logger.info("1. Use these fixed functions in training")
            logger.info("2. Monitor GPU utilization during actual training")
            logger.info("3. Check that batch sizes are large enough to utilize GPU")
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    if __name__ == "__main__":
        main()

except ImportError as e:
    print(f"Import error: {e}")
    print("Some dependencies may be missing. Install them with:")
    print("pip install -r requirements.txt")