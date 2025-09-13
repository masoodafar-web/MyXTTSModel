#!/usr/bin/env python3
"""
Memory Optimization Test Script for MyXTTS

This script tests the memory optimization fixes to ensure they prevent OOM errors
while maintaining good GPU utilization.
"""

import os
import sys
import time
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from myxtts.config.config import XTTSConfig
    from myxtts.training.trainer import XTTSTrainer
    from myxtts import get_xtts_model
    from gpu_monitor import GPUMonitor
    MYXTTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MyXTTS modules not fully available: {e}")
    MYXTTS_AVAILABLE = False


def test_memory_optimization(batch_sizes=[2, 4, 6, 8], duration=60):
    """Test memory optimization with different batch sizes."""
    
    if not MYXTTS_AVAILABLE:
        print("MyXTTS not available, skipping memory optimization test")
        return False
    
    if not tf.config.list_physical_devices('GPU'):
        print("No GPUs available for memory optimization test")
        return False
    
    print("=== Memory Optimization Test ===")
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        try:
            # Load memory-optimized configuration
            config = XTTSConfig(
                batch_size=batch_size,
                gradient_accumulation_steps=8 // batch_size if batch_size <= 8 else 1,
                enable_memory_cleanup=True,
                max_memory_fraction=0.85,
                mixed_precision=True,
                enable_xla=True
            )
            
            # Create model and trainer
            model = get_xtts_model()(config.model)
            trainer = XTTSTrainer(config, model)
            
            # Find optimal batch size
            optimal_batch_size = trainer.find_optimal_batch_size(
                start_batch_size=batch_size, 
                max_batch_size=batch_size + 2
            )
            
            print(f"  Optimal batch size found: {optimal_batch_size}")
            
            # Test training step with memory monitoring
            monitor = GPUMonitor(interval=1.0)
            monitor.start_monitoring()
            
            # Create synthetic training data
            text_seq = tf.random.uniform([batch_size, 50], maxval=1000, dtype=tf.int32)
            mel_spec = tf.random.normal([batch_size, 100, 80])
            text_len = tf.fill([batch_size], 50)
            mel_len = tf.fill([batch_size], 100)
            
            # Run multiple training steps
            start_time = time.time()
            step_count = 0
            
            while time.time() - start_time < duration:
                try:
                    if config.training.gradient_accumulation_steps > 1:
                        losses = trainer.train_step_with_accumulation(
                            (text_seq, mel_spec, text_len, mel_len),
                            accumulation_steps=config.training.gradient_accumulation_steps
                        )
                    else:
                        losses = trainer.train_step(text_seq, mel_spec, text_len, mel_len)
                    
                    step_count += 1
                    
                    if step_count % 10 == 0:
                        print(f"    Step {step_count}: loss = {losses['total_loss']:.4f}")
                    
                except tf.errors.ResourceExhaustedError as e:
                    print(f"    OOM error at step {step_count}: {e}")
                    break
                except Exception as e:
                    print(f"    Error at step {step_count}: {e}")
                    break
            
            monitor.stop_monitoring()
            
            # Get memory usage summary
            summary = monitor.get_summary_report()
            
            results[batch_size] = {
                "optimal_batch_size": optimal_batch_size,
                "steps_completed": step_count,
                "duration": time.time() - start_time,
                "avg_steps_per_sec": step_count / (time.time() - start_time),
                "memory_summary": summary,
                "success": step_count > 0
            }
            
            print(f"  Completed {step_count} steps in {time.time() - start_time:.1f}s")
            print(f"  Average: {step_count / (time.time() - start_time):.2f} steps/sec")
            
            # Clean up
            del model, trainer, text_seq, mel_spec, text_len, mel_len
            tf.keras.backend.clear_session()
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[batch_size] = {
                "success": False,
                "error": str(e)
            }
    
    return results


def test_gradient_accumulation():
    """Test gradient accumulation functionality."""
    
    if not MYXTTS_AVAILABLE:
        print("MyXTTS not available, skipping gradient accumulation test")
        return False
    
    print("\n=== Gradient Accumulation Test ===")
    
    try:
        # Test with different accumulation steps
        accumulation_steps = [1, 2, 4, 8]
        
        for steps in accumulation_steps:
            print(f"\nTesting {steps} accumulation steps")
            
            config = XTTSConfig(
                batch_size=2,  # Small batch size
                gradient_accumulation_steps=steps,
                enable_memory_cleanup=True
            )
            
            model = get_xtts_model()(config.model)
            trainer = XTTSTrainer(config, model)
            
            # Create synthetic data
            text_seq = tf.random.uniform([2, 50], maxval=1000, dtype=tf.int32)
            mel_spec = tf.random.normal([2, 100, 80])
            text_len = tf.fill([2], 50)
            mel_len = tf.fill([2], 100)
            
            # Test gradient accumulation
            start_time = time.time()
            losses = trainer.train_step_with_accumulation(
                (text_seq, mel_spec, text_len, mel_len),
                accumulation_steps=steps
            )
            end_time = time.time()
            
            print(f"  Completed in {end_time - start_time:.3f}s")
            print(f"  Loss: {losses['total_loss']:.4f}")
            
            # Clean up
            del model, trainer
            tf.keras.backend.clear_session()
        
        return True
        
    except Exception as e:
        print(f"Gradient accumulation test failed: {e}")
        return False


def test_memory_cleanup():
    """Test memory cleanup functionality."""
    
    if not tf.config.list_physical_devices('GPU'):
        print("No GPUs available for memory cleanup test")
        return False
    
    print("\n=== Memory Cleanup Test ===")
    
    try:
        # Monitor memory before and after cleanup
        monitor = GPUMonitor(interval=0.5)
        monitor.start_monitoring()
        
        # Create large tensors to use memory
        large_tensors = []
        for i in range(10):
            tensor = tf.random.normal([1000, 1000])
            large_tensors.append(tensor)
        
        time.sleep(2)  # Let monitoring capture high usage
        
        # Clear tensors and clean up
        del large_tensors
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        time.sleep(2)  # Let monitoring capture cleanup
        
        monitor.stop_monitoring()
        
        print("Memory cleanup test completed")
        print(monitor.get_summary_report())
        
        return True
        
    except Exception as e:
        print(f"Memory cleanup test failed: {e}")
        return False


def main():
    """Main function for memory optimization testing."""
    parser = argparse.ArgumentParser(description="Memory Optimization Test for MyXTTS")
    parser.add_argument("--batch-sizes", type=str, default="2,4,6,8",
                       help="Comma-separated batch sizes to test")
    parser.add_argument("--duration", type=int, default=30,
                       help="Duration for each batch size test in seconds")
    parser.add_argument("--skip-batch-test", action="store_true",
                       help="Skip batch size testing")
    parser.add_argument("--skip-gradient-test", action="store_true",
                       help="Skip gradient accumulation testing")
    parser.add_argument("--skip-cleanup-test", action="store_true",
                       help="Skip memory cleanup testing")
    
    args = parser.parse_args()
    
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(',')]
    
    print("MyXTTS Memory Optimization Test")
    print("=" * 50)
    
    results = {
        "batch_size_test": None,
        "gradient_accumulation_test": None,
        "memory_cleanup_test": None
    }
    
    # Test 1: Batch size optimization
    if not args.skip_batch_test:
        results["batch_size_test"] = test_memory_optimization(batch_sizes, args.duration)
    
    # Test 2: Gradient accumulation
    if not args.skip_gradient_test:
        results["gradient_accumulation_test"] = test_gradient_accumulation()
    
    # Test 3: Memory cleanup
    if not args.skip_cleanup_test:
        results["memory_cleanup_test"] = test_memory_cleanup()
    
    # Summary
    print(f"\n=== Test Summary ===")
    
    if results["batch_size_test"]:
        successful_batches = sum(1 for r in results["batch_size_test"].values() if r.get("success", False))
        total_batches = len(results["batch_size_test"])
        print(f"Batch size tests: {successful_batches}/{total_batches} successful")
        
        for batch_size, result in results["batch_size_test"].items():
            if result.get("success", False):
                print(f"  Batch {batch_size}: {result['steps_completed']} steps, "
                      f"{result['avg_steps_per_sec']:.2f} steps/sec")
    
    if results["gradient_accumulation_test"]:
        print("✓ Gradient accumulation test passed")
    else:
        print("✗ Gradient accumulation test failed")
    
    if results["memory_cleanup_test"]:
        print("✓ Memory cleanup test passed")
    else:
        print("✗ Memory cleanup test failed")
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    print("1. Use the memory-optimized configuration (config_memory_optimized.yaml)")
    print("2. Start with small batch sizes (2-4) and use gradient accumulation")
    print("3. Enable memory cleanup for long training sessions")
    print("4. Monitor GPU memory usage during training")
    
    if results["batch_size_test"]:
        successful_sizes = [bs for bs, r in results["batch_size_test"].items() if r.get("success", False)]
        if successful_sizes:
            recommended_size = max(successful_sizes)
            print(f"5. Recommended batch size based on testing: {recommended_size}")


if __name__ == "__main__":
    main()