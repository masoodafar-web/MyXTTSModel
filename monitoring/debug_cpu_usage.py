#!/usr/bin/env python3
"""
Debug script to identify CPU bottlenecks during actual training execution.
This will help identify where CPU is still being used inappropriately.
"""

import os
import time
import psutil
import threading
from pathlib import Path
import tensorflow as tf

# Add the project root to the path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from myxtts.config.config import XTTSConfig, DataConfig
from myxtts.data.ljspeech import LJSpeechDataset
from myxtts.models.xtts import XTTS
from myxtts.utils.commons import setup_gpu_strategy, get_device, configure_gpus

class CPUUsageMonitor:
    """Monitor CPU usage patterns during training to identify bottlenecks."""
    
    def __init__(self):
        self.cpu_usage_history = []
        self.monitoring = False
        self.thread = None
        
    def start_monitoring(self):
        """Start monitoring CPU usage in background thread."""
        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def stop_monitoring(self):
        """Stop monitoring and return results."""
        self.monitoring = False
        if self.thread:
            self.thread.join()
        return self.cpu_usage_history
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            timestamp = time.time()
            self.cpu_usage_history.append((timestamp, cpu_percent))
            time.sleep(0.1)

def test_gpu_device_placement():
    """Test if TensorFlow is actually using GPU for operations."""
    print("=== Testing GPU Device Placement ===")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu}")
    
    if not gpus:
        print("❌ No GPU detected!")
        return False
    
    # Test basic GPU operations
    print("\n--- Testing Basic GPU Operations ---")
    try:
        with tf.device('/GPU:0'):
            # Create tensors on GPU
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            
            # Perform computation
            start_time = time.time()
            c = tf.matmul(a, b)
            gpu_time = time.time() - start_time
            
            print(f"✅ GPU matrix multiplication successful: {gpu_time:.4f}s")
            print(f"   Result shape: {c.shape}")
            
        # Compare with CPU
        with tf.device('/CPU:0'):
            start_time = time.time()
            c_cpu = tf.matmul(a, b)
            cpu_time = time.time() - start_time
            
            print(f"   CPU time for comparison: {cpu_time:.4f}s")
            
        return True
        
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
        return False

def test_data_loading_device():
    """Test if data loading pipeline uses GPU properly."""
    print("\n=== Testing Data Loading Pipeline ===")
    
    # Create minimal config
    data_config = DataConfig()
    print(f"TF Native Loading: {data_config.use_tf_native_loading}")
    print(f"GPU Prefetch: {data_config.enhanced_gpu_prefetch}")
    
    # Test data loading (using minimal dummy data)
    try:
        # Create dummy dataset structure
        dummy_data_path = Path("/tmp/dummy_ljspeech")
        dummy_data_path.mkdir(exist_ok=True)
        
        # Create minimal metadata
        metadata_file = dummy_data_path / "metadata.csv"
        with open(metadata_file, 'w') as f:
            f.write("LJ001-0001|The quick brown fox jumps over the lazy dog.|The quick brown fox jumps over the lazy dog.\n")
            f.write("LJ001-0002|Hello world this is a test.|Hello world this is a test.\n")
        
        # Create dummy audio files directory
        wavs_dir = dummy_data_path / "wavs"
        wavs_dir.mkdir(exist_ok=True)
        
        # This will test the data loading but skip actual file operations
        dataset = LJSpeechDataset(
            str(dummy_data_path), 
            data_config, 
            subset="train", 
            download=False,
            preprocess=False
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Use TF native loading: {dataset.config.use_tf_native_loading}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False

def test_model_device_placement():
    """Test if model is properly placed on GPU."""
    print("\n=== Testing Model Device Placement ===")
    
    try:
        # Setup strategy (single GPU only)
        strategy = setup_gpu_strategy()
        print(f"Strategy: {type(strategy).__name__}")
        
        # Create model within strategy scope
        with strategy.scope():
            model_config = {
                'vocab_size': 100,
                'hidden_size': 256,
                'num_layers': 2,
                'mel_channels': 80
            }
            
            # Test model creation
            model = XTTS(model_config)
            print("✅ Model created successfully")
            
            # Test forward pass with GPU tensors
            with tf.device('/GPU:0'):
                dummy_text = tf.random.uniform([2, 10], 0, 100, dtype=tf.int32)
                dummy_mel = tf.random.normal([2, 20, 80])
                dummy_text_len = tf.constant([10, 8], dtype=tf.int32)
                dummy_mel_len = tf.constant([20, 15], dtype=tf.int32)
                
                print(f"Input tensors created on GPU")
                print(f"  Text shape: {dummy_text.shape}, device: {dummy_text.device}")
                print(f"  Mel shape: {dummy_mel.shape}, device: {dummy_mel.device}")
                
                # Forward pass
                start_time = time.time()
                outputs = model(
                    text_sequences=dummy_text,
                    mel_spectrograms=dummy_mel,
                    text_lengths=dummy_text_len,
                    mel_lengths=dummy_mel_len,
                    training=True
                )
                forward_time = time.time() - start_time
                
                print(f"✅ Forward pass successful: {forward_time:.4f}s")
                print(f"   Output shapes: {[o.shape for o in outputs]}")
                
        return True
        
    except Exception as e:
        print(f"❌ Model device placement test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_cpu_usage_analysis():
    """Run comprehensive CPU usage analysis during training operations."""
    print("\n=== CPU Usage Analysis ===")
    
    # Initialize monitor
    monitor = CPUUsageMonitor()
    
    # Baseline CPU usage
    print("Measuring baseline CPU usage...")
    monitor.start_monitoring()
    time.sleep(2)
    baseline_usage = monitor.stop_monitoring()
    baseline_avg = sum(usage[1] for usage in baseline_usage) / len(baseline_usage)
    print(f"Baseline CPU usage: {baseline_avg:.1f}%")
    
    # Test GPU operations CPU usage
    print("\nTesting GPU operations CPU usage...")
    monitor = CPUUsageMonitor()
    monitor.start_monitoring()
    
    success = test_gpu_device_placement()
    
    gpu_usage = monitor.stop_monitoring()
    gpu_avg = sum(usage[1] for usage in gpu_usage) / len(gpu_usage)
    print(f"CPU usage during GPU operations: {gpu_avg:.1f}%")
    
    if gpu_avg > baseline_avg * 2:
        print(f"⚠️  HIGH CPU usage during GPU operations! Expected low usage.")
        print(f"   This suggests GPU operations are falling back to CPU.")
        return False
    else:
        print(f"✅ Normal CPU usage during GPU operations")
        return True

def main():
    """Main debugging function."""
    print("MyXTTS CPU Usage Debug Tool")
    print("=" * 50)
    
    # Configure GPUs
    try:
        configure_gpus(memory_growth=True)
        print("✅ GPU configuration successful")
    except Exception as e:
        print(f"❌ GPU configuration failed: {e}")
    
    # Get device info
    device = get_device()
    print(f"Detected device: {device}")
    
    if device != "GPU":
        print("❌ GPU not available - this is the root cause!")
        print("   Ensure CUDA drivers and TensorFlow-GPU are properly installed.")
        return
    
    # Run tests
    tests = [
        ("GPU Device Placement", test_gpu_device_placement),
        ("Data Loading Pipeline", test_data_loading_device),
        ("Model Device Placement", test_model_device_placement),
        ("CPU Usage Analysis", run_cpu_usage_analysis),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("DEBUG SUMMARY:")
    
    failed_tests = [name for name, result in results if not result]
    
    if not failed_tests:
        print("🎉 All tests passed!")
        print("   GPU setup appears to be working correctly.")
        print("   The issue might be in the specific training script or configuration.")
    else:
        print(f"❌ {len(failed_tests)} test(s) failed:")
        for test_name in failed_tests:
            print(f"   - {test_name}")
        print("\n🔧 RECOMMENDED FIXES:")
        
        if "GPU Device Placement" in failed_tests:
            print("   1. Install CUDA drivers and TensorFlow-GPU")
            print("   2. Verify GPU is properly configured")
            
        if "Data Loading Pipeline" in failed_tests:
            print("   3. Check data loading configuration")
            
        if "Model Device Placement" in failed_tests:
            print("   4. Verify model is created within GPU strategy scope")
            
        if "CPU Usage Analysis" in failed_tests:
            print("   5. Check for hidden CPU fallbacks in operations")

if __name__ == "__main__":
    main()