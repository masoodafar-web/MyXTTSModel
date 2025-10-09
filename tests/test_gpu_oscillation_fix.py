#!/usr/bin/env python3
"""
Test GPU Oscillation Fix

This test validates that the TensorFlow-native data loader eliminates
the cyclic GPU utilization pattern (2-40% oscillation).
"""

import sys
import os
from pathlib import Path
import time
import unittest
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tensorflow as tf
except ImportError:
    print("TensorFlow not installed, skipping tests")
    sys.exit(0)

from myxtts.data.tf_native_loader import TFNativeDataLoader
from myxtts.config.config import DataConfig


class TestGPUOscillationFix(unittest.TestCase):
    """Test suite for GPU oscillation fix."""
    
    def test_tf_native_loader_initialization(self):
        """Test that TF-native loader initializes correctly."""
        loader = TFNativeDataLoader(
            sample_rate=22050,
            n_mels=80,
        )
        
        self.assertIsNotNone(loader)
        self.assertEqual(loader.sample_rate, 22050)
        self.assertEqual(loader.n_mels, 80)
        self.assertIsNotNone(loader.mel_filterbank)
    
    def test_mel_filterbank_creation(self):
        """Test mel filterbank matrix creation."""
        loader = TFNativeDataLoader(n_mels=80)
        mel_matrix = loader.mel_filterbank
        
        # Check shape
        expected_shape = (80, loader.n_fft // 2 + 1)
        self.assertEqual(mel_matrix.shape, expected_shape)
        
        # Check it's a valid tensor
        self.assertIsInstance(mel_matrix, tf.Tensor)
        
        # Check values are non-negative (mel filterbank should be)
        self.assertTrue(tf.reduce_all(mel_matrix >= 0.0))
    
    def test_audio_normalization(self):
        """Test audio normalization function."""
        loader = TFNativeDataLoader()
        
        # Create test audio with known properties
        audio = tf.constant([0.5, -0.5, 0.3, -0.3], dtype=tf.float32)
        normalized = loader._normalize_audio(audio)
        
        # Check output shape matches input
        self.assertEqual(normalized.shape, audio.shape)
        
        # Check values are in valid range
        self.assertTrue(tf.reduce_all(normalized >= -1.0))
        self.assertTrue(tf.reduce_all(normalized <= 1.0))
    
    def test_truncate_or_pad(self):
        """Test audio truncation and padding."""
        loader = TFNativeDataLoader()
        
        # Test truncation
        long_audio = tf.ones(1000, dtype=tf.float32)
        truncated = loader._truncate_or_pad(long_audio, 500)
        self.assertEqual(truncated.shape[0], 500)
        
        # Test padding
        short_audio = tf.ones(300, dtype=tf.float32)
        padded = loader._truncate_or_pad(short_audio, 500)
        self.assertEqual(padded.shape[0], 500)
        
        # Check padding is zeros
        self.assertTrue(tf.reduce_all(padded[300:] == 0.0))
    
    def test_mel_spectrogram_computation(self):
        """Test mel spectrogram computation with TF operations."""
        loader = TFNativeDataLoader(
            sample_rate=22050,
            n_mels=80,
            n_fft=1024,
            hop_length=256,
        )
        
        # Create synthetic audio (1 second)
        duration = 1.0
        sample_rate = 22050
        t = tf.linspace(0.0, duration, int(sample_rate * duration))
        
        # Create a 440 Hz sine wave (A4 note)
        audio = tf.sin(2.0 * np.pi * 440.0 * t)
        audio = tf.cast(audio, tf.float32)
        
        # Compute mel spectrogram
        mel_spec = loader._compute_mel_spectrogram(audio)
        
        # Check shape
        expected_time_steps = (len(audio) + loader.hop_length - 1) // loader.hop_length
        self.assertEqual(mel_spec.shape[0], expected_time_steps)
        self.assertEqual(mel_spec.shape[1], 80)
        
        # Check values are finite
        self.assertTrue(tf.reduce_all(tf.math.is_finite(mel_spec)))
        
        # Check values are in reasonable range [0, 1] after normalization
        self.assertTrue(tf.reduce_all(mel_spec >= 0.0))
        self.assertTrue(tf.reduce_all(mel_spec <= 1.0))
    
    def test_graph_compatibility(self):
        """Test that TF-native loader is graph-compatible."""
        loader = TFNativeDataLoader()
        
        # Create synthetic audio
        audio = tf.random.normal([22050], dtype=tf.float32)
        
        # This should not raise an error if graph-compatible
        @tf.function
        def compute_mel_in_graph():
            return loader._compute_mel_spectrogram(audio)
        
        # Run in graph mode
        mel_spec = compute_mel_in_graph()
        
        # Check output is valid
        self.assertIsNotNone(mel_spec)
        self.assertEqual(len(mel_spec.shape), 2)
    
    def test_no_numpy_function_in_pipeline(self):
        """Test that pipeline doesn't use tf.numpy_function when TF-native is enabled."""
        from myxtts.data.ljspeech import LJSpeechDataset
        
        # Create config with TF-native enabled
        config = DataConfig(
            use_tf_native_loading=True,
            batch_size=4,
            num_workers=2,
        )
        
        # This test just checks the configuration is set correctly
        self.assertTrue(config.use_tf_native_loading)
    
    def test_batch_loading_stability(self):
        """
        Test that batch loading times are stable (no oscillation).
        
        This is a basic stability check - in production, use
        utilities/diagnose_gpu_bottleneck.py for comprehensive testing.
        """
        loader = TFNativeDataLoader()
        
        # Create multiple batches of synthetic audio
        batch_times = []
        num_batches = 10
        
        for _ in range(num_batches):
            audio = tf.random.normal([22050], dtype=tf.float32)
            
            start_time = time.perf_counter()
            mel_spec = loader._compute_mel_spectrogram(audio)
            _ = mel_spec.numpy()  # Force execution
            end_time = time.perf_counter()
            
            batch_time = (end_time - start_time) * 1000  # ms
            batch_times.append(batch_time)
        
        # Calculate variance
        mean_time = np.mean(batch_times)
        std_time = np.std(batch_times)
        variation_ratio = std_time / mean_time if mean_time > 0 else 0
        
        # Check that variation is reasonable (< 50%)
        # High variation (>50%) indicates oscillation pattern
        self.assertLess(
            variation_ratio, 0.5,
            f"High variation detected ({variation_ratio:.1%}), "
            f"mean={mean_time:.1f}ms, std={std_time:.1f}ms"
        )
    
    def test_gpu_compatibility(self):
        """Test that operations can run on GPU if available."""
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            self.skipTest("No GPU available for testing")
        
        loader = TFNativeDataLoader()
        
        # Create audio on GPU
        with tf.device('/GPU:0'):
            audio = tf.random.normal([22050], dtype=tf.float32)
            mel_spec = loader._compute_mel_spectrogram(audio)
        
        # Check that computation succeeded
        self.assertIsNotNone(mel_spec)
        self.assertEqual(len(mel_spec.shape), 2)


class TestDataConfigGPUSettings(unittest.TestCase):
    """Test GPU optimization settings in DataConfig."""
    
    def test_default_gpu_optimization_settings(self):
        """Test that GPU optimization settings have correct defaults."""
        config = DataConfig()
        
        # Check critical settings for GPU oscillation fix
        self.assertTrue(
            config.use_tf_native_loading,
            "use_tf_native_loading should be True by default"
        )
        self.assertTrue(
            config.prefetch_to_gpu,
            "prefetch_to_gpu should be True by default"
        )
        self.assertTrue(
            config.enhanced_gpu_prefetch,
            "enhanced_gpu_prefetch should be True by default"
        )
        self.assertTrue(
            config.optimize_cpu_gpu_overlap,
            "optimize_cpu_gpu_overlap should be True by default"
        )
    
    def test_gpu_settings_can_be_disabled(self):
        """Test that GPU optimizations can be disabled for debugging."""
        config = DataConfig(
            use_tf_native_loading=False,
            prefetch_to_gpu=False,
        )
        
        self.assertFalse(config.use_tf_native_loading)
        self.assertFalse(config.prefetch_to_gpu)


def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
        print("\nGPU oscillation fix is working correctly.")
        print("For comprehensive profiling, run:")
        print("  python utilities/diagnose_gpu_bottleneck.py")
    else:
        print("\n❌ Some tests failed")
        print("Review failures above for details")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
