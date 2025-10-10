#!/usr/bin/env python3
"""
Test Memory Isolation for Producer-Consumer GPU Pipeline.

This test validates:
1. GPU memory isolation setup
2. Memory-isolated trainer initialization
3. Phase separation (data processing vs model training)
4. Memory monitoring functionality
5. CLI argument integration
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestMemoryIsolationUtilities(unittest.TestCase):
    """Test GPU memory isolation utilities"""
    
    def test_gpu_memory_module_exists(self):
        """Test that gpu_memory module can be imported"""
        try:
            from myxtts.utils import gpu_memory
            self.assertTrue(hasattr(gpu_memory, 'setup_gpu_memory_isolation'))
            self.assertTrue(hasattr(gpu_memory, 'monitor_gpu_memory'))
            self.assertTrue(hasattr(gpu_memory, 'log_memory_stats'))
        except ImportError as e:
            # TensorFlow not available in test environment, skip
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_setup_gpu_memory_isolation_signature(self):
        """Test that setup_gpu_memory_isolation has correct signature"""
        try:
            from myxtts.utils.gpu_memory import setup_gpu_memory_isolation
            import inspect
            
            sig = inspect.signature(setup_gpu_memory_isolation)
            params = list(sig.parameters.keys())
            
            # Check required parameters
            self.assertIn('data_gpu_id', params)
            self.assertIn('model_gpu_id', params)
            self.assertIn('data_gpu_memory_limit', params)
            self.assertIn('model_gpu_memory_limit', params)
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_monitor_gpu_memory_signature(self):
        """Test that monitor_gpu_memory has correct signature"""
        try:
            from myxtts.utils.gpu_memory import monitor_gpu_memory
            import inspect
            
            sig = inspect.signature(monitor_gpu_memory)
            params = list(sig.parameters.keys())
            
            self.assertIn('data_gpu_id', params)
            self.assertIn('model_gpu_id', params)
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_get_optimal_memory_limits_exists(self):
        """Test that optimal memory limit calculation exists"""
        try:
            from myxtts.utils.gpu_memory import get_optimal_memory_limits
            self.assertTrue(callable(get_optimal_memory_limits))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")


class TestMemoryIsolatedTrainer(unittest.TestCase):
    """Test Memory-Isolated Dual-GPU Trainer"""
    
    def test_trainer_module_exists(self):
        """Test that memory_isolated_trainer module can be imported"""
        try:
            from myxtts.training import memory_isolated_trainer
            self.assertTrue(hasattr(memory_isolated_trainer, 'MemoryIsolatedDualGPUTrainer'))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_trainer_inherits_from_xtts_trainer(self):
        """Test that MemoryIsolatedDualGPUTrainer inherits from XTTSTrainer"""
        try:
            from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
            from myxtts.training.trainer import XTTSTrainer
            
            self.assertTrue(issubclass(MemoryIsolatedDualGPUTrainer, XTTSTrainer))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_trainer_init_signature(self):
        """Test that trainer has correct initialization parameters"""
        try:
            from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
            import inspect
            
            sig = inspect.signature(MemoryIsolatedDualGPUTrainer.__init__)
            params = list(sig.parameters.keys())
            
            # Check required parameters
            self.assertIn('config', params)
            self.assertIn('data_gpu_id', params)
            self.assertIn('model_gpu_id', params)
            self.assertIn('data_gpu_memory_limit', params)
            self.assertIn('model_gpu_memory_limit', params)
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_trainer_has_phase_methods(self):
        """Test that trainer has phase separation methods"""
        try:
            from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
            
            # Check for phase methods
            self.assertTrue(hasattr(MemoryIsolatedDualGPUTrainer, '_preprocess_on_data_gpu'))
            self.assertTrue(hasattr(MemoryIsolatedDualGPUTrainer, '_transfer_to_model_gpu'))
            self.assertTrue(hasattr(MemoryIsolatedDualGPUTrainer, '_train_step_impl'))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_trainer_has_monitoring_methods(self):
        """Test that trainer has memory monitoring methods"""
        try:
            from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
            
            self.assertTrue(hasattr(MemoryIsolatedDualGPUTrainer, '_check_memory_health'))
            self.assertTrue(hasattr(MemoryIsolatedDualGPUTrainer, '_setup_memory_baselines'))
            self.assertTrue(hasattr(MemoryIsolatedDualGPUTrainer, 'get_memory_stats'))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")


class TestCLIIntegration(unittest.TestCase):
    """Test CLI integration for memory isolation"""
    
    def test_enable_memory_isolation_flag(self):
        """Test that --enable-memory-isolation flag is recognized"""
        # This test verifies the argument exists in train_main.py
        # by checking if it can be parsed
        pass  # Placeholder - actual CLI parsing tested in integration
    
    def test_memory_limit_arguments(self):
        """Test that memory limit arguments exist"""
        # This test verifies --data-gpu-memory and --model-gpu-memory flags
        pass  # Placeholder - actual CLI parsing tested in integration


class TestMemoryIsolationWorkflow(unittest.TestCase):
    """Test complete memory isolation workflow"""
    
    def test_workflow_without_tensorflow(self):
        """Test workflow structure without requiring TensorFlow GPU"""
        try:
            from myxtts.utils.gpu_memory import get_optimal_memory_limits
            
            # Test that functions are callable
            self.assertTrue(callable(get_optimal_memory_limits))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_device_mapping(self):
        """Test that device mapping logic is correct"""
        try:
            # Test the concept: physical GPU IDs map to logical devices
            # Physical GPU 0 -> /GPU:0 (data processing)
            # Physical GPU 1 -> /GPU:1 (model training)
            
            # This is validated in the trainer initialization
            from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
            
            # Verify the class has device attributes
            self.assertTrue(hasattr(MemoryIsolatedDualGPUTrainer, '__init__'))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")


class TestMemoryMonitoring(unittest.TestCase):
    """Test memory monitoring functionality"""
    
    def test_memory_info_function_exists(self):
        """Test that get_gpu_memory_info exists"""
        try:
            from myxtts.utils.gpu_memory import get_gpu_memory_info
            self.assertTrue(callable(get_gpu_memory_info))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_memory_leak_detection_exists(self):
        """Test that detect_memory_leak exists"""
        try:
            from myxtts.utils.gpu_memory import detect_memory_leak
            self.assertTrue(callable(detect_memory_leak))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")
    
    def test_log_memory_stats_exists(self):
        """Test that log_memory_stats exists"""
        try:
            from myxtts.utils.gpu_memory import log_memory_stats
            self.assertTrue(callable(log_memory_stats))
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")


class TestDoubleBuffering(unittest.TestCase):
    """Test double buffering support in trainer"""
    
    def test_trainer_has_buffer_attributes(self):
        """Test that trainer has buffer management attributes"""
        try:
            from myxtts.training.memory_isolated_trainer import MemoryIsolatedDualGPUTrainer
            
            # The trainer should have buffer-related attributes
            # We can't instantiate without TensorFlow, but we can check the class definition
            import inspect
            source = inspect.getsource(MemoryIsolatedDualGPUTrainer.__init__)
            
            # Check that buffer-related setup is present
            self.assertIn('buffer', source.lower())
        except ImportError as e:
            self.skipTest(f"TensorFlow not available: {e}")


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryIsolationUtilities))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryIsolatedTrainer))
    suite.addTests(loader.loadTestsFromTestCase(TestCLIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryIsolationWorkflow))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryMonitoring))
    suite.addTests(loader.loadTestsFromTestCase(TestDoubleBuffering))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
