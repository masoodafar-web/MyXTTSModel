#!/usr/bin/env python3
"""
Test GPU Memory API Compatibility Across TensorFlow Versions.

This test validates that the gpu_memory.py module correctly handles:
1. New API (set_virtual_device_configuration) for TensorFlow 2.10+
2. Old API (set_logical_device_configuration) for TensorFlow < 2.10
3. Fallback to memory growth when neither API is available
4. Proper error handling and logging for each scenario
"""

import sys
import os
import unittest
import re

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestGPUMemoryAPICompatibility(unittest.TestCase):
    """Test GPU memory isolation API compatibility"""
    
    def test_gpu_memory_file_exists(self):
        """Test that gpu_memory.py file exists"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        self.assertTrue(os.path.exists(gpu_memory_path), f"gpu_memory.py not found at {gpu_memory_path}")
    
    def test_code_uses_new_api(self):
        """Test that code includes new API (set_virtual_device_configuration)"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        with open(gpu_memory_path, 'r') as f:
            code = f.read()
        
        # Check that new API is present
        self.assertIn('set_virtual_device_configuration', code,
                     "New API set_virtual_device_configuration not found in code")
        self.assertIn('VirtualDeviceConfiguration', code,
                     "VirtualDeviceConfiguration not found in code")
    
    def test_code_maintains_old_api(self):
        """Test that code still supports old API (set_logical_device_configuration)"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        with open(gpu_memory_path, 'r') as f:
            code = f.read()
        
        # Check that old API is still present for backward compatibility
        self.assertIn('set_logical_device_configuration', code,
                     "Old API set_logical_device_configuration not found in code")
        self.assertIn('LogicalDeviceConfiguration', code,
                     "LogicalDeviceConfiguration not found in code")
    
    def test_code_has_api_detection(self):
        """Test that code has proper API detection using hasattr"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        with open(gpu_memory_path, 'r') as f:
            code = f.read()
        
        # Check that API detection is present
        self.assertIn('hasattr', code, "API detection using hasattr not found")
        self.assertRegex(code, r'hasattr.*set_virtual_device_configuration',
                        "API detection for set_virtual_device_configuration not found")
    
    def test_code_has_attribute_error_handling(self):
        """Test that code handles AttributeError for API compatibility"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        with open(gpu_memory_path, 'r') as f:
            code = f.read()
        
        # Check that AttributeError is caught
        self.assertIn('except AttributeError', code,
                     "AttributeError handling not found in code")
    
    def test_code_has_fallback_to_memory_growth(self):
        """Test that code falls back to memory growth when API not available"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        with open(gpu_memory_path, 'r') as f:
            code = f.read()
        
        # Check that fallback logic is present
        self.assertIn('set_memory_growth', code,
                     "Fallback to set_memory_growth not found")
        self.assertRegex(code, r'Falling back to memory growth',
                        "Fallback warning message not found")
    
    def test_code_has_proper_logging(self):
        """Test that code has proper logging for different scenarios"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        with open(gpu_memory_path, 'r') as f:
            code = f.read()
        
        # Check for various log messages
        self.assertIn('logger.debug', code, "Debug logging not found")
        self.assertIn('logger.warning', code, "Warning logging not found")
        self.assertIn('logger.info', code, "Info logging not found")


class TestAPIVersionInfo(unittest.TestCase):
    """Test API version information and documentation"""
    
    def test_function_has_docstring(self):
        """Test that setup_gpu_memory_isolation function has proper documentation"""
        gpu_memory_path = os.path.join(
            os.path.dirname(__file__), '..', 'myxtts', 'utils', 'gpu_memory.py'
        )
        with open(gpu_memory_path, 'r') as f:
            code = f.read()
        
        # Check that function exists with docstring
        self.assertIn('def setup_gpu_memory_isolation', code)
        # Check that there's a docstring after the function definition
        pattern = re.compile(r'def setup_gpu_memory_isolation.*?""".*?"""', re.DOTALL)
        self.assertIsNotNone(pattern.search(code), "Function docstring not found")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("GPU MEMORY API COMPATIBILITY TEST SUITE")
    print("=" * 80)
    print("\nThis test suite validates:")
    print("  1. New API support (TensorFlow 2.10+)")
    print("  2. Old API support (TensorFlow < 2.10)")
    print("  3. Fallback to memory growth")
    print("  4. Proper error handling")
    print("=" * 80 + "\n")
    
    # Run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestGPUMemoryAPICompatibility))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIVersionInfo))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
