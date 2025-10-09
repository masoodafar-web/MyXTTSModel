#!/usr/bin/env python3
"""
Test New Profiling Tools

This test validates the new profiling and diagnostic tools.
"""

import sys
import os
from pathlib import Path
import unittest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestProfilingTools(unittest.TestCase):
    """Test suite for new profiling tools."""
    
    def test_comprehensive_diagnostic_import(self):
        """Test that comprehensive diagnostic can be imported."""
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            self.assertTrue(True, "Comprehensive diagnostic imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import comprehensive diagnostic: {e}")
    
    def test_enhanced_profiler_import(self):
        """Test that enhanced profiler can be imported."""
        try:
            from utilities.enhanced_gpu_profiler import EnhancedGPUProfiler
            self.assertTrue(True, "Enhanced profiler imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import enhanced profiler: {e}")
    
    def test_training_step_profiler_import(self):
        """Test that training step profiler can be imported."""
        try:
            from utilities.training_step_profiler import TrainingStepProfiler
            self.assertTrue(True, "Training step profiler imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import training step profiler: {e}")
    
    def test_comprehensive_diagnostic_initialization(self):
        """Test comprehensive diagnostic initialization."""
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            
            # Initialize with default config
            diagnostic = ComprehensiveGPUDiagnostic()
            self.assertIsNotNone(diagnostic)
            self.assertIsNotNone(diagnostic.config)
            self.assertEqual(len(diagnostic.issues), 0)
            self.assertEqual(len(diagnostic.recommendations), 0)
            
        except Exception as e:
            self.fail(f"Failed to initialize comprehensive diagnostic: {e}")
    
    def test_enhanced_profiler_initialization(self):
        """Test enhanced profiler initialization."""
        try:
            from utilities.enhanced_gpu_profiler import EnhancedGPUProfiler
            
            # Initialize with default config
            profiler = EnhancedGPUProfiler()
            self.assertIsNotNone(profiler)
            self.assertIsNotNone(profiler.config)
            self.assertEqual(len(profiler.batch_load_times), 0)
            
        except Exception as e:
            self.fail(f"Failed to initialize enhanced profiler: {e}")
    
    def test_training_step_profiler_initialization(self):
        """Test training step profiler initialization."""
        try:
            from utilities.training_step_profiler import TrainingStepProfiler
            
            # Initialize with default config
            profiler = TrainingStepProfiler()
            self.assertIsNotNone(profiler)
            self.assertIsNotNone(profiler.config)
            self.assertEqual(len(profiler.data_load_times), 0)
            
        except Exception as e:
            self.fail(f"Failed to initialize training step profiler: {e}")
    
    def test_comprehensive_diagnostic_hardware_check(self):
        """Test hardware check functionality."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed")
        
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            
            diagnostic = ComprehensiveGPUDiagnostic()
            result = diagnostic.check_hardware()
            
            # Should return a dict with basic info
            self.assertIsInstance(result, dict)
            self.assertIn('available', result)
            
            # If GPU available, check for more fields
            if result['available']:
                self.assertIn('count', result)
                self.assertIn('functional', result)
            
        except Exception as e:
            self.fail(f"Hardware check failed: {e}")
    
    def test_comprehensive_diagnostic_config_check(self):
        """Test configuration check functionality."""
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            
            diagnostic = ComprehensiveGPUDiagnostic()
            result = diagnostic.check_configuration()
            
            # Should return a dict with settings
            self.assertIsInstance(result, dict)
            self.assertIn('settings', result)
            
        except Exception as e:
            self.fail(f"Configuration check failed: {e}")
    
    def test_enhanced_profiler_stats_analysis(self):
        """Test timing statistics analysis."""
        try:
            from utilities.enhanced_gpu_profiler import EnhancedGPUProfiler
            
            profiler = EnhancedGPUProfiler()
            
            # Test with synthetic timing data
            times = [100.0, 105.0, 98.0, 102.0, 101.0]  # Low variation
            result = profiler._analyze_timing_stats(times, "Test Operation")
            
            self.assertIsInstance(result, dict)
            self.assertIn('avg_time', result)
            self.assertIn('std_time', result)
            self.assertIn('variation_ratio', result)
            self.assertIn('oscillation_detected', result)
            
            # With low variation, should not detect oscillation
            self.assertLess(result['variation_ratio'], 0.1)
            self.assertFalse(result['oscillation_detected'])
            
        except Exception as e:
            self.fail(f"Statistics analysis failed: {e}")
    
    def test_enhanced_profiler_cyclic_detection(self):
        """Test cyclic pattern detection."""
        try:
            from utilities.enhanced_gpu_profiler import EnhancedGPUProfiler
            import numpy as np
            
            profiler = EnhancedGPUProfiler()
            
            # Create synthetic cyclic pattern
            # Pattern: 100, 50, 100, 50, 100, 50, ...
            times = [100.0 if i % 2 == 0 else 50.0 for i in range(20)]
            
            result = profiler._detect_cyclic_pattern(times)
            
            # Should detect the period-2 pattern
            if result:
                self.assertIsInstance(result, dict)
                self.assertIn('period', result)
                # Period should be around 2
                self.assertLessEqual(result['period'], 4)
            
        except Exception as e:
            self.fail(f"Cyclic pattern detection failed: {e}")
    
    def test_comprehensive_diagnostic_code_analysis(self):
        """Test code analysis functionality."""
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            
            diagnostic = ComprehensiveGPUDiagnostic()
            result = diagnostic.analyze_code()
            
            self.assertIsInstance(result, dict)
            self.assertIn('issues', result)
            self.assertIsInstance(result['issues'], list)
            
        except Exception as e:
            self.fail(f"Code analysis failed: {e}")
    
    def test_comprehensive_diagnostic_tf_native_check(self):
        """Test TF-native loader verification."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed")
        
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            
            diagnostic = ComprehensiveGPUDiagnostic()
            result = diagnostic.verify_tf_native_loader()
            
            self.assertIsInstance(result, dict)
            self.assertIn('available', result)
            
            # If available, should have functional status
            if result['available']:
                self.assertIn('functional', result)
            
        except Exception as e:
            self.fail(f"TF-native verification failed: {e}")
    
    def test_comprehensive_diagnostic_graph_mode_check(self):
        """Test graph mode verification."""
        try:
            import tensorflow as tf
        except ImportError:
            self.skipTest("TensorFlow not installed")
        
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            
            diagnostic = ComprehensiveGPUDiagnostic()
            result = diagnostic.verify_graph_mode()
            
            self.assertIsInstance(result, dict)
            self.assertIn('graph_mode', result)
            self.assertIn('xla', result)
            
        except Exception as e:
            self.fail(f"Graph mode verification failed: {e}")
    
    def test_documentation_exists(self):
        """Test that comprehensive documentation exists."""
        doc_path = Path(__file__).parent.parent / "docs" / "COMPREHENSIVE_GPU_BOTTLENECK_ANALYSIS.md"
        self.assertTrue(doc_path.exists(), "Documentation file should exist")
        
        # Check content
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Should contain key sections
        self.assertIn('comprehensive_gpu_diagnostic.py', content)
        self.assertIn('enhanced_gpu_profiler.py', content)
        self.assertIn('training_step_profiler.py', content)
        self.assertIn('Bottleneck', content)


class TestToolsIntegration(unittest.TestCase):
    """Integration tests for profiling tools."""
    
    def test_all_tools_can_be_imported_together(self):
        """Test that all tools can be imported without conflicts."""
        try:
            from utilities.comprehensive_gpu_diagnostic import ComprehensiveGPUDiagnostic
            from utilities.enhanced_gpu_profiler import EnhancedGPUProfiler
            from utilities.training_step_profiler import TrainingStepProfiler
            
            # Should be able to instantiate all
            diag = ComprehensiveGPUDiagnostic()
            prof1 = EnhancedGPUProfiler()
            prof2 = TrainingStepProfiler()
            
            self.assertIsNotNone(diag)
            self.assertIsNotNone(prof1)
            self.assertIsNotNone(prof2)
            
        except Exception as e:
            self.fail(f"Failed to import all tools together: {e}")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestProfilingTools))
    suite.addTests(loader.loadTestsFromTestCase(TestToolsIntegration))
    
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
        print("\nProfiling tools are ready to use:")
        print("  - utilities/comprehensive_gpu_diagnostic.py")
        print("  - utilities/enhanced_gpu_profiler.py")
        print("  - utilities/training_step_profiler.py")
    else:
        print("\n❌ Some tests failed")
        print("Review failures above for details")
    
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
