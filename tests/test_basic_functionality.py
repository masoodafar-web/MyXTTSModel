#!/usr/bin/env python3
"""
Simple test to verify that core functionality works without detailed monitoring.
"""

def test_gpu_detection():
    """Test basic GPU detection without monitoring overhead."""
    try:
        # Test our simplified gpu_monitor
        from gpu_monitor import get_gpu_info, check_gpu_availability
        
        print("=== Testing Basic GPU Detection ===")
        
        # Test basic availability check
        has_gpu = check_gpu_availability()
        print(f"GPU Available: {has_gpu}")
        
        # Test GPU info
        gpu_info = get_gpu_info()
        print(f"GPU Count: {gpu_info['count']}")
        print(f"TensorFlow Detected: {gpu_info['tensorflow_detected']}")
        
        print("✓ Basic GPU detection works")
        return True
        
    except Exception as e:
        print(f"✗ GPU detection failed: {e}")
        return False

def test_performance_basics():
    """Test basic performance monitoring (CPU/memory only)."""
    try:
        from myxtts.utils.performance import PerformanceMonitor
        
        print("\n=== Testing Basic Performance Monitoring ===")
        
        # Test monitor creation
        monitor = PerformanceMonitor()
        print("✓ PerformanceMonitor created")
        
        # Test timing context manager
        import time
        with monitor.time_operation("test_operation"):
            time.sleep(0.01)
        print("✓ Timing operation works")
        
        # Test report generation (should work without GPU monitoring)
        report = monitor.get_summary_report()
        print("✓ Summary report generated")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring failed: {e}")
        return False

def main():
    """Run basic functionality tests."""
    print("Testing MyXTTS without detailed monitoring...")
    
    success = True
    success &= test_gpu_detection()
    success &= test_performance_basics()
    
    if success:
        print("\n✅ All basic functionality tests passed!")
        print("The codebase works correctly without detailed monitoring.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")
    
    return success

if __name__ == "__main__":
    main()