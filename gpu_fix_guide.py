#!/usr/bin/env python3
"""
راهنمای کامل رفع مسئله GPU Utilization
======================================

مسئله شما: GPU utilization بین 40% و 2% نوسان می‌کند
علت: Data loading inefficiency و CPU-GPU synchronization مشکلات

راه‌حل‌های پیاده‌سازی شده:
1. GPU Utilization Optimizer
2. Async Data Prefetching  
3. Optimized DataLoader
4. Real-time Monitoring
5. Memory Management

"""

import subprocess
import time
import sys
from pathlib import Path

def check_requirements():
    """بررسی requirements لازم"""
    print("🔍 Checking requirements...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except ImportError:
        print("❌ PyTorch not found")
        return False
    
    try:
        import pynvml
        print("✅ pynvml available for GPU monitoring")
    except ImportError:
        print("⚠️  pynvml not found, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "nvidia-ml-py"], check=True)
    
    try:
        import psutil
        print("✅ psutil available")
    except ImportError:
        print("⚠️  psutil not found, installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    
    return True

def show_gpu_optimization_guide():
    """نمایش راهنمای بهینه‌سازی GPU"""
    print("""
🎯 راهنمای رفع مسئله GPU Utilization
=====================================

مسئله شما:
- GPU utilization بین 40% و 2% نوسان می‌کند
- مدل در حال training است ولی GPU به طور کامل استفاده نمی‌شود
- داده‌ها تیکه تیکه وارد GPU می‌شوند

علل اصلی:
1. Data loading bottleneck (CPU کند است)
2. Synchronous data transfer (CPU منتظر GPU می‌ماند)
3. Insufficient prefetching (داده‌ها آماده نیستند)
4. Memory management ناکارآمد
5. Worker thread های کم

راه‌حل‌های پیاده‌سازی شده:
==========================

1. 🚀 GPU Utilization Optimizer:
   - Async data prefetching
   - Multi-threaded data loading
   - GPU memory pool management
   - Real-time monitoring

2. 📊 DataLoader Optimizations:
   - Increased num_workers (8-16)
   - Higher prefetch_factor (4-8)
   - Persistent workers
   - Pin memory enabled
   - Drop last batch for consistency

3. 🧠 Memory Management:
   - Optimized memory fraction
   - Gradient checkpointing
   - Memory cleanup intervals
   - Async GPU transfers

4. 📈 Real-time Monitoring:
   - GPU utilization tracking
   - Memory usage monitoring
   - Performance recommendations
   - Automatic optimization

استفاده:
========

روش 1 - اسکریپت آماده:
./train_gpu_optimized.sh

روش 2 - دستی:
python3 train_main.py --config config_gpu_utilization_optimized.yaml --model-size tiny

روش 3 - با مانیتورینگ کامل:
python3 train_main.py --model-size tiny --optimization-level enhanced --apply-fast-convergence

نتایج مورد انتظار:
==================
- GPU utilization: 80-95% (stable)
- Memory usage: 70-85%
- Training speed: 2-3x بهتر
- کاهش data loading latency
- Stable performance بدون نوسان

""")

def run_gpu_benchmark():
    """اجرای benchmark GPU utilization"""
    print("🧪 Running GPU utilization benchmark...")
    
    try:
        from gpu_utilization_optimizer import create_gpu_optimizer
        import torch
        
        if not torch.cuda.is_available():
            print("❌ CUDA not available")
            return
        
        device = torch.device('cuda')
        optimizer = create_gpu_optimizer(device)
        
        print("📊 Monitoring GPU for 30 seconds...")
        
        for i in range(30):
            stats = optimizer.monitor_gpu_utilization()
            if stats:
                gpu_util = stats.get('gpu_utilization', 0)
                memory_util = stats.get('memory_used_percent', 0)
                print(f"Step {i+1:2d}: GPU={gpu_util:3.0f}%, Memory={memory_util:5.1f}%")
            
            time.sleep(1)
        
        recommendations = optimizer.get_optimization_recommendations()
        if recommendations.get('recommendations'):
            print("\n💡 Recommendations:")
            for rec in recommendations['recommendations']:
                print(f"   - {rec['issue']}: {rec['solution']}")
        
        optimizer.stop_optimization()
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def test_optimized_training():
    """تست training بهینه‌شده"""
    print("\n🎯 Testing optimized training setup...")
    
    config_file = "config_gpu_utilization_optimized.yaml"
    if not Path(config_file).exists():
        print(f"❌ Config file not found: {config_file}")
        print("Run: python3 gpu_utilization_config.py")
        return
    
    print(f"✅ Config file found: {config_file}")
    
    # Test with dry run
    cmd = [
        "python3", "train_main.py",
        "--config", config_file,
        "--model-size", "tiny",
        "--epochs", "1",  # Just one epoch for testing
        "--optimization-level", "enhanced"
    ]
    
    print(f"Test command: {' '.join(cmd)}")
    print("This would start optimized training with GPU utilization monitoring.")
    print("The actual training will show real-time GPU stats and recommendations.")

def main():
    """اجرای اصلی"""
    print("🔧 MyXTTS GPU Utilization Fix")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("❌ Requirements check failed")
        return
    
    # Show guide
    show_gpu_optimization_guide()
    
    # Interactive menu
    while True:
        print("\nانتخاب کنید:")
        print("1. اجرای benchmark GPU utilization")
        print("2. تست training بهینه‌شده")
        print("3. مشاهده راهنمای کامل")
        print("4. خروج")
        
        choice = input("انتخاب شما (1-4): ").strip()
        
        if choice == "1":
            run_gpu_benchmark()
        elif choice == "2":
            test_optimized_training()
        elif choice == "3":
            show_gpu_optimization_guide()
        elif choice == "4":
            print("✅ خروج")
            break
        else:
            print("❌ انتخاب نامعتبر")

if __name__ == "__main__":
    main()