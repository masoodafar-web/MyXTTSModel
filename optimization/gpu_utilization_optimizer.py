#!/usr/bin/env python3
"""
GPU Utilization Optimizer for MyXTTS Training
==============================================

راه‌حل مسئله GPU utilization که بین 40% و 2% نوسان می‌کند
این مسئله ناشی از:
1. Inefficient data loading
2. GPU memory management issues
3. CPU-GPU synchronization problems
4. Poor prefetching strategies

این اسکریپت راه‌حل جامعی ارائه می‌دهد.
"""

import torch
import threading
import queue
import time
import psutil
import gc
from typing import Dict, Any, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class GPUUtilizationOptimizer:
    """بهینه‌ساز GPU utilization برای رفع مشکل نوسان GPU usage"""
    
    def __init__(self, 
                 device: torch.device,
                 max_prefetch_batches: int = 8,
                 enable_async_loading: bool = True,
                 memory_fraction: float = 0.85):
        """
        Args:
            device: GPU device
            max_prefetch_batches: تعداد batch های prefetch شده
            enable_async_loading: فعال‌سازی async data loading
            memory_fraction: درصد memory GPU که استفاده می‌شود
        """
        self.device = device
        self.max_prefetch_batches = max_prefetch_batches
        self.enable_async_loading = enable_async_loading
        self.memory_fraction = memory_fraction
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Prefetch queue
        self.prefetch_queue = queue.Queue(maxsize=max_prefetch_batches)
        self.prefetch_thread = None
        self.stop_prefetch = threading.Event()
        
        # Memory management
        self._setup_memory_management()
        
        # Statistics
        self.stats = {
            'gpu_utilization': [],
            'memory_usage': [],
            'data_loading_time': [],
            'processing_time': []
        }
    
    def _setup_memory_management(self):
        """تنظیم memory management برای GPU"""
        if torch.cuda.is_available():
            # Set memory fraction
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            
            # Enable memory efficient attention if available
            if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                torch.backends.cuda.enable_math_sdp(True)
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            
            # Allocate memory pool
            torch.cuda.empty_cache()
            
            print(f"✅ GPU Memory management configured:")
            print(f"   - Memory fraction: {self.memory_fraction}")
            print(f"   - Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    def create_optimized_dataloader(self, dataset, batch_size: int, num_workers: int = None) -> torch.utils.data.DataLoader:
        """ایجاد DataLoader بهینه‌شده برای GPU utilization"""
        
        # Auto-determine optimal num_workers
        if num_workers is None:
            cpu_count = psutil.cpu_count(logical=False)
            num_workers = min(cpu_count * 2, 16)  # حداکثر 16 worker
        
        # Optimized DataLoader settings
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,  # افزایش prefetch factor
            drop_last=True,
            multiprocessing_context='spawn' if num_workers > 0 else None,
            worker_init_fn=self._worker_init_fn,
        )
        
        print(f"✅ Optimized DataLoader created:")
        print(f"   - Batch size: {batch_size}")
        print(f"   - Num workers: {num_workers}")
        print(f"   - Prefetch factor: 4")
        print(f"   - Persistent workers: True")
        
        return dataloader
    
    def _worker_init_fn(self, worker_id):
        """تنظیمات اولیه برای worker threads"""
        # Set CPU affinity for better performance
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Distribute workers across CPU cores
            cpu_count = psutil.cpu_count(logical=False)
            cpu_id = worker_id % cpu_count
            try:
                psutil.Process().cpu_affinity([cpu_id])
            except:
                pass  # Ignore if can't set affinity
    
    def start_async_prefetch(self, dataloader):
        """شروع async prefetching برای کاهش data loading latency"""
        if not self.enable_async_loading:
            return
        
        def prefetch_worker():
            data_iterator = iter(dataloader)
            while not self.stop_prefetch.is_set():
                try:
                    batch = next(data_iterator)
                    # Move to GPU asynchronously
                    if isinstance(batch, (list, tuple)):
                        batch = [item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item 
                                for item in batch]
                    elif isinstance(batch, dict):
                        batch = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                                for k, v in batch.items()}
                    else:
                        batch = batch.to(self.device, non_blocking=True)
                    
                    self.prefetch_queue.put(batch, timeout=1.0)
                    
                except StopIteration:
                    data_iterator = iter(dataloader)
                    continue
                except queue.Full:
                    continue
                except Exception as e:
                    print(f"⚠️ Prefetch error: {e}")
                    continue
        
        self.stop_prefetch.clear()
        self.prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        
        print("✅ Async prefetching started")
    
    def get_next_batch(self, fallback_batch=None, timeout: float = 0.1):
        """دریافت batch بعدی با prefetching"""
        if not self.enable_async_loading or self.prefetch_queue.empty():
            return fallback_batch
        
        try:
            return self.prefetch_queue.get(timeout=timeout)
        except queue.Empty:
            return fallback_batch
    
    def optimize_training_step(self, model, optimizer, batch, loss_fn, scaler=None):
        """بهینه‌سازی training step برای maximum GPU utilization"""
        
        start_time = time.time()
        
        # Ensure model is in training mode
        model.train()
        
        # Zero gradients with set_to_none for better performance
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(batch)
                loss = loss_fn(outputs, batch)
            
            # Scale loss and backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(batch)
            loss = loss_fn(outputs, batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        # Record statistics
        processing_time = time.time() - start_time
        self.stats['processing_time'].append(processing_time)
        
        # Memory management
        if len(self.stats['processing_time']) % 50 == 0:
            self._cleanup_memory()
        
        return loss, outputs
    
    def _cleanup_memory(self):
        """پاکسازی memory برای جلوگیری از memory fragmentation"""
        torch.cuda.empty_cache()
        gc.collect()
    
    def monitor_gpu_utilization(self) -> Dict[str, float]:
        """مانیتورینگ GPU utilization"""
        if not torch.cuda.is_available():
            return {}
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = util.gpu
            memory_util = util.memory
            
            # Memory usage
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used / mem_info.total * 100
            
            stats = {
                'gpu_utilization': gpu_util,
                'memory_utilization': memory_util,
                'memory_used_percent': memory_used,
                'memory_used_gb': mem_info.used / 1e9,
                'memory_total_gb': mem_info.total / 1e9
            }
            
            # Store for analysis
            self.stats['gpu_utilization'].append(gpu_util)
            self.stats['memory_usage'].append(memory_used)
            
            return stats
            
        except ImportError:
            print("⚠️ pynvml not available, using basic monitoring")
            return self._basic_gpu_monitoring()
        except Exception as e:
            print(f"⚠️ GPU monitoring error: {e}")
            return {}
    
    def _basic_gpu_monitoring(self) -> Dict[str, float]:
        """مانیتورینگ پایه GPU بدون pynvml"""
        if not torch.cuda.is_available():
            return {}
        
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        return {
            'memory_allocated_gb': memory_allocated,
            'memory_reserved_gb': memory_reserved,
            'memory_total_gb': memory_total,
            'memory_used_percent': (memory_reserved / memory_total) * 100
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """توصیه‌های بهینه‌سازی بر اساس آمار جمع‌آوری شده"""
        if not self.stats['gpu_utilization']:
            return {"message": "No statistics available yet"}
        
        avg_gpu_util = np.mean(self.stats['gpu_utilization'][-100:])  # Last 100 measurements
        gpu_util_std = np.std(self.stats['gpu_utilization'][-100:])
        
        recommendations = {
            'average_gpu_utilization': avg_gpu_util,
            'gpu_utilization_stability': gpu_util_std,
            'recommendations': []
        }
        
        # GPU utilization analysis
        if avg_gpu_util < 70:
            recommendations['recommendations'].append({
                'issue': 'Low GPU utilization',
                'solution': 'Increase batch size or reduce data loading time',
                'priority': 'high'
            })
        
        if gpu_util_std > 20:
            recommendations['recommendations'].append({
                'issue': 'Unstable GPU utilization',
                'solution': 'Enable async prefetching and increase prefetch buffer',
                'priority': 'high'
            })
        
        # Memory analysis
        if self.stats['memory_usage']:
            avg_memory = np.mean(self.stats['memory_usage'][-100:])
            if avg_memory > 90:
                recommendations['recommendations'].append({
                    'issue': 'High memory usage',
                    'solution': 'Reduce batch size or enable gradient checkpointing',
                    'priority': 'medium'
                })
        
        return recommendations
    
    def print_status(self):
        """چاپ وضعیت فعلی GPU"""
        gpu_stats = self.monitor_gpu_utilization()
        if gpu_stats:
            print(f"\n📊 GPU Status:")
            print(f"   - GPU Utilization: {gpu_stats.get('gpu_utilization', 'N/A')}%")
            print(f"   - Memory Usage: {gpu_stats.get('memory_used_percent', 'N/A'):.1f}%")
            print(f"   - Memory Used: {gpu_stats.get('memory_used_gb', 'N/A'):.1f} GB")
            print(f"   - Prefetch Queue Size: {self.prefetch_queue.qsize()}")
    
    def stop_optimization(self):
        """توقف بهینه‌سازی و تمیز کردن resources"""
        if self.prefetch_thread and self.prefetch_thread.is_alive():
            self.stop_prefetch.set()
            self.prefetch_thread.join(timeout=2.0)
        
        self.executor.shutdown(wait=False)
        self._cleanup_memory()
        
        print("✅ GPU optimization stopped")

def create_gpu_optimizer(device: torch.device, **kwargs) -> GPUUtilizationOptimizer:
    """Factory function برای ایجاد GPU optimizer"""
    return GPUUtilizationOptimizer(device, **kwargs)

# Test function
def test_gpu_optimizer():
    """تست GPU optimizer"""
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return
    
    device = torch.device('cuda')
    optimizer = create_gpu_optimizer(device)
    
    # Test monitoring
    print("🧪 Testing GPU monitoring...")
    stats = optimizer.monitor_gpu_utilization()
    print(f"   Stats: {stats}")
    
    # Test recommendations
    print("\n🧪 Testing recommendations...")
    # Simulate some stats
    optimizer.stats['gpu_utilization'] = [30, 50, 20, 60, 25] * 20
    recommendations = optimizer.get_optimization_recommendations()
    print(f"   Recommendations: {recommendations}")
    
    optimizer.stop_optimization()
    print("✅ Test completed")

if __name__ == "__main__":
    test_gpu_optimizer()