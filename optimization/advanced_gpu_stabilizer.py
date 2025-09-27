#!/usr/bin/env python3
"""
Advanced GPU Utilization Stabilizer
Comprehensive solution for MyXTTS GPU utilization fluctuation issues

This module provides advanced GPU optimization specifically designed to solve
the fluctuating GPU utilization problem in MyXTTS training.

Key Features:
- Aggressive async data prefetching with multiple threads
- GPU memory pinning and optimization
- Smart batch queue management
- Real-time GPU utilization monitoring and adjustment
- Automatic bottleneck detection and resolution
"""

import os
import time
import threading
import queue
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple, Any
import logging

import tensorflow as tf
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import psutil

logger = logging.getLogger(__name__)

class AdvancedGPUStabilizer:
    """Advanced GPU utilization stabilizer for consistent performance"""
    
    def __init__(self, 
                 max_prefetch_batches: int = 16,
                 num_prefetch_threads: int = 8,
                 memory_fraction: float = 0.9,
                 enable_memory_pinning: bool = True,
                 aggressive_mode: bool = True):
        
        self.max_prefetch_batches = max_prefetch_batches
        self.num_prefetch_threads = num_prefetch_threads
        self.memory_fraction = memory_fraction
        self.enable_memory_pinning = enable_memory_pinning
        self.aggressive_mode = aggressive_mode
        
        # State management
        self.is_active = False
        self.prefetch_queue = queue.Queue(maxsize=max_prefetch_batches)
        self.executor = None
        self.monitoring_thread = None
        self.stats = {
            'batches_processed': 0,
            'gpu_utilization_history': [],
            'data_loading_times': [],
            'gpu_waiting_times': [],
            'last_adjustment': time.time()
        }
        
        # Configure GPU memory
        self._configure_gpu_memory()
        
        # Initialize monitoring
        self._setup_monitoring()
        
        logger.info(f"🚀 Advanced GPU Stabilizer initialized:")
        logger.info(f"   • Prefetch batches: {max_prefetch_batches}")
        logger.info(f"   • Prefetch threads: {num_prefetch_threads}")
        logger.info(f"   • Memory fraction: {memory_fraction}")
        logger.info(f"   • Aggressive mode: {aggressive_mode}")
    
    def _configure_gpu_memory(self):
        """Configure GPU memory for optimal performance"""
        try:
            # TensorFlow GPU configuration
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    
                # Set memory fraction
                tf.config.experimental.set_virtual_device_configuration(
                    physical_devices[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=int(25600 * self.memory_fraction)  # 25.6GB * fraction
                    )]
                )
                
                logger.info(f"✅ GPU memory configured: {self.memory_fraction*100:.1f}% allocation")
            
            # PyTorch GPU configuration
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
                if self.enable_memory_pinning:
                    torch.cuda.empty_cache()
                    
                logger.info(f"✅ PyTorch GPU memory configured")
                
        except Exception as e:
            logger.warning(f"GPU memory configuration failed: {e}")
    
    def _setup_monitoring(self):
        """Setup GPU utilization monitoring"""
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitor_gpu_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("✅ GPU monitoring thread started")
    
    def _monitor_gpu_loop(self):
        """Continuous GPU monitoring loop"""
        while self.monitoring_active:
            try:
                gpu_util = self._get_gpu_utilization()
                if gpu_util is not None:
                    self.stats['gpu_utilization_history'].append({
                        'timestamp': time.time(),
                        'utilization': gpu_util
                    })
                    
                    # Keep only last 100 readings
                    if len(self.stats['gpu_utilization_history']) > 100:
                        self.stats['gpu_utilization_history'] = self.stats['gpu_utilization_history'][-100:]
                    
                    # Auto-adjust if utilization is low
                    if gpu_util < 50 and self.is_active:
                        self._auto_adjust_settings()
                
                time.sleep(0.5)  # Monitor every 500ms
                
            except Exception as e:
                logger.debug(f"Monitoring error: {e}")
                time.sleep(1.0)
    
    def _get_gpu_utilization(self) -> Optional[float]:
        """Get current GPU utilization"""
        try:
            import subprocess
            result = subprocess.run([
                'nvidia-smi', '--query-gpu=utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if lines and lines[0].strip():
                    return float(lines[0].strip())
        except Exception:
            pass
        return None
    
    def _auto_adjust_settings(self):
        """Automatically adjust settings for better GPU utilization"""
        current_time = time.time()
        if current_time - self.stats['last_adjustment'] < 10:  # Don't adjust too frequently
            return
            
        self.stats['last_adjustment'] = current_time
        
        # Get recent utilization
        recent_utils = [entry['utilization'] for entry in self.stats['gpu_utilization_history'][-10:]]
        if not recent_utils:
            return
            
        avg_util = sum(recent_utils) / len(recent_utils)
        
        if avg_util < 30:
            # Very low utilization - increase aggressiveness
            self.max_prefetch_batches = min(32, self.max_prefetch_batches + 4)
            self.num_prefetch_threads = min(16, self.num_prefetch_threads + 2)
            logger.info(f"📈 Auto-adjusted: prefetch_batches={self.max_prefetch_batches}, threads={self.num_prefetch_threads}")
            
        elif avg_util < 50:
            # Low utilization - moderate increase
            self.max_prefetch_batches = min(24, self.max_prefetch_batches + 2)
            logger.info(f"📈 Auto-adjusted: prefetch_batches={self.max_prefetch_batches}")
    
    def create_optimized_dataloader(self, 
                                  dataset: Dataset,
                                  batch_size: int,
                                  num_workers: int = None,
                                  shuffle: bool = True) -> DataLoader:
        """Create highly optimized DataLoader for stable GPU utilization"""
        
        if num_workers is None:
            # Use more workers for better parallelism
            num_workers = min(16, max(8, mp.cpu_count()))
        
        # Optimized DataLoader settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=self.enable_memory_pinning and torch.cuda.is_available(),
            drop_last=True,  # Consistent batch sizes
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=8,  # Aggressive prefetching
            multiprocessing_context='spawn' if os.name != 'nt' else None,
            timeout=60,  # Prevent hanging
        )
        
        logger.info(f"✅ Optimized DataLoader created:")
        logger.info(f"   • Batch size: {batch_size}")
        logger.info(f"   • Workers: {num_workers}")
        logger.info(f"   • Pin memory: {dataloader.pin_memory}")
        logger.info(f"   • Prefetch factor: {dataloader.prefetch_factor}")
        
        return dataloader
    
    def start_aggressive_prefetching(self, dataloader: DataLoader):
        """Start aggressive data prefetching to eliminate GPU starvation"""
        if self.is_active:
            logger.warning("Prefetching already active")
            return
        
        self.is_active = True
        self.dataloader = dataloader
        
        # Create prefetch executor
        self.executor = ThreadPoolExecutor(max_workers=self.num_prefetch_threads)
        
        # Start prefetch threads
        for i in range(self.num_prefetch_threads):
            self.executor.submit(self._prefetch_worker, i)
        
        logger.info(f"🚀 Aggressive prefetching started with {self.num_prefetch_threads} threads")
    
    def start_aggressive_prefetching_tf(self, dataset_iter):
        """Start aggressive data prefetching for TensorFlow datasets"""
        if self.is_active:
            logger.warning("Prefetching already active")
            return
        
        self.is_active = True
        self.dataset_iter = dataset_iter
        
        # Create prefetch executor
        self.executor = ThreadPoolExecutor(max_workers=self.num_prefetch_threads)
        
        # Start prefetch threads for TensorFlow
        for i in range(self.num_prefetch_threads):
            self.executor.submit(self._prefetch_worker_tf, i)
        
        logger.info(f"🚀 TensorFlow aggressive prefetching started with {self.num_prefetch_threads} threads")
    
    def _prefetch_worker(self, worker_id: int):
        """Worker thread for aggressive data prefetching"""
        logger.debug(f"Prefetch worker {worker_id} started")
        
        data_iter = iter(self.dataloader)
        
        while self.is_active:
            try:
                # Load next batch
                start_time = time.time()
                batch = next(data_iter)
                load_time = time.time() - start_time
                
                # Record loading time
                self.stats['data_loading_times'].append(load_time)
                if len(self.stats['data_loading_times']) > 100:
                    self.stats['data_loading_times'] = self.stats['data_loading_times'][-100:]
                
                # Pin to GPU memory if enabled
                if self.enable_memory_pinning and torch.cuda.is_available():
                    if isinstance(batch, (list, tuple)):
                        batch = [item.cuda(non_blocking=True) if hasattr(item, 'cuda') else item for item in batch]
                    elif hasattr(batch, 'cuda'):
                        batch = batch.cuda(non_blocking=True)
                
                # Add to prefetch queue (blocks if queue is full)
                self.prefetch_queue.put(batch, timeout=30)
                self.stats['batches_processed'] += 1
                
            except StopIteration:
                # Restart iterator for continuous training
                data_iter = iter(self.dataloader)
                continue
            except queue.Full:
                # Queue is full, GPU is processing - this is good!
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"Prefetch worker {worker_id} error: {e}")
                time.sleep(0.1)
                continue
    
    def _prefetch_worker_tf(self, worker_id: int):
        """Worker thread for TensorFlow dataset prefetching"""
        logger.debug(f"TF Prefetch worker {worker_id} started")
        
        while self.is_active:
            try:
                # Load next batch from TensorFlow dataset
                start_time = time.time()
                batch = next(self.dataset_iter)
                load_time = time.time() - start_time
                
                # Record loading time
                self.stats['data_loading_times'].append(load_time)
                if len(self.stats['data_loading_times']) > 100:
                    self.stats['data_loading_times'] = self.stats['data_loading_times'][-100:]
                
                # TensorFlow tensors are already on GPU if placed correctly
                # No need for manual CUDA placement
                
                # Add to prefetch queue (blocks if queue is full)
                self.prefetch_queue.put(batch, timeout=30)
                self.stats['batches_processed'] += 1
                
            except StopIteration:
                # Dataset exhausted - this is normal for TF datasets
                logger.debug(f"TF worker {worker_id}: Dataset exhausted")
                time.sleep(0.1)
                continue
            except queue.Full:
                # Queue is full, GPU is processing - this is good!
                time.sleep(0.01)
                continue
            except Exception as e:
                logger.error(f"TF Prefetch worker {worker_id} error: {e}")
                time.sleep(0.1)
                continue
    
    def get_next_batch(self, timeout: float = 30) -> Any:
        """Get next batch from prefetch queue"""
        try:
            start_time = time.time()
            batch = self.prefetch_queue.get(timeout=timeout)
            wait_time = time.time() - start_time
            
            # Record GPU waiting time
            self.stats['gpu_waiting_times'].append(wait_time)
            if len(self.stats['gpu_waiting_times']) > 100:
                self.stats['gpu_waiting_times'] = self.stats['gpu_waiting_times'][-100:]
            
            return batch
            
        except queue.Empty:
            logger.warning("⚠️  GPU starved! No batch available in prefetch queue")
            raise RuntimeError("GPU starvation detected - data loading too slow")
    
    def get_next_batch_tf(self, timeout: float = 30) -> Any:
        """Get next batch from prefetch queue for TensorFlow"""
        return self.get_next_batch(timeout=timeout)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics"""
        recent_gpu_utils = [entry['utilization'] for entry in self.stats['gpu_utilization_history'][-10:]]
        avg_gpu_util = sum(recent_gpu_utils) / len(recent_gpu_utils) if recent_gpu_utils else 0
        
        recent_load_times = self.stats['data_loading_times'][-10:] if self.stats['data_loading_times'] else []
        avg_load_time = sum(recent_load_times) / len(recent_load_times) if recent_load_times else 0
        
        recent_wait_times = self.stats['gpu_waiting_times'][-10:] if self.stats['gpu_waiting_times'] else []
        avg_wait_time = sum(recent_wait_times) / len(recent_wait_times) if recent_wait_times else 0
        
        queue_size = self.prefetch_queue.qsize() if hasattr(self, 'prefetch_queue') else 0
        
        return {
            'is_active': self.is_active,
            'avg_gpu_utilization': avg_gpu_util,
            'avg_data_load_time': avg_load_time,
            'avg_gpu_wait_time': avg_wait_time,
            'prefetch_queue_size': queue_size,
            'max_queue_size': self.max_prefetch_batches,
            'prefetch_threads': self.num_prefetch_threads,
            'batches_processed': self.stats['batches_processed'],
            'status': self._get_status_message()
        }
    
    def _get_status_message(self) -> str:
        """Get human-readable status message"""
        recent_utils = [entry['utilization'] for entry in self.stats['gpu_utilization_history'][-5:]]
        if not recent_utils:
            return "Initializing..."
        
        avg_util = sum(recent_utils) / len(recent_utils)
        
        if avg_util >= 85:
            return "🟢 Excellent GPU utilization"
        elif avg_util >= 70:
            return "🟡 Good GPU utilization"
        elif avg_util >= 50:
            return "🟠 Moderate GPU utilization"
        else:
            return "🔴 Poor GPU utilization - investigating..."
    
    def print_detailed_status(self):
        """Print detailed status information"""
        status = self.get_optimization_status()
        
        print("\n" + "="*60)
        print("🔧 ADVANCED GPU STABILIZER STATUS")
        print("="*60)
        print(f"Status: {status['status']}")
        print(f"GPU Utilization: {status['avg_gpu_utilization']:.1f}%")
        print(f"Data Load Time: {status['avg_data_load_time']*1000:.1f}ms")
        print(f"GPU Wait Time: {status['avg_gpu_wait_time']*1000:.1f}ms")
        print(f"Prefetch Queue: {status['prefetch_queue_size']}/{status['max_queue_size']}")
        print(f"Prefetch Threads: {status['prefetch_threads']}")
        print(f"Batches Processed: {status['batches_processed']}")
        print("="*60)
        
        # Show recommendations
        recommendations = self._get_recommendations(status)
        if recommendations:
            print("💡 OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"{i}. {rec}")
            print("="*60)
    
    def _get_recommendations(self, status: Dict) -> List[str]:
        """Get optimization recommendations"""
        recommendations = []
        
        if status['avg_gpu_utilization'] < 50:
            recommendations.append("Increase batch size or prefetch threads")
            recommendations.append("Check for CPU bottlenecks in data preprocessing")
        
        if status['avg_data_load_time'] > 0.1:  # >100ms
            recommendations.append("Data loading is slow - consider caching or faster storage")
        
        if status['avg_gpu_wait_time'] > 0.01:  # >10ms
            recommendations.append("GPU is waiting for data - increase prefetch buffer")
        
        if status['prefetch_queue_size'] < status['max_queue_size'] * 0.5:
            recommendations.append("Prefetch queue underutilized - increase threads or batch size")
        
        return recommendations
    
    def stop_optimization(self):
        """Stop all optimization processes"""
        logger.info("🛑 Stopping GPU optimization...")
        
        self.is_active = False
        self.monitoring_active = False
        
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Clear prefetch queue
        if hasattr(self, 'prefetch_queue'):
            while not self.prefetch_queue.empty():
                try:
                    self.prefetch_queue.get_nowait()
                except queue.Empty:
                    break
        
        logger.info("✅ GPU optimization stopped")


def create_advanced_gpu_stabilizer(**kwargs) -> AdvancedGPUStabilizer:
    """Factory function to create Advanced GPU Stabilizer"""
    return AdvancedGPUStabilizer(**kwargs)


# Integration wrapper for existing code
class StabilizedDataLoaderWrapper:
    """Wrapper to integrate stabilizer with existing training code"""
    
    def __init__(self, dataloader: DataLoader, stabilizer: AdvancedGPUStabilizer):
        self.dataloader = dataloader
        self.stabilizer = stabilizer
        self.started = False
    
    def __iter__(self):
        if not self.started:
            self.stabilizer.start_aggressive_prefetching(self.dataloader)
            self.started = True
        
        return self
    
    def __next__(self):
        return self.stabilizer.get_next_batch()
    
    def __len__(self):
        return len(self.dataloader)


# Quick setup function
def setup_advanced_gpu_optimization(dataloader: DataLoader, 
                                   aggressive: bool = True) -> Tuple[AdvancedGPUStabilizer, StabilizedDataLoaderWrapper]:
    """Quick setup for advanced GPU optimization"""
    
    # Create stabilizer with aggressive settings
    stabilizer = create_advanced_gpu_stabilizer(
        max_prefetch_batches=32 if aggressive else 16,
        num_prefetch_threads=12 if aggressive else 8,
        memory_fraction=0.9,
        enable_memory_pinning=True,
        aggressive_mode=aggressive
    )
    
    # Create wrapper
    wrapper = StabilizedDataLoaderWrapper(dataloader, stabilizer)
    
    logger.info("🚀 Advanced GPU optimization setup complete!")
    
    return stabilizer, wrapper


if __name__ == "__main__":
    # Test the stabilizer
    print("🧪 Testing Advanced GPU Stabilizer...")
    
    stabilizer = create_advanced_gpu_stabilizer(aggressive_mode=True)
    
    # Print status for 10 seconds
    for i in range(20):
        time.sleep(0.5)
        if i % 4 == 0:  # Every 2 seconds
            stabilizer.print_detailed_status()
    
    stabilizer.stop_optimization()
    print("✅ Test completed")