"""
Performance monitoring and profiling utilities for MyXTTS.

This module provides tools to monitor CPU/GPU utilization, data loading 
bottlenecks, and overall training performance.
"""

import time
import threading
import psutil
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_memory_total: float = 0.0
    gpu_utilization: float = 0.0
    data_loading_time: float = 0.0
    model_compute_time: float = 0.0
    total_step_time: float = 0.0
    batch_size: int = 0
    samples_per_second: float = 0.0


class PerformanceMonitor:
    """
    Monitor system and training performance to identify bottlenecks.
    
    This class provides real-time monitoring of CPU, GPU, and memory usage,
    as well as detailed timing of data loading and model computation phases.
    """
    
    def __init__(self, monitor_interval: float = 1.0, history_size: int = 1000):
        """
        Initialize performance monitor.
        
        Args:
            monitor_interval: How often to sample system metrics (seconds)
            history_size: Number of metrics to keep in history
        """
        self.monitor_interval = monitor_interval
        self.history_size = history_size
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=history_size)
        self.timing_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # GPU availability check
        self.has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        
        # Initialize GPU monitoring if available
        if self.has_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_available = True
            except (ImportError, Exception):
                self._gpu_available = False
        else:
            self._gpu_available = False
    
    def start_monitoring(self):
        """Start background monitoring of system metrics."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        print("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitor_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        metrics = PerformanceMetrics()
        
        # CPU and memory
        metrics.cpu_percent = psutil.cpu_percent(interval=None)
        metrics.memory_percent = psutil.virtual_memory().percent
        
        # GPU metrics if available
        if self._gpu_available:
            try:
                import pynvml
                # Memory usage
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                metrics.gpu_memory_used = mem_info.used / 1024**3  # GB
                metrics.gpu_memory_total = mem_info.total / 1024**3  # GB
                
                # GPU utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                metrics.gpu_utilization = util.gpu
            except Exception:
                pass
        
        return metrics
    
    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            
        Example:
            with monitor.time_operation("data_loading"):
                batch = next(data_iterator)
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration = end_time - start_time
            self.timing_stats[operation_name].append(duration)
            
            # Keep only recent timings
            if len(self.timing_stats[operation_name]) > self.history_size:
                self.timing_stats[operation_name] = self.timing_stats[operation_name][-self.history_size:]
    
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """
        Analyze current performance and identify bottlenecks.
        
        Returns:
            Dictionary with bottleneck analysis and recommendations
        """
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        
        # Calculate averages
        avg_cpu = np.mean([m.cpu_percent for m in recent_metrics])
        avg_memory = np.mean([m.memory_percent for m in recent_metrics])
        avg_gpu_util = np.mean([m.gpu_utilization for m in recent_metrics])
        
        # Analyze timing stats
        timing_analysis = {}
        for op_name, timings in self.timing_stats.items():
            if timings:
                recent_timings = timings[-50:]  # Last 50 operations
                timing_analysis[op_name] = {
                    "mean_time": np.mean(recent_timings),
                    "std_time": np.std(recent_timings),
                    "min_time": np.min(recent_timings),
                    "max_time": np.max(recent_timings),
                    "samples": len(recent_timings)
                }
        
        # Bottleneck detection
        bottlenecks = []
        recommendations = []
        
        # High CPU usage suggests CPU bottleneck
        if avg_cpu > 90:
            bottlenecks.append("high_cpu_usage")
            recommendations.append("CPU usage is very high. Consider optimizing data preprocessing or using more efficient algorithms.")
        
        # Low GPU utilization suggests data loading bottleneck
        if self.has_gpu and avg_gpu_util < 30:
            bottlenecks.append("critical_low_gpu_utilization")
            recommendations.append(f"GPU utilization is critically low ({avg_gpu_util:.1f}%). This strongly suggests a data loading bottleneck or model not properly placed on GPU. Check: 1) Data preprocessing happening on CPU during training, 2) Insufficient data prefetching, 3) Model device placement, 4) Batch size too small.")
        elif self.has_gpu and avg_gpu_util < 60:
            bottlenecks.append("low_gpu_utilization")
            recommendations.append(f"GPU utilization is suboptimal ({avg_gpu_util:.1f}%). Consider: 1) Increasing prefetch buffer size, 2) Using more parallel data loading workers, 3) Optimizing data preprocessing, 4) Increasing batch size if memory allows.")
        
        # High memory usage
        if avg_memory > 85:
            bottlenecks.append("high_memory_usage")
            recommendations.append("Memory usage is high. Consider reducing batch size or using memory mapping for large datasets.")
        
        # Data loading vs compute time analysis
        if "data_loading" in timing_analysis and "model_compute" in timing_analysis:
            data_time = timing_analysis["data_loading"]["mean_time"]
            compute_time = timing_analysis["model_compute"]["mean_time"]
            
            if data_time > compute_time * 0.5:  # Data loading takes more than 50% of compute time
                bottlenecks.append("data_loading_bottleneck")
                recommendations.append(f"Data loading time ({data_time:.3f}s) is significant compared to compute time ({compute_time:.3f}s). Consider optimizing data loading pipeline.")
        
        return {
            "system_metrics": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "gpu_utilization": avg_gpu_util,
                "has_gpu": self.has_gpu
            },
            "timing_analysis": timing_analysis,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "total_samples": len(self.metrics_history)
        }
    
    def get_summary_report(self) -> str:
        """Generate a human-readable performance summary."""
        analysis = self.get_bottleneck_analysis()
        
        if "error" in analysis:
            return analysis["error"]
        
        report = "=== MyXTTS Performance Analysis ===\n\n"
        
        # System metrics
        metrics = analysis["system_metrics"]
        report += f"System Metrics (Average):\n"
        report += f"  CPU Usage: {metrics['cpu_percent']:.1f}%\n"
        report += f"  Memory Usage: {metrics['memory_percent']:.1f}%\n"
        if metrics["has_gpu"]:
            report += f"  GPU Utilization: {metrics['gpu_utilization']:.1f}%\n"
        else:
            report += f"  GPU: Not available\n"
        report += "\n"
        
        # Timing analysis
        timing = analysis["timing_analysis"]
        if timing:
            report += "Operation Timings:\n"
            for op_name, stats in timing.items():
                report += f"  {op_name}: {stats['mean_time']:.3f}s ±{stats['std_time']:.3f}s (n={stats['samples']})\n"
            report += "\n"
        
        # Bottlenecks and recommendations
        if analysis["bottlenecks"]:
            report += "⚠️  Detected Bottlenecks:\n"
            for bottleneck in analysis["bottlenecks"]:
                report += f"  - {bottleneck}\n"
            report += "\n"
        
        if analysis["recommendations"]:
            report += "💡 Recommendations:\n"
            for i, rec in enumerate(analysis["recommendations"], 1):
                report += f"  {i}. {rec}\n"
        else:
            report += "✅ No major bottlenecks detected!\n"
        
        return report
    
    def log_step_timing(
        self, 
        data_time: float, 
        compute_time: float, 
        batch_size: int
    ):
        """
        Log timing for a complete training step.
        
        Args:
            data_time: Time spent loading data
            compute_time: Time spent in model computation
            batch_size: Number of samples in the batch
        """
        total_time = data_time + compute_time
        samples_per_second = batch_size / total_time if total_time > 0 else 0
        
        # Store in timing stats
        self.timing_stats["data_loading"].append(data_time)
        self.timing_stats["model_compute"].append(compute_time)
        self.timing_stats["total_step"].append(total_time)
        self.timing_stats["samples_per_second"].append(samples_per_second)
        
        # Update latest metrics
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.data_loading_time = data_time
            latest.model_compute_time = compute_time
            latest.total_step_time = total_time
            latest.batch_size = batch_size
            latest.samples_per_second = samples_per_second


class DataLoadingProfiler:
    """
    Specialized profiler for data loading operations.
    
    This class provides detailed profiling of the data loading pipeline
    to identify specific bottlenecks in dataset preparation.
    """
    
    def __init__(self):
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.cache_stats = {"hits": 0, "misses": 0, "errors": 0}
    
    @contextmanager
    def profile_operation(self, operation: str):
        """Profile a specific data loading operation."""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            end_time = time.perf_counter()
            self.operation_times[operation].append(end_time - start_time)
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_stats["hits"] += 1
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_stats["misses"] += 1
    
    def record_cache_error(self):
        """Record a cache error."""
        self.cache_stats["errors"] += 1
    
    def get_cache_efficiency(self) -> float:
        """Calculate cache hit rate."""
        total = sum(self.cache_stats.values())
        if total == 0:
            return 0.0
        return self.cache_stats["hits"] / total
    
    def get_report(self) -> str:
        """Generate detailed data loading report."""
        report = "=== Data Loading Profile ===\n\n"
        
        # Cache statistics
        efficiency = self.get_cache_efficiency()
        report += f"Cache Efficiency: {efficiency:.1%}\n"
        report += f"  Hits: {self.cache_stats['hits']}\n"
        report += f"  Misses: {self.cache_stats['misses']}\n"
        report += f"  Errors: {self.cache_stats['errors']}\n\n"
        
        # Operation timings
        if self.operation_times:
            report += "Operation Timings:\n"
            for op, times in self.operation_times.items():
                if times:
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    report += f"  {op}: {mean_time:.4f}s ±{std_time:.4f}s (n={len(times)})\n"
        
        return report


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def start_performance_monitoring():
    """Start global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.start_monitoring()


def stop_performance_monitoring():
    """Stop global performance monitoring."""
    monitor = get_performance_monitor()
    monitor.stop_monitoring()


def print_performance_report():
    """Print the current performance report."""
    monitor = get_performance_monitor()
    print(monitor.get_summary_report())


# Context manager for easy timing
@contextmanager
def time_operation(operation_name: str):
    """Time an operation using the global monitor."""
    monitor = get_performance_monitor()
    with monitor.time_operation(operation_name):
        yield