"""
Performance monitoring and profiling utilities for MyXTTS.

This module provides basic performance monitoring for CPU and memory usage,
focusing on training performance without detailed GPU monitoring.
"""

import time
import threading
import psutil
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    data_loading_time: float = 0.0
    model_compute_time: float = 0.0
    total_step_time: float = 0.0
    batch_size: int = 0
    samples_per_second: float = 0.0


class PerformanceMonitor:
    """
    Monitor system and training performance to identify bottlenecks.
    
    This class provides basic monitoring of CPU and memory usage,
    as well as timing of data loading and model computation phases.
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
    
    def start_monitoring(self):
        """Start background monitoring of system metrics."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        # Silence startup message to reduce notebook logs
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        # Silence stop message to reduce notebook logs
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                time.sleep(self.monitor_interval)
            except Exception as e:
                # Suppress noisy monitoring loop errors from flooding logs
                pass
                time.sleep(self.monitor_interval)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        metrics = PerformanceMetrics()
        
        # CPU and memory
        metrics.cpu_percent = psutil.cpu_percent(interval=None)
        metrics.memory_percent = psutil.virtual_memory().percent
        
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
                "memory_percent": avg_memory
            },
            "timing_analysis": timing_analysis,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "total_samples": len(self.metrics_history)
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a concise, program-friendly summary of recent performance.

        Returns:
            Dict with keys such as avg_step_time, avg_data_time,
            avg_compute_time, avg_samples_per_second, and recent system metrics.
        """
        # Timing summaries
        def _avg(name: str, last_n: int = 50) -> float:
            vals = self.timing_stats.get(name, [])
            if not vals:
                return 0.0
            vals = vals[-last_n:]
            try:
                return float(np.mean(vals))
            except Exception:
                return 0.0

        avg_data_time = _avg("data_loading")
        avg_compute_time = _avg("model_compute")
        avg_step_time = _avg("total_step") or (avg_data_time + avg_compute_time)
        avg_sps = _avg("samples_per_second")

        # System metrics (recent average)
        if self.metrics_history:
            recent = list(self.metrics_history)[-10:]
            avg_cpu = float(np.mean([m.cpu_percent for m in recent]))
            avg_mem = float(np.mean([m.memory_percent for m in recent]))
            latest = recent[-1]
            batch_size = int(latest.batch_size or 0)
        else:
            avg_cpu = avg_mem = 0.0
            batch_size = 0

        return {
            "avg_step_time": avg_step_time,
            "avg_data_time": avg_data_time,
            "avg_compute_time": avg_compute_time,
            "avg_samples_per_second": avg_sps,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_mem,
            "batch_size": batch_size,
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
