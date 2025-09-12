#!/usr/bin/env python3
"""
GPU Monitoring Script for MyXTTS Training

This script provides real-time monitoring of GPU utilization, memory usage,
and training performance to help diagnose GPU utilization issues.
"""

import time
import threading
import argparse
import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import pynvml
    NVIDIA_ML_AVAILABLE = True
except ImportError:
    NVIDIA_ML_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be limited.")


@dataclass
class GPUMetrics:
    """Container for GPU metrics."""
    gpu_utilization: float = 0.0
    memory_used: float = 0.0  # GB
    memory_total: float = 0.0  # GB
    memory_utilization: float = 0.0  # %
    temperature: float = 0.0  # Celsius
    power_draw: float = 0.0  # Watts
    fan_speed: float = 0.0  # %


class GPUMonitor:
    """
    Real-time GPU monitoring for training diagnostics.
    """
    
    def __init__(self, interval: float = 1.0, log_to_file: bool = False):
        """
        Initialize GPU monitor.
        
        Args:
            interval: Monitoring interval in seconds
            log_to_file: Whether to log metrics to a file
        """
        self.interval = interval
        self.log_to_file = log_to_file
        self.monitoring = False
        self.metrics_history: List[GPUMetrics] = []
        
        # Initialize NVIDIA ML if available
        self.nvidia_ml_available = NVIDIA_ML_AVAILABLE
        if self.nvidia_ml_available:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) 
                    for i in range(self.device_count)
                ]
                print(f"Initialized NVIDIA ML for {self.device_count} GPU(s)")
            except Exception as e:
                print(f"Failed to initialize NVIDIA ML: {e}")
                self.nvidia_ml_available = False
                self.device_count = 0
                self.gpu_handles = []
        else:
            self.device_count = 0
            self.gpu_handles = []
        
        # Initialize TensorFlow GPU detection
        self.tf_gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow detected {len(self.tf_gpus)} GPU(s)")
        
        if self.log_to_file:
            self.log_file = open(f"gpu_monitor_{int(time.time())}.csv", "w")
            self.log_file.write("timestamp,gpu_id,utilization,memory_used_gb,memory_total_gb,memory_util_pct,temp_c,power_w,fan_pct\n")
    
    def get_gpu_metrics(self, gpu_id: int = 0) -> GPUMetrics:
        """
        Get current GPU metrics for a specific GPU.
        
        Args:
            gpu_id: GPU index (0-based)
            
        Returns:
            GPUMetrics object with current metrics
        """
        metrics = GPUMetrics()
        
        if not self.nvidia_ml_available or gpu_id >= len(self.gpu_handles):
            return metrics
        
        try:
            handle = self.gpu_handles[gpu_id]
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics.gpu_utilization = util.gpu
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.memory_used = mem_info.used / (1024**3)  # Convert to GB
            metrics.memory_total = mem_info.total / (1024**3)  # Convert to GB
            metrics.memory_utilization = (mem_info.used / mem_info.total) * 100
            
            # Temperature
            try:
                metrics.temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                pass
            
            # Power draw
            try:
                metrics.power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                pass
            
            # Fan speed
            try:
                metrics.fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except:
                pass
                
        except Exception as e:
            print(f"Error getting GPU metrics: {e}")
        
        return metrics
    
    def start_monitoring(self):
        """Start background GPU monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("GPU monitoring started")
    
    def stop_monitoring(self):
        """Stop background GPU monitoring."""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=2.0)
        
        if self.log_to_file and hasattr(self, 'log_file'):
            self.log_file.close()
        
        print("GPU monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                timestamp = time.time()
                
                for gpu_id in range(self.device_count):
                    metrics = self.get_gpu_metrics(gpu_id)
                    self.metrics_history.append(metrics)
                    
                    # Print real-time metrics
                    print(f"\rGPU {gpu_id}: {metrics.gpu_utilization:3.0f}% | "
                          f"Mem: {metrics.memory_used:.1f}/{metrics.memory_total:.1f}GB "
                          f"({metrics.memory_utilization:.1f}%) | "
                          f"Temp: {metrics.temperature:.0f}°C | "
                          f"Power: {metrics.power_draw:.0f}W", end="")
                    
                    # Log to file if enabled
                    if self.log_to_file:
                        self.log_file.write(f"{timestamp},{gpu_id},{metrics.gpu_utilization},"
                                          f"{metrics.memory_used},{metrics.memory_total},"
                                          f"{metrics.memory_utilization},{metrics.temperature},"
                                          f"{metrics.power_draw},{metrics.fan_speed}\n")
                        self.log_file.flush()
                
                time.sleep(self.interval)
            except Exception as e:
                print(f"\nError in monitoring loop: {e}")
                time.sleep(self.interval)
    
    def get_summary_report(self) -> str:
        """Generate a summary report of GPU utilization."""
        if not self.metrics_history:
            return "No GPU metrics collected"
        
        recent_metrics = self.metrics_history[-100:]  # Last 100 samples
        
        if not recent_metrics:
            return "No recent GPU metrics available"
        
        avg_util = np.mean([m.gpu_utilization for m in recent_metrics])
        max_util = np.max([m.gpu_utilization for m in recent_metrics])
        min_util = np.min([m.gpu_utilization for m in recent_metrics])
        
        avg_mem = np.mean([m.memory_utilization for m in recent_metrics])
        max_mem = np.max([m.memory_utilization for m in recent_metrics])
        
        avg_temp = np.mean([m.temperature for m in recent_metrics if m.temperature > 0])
        max_temp = np.max([m.temperature for m in recent_metrics if m.temperature > 0])
        
        report = f"""
=== GPU Utilization Summary ===
GPU Utilization:
  Average: {avg_util:.1f}%
  Maximum: {max_util:.1f}%
  Minimum: {min_util:.1f}%

Memory Utilization:
  Average: {avg_mem:.1f}%
  Maximum: {max_mem:.1f}%

Temperature:
  Average: {avg_temp:.1f}°C
  Maximum: {max_temp:.1f}°C

Samples: {len(recent_metrics)}
"""
        
        # Add diagnostic recommendations
        if avg_util < 30:
            report += "\n⚠️  LOW GPU UTILIZATION DETECTED!"
            report += "\nPossible causes:"
            report += "\n  - Data loading bottleneck (CPU preprocessing too slow)"
            report += "\n  - Small batch size"
            report += "\n  - Model not properly placed on GPU"
            report += "\n  - Inefficient data pipeline"
            report += "\n  - CPU-bound operations in training loop"
        
        if avg_mem < 50:
            report += "\n💡 Memory utilization is low - consider increasing batch size"
        
        return report
    
    def diagnose_training_bottlenecks(self) -> Dict[str, str]:
        """
        Analyze GPU metrics to diagnose training bottlenecks.
        
        Returns:
            Dictionary with bottleneck analysis and recommendations
        """
        if not self.metrics_history:
            return {"error": "No metrics available for analysis"}
        
        recent_metrics = self.metrics_history[-50:]  # Last 50 samples
        avg_util = np.mean([m.gpu_utilization for m in recent_metrics])
        avg_mem = np.mean([m.memory_utilization for m in recent_metrics])
        
        bottlenecks = []
        recommendations = []
        
        if avg_util < 20:
            bottlenecks.append("severe_gpu_underutilization")
            recommendations.append("GPU utilization is critically low. Check data loading pipeline and ensure model is on GPU.")
        elif avg_util < 50:
            bottlenecks.append("gpu_underutilization")
            recommendations.append("GPU utilization is suboptimal. Consider optimizing data loading or increasing batch size.")
        
        if avg_mem < 30:
            bottlenecks.append("low_memory_usage")
            recommendations.append("GPU memory usage is low. Consider increasing batch size for better utilization.")
        
        return {
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "avg_gpu_utilization": avg_util,
            "avg_memory_utilization": avg_mem
        }


def main():
    """Main function for standalone GPU monitoring."""
    parser = argparse.ArgumentParser(description="GPU Monitor for MyXTTS Training")
    parser.add_argument("--interval", type=float, default=1.0, 
                       help="Monitoring interval in seconds (default: 1.0)")
    parser.add_argument("--log-file", action="store_true",
                       help="Log metrics to CSV file")
    parser.add_argument("--duration", type=int, default=0,
                       help="Monitoring duration in seconds (0 = indefinite)")
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(interval=args.interval, log_to_file=args.log_file)
    
    if monitor.device_count == 0:
        print("No GPUs detected. Exiting.")
        return
    
    print(f"Starting GPU monitoring for {monitor.device_count} GPU(s)...")
    print("Press Ctrl+C to stop")
    
    monitor.start_monitoring()
    
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        monitor.stop_monitoring()
        print("\n" + monitor.get_summary_report())


if __name__ == "__main__":
    main()