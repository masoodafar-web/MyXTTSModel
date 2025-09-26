#!/usr/bin/env python3
"""
GPU Utilization Live Monitor
Real-time monitoring of GPU utilization during MyXTTS training
"""

import time
import subprocess
import os
import signal
import sys
from datetime import datetime

class GPUMonitor:
    def __init__(self, interval=2.0):
        self.interval = interval
        self.running = True
        
        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\n🛑 Stopping GPU monitor...")
        self.running = False
    
    def get_gpu_stats(self):
        """Get GPU utilization and memory usage"""
        try:
            # Query GPU stats using nvidia-smi
            cmd = [
                'nvidia-smi', 
                '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None
                
            lines = result.stdout.strip().split('\n')
            gpus = []
            
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 6:
                        gpu_info = {
                            'index': int(parts[0]),
                            'name': parts[1],
                            'utilization': int(parts[2]) if parts[2] != '[Not Supported]' else 0,
                            'memory_used': int(parts[3]),
                            'memory_total': int(parts[4]),
                            'temperature': int(parts[5]) if parts[5] != '[Not Supported]' else 0,
                            'power_draw': float(parts[6]) if len(parts) > 6 and parts[6] != '[Not Supported]' else 0,
                            'power_limit': float(parts[7]) if len(parts) > 7 and parts[7] != '[Not Supported]' else 0
                        }
                        gpus.append(gpu_info)
            
            return gpus
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            return None
    
    def format_gpu_bar(self, value, max_value, width=20):
        """Create a progress bar for GPU metrics"""
        if max_value == 0:
            percentage = 0
        else:
            percentage = min(100, (value / max_value) * 100)
        
        filled = int((percentage / 100) * width)
        bar = '█' * filled + '░' * (width - filled)
        return f"{bar} {percentage:5.1f}%"
    
    def print_gpu_status(self, gpus):
        """Print formatted GPU status"""
        if not gpus:
            print("❌ No GPU data available")
            return
        
        # Clear screen and move cursor to top
        os.system('clear')
        
        print("🖥️  MyXTTS GPU Live Monitor")
        print("=" * 80)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        for gpu in gpus:
            print(f"🔥 GPU {gpu['index']}: {gpu['name']}")
            print(f"   GPU Usage:  {self.format_gpu_bar(gpu['utilization'], 100)}")
            
            memory_pct = (gpu['memory_used'] / gpu['memory_total']) * 100
            print(f"   Memory:     {self.format_gpu_bar(gpu['memory_used'], gpu['memory_total'])} ({gpu['memory_used']:,}MB / {gpu['memory_total']:,}MB)")
            
            temp_color = "🟢" if gpu['temperature'] < 70 else "🟡" if gpu['temperature'] < 80 else "🔴"
            print(f"   Temperature: {temp_color} {gpu['temperature']}°C")
            
            if gpu['power_draw'] > 0:
                power_pct = (gpu['power_draw'] / gpu['power_limit']) * 100 if gpu['power_limit'] > 0 else 0
                print(f"   Power:      {gpu['power_draw']:.1f}W / {gpu['power_limit']:.1f}W ({power_pct:.1f}%)")
            
            print()
        
        # Show some helpful information
        print("📊 Monitoring Tips:")
        print("   • مطلوب: GPU utilization بالای 80% برای training مؤثر")
        print("   • مشکل: Fluctuation بین 40% و 2% (که ما حل کردیم!)")
        print("   • خوب: Memory usage پایدار و مناسب")
        print()
        print("⏹️  Press Ctrl+C to stop monitoring")
    
    def run(self):
        """Main monitoring loop"""
        print("🚀 Starting GPU monitor...")
        print("Press Ctrl+C to stop")
        print("-" * 50)
        
        while self.running:
            try:
                gpus = self.get_gpu_stats()
                self.print_gpu_status(gpus)
                time.sleep(self.interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.interval)
        
        print("\n✅ GPU monitoring stopped")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Live GPU utilization monitor for MyXTTS training')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Update interval in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    monitor = GPUMonitor(interval=args.interval)
    monitor.run()

if __name__ == "__main__":
    main()