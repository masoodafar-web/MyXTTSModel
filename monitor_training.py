#!/usr/bin/env python3
"""
Real-time Training Monitor
نمایش وضعیت لحظه‌ای training
"""

import time
import subprocess
import psutil
import re
import os

def get_process_info(pid):
    """Get process information"""
    try:
        process = psutil.Process(pid)
        return {
            'cpu_percent': process.cpu_percent(),
            'memory_percent': process.memory_percent(),
            'memory_mb': process.memory_info().rss / 1024 / 1024,
            'status': process.status(),
            'create_time': process.create_time(),
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def get_gpu_usage():
    """Get GPU usage information"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                               '--format=csv,noheader,nounits'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        gpus = []
        for i, line in enumerate(lines):
            parts = line.split(', ')
            if len(parts) >= 4:
                gpus.append({
                    'gpu_id': i,
                    'utilization': int(parts[0]),
                    'memory_used': int(parts[1]),
                    'memory_total': int(parts[2]),
                    'temperature': int(parts[3])
                })
        return gpus
    except Exception as e:
        return []

def check_training_log():
    """Check training log for recent activity"""
    log_patterns = [
        r'Epoch \d+:.*?(\d+)%.*?(\d+/\d+)',
        r'loss.*?(\d+\.\d+)',
        r'Step \d+:',
        r'Starting Epoch'
    ]
    
    # Look for log files or recent output
    possible_logs = ['training.log', 'myxtts.log']
    recent_activity = []
    
    for log_file in possible_logs:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-10:]  # Last 10 lines
                    for line in lines:
                        for pattern in log_patterns:
                            if re.search(pattern, line):
                                recent_activity.append(line.strip())
                                break
            except Exception:
                pass
    
    return recent_activity

def main():
    training_pid = 1090207  # PID فرایند training
    
    print("🔍 Training Process Monitor")
    print("=" * 50)
    print(f"Monitoring PID: {training_pid}")
    print("Press Ctrl+C to stop monitoring")
    print()
    
    start_time = time.time()
    
    try:
        while True:
            # Process info
            proc_info = get_process_info(training_pid)
            if not proc_info:
                print("❌ Training process not found!")
                break
            
            # GPU info
            gpu_info = get_gpu_usage()
            
            # Time info
            elapsed = time.time() - start_time
            process_uptime = time.time() - proc_info['create_time']
            
            # Clear screen and show info
            os.system('clear')
            print("🔍 Training Process Monitor")
            print("=" * 50)
            print(f"⏱️  Monitoring time: {elapsed:.0f}s")
            print(f"🔄 Process uptime: {process_uptime/60:.1f} minutes")
            print(f"💻 CPU: {proc_info['cpu_percent']:.1f}%")
            print(f"🧠 Memory: {proc_info['memory_mb']:.0f}MB ({proc_info['memory_percent']:.1f}%)")
            print(f"📊 Status: {proc_info['status']}")
            print()
            
            if gpu_info:
                print("🎮 GPU Status:")
                for gpu in gpu_info:
                    print(f"   GPU {gpu['gpu_id']}: {gpu['utilization']}% | "
                          f"{gpu['memory_used']}/{gpu['memory_total']}MB | "
                          f"{gpu['temperature']}°C")
                print()
            
            # Recent log activity
            recent_logs = check_training_log()
            if recent_logs:
                print("📝 Recent Activity:")
                for log in recent_logs[-3:]:  # Show last 3
                    print(f"   {log}")
            else:
                print("📝 No recent log activity detected")
            
            print()
            print("💡 Tips:")
            if proc_info['cpu_percent'] > 90:
                print("   - High CPU usage - likely compiling TensorFlow graph")
            if all(gpu['utilization'] < 5 for gpu in gpu_info):
                print("   - Low GPU usage - still in setup/compilation phase")
            if process_uptime > 600:  # 10 minutes
                print("   - Process running for >10 min - might be stuck")
                print("   - Consider using quick_restart.sh script")
            
            print("\nPress Ctrl+C to stop monitoring")
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()