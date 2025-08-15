#!/usr/bin/env python3
import subprocess
import time
import re
from datetime import datetime

def get_gpu_stats():
    """Get GPU utilization and memory usage using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        lines = result.stdout.strip().split('\n')
        gpu_stats = []
        
        for i, line in enumerate(lines):
            parts = [part.strip() for part in line.split(',')]
            if len(parts) >= 5:
                gpu_util = int(parts[0])
                mem_used = int(parts[1])
                mem_total = int(parts[2])
                temp = int(parts[3])
                name = parts[4]
                mem_percent = (mem_used / mem_total) * 100
                
                gpu_stats.append({
                    'id': i,
                    'name': name,
                    'gpu_util': gpu_util,
                    'mem_used': mem_used,
                    'mem_total': mem_total,
                    'mem_percent': mem_percent,
                    'temp': temp
                })
        
        return gpu_stats
    
    except subprocess.CalledProcessError:
        return None
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        return None

def format_memory(mb):
    """Format memory in MB to human readable format"""
    if mb >= 1024:
        return f"{mb/1024:.1f}GB"
    return f"{mb}MB"

def print_gpu_stats(gpu_stats):
    """Print formatted GPU statistics"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] GPU Status:")
    print("-" * 80)
    
    for gpu in gpu_stats:
        print(f"GPU {gpu['id']}: {gpu['name']}")
        print(f"  Compute: {gpu['gpu_util']:3d}% | "
              f"Memory: {gpu['mem_percent']:5.1f}% "
              f"({format_memory(gpu['mem_used'])}/{format_memory(gpu['mem_total'])}) | "
              f"Temp: {gpu['temp']}°C")

def main():
    print("GPU Monitor - Press Ctrl+C to stop")
    print("=" * 80)
    
    try:
        while True:
            gpu_stats = get_gpu_stats()
            
            if gpu_stats is None:
                print("Failed to get GPU stats")
                time.sleep(5)
                continue
            
            if not gpu_stats:
                print("No GPUs detected")
                time.sleep(5)
                continue
            
            print_gpu_stats(gpu_stats)
            time.sleep(2)  # Update every 2 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()