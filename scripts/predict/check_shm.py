import os
import sys
import psutil
import subprocess
import multiprocessing as mp
from multiprocessing import Array
import platform

def print_system_info():
    print("=== System Information ===")
    print(f"Node name: {platform.node()}")
    print(f"OS: {platform.system()} {platform.release()}\n")

def print_shm_info():
    print("=== /dev/shm Information ===")
    try:
        # Use df command to get /dev/shm info
        df = subprocess.Popen(['df', '-BM', '/dev/shm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = df.communicate()
        if df.returncode == 0:
            lines = stdout.decode().strip().split('\n')
            if len(lines) > 1:
                _, total, used, free, *_ = lines[1].split()
                print(f"Total space: {total}")
                print(f"Used space: {used}")
                print(f"Free space: {free}\n")
    except Exception as e:
        print(f"Error getting /dev/shm information: {e}\n")

    print("=== Contents of /dev/shm ===")
    try:
        ls = subprocess.Popen(['ls', '-lh', '/dev/shm'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = ls.communicate()
        if ls.returncode == 0:
            print(stdout.decode())
    except Exception as e:
        print(f"Error getting /dev/shm contents: {e}\n")

def print_process_info():
    print("=== Current Process Information ===")
    process = psutil.Process()
    print(f"Process ID: {process.pid}")
    print(f"Memory Info: {process.memory_info()}")
    print(f"Memory Percent: {process.memory_percent()}%")
    print(f"CPU Percent: {process.cpu_percent()}%\n")

def test_shared_memory():
    print("=== Shared Memory Allocation Test ===\n")
    sizes_mb = [100, 500, 1000, 2000]
    
    for size in sizes_mb:
        print(f"Trying to allocate {size} MB...")
        try:
            # Allocate shared memory using Array
            n = size * 1024 * 1024 // 8  # Convert MB to number of doubles (8 bytes each)
            shm = Array('d', n)
            print(f"Successfully allocated {size} MB")
            del shm  # Release the memory
        except Exception as e:
            print(f"Failed to allocate {size} MB: {e}")
        print()

if __name__ == "__main__":
    print_system_info()
    print_shm_info()
    print_process_info()
    test_shared_memory() 