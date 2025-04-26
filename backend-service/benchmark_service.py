import os
import platform
import psutil
import time
import numpy as np
import cv2
import scipy
from typing import Dict, List, Any

def get_system_info() -> Dict[str, str]:
    """
    Get system information for benchmarking context.
    """
    try:
        # Get CPU information
        cpu_info = platform.processor()
        if not cpu_info:
            cpu_info = "Unknown CPU"
        
        # Get number of cores
        cpu_count = os.cpu_count()
        if cpu_count:
            cores_info = f"{cpu_count} cores"
        else:
            cores_info = "Unknown cores"
            
        # Add CPU frequency if available
        try:
            cpu_freq = psutil.cpu_freq()
            if cpu_freq and cpu_freq.current:
                cores_info += f" @ {int(cpu_freq.current)}MHz"
        except:
            pass
        
        # Get memory information
        mem_info = psutil.virtual_memory()
        memory_gb = round(mem_info.total / (1024**3), 2)
        memory_info = f"{memory_gb:.2f}GB"
        
        # Get architecture
        architecture = platform.machine()
        if not architecture:
            architecture = "Unknown architecture"
            
        return {
            "cpu": cpu_info,
            "cores": cores_info,
            "memory": memory_info,
            "architecture": architecture
        }
    except Exception as e:
        print(f"Error getting system info: {str(e)}")
        # Return fallback values if there's an error
        return {
            "cpu": "Unknown CPU",
            "cores": "Unknown cores",
            "memory": "Unknown memory",
            "architecture": platform.machine() or "Unknown architecture"
        }

def benchmark_empty_page() -> Dict[str, float]:
    """
    Benchmark an empty page with minimal processing.
    """
    # Measure load time
    start_time = time.time()
    # Minimal processing - just basic operations
    for _ in range(10):
        np.random.rand(10, 10)
    load_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Measure memory usage
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    # Minimal memory operations
    temp = []
    for _ in range(5):
        temp.append(np.zeros((10, 10)))
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_delta = end_mem - start_mem
    
    # Measure CPU usage
    start_cpu = time.process_time()
    # Minimal CPU operations
    for _ in range(10):
        np.random.rand(10, 10).sum()
    cpu_time = (time.process_time() - start_cpu) * 1000  # Convert to ms
    
    return {
        "load_time": load_time,
        "memory_delta": memory_delta,
        "cpu_time": cpu_time
    }

def benchmark_simple_static() -> Dict[str, float]:
    """
    Benchmark a simple static page with basic image processing.
    """
    # Measure load time
    start_time = time.time()
    # Simple image processing
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    load_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Measure memory usage
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    # Simple memory operations
    images = []
    for _ in range(5):
        images.append(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
        cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_delta = end_mem - start_mem
    
    # Measure CPU usage
    start_cpu = time.process_time()
    # Simple CPU operations
    for _ in range(5):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        edges = cv2.Canny(img, 100, 200)
    cpu_time = (time.process_time() - start_cpu) * 1000  # Convert to ms
    
    return {
        "load_time": load_time,
        "memory_delta": memory_delta,
        "cpu_time": cpu_time
    }

def benchmark_text_heavy() -> Dict[str, float]:
    """
    Benchmark a text-heavy page with text processing operations.
    """
    # Generate some text data
    text = "The quick brown fox jumps over the lazy dog. " * 1000
    words = text.split()
    
    # Measure load time
    start_time = time.time()
    # Text processing operations
    word_count = {}
    for word in words:
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    load_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Measure memory usage
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    # Text memory operations
    text_copies = []
    for _ in range(10):
        text_copies.append(text + str(np.random.rand()))
        text_copies[-1].split()
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_delta = end_mem - start_mem
    
    # Measure CPU usage
    start_cpu = time.process_time()
    # Text CPU operations
    for _ in range(5):
        word_freq = {}
        for word in words:
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1
        sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    cpu_time = (time.process_time() - start_cpu) * 1000  # Convert to ms
    
    return {
        "load_time": load_time,
        "memory_delta": memory_delta,
        "cpu_time": cpu_time
    }

def benchmark_dynamic_content() -> Dict[str, float]:
    """
    Benchmark a dynamic content page with complex operations.
    """
    # Measure load time
    start_time = time.time()
    # Complex image processing
    img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
    # Apply multiple filters
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    load_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Measure memory usage
    start_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    # Complex memory operations
    large_arrays = []
    for _ in range(5):
        large_arrays.append(np.random.rand(500, 500, 3))
        # Apply FFT
        scipy.fft.fft2(large_arrays[-1][:,:,0])
    end_mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    memory_delta = end_mem - start_mem
    
    # Measure CPU usage
    start_cpu = time.process_time()
    # Complex CPU operations
    for _ in range(3):
        # Matrix operations
        a = np.random.rand(300, 300)
        b = np.random.rand(300, 300)
        # Matrix multiplication
        np.matmul(a, b)
        # SVD decomposition
        u, s, vh = np.linalg.svd(a)
    cpu_time = (time.process_time() - start_cpu) * 1000  # Convert to ms
    
    return {
        "load_time": load_time,
        "memory_delta": memory_delta,
        "cpu_time": cpu_time
    }

def run_benchmarks() -> Dict[str, Any]:
    """
    Run all benchmarks and return formatted results.
    """
    print("Starting benchmarks...")
    
    # Run each benchmark
    print("Running empty page benchmark...")
    empty_results = benchmark_empty_page()
    
    print("Running simple static benchmark...")
    simple_results = benchmark_simple_static()
    
    print("Running text heavy benchmark...")
    text_results = benchmark_text_heavy()
    
    print("Running dynamic content benchmark...")
    dynamic_results = benchmark_dynamic_content()
    
    # Collect all results
    load_times = [
        empty_results["load_time"],
        simple_results["load_time"],
        text_results["load_time"],
        dynamic_results["load_time"]
    ]
    
    memory_deltas = [
        empty_results["memory_delta"],
        simple_results["memory_delta"],
        text_results["memory_delta"],
        dynamic_results["memory_delta"]
    ]
    
    cpu_times = [
        empty_results["cpu_time"],
        simple_results["cpu_time"],
        text_results["cpu_time"],
        dynamic_results["cpu_time"]
    ]
    
    # Calculate averages
    avg_load_time = sum(load_times) / len(load_times) / 1000  # Convert to seconds
    avg_memory_delta = sum(memory_deltas) / len(memory_deltas)
    avg_cpu_time = sum(cpu_times) / len(cpu_times)
    
    # Get system information
    system_info = get_system_info()
    
    print("Benchmarks completed.")
    
    # Format results to match frontend expectations
    return {
        "loadTime": {
            "average": avg_load_time,
            "data": load_times
        },
        "memory": {
            "delta": avg_memory_delta,
            "data": memory_deltas
        },
        "cpu": {
            "total": avg_cpu_time,
            "data": cpu_times
        },
        "system": system_info
    }

# For testing
if __name__ == "__main__":
    results = run_benchmarks()
    print(results)
