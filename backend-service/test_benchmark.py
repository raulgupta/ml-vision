import requests
import json
import time

def test_benchmark_endpoint():
    """
    Test the benchmark endpoint by sending a POST request and printing the results.
    """
    print("Testing benchmark endpoint...")
    
    # URL of the benchmark endpoint
    url = "http://localhost:8000/benchmark"
    
    try:
        # Send POST request to the benchmark endpoint
        start_time = time.time()
        response = requests.post(url)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()
            
            # Calculate request time
            request_time = time.time() - start_time
            
            # Print the results
            print(f"Benchmark completed in {request_time:.2f} seconds")
            print("\nResults:")
            print(f"Load Time Average: {data['loadTime']['average']:.3f} seconds")
            print(f"Memory Delta: {data['memory']['delta']:.3f} MB")
            print(f"CPU Time: {data['cpu']['total']:.3f} ms")
            
            print("\nSystem Information:")
            print(f"CPU: {data['system']['cpu']}")
            print(f"Cores: {data['system']['cores']}")
            print(f"Memory: {data['system']['memory']}")
            print(f"Architecture: {data['system']['architecture']}")
            
            print("\nDetailed Results:")
            print("Load Times (ms):", [f"{x:.2f}" for x in data['loadTime']['data']])
            print("Memory Deltas (MB):", [f"{x:.2f}" for x in data['memory']['data']])
            print("CPU Times (ms):", [f"{x:.2f}" for x in data['cpu']['data']])
            
            return True
        else:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_benchmark_endpoint()
