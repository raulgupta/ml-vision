import os
import urllib.request

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

def main():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download YOLOv3-tiny weights
    yolov3_tiny_url = "https://https://github.com/smarthomefans/darknet-test/blob/master/yolov3-tiny.weights"
    yolov3_tiny_path = os.path.join("models", "yolov3-tiny.weights")
    
    if not os.path.exists(yolov3_tiny_path):
        download_file(yolov3_tiny_url, yolov3_tiny_path)
    else:
        print(f"File {yolov3_tiny_path} already exists. Skipping download.")

if __name__ == "__main__":
    main()
