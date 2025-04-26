FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV, git, and git-lfs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Clone the public repo and pull LFS files
RUN git clone https://github.com/raulgupta/ml-vision.git /app && \
    cd /app && \
    git lfs install && \
    git lfs pull

# Install Python dependencies
RUN pip install --no-cache-dir -r backend-service/requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "backend-service/server.py"]