FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV, git, and git-lfs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy the .git directory and .gitattributes to set up Git context
COPY .git .git
COPY .gitattributes .

# Initialize Git repository and configure LFS
RUN git init && \
    git remote add origin https://github.com/raulgupta/ml-vision.git && \
    git fetch origin && \
    git checkout main && \
    git lfs install && \
    git lfs pull

# Copy the rest of the application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "server.py"]
