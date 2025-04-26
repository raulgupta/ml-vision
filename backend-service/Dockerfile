FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install

# Copy .git directory and .gitattributes first
COPY .git ./.git
COPY .gitattributes .

# Initialize Git repository
RUN git init && \
    git remote add origin https://github.com/raulgupta/ml-vision.git && \
    git fetch origin && \
    git checkout main

# Pull LFS files
RUN git lfs pull

# Copy the rest of the application code
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "server.py"]
