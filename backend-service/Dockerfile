FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV and git-lfs
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Copy application code first (including .git and .gitattributes)
COPY . .

# Initialize git-lfs and pull files
RUN git lfs install && git lfs pull

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["python", "server.py"]
