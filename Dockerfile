FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install system dependencies (FFmpeg, 7-Zip, Par2)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    p7zip-full \
    par2 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python specific dependencies
# (numpy and torch are already included in the base image)
RUN pip install --no-cache-dir \
    pillow \
    numpy

# Set the working directory
WORKDIR /app

# By default, drop into a shell, or you can set the entrypoint to the script
CMD ["/bin/bash"]