# Use official PyTorch image with CUDA 12.1 support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    p7zip-full \
    par2 \
    git \
    wget \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Python Dependencies
RUN pip install --no-cache-dir \
    pillow \
    numpy

# 3. Install FFmpeg 7.0 (Stable)
# This version correctly detects NVENC libraries on WSL2/Docker
RUN wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n7.0-latest-linux64-gpl-7.0.tar.xz -O /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && mv /tmp/ffmpeg-n7.0-latest-linux64-gpl-7.0/bin/ffmpeg /usr/local/bin/ffmpeg \
    && mv /tmp/ffmpeg-n7.0-latest-linux64-gpl-7.0/bin/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg \
    && chmod +x /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg*

# 4. Remove Conda FFmpeg (Critical: ensures we use the one we just installed)
RUN rm -f /opt/conda/bin/ffmpeg \
    && rm -f /opt/conda/bin/ffprobe

WORKDIR /app
CMD ["/bin/bash"]