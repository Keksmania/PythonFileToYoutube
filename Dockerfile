# Use official PyTorch image with CUDA 12.1 support
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Prevent interactive prompts during installation
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


RUN wget https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -O /tmp/ffmpeg.tar.xz \
    && tar -xf /tmp/ffmpeg.tar.xz -C /tmp \
    && mv /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffmpeg /usr/local/bin/ffmpeg \
    && mv /tmp/ffmpeg-master-latest-linux64-gpl/bin/ffprobe /usr/local/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg \
    && chmod +x /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg*


# 4. Remove the old Conda FFmpeg (Critical)
RUN rm -f /opt/conda/bin/ffmpeg \
    && rm -f /opt/conda/bin/ffprobe

# 5. Set working directory
WORKDIR /app

# 6. Default command
CMD ["/bin/bash"]