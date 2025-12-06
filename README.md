# PythonFileToYoutube

Turn any file or folder into one or more MP4 videos (and back again) using the `file_to_video_torch.py` script provided in this repo.
The script is optimized for YouTube. Meaning the default config is good enough that a file should be able to survive a full round trip to YouTube and back.

## Features

- **End-to-end pipeline** – archive → error-correct → encode to frames → stream to FFmpeg, plus the reverse decode path.
- **GPU-accelerated** – uses PyTorch to process large frame batches efficiently (CUDA strongly recommended).
- **NVENC Support** – Optional hardware encoding for massive speed gains.
- **Asynchronous Processing** – Decoupled CPU unpacking and GPU encoding threads for maximum throughput.
- **Robust Redundancy** – Splits large files into 1GB volumes and generates optimized 128KB PAR2 blocks.
- **Segmented output** – Automatically rotates MP4 files when they exceed the configured length (default ~11 hours).
- **Cross-platform** – Works on Windows and Linux via Docker (recommended) or local installation.

## Prerequisites

If you use **Docker (Recommended)**, the container handles Python, FFmpeg, 7-Zip, PAR2, and PyTorch automatically. You only need to provide the hardware drivers on your host machine.

### 1. Host Machine Requirements (for Docker)
You must install these on your physical machine before running the container:

*   **NVIDIA GPU Drivers**: Install the latest specific drivers for your card from [NVIDIA](https://www.nvidia.com/Download/index.aspx).
*   **Docker**:
    *   **Windows**: Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
        *   *Critical:* Enable **"Use the WSL 2 based engine"** in Docker Settings > General.
    *   **Linux**: Install [Docker Engine](https://docs.docker.com/engine/install/).
        *   *Critical:* You **must** install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) so Docker can see your GPU.
        ```bash
        # Linux only: Add repo and install toolkit
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        ```

### 2. Local Install Requirements (Manual Method)
*Only required if you are NOT using Docker.*
*   Python 3.11
*   PyTorch (GPU build)
*   7-Zip (added to PATH)
*   PAR2 (added to PATH)
*   FFmpeg (added to PATH) - **Must be a build with `enable-libx264` and `enable-ffnvcodec`/`h264_nvenc` support.**

---

## Installation (Docker)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Keksmania/PythonFileToYoutube.git
    cd PythonFileToYoutube
    ```

2.  **Build the Container:**
    ```bash
    docker compose up -d --build
    ```
    *This downloads PyTorch and compiles a custom FFmpeg build inside the container. It may take a few minutes.*

3.  **Verify GPU Access:**
    ```bash
    docker compose exec f2yt nvidia-smi
    ```
    *You should see your GPU stats.*

## Usage

The Docker setup uses a **Bind Mount**. The folder `PythonFileToYoutube` on your computer is mapped directly to `/app` inside the container.
*   **To Encode:** Put files in your project folder. Inside Docker, access them via `/app/filename`.
*   **To Retrieve:** Output videos appear immediately in your project folder.

### Encoding
```bash
# Docker
docker compose exec f2yt python file_to_video_torch.py -mode encode -input "file_to_video_torch.py"

# Local
python file_to_video_torch.py -mode encode -input "file_to_video_torch.py"
```
*   **Optional:** Add `-p yourpassword` to encrypt the payload.

### Decoding
```bash
# Docker (Point to the folder containing the video parts)
docker compose exec f2yt python file_to_video_torch.py -mode decode -input "file_to_video_torch_F2YT_Output"

# Local
python file_to_video_torch.py -mode decode -input "file_to_video_torch_F2YT_Output"
```
*   The script automatically sorts video parts (part001, part002, etc).
*   Files are extracted to: `<input_folder>/Decoded_Files`.

## Configuration (`f2yt_config.json`)

The default configuration is optimized for YouTube's compression.

| Key | Default | Description |
| --- | --- | --- |
| `VIDEO_WIDTH` | 720 | Output resolution width. |
| `VIDEO_HEIGHT` | 720 | Output resolution height. |
| `DATA_K_SIDE` | 180 | Data grid size per frame. |
| `DATA_HAMMING_N` | 127 | Hamming Code Block size (Data density). |
| `DATA_HAMMING_K` | 120 | Hamming Code Data size (High efficiency). |
| `ENABLE_NVENC` | true | Uses GPU hardware encoding (fast). Set `false` for CPU. |
| `GPU_PROCESSOR_BATCH_SIZE` | 512 | Frames processed per batch on GPU. |

## Troubleshooting

- **"Could not select device driver"**: Your host machine (Windows/Linux) is missing NVIDIA drivers, or the NVIDIA Container Toolkit is not configured.
- **FFmpeg Error / Broken Pipe**: Ensure you are using the provided Dockerfile. It installs a specific static build of FFmpeg compatible with NVENC.
- **Permission Denied (Linux)**: Add your user to the docker group (`sudo usermod -aG docker $USER`) or run with `sudo`.

## License

MIT.