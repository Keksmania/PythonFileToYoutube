# PythonFileToYoutube

Turn any file or folder into one or more MP4 videos (and back again) using the `file_to_video_torch.py` script provided in this repo.
The script is optimized for YouTube. Meaning the default config is good enough that a file should be able to survive a full round trip to YouTube and back.

## Features

- **End-to-end pipeline** – archive → error-correct → encode to frames → stream to FFmpeg, plus the reverse decode path.
- **GPU-accelerated** – uses PyTorch to process large frame batches efficiently (CUDA strongly recommended).
- **NVENC Support** – Optional hardware encoding for massive speed gains (see configuration).
- **Asynchronous Processing** – Decoupled CPU unpacking and GPU encoding threads for maximum throughput and smooth resource usage.
- **Robust Redundancy** – Splits large files into 1GB volumes and generates optimized 128KB PAR2 blocks for granular recovery.
- **Segmented output** – Automatically rotates MP4 files when they exceed the configured length (default ~11 hours at 60fps).
- **Robust metadata** – Barcode + info frames describe payload size, frame counts, and optional password protection.
- **Cross-platform** – Works on Windows and Linux as long as the external tools are installed (or via Docker).

## Requirements

| Component | Notes |
| --- | --- |
| Python | 3.11 (system install). Earlier versions are untested. |
| PyTorch | GPU build (2.9.0+ with CUDA 13.x recommended). Install from https://pytorch.org. |
| CUDA | NVIDIA GPU drivers + CUDA toolkit matching your PyTorch build. `nvidia-smi` must work. |
| FFmpeg | Must include `libx264` encoder. For GPU encoding, it must support `h264_nvenc`. |
| 7-Zip CLI | `7z` binary (`p7zip-full` on Linux, official installer on Windows). |
| PAR2 | `par2`/`par2cmdline` binary for creating redundancy blocks. |
| Disk space | Enough to store the temp archive, PAR2 set, intermediate frames, and final MP4(s). |

## Installation & Setup

You have two options: **Local Install** (manual dependency management) or **Docker** (recommended for ease of use).

### Option 1: Docker (Recommended)

Docker handles all software dependencies (Python, FFmpeg, 7-Zip, PAR2, PyTorch). You only need to ensure your host machine has the GPU drivers installed.

#### Prerequisites

1.  **NVIDIA GPU Drivers:** Install the latest drivers for your graphics card on your host machine.
2.  **Docker:** Install Docker for your OS.

#### Windows (WSL2) Setup
1.  **Install Docker Desktop:** Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
2.  **Enable WSL2 Backend:** During installation (or in Settings > General), ensure "Use the WSL 2 based engine" is checked.
3.  **Verify GPU:** Open PowerShell and run `nvidia-smi`. If you see your GPU, Docker will automatically be able to use it via WSL2.

#### Linux Setup
1.  **Install Docker Engine:** Follow the official guide for [Ubuntu/Debian/CentOS](https://docs.docker.com/engine/install/).
2.  **Install NVIDIA Container Toolkit:** This is required for Docker to see your GPU.
    ```bash
    # Add the package repositories
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    # Install the toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit

    # Configure Docker
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

#### Running with Docker

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Keksmania/PythonFileToYoutube.git
    cd PythonFileToYoutube
    ```
2.  **Build and Start the Container:**
    ```bash
    docker-compose up -d --build
    ```
3.  **Run the script:**
    The `docker-compose.yml` maps the current folder to `/app` inside the container.
    ```bash
    # Example Encode
    docker-compose exec f2yt python file_to_video_torch.py -mode encode -input "my_folder"

    # Example Decode
    docker-compose exec f2yt python file_to_video_torch.py -mode decode -input "my_folder_F2YT"
    ```

---

### Option 2: Local Install (Manual)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Keksmania/PythonFileToYoutube.git
    cd PythonFileToYoutube
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install external tools:**
    *   **Windows:** Download and install FFmpeg, 7-Zip, and PAR2. Add their folders to your System `PATH`.
    *   **Linux:** `sudo apt install ffmpeg p7zip-full par2`
4.  **Verify Environment:**
    *   Windows: `.\tools\check_deps_windows.cmd`
    *   Linux: `./tools/check_deps_linux.sh`
5.  **Run:**
    ```bash
    python file_to_video_torch.py -mode encode -input "my_folder"
    ```

## Usage Guide

### Encoding
```powershell
python file_to_video_torch.py -mode encode -input "C:\path\to\file_or_folder" [-output "C:\path\to\output_dir"] [-p password]
```
- The output directory will contain `*_F2YT.mp4` (or segmented `_part###.mp4`) files.
- Set `-p` to password-protect the 7z archive.

### Decoding
```powershell
python file_to_video_torch.py -mode decode -input "video_part001.mp4,video_part002.mp4" [-output "C:\path\to\restore"] [-p password]
```
- **Folder Input:** You can point `-input` to a folder containing the video parts; the script will sort them automatically (recommended).
- **File Input:** If providing a list, use a comma-separated string.
- If no output directory is specified, it defaults to `_F2YT_Output` next to the first input file.
- Decoded files end up in `<output>/Decoded_Files` once 7-Zip and PAR2 complete.

## Configuration Reference (`f2yt_config.json`)

| Key | Description |
| --- | --- |
| `FFMPEG_PATH` | Path to the FFmpeg binary. |
| `SEVENZIP_PATH` | Path to the 7-Zip CLI (`7z`). |
| `PAR2_PATH` | Path to the PAR2 executable. |
| `VIDEO_WIDTH`/`VIDEO_HEIGHT` | Output resolution (default 720x720). |
| `VIDEO_FPS` | Frames per second (default 60). |
| `DATA_K_SIDE` | Data frame size (default 180). Must scale cleanly into video resolution (e.g., 720/180 = 4). |
| `NUM_COLORS_DATA` | Palette size (power of two). |
| `DATA_HAMMING_N` | Hamming Block Size (default 127). |
| `DATA_HAMMING_K` | Hamming Data Size (default 120). Provides 94% efficiency. |
| `PAR2_REDUNDANCY_PERCENT` | PAR2 redundancy percentage. |
| `X264_CRF` | Quality parameter for x264/NVENC (lower = larger files/better quality). |
| `ENABLE_NVENC` | Set to `true` to use GPU hardware encoding (faster, but larger files). |
| `CPU_PRODUCER_CHUNK_MB` | File chunk size for the producer thread. |
| `GPU_PROCESSOR_BATCH_SIZE` | Number of frames processed per GPU batch. |
| `GPU_OVERLAP_STREAMS` | Number of CUDA streams for async processing. |
| `PIPELINE_QUEUE_DEPTH` | Buffer size for thread queues to smooth workload spikes. |
| `MAX_VIDEO_SEGMENT_HOURS` | Approx hours per MP4 before rolling to a new segment (default 11). |

## Workflow Overview

1. **Preparation**: `prepare_files_for_encoding` compresses the payload with 7-Zip. It enforces a **1GB volume split** and generates **128KB-block PAR2** files for each volume to ensure recovery works even on large datasets with heavy video compression.
2. **Metadata**: Info frames + barcode encode session metadata (JSON) using Hamming(7,4).
3. **Async Encoding**: `DataProducerThread` reads raw bytes and unpacks them asynchronously on the CPU.
4. **GPU Packing**: `encode_data_frames_gpu` packs unpacked bits into RGB tiles on the GPU using CUDA (FP16 optimized).
5. **Video Stream**: `FFmpegConsumerThread` pipes the RGB stream into `ffmpeg` (via CPU `libx264` or GPU `h264_nvenc`) to produce MP4 segments.
6. **Decoding**: Runs the inverse steps: `ContinuousPipeFrameReader` stitches multiple video segments seamlessly using FFmpeg's concat protocol -> GPU Decodes (Hamming corrected) -> `DataWriterThread` writes to disk -> PAR2 Repairs -> 7-Zip Extracts.

## Troubleshooting

- **Dependency missing** – Run the OS-specific dependency script; install any reported tool and rerun.
- **CUDA unavailable** – Confirm `nvidia-smi` works and that your PyTorch build reports `torch.cuda.is_available() == True`.
- **FFmpeg errors** – Inspect the log output. Ensure your `ffmpeg` build supports `libx264` (or `h264_nvenc` if enabled).
- **Decode hangs or stops early** – The script uses FFmpeg pipes. Ensure your disk has enough space for temporary files.
- **Output Resolution** – The script forces strict resolution scaling filters to avoid "mod-16" cropping issues. Ensure `VIDEO_WIDTH` is divisible by `DATA_K_SIDE`.

## Contributing

Issues and PRs are welcome! Please include:
- A description of the problem / feature.
- Hardware/OS details, especially GPU and CUDA versions.

## License

MIT.