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
| PyTorch | GPU build (2.9.0+ with CUDA 13.x recommended). |
| CUDA | NVIDIA GPU drivers + CUDA toolkit matching your PyTorch build. `nvidia-smi` must work. |
| FFmpeg | **Crucial:** Must support `h264_nvenc` for GPU acceleration. Standard CPU builds will be very slow. |
| 7-Zip CLI | `7z` binary (`p7zip-full` on Linux, official installer on Windows). |
| PAR2 | `par2`/`par2cmdline` binary for creating redundancy blocks. |
| Disk space | Enough to store the temp archive, PAR2 set, intermediate frames, and final MP4(s). |

---

## Installation & Setup

### Step 1: Get the Code (Required for both methods)
Regardless of how you run the script, you first need to download the code to your machine.

```bash
git clone https://github.com/Keksmania/PythonFileToYoutube.git
cd PythonFileToYoutube
```

### Step 2: Choose your Installation Method

#### Option A: Docker (Recommended)
Docker handles all software dependencies (Python, FFmpeg, 7-Zip, PAR2, PyTorch) automatically. You only need to ensure your host machine has the GPU drivers and container toolkit installed.

**1. Install GPU Support for Docker:**

*   **Windows Users:**
    *   Install **[Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/)**.
    *   During installation (or in Settings > General), ensure **"Use the WSL 2 based engine"** is checked.
    *   *Note:* The NVIDIA Container Toolkit is included with Docker Desktop on Windows. You do not need to install it separately.

*   **Linux Users:**
    *   Install the **[Docker Engine](https://docs.docker.com/engine/install/)**.
    *   **Crucial:** You must install the **[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** so Docker can access your GPU.
    *   *Quick command for apt-based systems (Ubuntu/Debian):*
        ```bash
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
        && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
        
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        sudo nvidia-ctk runtime configure --runtime=docker
        sudo systemctl restart docker
        ```

**2. Build and Start the Container:**
Make sure you are inside the `PythonFileToYoutube` folder.
```bash
docker-compose up -d --build
```
*This command downloads the heavy PyTorch image and sets up the environment. It may take a few minutes.*

**3. Understanding Data Transfer:**
The Docker setup uses a **Bind Mount**. The folder `PythonFileToYoutube` on your computer is mapped directly to `/app` inside the Docker container.
*   **To Encode:** Put your files in the `PythonFileToYoutube` folder on your computer. Inside Docker, they appear at `/app/filename`.
*   **To Retrieve:** The output videos created by Docker will instantly appear in your `PythonFileToYoutube` folder on your computer.

---

#### Option B: Local Install (Manual)
Use this if you prefer to manage dependencies yourself directly on your host OS.

1.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install External Tools:**
    *   **Windows:** 
        *   **FFmpeg:** Download a "Full" build from [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or [BtbN](https://github.com/BtbN/FFmpeg-Builds/releases). 
        *   **Important:** Ensure you download a version that explicitly lists **hardware acceleration** or **non-free** support. The generic "essentials" build sometimes lacks NVENC.
        *   **7-Zip:** Download and install from [7-zip.org](https://www.7-zip.org/download.html).
        *   **PAR2:** Download `par2cmdline` for Windows.
        *   **Path:** You **must** add the folders containing `ffmpeg.exe`, `7z.exe`, and `par2.exe` to your Windows System `PATH` environment variable.

    *   **Linux:**
        ```bash
        sudo apt update
        sudo apt install ffmpeg p7zip-full par2
        ```
        *   *Note:* Verify your installed FFmpeg supports NVENC by running `ffmpeg -encoders | grep nvenc`. If it produces no output, you may need to compile FFmpeg from source with `--enable-nvenc` or find a static build that includes it.

3.  **Verify Environment:**
    Run the included check script to ensure all tools are visible.
    *   **Windows PowerShell:** `.\tools\check_deps_windows.cmd`
    *   **Linux/macOS Bash:** `./tools/check_deps_linux.sh`

---

## Usage

### 1. Encoding (File -> Video)

**Using Docker:**
```bash
# Note: Input path must start with /app/
docker-compose exec f2yt python file_to_video_torch.py -mode encode -input "/app/my_secret_data.zip"
```

**Using Local Install:**
```powershell
python file_to_video_torch.py -mode encode -input "C:\path\to\my_secret_data.zip" [-output "C:\path\to\output_dir"]
```

*   **Output:** The script will create `*_F2YT.mp4` (or `_part001.mp4`) files in the output directory.
*   **Options:** Add `-p mypassword` to password-protect the internal archive.

### 2. Decoding (Video -> File)

**Using Docker:**
```bash
# Point to the FOLDER containing the video parts (Recommended)
docker-compose exec f2yt python file_to_video_torch.py -mode decode -input "/app/my_secret_data_F2YT_Output"
```

**Using Local Install:**
```powershell
# Point to the FOLDER containing the video parts
python file_to_video_torch.py -mode decode -input "C:\path\to\my_secret_data_F2YT_Output"
```

*   **Sorting:** If you point input to a folder, the script automatically finds and sorts valid video files.
*   **Result:** Decoded files will appear in `<output_dir>/Decoded_Files` once 7-Zip and PAR2 recovery are complete.

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

- **Dependency missing** – Run the OS-specific dependency script (Option B steps); install any reported tool and rerun.
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