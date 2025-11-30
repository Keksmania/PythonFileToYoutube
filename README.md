# PythonFileToYoutube

Turn any file or folder into one or more MP4 videos (and back again) using the file_to_video_torch.py script provided in this repo.
The script is optimized for youtube. Meaning the default config is good enough that a file should be able to survice a full round trip to youtube and back. 

The software has a CLI.


## Features

- **End-to-end pipeline** – archive → error-correct → encode to frames → stream to FFmpeg, plus the reverse decode path.
- **GPU-accelerated** – uses PyTorch to process large frame batches efficiently (CUDA strongly recommended).
- **NVENC Support** – Optional hardware encoding for massive speed gains (see configuration).
- **Asynchronous Processing** – Decoupled CPU unpacking and GPU encoding threads for maximum throughput and smooth resource usage.
- **Robust Redundancy** – Splits large files into 1GB volumes and generates optimized 128KB PAR2 blocks for granular recovery.
- **Segmented output** – Automatically rotates MP4 files when they exceed the configured length (default ~11 hours at 60fps).
- **Robust metadata** – Barcode + info frames describe payload size, frame counts, and optional password protection.
- **Cross-platform** – Works on Windows and Linux as long as the external tools are installed.

## Requirements

| Component | Notes |
| --- | --- |
| Python | 3.11 (system install). Earlier versions are untested. |
| PyTorch | GPU build (2.9.0+ with CUDA 13.x). Install from https://pytorch.org. |
| CUDA | NVIDIA GPU drivers + CUDA toolkit matching your PyTorch build. `nvidia-smi` must work. |
| FFmpeg | Must include `libx264` encoder. For GPU encoding, it must support `h264_nvenc`. |
| 7-Zip CLI | `7z` binary (`p7zip-full` on Linux, official installer on Windows). |
| PAR2 | `par2`/`par2cmdline` binary for creating redundancy blocks. |
| Disk space | Enough to store the temp archive, PAR2 set, intermediate frames, and final MP4(s). |

Use the provided dependency scripts to verify your environment quickly:

- Windows PowerShell: `.\tools\check_deps_windows.cmd`
- Linux/macOS Bash: `./tools/check_deps_linux.sh`

Each script lists missing dependencies and suggests installation steps (including CUDA checks).

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Keksmania/PythonFileToYoutube.git
   cd PythonFileToYoutube
   ```
2. **Install Python dependencies** – the project relies solely on the standard library, `numpy` and `torch`.
3. **Install external tools** – ensure `ffmpeg`, `7z`, and `par2` commands are on your PATH.
4. **Configure** – edit `f2yt_config.json` if you need custom paths or encoding settings. Defaults assume 720x720 video, 60fps, and RGB data frames.
5. **Verify environment** – run the OS-appropriate dependency script listed above.

## Usage

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
- You **must** provide a comma-separated list of all segment files if the payload spans multiple videos.
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
4. **GPU Packing**: `encode_data_frames_gpu` packs unpacked bits into RGB tiles on the GPU using CUDA.
5. **Video Stream**: `FFmpegConsumerThread` pipes the RGB stream into `ffmpeg` (via CPU `libx264` or GPU `h264_nvenc`) to produce MP4 segments.
6. **Decoding**: Runs the inverse steps: `FrameProducerThread` fetches frames -> GPU Decodes -> `DataWriterThread` writes to disk -> PAR2 Repairs -> 7-Zip Extracts.

## Troubleshooting

- **Dependency missing** – Run the OS-specific dependency script; install any reported tool and rerun.
- **CUDA unavailable** – Confirm `nvidia-smi` works and that your PyTorch build reports `torch.cuda.is_available() == True`.
- **FFmpeg errors** – Inspect the log output. Ensure your `ffmpeg` build supports `libx264` (or `h264_nvenc` if enabled).
- **Decode issues** – Verify that the info frame count matches (watch for warnings) and that you supply **all** segment files in the correct order.
- **Output Resolution** – The script forces strict resolution scaling filters to avoid "mod-16" cropping issues. Ensure `VIDEO_WIDTH` is divisible by `DATA_K_SIDE`.

## Contributing

Issues and PRs are welcome! Please include:
- A description of the problem / feature.
- Hardware/OS details, especially GPU and CUDA versions.

## License

MIT.