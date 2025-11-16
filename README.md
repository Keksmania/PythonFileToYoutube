# PythonFileToYoutube

Turn any file or folder into one or more MP4 videos (and back again) using PyTorch-powered encoding, 7-Zip compression, PAR2 redundancy, and FFmpeg/x264. This repository contains the reference implementation plus tools for dependency checking and testing.

## Features

- **End-to-end pipeline** – archive → error-correct → encode to frames → stream to FFmpeg, plus the reverse decode path.
- **GPU-accelerated** – uses PyTorch to process large frame batches efficiently (CUDA strongly recommended).
- **Segmented output** – automatically rotates MP4 files when they exceed the configured length (default ~11 hours at 60 fps).
- **Robust metadata** – barcode + info frames describe payload size, frame counts, and optional password protection.
- **Test suite** – run `python file_to_video_torch.py -test` to exercise Hamming codecs, frame round trips, and a miniature encode/decode flow.
- **Cross-platform** – works on Windows and Linux as long as the external tools are installed.

## Requirements

| Component | Notes |
| --- | --- |
| Python | 3.11 (system install). Earlier versions are untested. |
| PyTorch | GPU build (2.9.0+ with CUDA 13.x). Install from https://pytorch.org. |
| CUDA | NVIDIA GPU drivers + CUDA toolkit matching your PyTorch build. `nvidia-smi` must work. |
| FFmpeg | Must include `libx264` encoder. The default config invokes `ffmpeg` from PATH. |
| 7-Zip CLI | `7z` binary (`p7zip-full` on Linux, official installer on Windows). |
| PAR2 | `par2`/`par2cmdline` binary for creating redundancy blocks. |
| Disk space | Enough to store the temp archive, PAR2 set, intermediate frames, and final MP4(s). |

Use the provided dependency scripts to verify your environment quickly:

- Windows PowerShell: `.\tools\check_deps_windows.ps1`
- Linux/macOS Bash: `./tools/check_deps_linux.sh`

Each script lists missing dependencies and suggests installation steps (including CUDA checks).

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<you>/PythonFileToYoutube.git
   cd PythonFileToYoutube
   ```
2. **Install Python dependencies** – the project relies solely on the standard library and PyTorch (installed globally via the PyTorch instructions).
3. **Install external tools** – ensure `ffmpeg`, `7z`, and `par2` commands are on your PATH.
4. **Configure** – edit `f2yt_config.json` if you need custom paths or encoding settings. Defaults assume 720×720 video, 60 fps, and RGB data frames.
5. **Verify environment** – run the OS-appropriate dependency script listed above.

## Usage

### Encoding
```powershell
python file_to_video_torch.py -mode encode -input C:\path\to\file_or_folder -output C:\path\to\output_dir [-p password]
```
- The output directory will gain `*_F2YT.mp4` (or segmented `_part###.mp4`) files plus temporary working folders.
- Set `-p` to password-protect the 7z archive (`-mhe=on`), which keeps filenames encrypted.

### Decoding
```powershell
python file_to_video_torch.py -mode decode -input "video.mp4[,video_part002.mp4,...]" -output C:\path\to\restore [-p password]
```
- You may provide a comma-separated list if the payload spans multiple segments.
- Decoded files end up in `<output>/Decoded_Files` once 7-Zip and PAR2 complete.

### Running the Test Suite
```powershell
python file_to_video_torch.py -test
```
The suite validates:
- Hamming(7,4) codec integrity.
- Pixel conversion round trips for info/data frames.
- Info block encode/decode consistency.
- Data frame encode/decode at 32×32.
- Large (100 KB) stress test.
- Password-protected encode/decode flow.

## Configuration Reference (`f2yt_config.json`)

| Key | Description |
| --- | --- |
| `FFMPEG_PATH` | Path to the FFmpeg binary. |
| `SEVENZIP_PATH` | Path to the 7-Zip CLI (`7z`). |
| `PAR2_PATH` | Path to the PAR2 executable. |
| `VIDEO_WIDTH`/`VIDEO_HEIGHT` | Output resolution (default 720×720). |
| `VIDEO_FPS` | Frames per second (default 60). |
| `DATA_K_SIDE` | Data frame size (default 180). |
| `NUM_COLORS_DATA` | Palette size (power of two). |
| `PAR2_REDUNDANCY_PERCENT` | PAR2 redundancy percentage (default 30). |
| `X264_CRF` | Quality parameter for x264 (lower = larger files). |
| `CPU_PRODUCER_CHUNK_MB` | File chunk size for the producer thread. |
| `GPU_PROCESSOR_BATCH_SIZE` | Number of frames processed per GPU batch. |
| `MAX_VIDEO_SEGMENT_HOURS` | Approx hours per MP4 before rolling to a new segment (default 11). |
| `MAX_VIDEO_SEGMENT_FRAMES_OVERRIDE` | Optional explicit frame cap for testing segmentation. |

## Workflow Overview

1. `prepare_files_for_encoding` compresses the payload with 7-Zip and creates PAR2 files.
2. Info frames + barcode encode session metadata (JSON) using Hamming(7,4).
3. `DataProducerThread` streams payload bits to `encode_data_frames_gpu`, which packs them into RGB tiles.
4. `FFmpegConsumerThread` pipes the RGB stream into `ffmpeg -f rawvideo ... -c:v libx264` to produce MP4 segments.
5. Decoding runs the inverse steps, reconstructing files and verifying hash sizes via the manifest.

## Troubleshooting

- **Dependency missing** – run the OS-specific dependency script; install any reported tool and rerun.
- **CUDA unavailable** – confirm `nvidia-smi` works and that your PyTorch build reports `torch.cuda.is_available() == True`.
- **FFmpeg errors** – inspect the log output (already printed). Ensure your `ffmpeg` build supports `libx264`.
- **Segment naming** – the encoder now normalizes all files to end in `.mp4`; check permissions if renames fail.
- **Decode issues** – verify that the info frame count matches (watch for warnings) and that you supply all segment files in order.

## Contributing

Issues and PRs are welcome! Please include:
- A description of the problem / feature.
- Steps to reproduce (or new tests).
- Hardware/OS details, especially GPU and CUDA versions.

## License

MIT.
