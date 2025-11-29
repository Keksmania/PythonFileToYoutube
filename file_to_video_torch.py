# file_to_video_torch.py

import argparse
import getpass
import json
import logging
import math
import os
import shlex
import subprocess
import sys
import threading
import queue
from collections import deque
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Deque, Union
import io
from PIL import Image
import filecmp
import tempfile
import shutil
import time

import numpy as np
import torch

# --- Constants ---
CONFIG_FILENAME = "f2yt_config.json"
TEMP_ENCODE_DIR = "temp_encode_processing"
TEMP_DECODE_DIR = "temp_decode_processing"
ASSEMBLED_FILES_DIR = "assembled_files"
BARCODE_NUM_BITS = 32
PIXEL_CHANNELS = 3

# Config metadata embedded into info frames so decoders can reproduce settings
DECODE_CONFIG_EXPORT_KEYS = [
    "DATA_K_SIDE",
    "NUM_COLORS_DATA"
]

# --- PyTorch Constants ---
INFO_HAMMING_K = 4
INFO_HAMMING_N = 7
INFO_K_SIDE = 16
INFO_BITS_PER_PALETTE_COLOR = 1
INFO_COLOR_PALETTE_TENSOR = torch.tensor([[0, 0, 0], [255, 255, 255]], dtype=torch.uint8)
INFO_G_MATRIX_TENSOR = torch.tensor([
    [1,0,0,0,1,1,1], [0,1,0,0,1,1,0], [0,0,1,0,1,0,1], [0,0,0,1,0,1,1]
], dtype=torch.uint8)
INFO_H_MATRIX_TENSOR = torch.tensor([
    [1,1,1,0,1,0,0], [1,1,0,1,0,1,0], [1,0,1,1,0,0,1]
], dtype=torch.uint8)


# Data Frames - Use same Hamming(7,4) as Info Frames
DATA_HAMMING_K = 4
DATA_HAMMING_N = 7
DATA_G_MATRIX_TENSOR = torch.tensor([
    [1,0,0,0,1,1,1], [0,1,0,0,1,1,0], [0,0,1,0,1,0,1], [0,0,0,1,0,1,1]
], dtype=torch.uint8)
DATA_H_MATRIX_TENSOR = torch.tensor([
    [1,1,1,0,1,0,0], [1,1,0,1,0,1,0], [1,0,1,1,0,0,1]
], dtype=torch.uint8)

# --- Setup and Configuration ---

def setup_logging(level=logging.INFO):
    logging.basicConfig(level=level, format="[%(levelname)s] %(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", stream=sys.stdout)

def setup_pytorch() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(device)
        logging.info(f"Found GPU: {props.name} with {props.total_memory / 1e9:.2f} GB of memory.")
        logging.info(f"PyTorch version: {torch.__version__}")
        try:
            major, minor = torch.cuda.get_device_capability(device)
            logging.info(f"CUDA Runtime version: {torch.version.cuda} | GPU Capability: {major}.{minor}")
        except Exception as e:
            logging.warning(f"Could not retrieve full CUDA details: {e}")
    else:
        logging.warning("CUDA is not available. Running on CPU. This will be very slow.")
        device = torch.device("cpu")
    return device

def load_config() -> Dict[str, Any]:
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / CONFIG_FILENAME
    default_config = {
        "FFMPEG_PATH": "ffmpeg", "SEVENZIP_PATH": "7z", "PAR2_PATH": "par2",
        "VIDEO_WIDTH": 720, "VIDEO_HEIGHT": 720, "VIDEO_FPS": 60,
        "DATA_K_SIDE": 180, "NUM_COLORS_DATA": 2,
        "PAR2_REDUNDANCY_PERCENT": 5, "X264_CRF": 40,
        "KEYINT_MAX": 2,
        "CPU_PRODUCER_CHUNK_MB": 128, 
        "GPU_PROCESSOR_BATCH_SIZE": 1024,
        "MAX_VIDEO_SEGMENT_HOURS": 11,
        "GPU_OVERLAP_STREAMS": 8,
        "PIPELINE_QUEUE_DEPTH": 64, 
        "CPU_WORKER_THREADS": 2,
        "ENABLE_NVENC": False
    }
    if not config_path.exists():
        logging.info(f"Config file not found. Creating default at '{config_path}'")
        try:
            with open(config_path, 'w') as f: json.dump(default_config, f, indent=4)
            return default_config
        except IOError as e:
            logging.error(f"Could not create default config file: {e}")
            logging.warning("Using internal default configuration."); return default_config
    logging.info(f"Loading configuration from '{config_path}'")
    try:
        with open(config_path, 'r') as f: user_config = json.load(f)
        default_config.update(user_config)
        return default_config
    except (IOError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load or parse config file: {e}")
        logging.warning("Using internal default configuration."); return default_config


def capture_decode_config_snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Collect critical encode settings required for decoding on another machine."""
    snapshot = {}
    for key in DECODE_CONFIG_EXPORT_KEYS:
        if key in config and config[key] is not None:
            snapshot[key] = config[key]
    # Include informational constants for completeness/debugging
    snapshot["INFO_K_SIDE"] = INFO_K_SIDE
    snapshot["INFO_BITS_PER_PALETTE_COLOR"] = INFO_BITS_PER_PALETTE_COLOR
    snapshot["DATA_HAMMING_K"] = DATA_HAMMING_K
    snapshot["DATA_HAMMING_N"] = DATA_HAMMING_N
    return snapshot


def apply_snapshot_to_config(config: Dict[str, Any], snapshot: Optional[Dict[str, Any]]) -> None:
    """Override local config with manifest-provided settings when available."""
    if not snapshot:
        logging.warning("Manifest missing encode_config_snapshot; using local configuration values.")
        return
    applied_keys = []
    missing_keys = []
    for key in DECODE_CONFIG_EXPORT_KEYS:
        if key in snapshot:
            config[key] = snapshot[key]
            applied_keys.append(key)
        else:
            missing_keys.append(key)
    if applied_keys:
        logging.info(f"Applied decode settings from manifest: {', '.join(applied_keys)}")
    if missing_keys:
        logging.warning(f"Manifest snapshot missing these decode keys: {', '.join(missing_keys)}")


def cleanup_temp_dir(temp_path: Path, label: str) -> None:
    """Best-effort removal of a temporary working directory."""
    if temp_path.exists():
        try:
            shutil.rmtree(temp_path)
            logging.info(f"Cleaned up {label} directory: {temp_path}")
        except Exception as e:
            logging.warning(f"Failed to remove {label} directory '{temp_path}': {e}")


def pin_tensor_if_possible(tensor: torch.Tensor) -> torch.Tensor:
    """Pin CPU tensor memory to enable async GPU transfers when CUDA is available."""
    if tensor.device.type != "cpu":
        return tensor
    if not torch.cuda.is_available():
        return tensor
    try:
        return tensor.pin_memory()
    except RuntimeError:
        logging.debug("pin_memory not supported for this tensor; continuing without pinning.")
        return tensor

# --- Core Utility Functions ---

def run_command(command: List[str], cwd: Optional[str] = None, stream_output: bool = False) -> bool:
    """
    Run a subprocess command.
    """
    logging.info(f"Running command: {shlex.join(command)}")
    try:
        if stream_output:
            process = subprocess.run(command, check=False, cwd=cwd, text=True)
        else:
            process = subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd)
        
        if not stream_output:
            if process.stdout: logging.info(f"STDOUT:\n{process.stdout.strip()}")
            if process.stderr:
                is_ffmpeg_progress = 'frame=' in process.stderr and 'fps=' in process.stderr
                log_level = logging.ERROR if process.returncode != 0 else logging.INFO
                if not (is_ffmpeg_progress and process.returncode == 0):
                     logging.log(log_level, f"STDERR:\n{process.stderr.strip()}")
        
        if process.returncode != 0:
            logging.error(f"Command failed with exit code {process.returncode}."); return False
        return True
    except FileNotFoundError:
        logging.error(f"Command not found: '{command[0]}'. Please check your PATH or {CONFIG_FILENAME}."); return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while running command: {e}"); return False

# --- PyTorch Utility Functions ---

def bytes_to_bit_tensor(data_bytes: bytes, device: torch.device) -> torch.Tensor:
    np_bytes = np.frombuffer(data_bytes, dtype=np.uint8)
    np_bits = np.unpackbits(np_bytes)
    return torch.from_numpy(np_bits).to(device)

def tensor_to_frame(bits_tensor: torch.Tensor, k_side: int, palette: torch.Tensor, bits_per_color: int) -> torch.Tensor:
    device = bits_tensor.device
    num_pixels = k_side * k_side
    required_bits = num_pixels * bits_per_color
    if bits_tensor.numel() < required_bits:
        pad_n = required_bits - bits_tensor.numel()
        logging.debug(f"tensor_to_frame: padding {pad_n} bits (required {required_bits}, got {bits_tensor.numel()})")
        padding = torch.zeros(pad_n, dtype=torch.uint8, device=device)
        bits_tensor = torch.cat((bits_tensor, padding))
    bits_for_frame = bits_tensor[:required_bits]
    bit_groups = bits_for_frame.view(num_pixels, bits_per_color)
    powers_of_2 = 2 ** torch.arange(bits_per_color - 1, -1, -1, device=device, dtype=torch.long)
    palette = palette.to(device)
    palette_indices = torch.matmul(bit_groups.float(), powers_of_2.float()).long()
    pixel_data = torch.nn.functional.embedding(palette_indices, palette)
    num_channels = palette.shape[1]
    frame_tensor = pixel_data.view(k_side, k_side, num_channels)
    return frame_tensor.to(torch.uint8)


def bit_chunks_to_frame_batch(
    frame_bit_chunks: torch.Tensor,
    k_side: int,
    palette: torch.Tensor,
    bits_per_color: int
) -> torch.Tensor:
    """Convert encoded bit chunks for many frames into actual RGB tensors in one shot."""
    if frame_bit_chunks.numel() == 0:
        return torch.empty(0, k_side, k_side, palette.shape[1], dtype=torch.uint8, device=frame_bit_chunks.device)

    device = frame_bit_chunks.device
    num_frames = frame_bit_chunks.shape[0]
    num_pixels = k_side * k_side
    required_bits = num_pixels * bits_per_color
    current_bits = frame_bit_chunks.shape[1]
    if current_bits < required_bits:
        pad_n = required_bits - current_bits
        frame_bit_chunks = torch.nn.functional.pad(frame_bit_chunks, (0, pad_n))
    elif current_bits > required_bits:
        frame_bit_chunks = frame_bit_chunks[:, :required_bits]

    bit_groups = frame_bit_chunks.view(num_frames, num_pixels, bits_per_color)
    powers_of_2 = 2 ** torch.arange(bits_per_color - 1, -1, -1, device=device, dtype=torch.long)
    palette = palette.to(device)
    palette_indices = torch.matmul(bit_groups.float(), powers_of_2.float()).long()
    pixel_data = torch.nn.functional.embedding(palette_indices, palette)
    num_channels = palette.shape[1]
    return pixel_data.view(num_frames, k_side, k_side, num_channels).to(torch.uint8)

def generate_palette_tensor(num_colors: int, device: torch.device) -> torch.Tensor:
    if num_colors <= 0 or (num_colors & (num_colors - 1)) != 0:
        logging.warning(f"num_colors must be a power of 2. Got {num_colors}. Defaulting to 2.")
        num_colors = 2
    palettes = {
        2: [[0, 0, 0], [255, 255, 255]],
        4: [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]],
        8: [[0,0,0], [255,0,0], [0,255,0], [0,0,255], [255,255,0], [255,0,255], [0,255,255], [255,255,255]],
    }
    if num_colors not in palettes:
        logging.warning(f"No predefined palette for {num_colors} colors. Generating a grayscale ramp.")
        palette_rgb = [[int(255 * i / (num_colors - 1))] * 3 for i in range(num_colors)]
    else:
        palette_rgb = palettes[num_colors]
    return torch.tensor(palette_rgb, dtype=torch.uint8, device=device)

def frame_to_bits_batch(frame_batch: torch.Tensor, palette: torch.Tensor, bits_per_color: int) -> torch.Tensor:
    device = frame_batch.device
    num_frames, h, w, c = frame_batch.shape

    pixels = frame_batch.view(num_frames * h * w, c).float()
    palette = palette.to(device).float()
    distances = torch.cdist(pixels, palette)
    indices = torch.argmin(distances, dim=1)
    
    unpacked_bits = []
    for i in range(bits_per_color - 1, -1, -1):
        unpacked_bits.append((indices >> i) & 1)
    result = torch.stack(unpacked_bits, dim=1).view(-1).to(torch.uint8)
    # Diagnostic: report counts and small sample of bits at INFO level
    total_bits = result.numel()
    logging.info(f"frame_to_bits_batch: extracted {total_bits} bits from {num_frames} frames ({h}x{w}), bits_per_color={bits_per_color}")
    return result


def build_syndrome_lookup_table(h_matrix: torch.Tensor) -> torch.Tensor:
    n, m = h_matrix.shape[1], h_matrix.shape[0]
    device = h_matrix.device
    powers_of_2 = 2 ** torch.arange(m - 1, -1, -1, device=device, dtype=torch.long)
    error_patterns = torch.zeros(2**m, n, dtype=torch.uint8, device=device)
    for i in range(n):
        syndrome_vec = h_matrix[:, i]
        syndrome_idx = torch.matmul(syndrome_vec.float(), powers_of_2.float()).long()
        if syndrome_idx > 0:
            if torch.sum(error_patterns[syndrome_idx]) == 0:
                error_patterns[syndrome_idx, i] = 1
    return error_patterns

def hamming_decode_gpu(
    received_bits: torch.Tensor,
    h_matrix: torch.Tensor,
    k: int,
    syndrome_table: torch.Tensor,
    debug: bool = False,
    return_error_tensor: bool = False
) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
    """
    Decodes Hamming encoded bits.
    NOTE: If more than 1 error is present in a block (which Hamming(7,4) cannot handle),
    this function will still output a 'corrected' codeword based on the syndrome.
    This implies we DO NOT throw away blocks with multiple errors. We pass the 'best guess'
    miscorrected data to PAR2, ensuring synchronization of the bitstream is maintained.
    """
    n = h_matrix.shape[1]
    device = received_bits.device
    codewords = received_bits.view(-1, n)
    h_t = h_matrix.t().float().to(device)
    syndrome = (torch.matmul(codewords.float(), h_t) % 2).long()
    m = h_matrix.shape[0]
    powers_of_2 = 2 ** torch.arange(m - 1, -1, -1, device=device, dtype=torch.long)
    syndrome_indices = torch.matmul(syndrome.float(), powers_of_2.float()).long()
    
    error_count_tensor = (syndrome_indices > 0).sum()
    if return_error_tensor:
        num_errors: Union[int, torch.Tensor] = error_count_tensor
    else:
        num_errors = int(error_count_tensor.item())
    
    if debug and codewords.shape[0] > 0:
        # Log first 10 codewords' syndrome info
        logging.debug(f"Hamming decode debug: first 10 codewords:")
        for i in range(min(10, codewords.shape[0])):
            logging.debug(f"  CW {i}: syndrome_idx={syndrome_indices[i].item()}, has_error={syndrome_indices[i].item() > 0}")
    
    error_mask = syndrome_table[syndrome_indices]
    corrected_codewords = codewords ^ error_mask
    return corrected_codewords[:, :k].reshape(-1), num_errors

# --- Encoding Pipeline ---

def prepare_files_for_encoding(input_path: Path, temp_dir: Path, config: Dict, password: Optional[str]) -> Optional[Tuple[List[Path], Dict]]:
    logging.info("Step 1: Preparing and archiving files...")
    sz_path, par2_path = config["SEVENZIP_PATH"], config["PAR2_PATH"]
    input_path = input_path.resolve()
    if not input_path.exists(): logging.error(f"Input path does not exist: {input_path}"); return None
    file_to_compress = input_path
    if input_path.is_dir():
        logging.info("Input is a directory. Archiving its contents...")
        archive_path = temp_dir / f"{input_path.name}.7z"
        cmd = [sz_path, "a", "-y", str(archive_path), str(input_path) + '/*']
        if not run_command(cmd): logging.error("Failed to archive directory."); return None
        file_to_compress = archive_path
    
    logging.info("Compressing data payload with 7-Zip (splitting at 1GB)...")
    payload_archive_base = temp_dir / f"{file_to_compress.stem}_data_archive.7z"
    cmd = [sz_path, "a", "-v1024m", "-y", str(payload_archive_base), str(file_to_compress)]
    if password: cmd.extend([f"-p{password}", "-mhe=on"])
    if not run_command(cmd): logging.error("Failed to compress data payload."); return None
    
    archive_files = sorted(list(temp_dir.glob(f"{file_to_compress.stem}_data_archive.7z*")))
    if not archive_files:
        logging.error("Could not find generated archive files.")
        return None

    redundancy = config["PAR2_REDUNDANCY_PERCENT"]
    # Fixed 128KB block size (128 * 1024)
    block_size = 131072

    logging.info(f"Creating PAR2 recovery files for {len(archive_files)} volume(s)...")
    
    for vol_file in archive_files:
        if vol_file.suffix == '.par2': continue
        par2_base_name = vol_file.name + ".recovery"
        par2_full_path = temp_dir / par2_base_name
        logging.info(f"  Generating PAR2 for volume: {vol_file.name} (Block size: 256KB)")
        cmd = [
            par2_path, "c", "-qq", 
            f"-r{redundancy}", 
            f"-s{block_size}", 
            str(par2_full_path) + ".par2", 
            str(vol_file)
        ]
        if not run_command(cmd): 
            logging.error(f"Failed to create PAR2 for {vol_file.name}"); 
            return None

    logging.info("Generating file manifest...")
    files_to_encode, file_manifest = [], []
    all_files = sorted(temp_dir.glob("*"))
    for f_path in all_files:
        if f_path.is_file():
            if "data_archive.7z" in f_path.name and ".par2" not in f_path.name:
                file_type = "sz_vol" 
            elif f_path.name.endswith(".par2") and ".vol" not in f_path.name:
                file_type = "par2_main"
            elif ".vol" in f_path.name and f_path.name.endswith(".par2"):
                file_type = "par2_vol"
            else:
                continue 
            
            files_to_encode.append(f_path)
            file_manifest.append({"name": f_path.name, "size": f_path.stat().st_size, "type": file_type})

    if not files_to_encode: logging.error("File preparation resulted in no files to encode."); return None
    logging.info(f"File preparation complete. {len(files_to_encode)} files generated.")
    return files_to_encode, {"file_manifest_detailed": file_manifest}

def generate_barcode_frame(num_info_frames: int, config: Dict, device: torch.device) -> Optional[torch.Tensor]:
    logging.info("Step 2: Generating barcode frame...")
    video_width, video_height, encoder_fps = config['VIDEO_WIDTH'], config['VIDEO_HEIGHT'], config['VIDEO_FPS']
    if num_info_frames > 0xFFFF: logging.error(f"Too many info frames ({num_info_frames}) for 16-bit barcode."); return None
    if encoder_fps > 0xFF: logging.error(f"Encoder FPS ({encoder_fps}) too high for 8-bit barcode."); return None
    combined_value = (encoder_fps << 16) | num_info_frames
    bit_string = f'{combined_value:0{BARCODE_NUM_BITS}b}'
    colors = torch.tensor([255 if bit == '1' else 0 for bit in bit_string], dtype=torch.uint8, device=device)
    frame_tensor = torch.zeros((video_height, video_width, PIXEL_CHANNELS), dtype=torch.uint8, device=device)
    bar_width = video_width / float(BARCODE_NUM_BITS)
    for i in range(BARCODE_NUM_BITS):
        x_start, x_end = int(i * bar_width), int((i + 1) * bar_width)
        frame_tensor[:, x_start:x_end, :3] = colors[i]
    logging.info(f"Generated barcode for {num_info_frames} info frames and {encoder_fps} FPS.")
    return frame_tensor.unsqueeze(0)

def generate_info_artifacts(file_manifest: Dict, config: Dict, device: torch.device, derived_params: Optional[Dict] = None) -> Optional[torch.Tensor]:
    logging.info("Step 3: Generating info frame artifacts...")
    info_json_obj = file_manifest.copy()
    info_json_obj.update({"version": "Python_FileToYoutube_v1.0.1", "is_password_protected": config.get("is_password_protected", False), "info_frame_count": 0, "data_frame_count": 0})
    info_json_obj["encode_config_snapshot"] = capture_decode_config_snapshot(config)

    bits_per_frame = INFO_K_SIDE * INFO_K_SIDE * INFO_BITS_PER_PALETTE_COLOR

    if derived_params is None:
        derived_params = get_derived_encoding_params(config, device)
    payload_bytes_per_frame = derived_params['payload_bits_data'] / 8
    total_payload_bytes = sum(f['size'] for f in file_manifest['file_manifest_detailed'])
    actual_data_frames = math.ceil(total_payload_bytes / payload_bytes_per_frame) if payload_bytes_per_frame > 0 else 0
    info_json_obj["data_frame_count"] = actual_data_frames  # tells decoder how many real data frames exist

    total_payload_bits = total_payload_bytes * 8
    hamming_k = derived_params["hamming_k"]
    hamming_n = derived_params["hamming_n"]
    num_codewords = math.ceil(total_payload_bits / hamming_k) if hamming_k > 0 else 0
    info_json_obj["total_payload_bits"] = total_payload_bits
    info_json_obj["total_encoded_data_bits"] = num_codewords * hamming_n

    g_matrix = INFO_G_MATRIX_TENSOR.to(device).float()

    def encode_manifest_payload() -> Tuple[bytes, torch.Tensor, torch.Tensor, torch.Tensor]:
        final_json_bytes_local = json.dumps(info_json_obj, separators=(',', ':')).encode('utf-8')
        len_bytes = len(final_json_bytes_local).to_bytes(4, 'big')
        payload_bytes = len_bytes + final_json_bytes_local
        payload_bits_local = bytes_to_bit_tensor(payload_bytes, device)
        rem = payload_bits_local.numel() % INFO_HAMMING_K
        if rem != 0:
            padding = torch.zeros(INFO_HAMMING_K - rem, dtype=torch.uint8, device=device)
            payload_bits_local = torch.cat((payload_bits_local, padding))
        data_chunks_local = payload_bits_local.view(-1, INFO_HAMMING_K)
        coded_chunks_local = (torch.matmul(data_chunks_local.float(), g_matrix) % 2).to(torch.uint8)
        return final_json_bytes_local, payload_bits_local, data_chunks_local, coded_chunks_local

    info_json_obj["info_frame_count"] = max(info_json_obj.get("info_frame_count", 0), 0)
    final_json_bytes = b""
    payload_bits = torch.empty(0, dtype=torch.uint8, device=device)
    data_chunks = torch.empty(0, INFO_HAMMING_K, dtype=torch.uint8, device=device)
    coded_chunks = torch.empty(0, INFO_HAMMING_N, dtype=torch.uint8, device=device)

    for iteration in range(10):
        final_json_bytes, payload_bits, data_chunks, coded_chunks = encode_manifest_payload()
        coded_bits_flat = coded_chunks.view(-1)
        required_frames = math.ceil(coded_bits_flat.numel() / bits_per_frame)
        if required_frames == info_json_obj["info_frame_count"]:
            break
        info_json_obj["info_frame_count"] = required_frames
    else:
        logging.warning("Info manifest frame count did not stabilize after 10 iterations; using last computed value.")
        info_json_obj["info_frame_count"] = required_frames
        coded_bits_flat = coded_chunks.view(-1)

    final_frame_count = max(info_json_obj["info_frame_count"], 1)

    logging.info("--- ENCODING SUMMARY (Info Frames) ---")
    logging.info(f"JSON length: {len(final_json_bytes)} bytes. Total payload with header: {payload_bits.numel() // 8} bytes.")
    logging.info(f"Total payload bits: {payload_bits.numel()}")
    num_h_blocks = payload_bits.numel() // INFO_HAMMING_K
    logging.info(f"Total Hamming({INFO_HAMMING_N},{INFO_HAMMING_K}) blocks: {num_h_blocks}")
    for i in range(min(10, num_h_blocks)):
        logging.info(f"  Block {i:<2} Data: {data_chunks[i].cpu().numpy()} -> Encoded: {coded_chunks[i].cpu().numpy()}")

    coded_bits_flat = coded_chunks.view(-1)
    total_bits_for_all_frames = final_frame_count * bits_per_frame
    if coded_bits_flat.numel() < total_bits_for_all_frames:
        padding = torch.zeros(total_bits_for_all_frames - coded_bits_flat.numel(), dtype=torch.uint8, device=device)
        coded_bits_flat = torch.cat((coded_bits_flat, padding))
    else:
        coded_bits_flat = coded_bits_flat[:total_bits_for_all_frames]

    frame_data_block = coded_bits_flat.view(final_frame_count, bits_per_frame)
    logging.info(f"   Storing in manifest: info_frame_count={final_frame_count}, data_frame_count={actual_data_frames}")
    all_frames = [tensor_to_frame(frame_data_block[i], INFO_K_SIDE, INFO_COLOR_PALETTE_TENSOR, INFO_BITS_PER_PALETTE_COLOR) for i in range(final_frame_count)]
    
    if not all_frames: return None
    result_frames = torch.stack(all_frames)
    logging.info(f"Generated {final_frame_count} info frames. Result tensor shape: {result_frames.shape}")
    
    return result_frames

def encode_data_frames_gpu(bits_tensor_gpu: torch.Tensor, derived_params: Dict) -> torch.Tensor:
    device = bits_tensor_gpu.device
    k_side = derived_params['DATA_K_SIDE']
    palette = derived_params['data_palette']
    bits_per_color = derived_params['bits_per_pixel_data']
    g_matrix = derived_params['g_matrix'].to(device).float()
    hamming_k = derived_params['hamming_k']
    bits_flat = bits_tensor_gpu.view(-1).to(device).to(torch.uint8)
    actual_payload_bits = bits_flat.numel()  # Track actual payload before padding
    rem = bits_flat.numel() % hamming_k
    if rem != 0:
        pad_n = hamming_k - rem
        bits_flat = torch.cat((bits_flat, torch.zeros(pad_n, dtype=torch.uint8, device=device)))
    data_chunks = bits_flat.view(-1, hamming_k)
    if g_matrix.shape[0] != hamming_k:
        if g_matrix.shape[1] == hamming_k:
            g_matrix = g_matrix.t()
        else:
            raise ValueError(f"G matrix shape {tuple(g_matrix.shape)} incompatible with hamming_k={hamming_k}.")
    coded_chunks = (torch.matmul(data_chunks.float(), g_matrix) % 2).to(torch.uint8)
    coded_bits_flat = coded_chunks.view(-1)
    payload_chunks_per_frame = derived_params['payload_chunks_per_frame']
    if payload_chunks_per_frame <= 0:
        raise ValueError("payload_chunks_per_frame must be > 0")
    num_frames = coded_chunks.shape[0] // payload_chunks_per_frame
    logging.debug(f"encode_data_frames_gpu: coded_chunks={coded_chunks.shape}, payload_chunks_per_frame={payload_chunks_per_frame}, num_frames={num_frames}")
    if num_frames == 0:
        return torch.empty(0, k_side, k_side, 4, dtype=torch.uint8, device=device)
    bits_per_frame_encoded = derived_params['total_encoded_bits_to_store_data']
    needed = num_frames * bits_per_frame_encoded
    if coded_bits_flat.numel() < needed:
        pad_n = needed - coded_bits_flat.numel()
        coded_bits_flat = torch.cat((coded_bits_flat, torch.zeros(pad_n, dtype=torch.uint8, device=device)))
    else:
        coded_bits_flat = coded_bits_flat[:needed]
    frame_bit_chunks = coded_bits_flat.view(num_frames, bits_per_frame_encoded)
    frames_tensor = bit_chunks_to_frame_batch(frame_bit_chunks, k_side, palette, bits_per_color)
    logging.info(
        f"encode_data_frames_gpu: produced {frames_tensor.shape[0]} frames (k_side={k_side}), bits_per_frame={bits_per_frame_encoded}, actual_payload={actual_payload_bits} bits"
    )
    return frames_tensor

def get_derived_encoding_params(config: Dict, device: torch.device) -> Dict:
    params = config.copy()
    params['bits_per_pixel_data'] = int(math.log2(config['NUM_COLORS_DATA']))
    raw_bits_in_data_block = (config['DATA_K_SIDE'] ** 2) * params['bits_per_pixel_data']
    params['hamming_k'], params['hamming_n'] = DATA_HAMMING_K, DATA_HAMMING_N
    params['g_matrix'] = DATA_G_MATRIX_TENSOR.to(device)
    num_potential_groups = raw_bits_in_data_block // params['hamming_n']
    groups_for_byte_align = 8 // math.gcd(8, params['hamming_k'])
    actual_num_groups = (num_potential_groups // groups_for_byte_align) * groups_for_byte_align
    params['payload_bits_data'] = actual_num_groups * params['hamming_k']
    params['total_encoded_bits_to_store_data'] = actual_num_groups * params['hamming_n']
    logging.debug(f"Derived params: DATA_K_SIDE={config['DATA_K_SIDE']}, bits_per_pixel_data={params['bits_per_pixel_data']}, raw_bits_in_data_block={raw_bits_in_data_block}")
    logging.debug(f"Groups: num_potential_groups={num_potential_groups}, groups_for_byte_align={groups_for_byte_align}, actual_num_groups={actual_num_groups}")
    logging.debug(f"Payload bits: payload_bits_data={params['payload_bits_data']}, total_encoded_bits_to_store_data={params['total_encoded_bits_to_store_data']}")
    if params['payload_bits_data'] == 0:
        raise ValueError("Config results in zero payload bits. Increase DATA_K_SIDE or NUM_COLORS_DATA.")
    params['payload_chunks_per_frame'] = actual_num_groups
    params['bits_per_gpu_batch'] = config['GPU_PROCESSOR_BATCH_SIZE'] * params['payload_bits_data']
    params['data_palette'] = generate_palette_tensor(config['NUM_COLORS_DATA'], device)
    params['payload_chunks_per_frame'] = int(params['payload_chunks_per_frame'])
    params['bits_per_gpu_batch'] = int(params['bits_per_gpu_batch'])
    return params

# --- Parallel Producer Architecture ---

class DataProducerThread(threading.Thread):
    def __init__(self, files_to_encode: List[Path], data_queue: queue.Queue, derived_params: Dict):
        super().__init__(daemon=True)
        self.files_to_encode = files_to_encode
        self.data_queue = data_queue
        self.chunk_size_bytes = derived_params['CPU_PRODUCER_CHUNK_MB'] * 1024 * 1024
        self.bits_per_batch = derived_params['bits_per_gpu_batch']
        self.num_workers = derived_params.get('CPU_WORKER_THREADS', 2)
        self.stop_event = threading.Event()
        self.raw_queue = queue.Queue(maxsize=4) # Buffer for read-ahead

    def file_reader(self):
        """Thread that just reads bytes from disk to keep IO busy."""
        for file_path in self.files_to_encode:
            if self.stop_event.is_set(): break
            logging.info(f"Reading file: {file_path.name}")
            try:
                with open(file_path, 'rb') as f:
                    while not self.stop_event.is_set():
                        chunk = f.read(self.chunk_size_bytes)
                        if not chunk: break
                        self.raw_queue.put(chunk)
            except Exception as e:
                logging.error(f"File read error: {e}")
        self.raw_queue.put(None) # Signal workers to stop

    def bit_packer_worker(self):
        """Worker thread that unpacks bits (CPU intensive)."""
        bit_buffer = np.array([], dtype=np.uint8)
        
        while not self.stop_event.is_set():
            # If we have enough in buffer, push to main queue
            while len(bit_buffer) >= self.bits_per_batch:
                if self.stop_event.is_set(): return
                batch_data = bit_buffer[:self.bits_per_batch]
                self.data_queue.put(torch.from_numpy(batch_data).to(torch.uint8))
                bit_buffer = bit_buffer[self.bits_per_batch:]
            
            # Fetch more data
            raw_chunk = self.raw_queue.get()
            if raw_chunk is None:
                # End of data stream. Push remaining bits.
                if len(bit_buffer) > 0:
                    padding = self.bits_per_batch - len(bit_buffer)
                    padded_data = np.pad(bit_buffer, (0, padding), 'constant')
                    self.data_queue.put(torch.from_numpy(padded_data).to(torch.uint8))
                
                # Signal completion to main thread
                self.raw_queue.put(None) 
                return

            # Heavy CPU operation
            np_bits = np.unpackbits(np.frombuffer(raw_chunk, dtype=np.uint8))
            bit_buffer = np.concatenate((bit_buffer, np_bits))

    def run(self):
        logging.info(f"DataProducerThread started with {self.num_workers} workers.")
        
        # Start File Reader
        reader_thread = threading.Thread(target=self.file_reader, daemon=True)
        reader_thread.start()

        self.bit_packer_worker()
        
        self.data_queue.put(None) # Signal main loop
        logging.info("DataProducerThread finished.")

    def stop(self):
        self.stop_event.set()

class FrameProducerThread(threading.Thread):
    """Fetches frames from FFmpeg in a background thread."""
    def __init__(self, video_path: Path, start_frame: int, total_frames: int, batch_size: int, config: Dict, queue: queue.Queue):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.current_frame = start_frame
        self.total_frames = total_frames
        self.batch_size = batch_size
        self.config = config
        self.out_queue = queue
        self.stop_event = threading.Event()

    def run(self):
        logging.info("FrameProducerThread started.")
        while self.total_frames > 0 and not self.stop_event.is_set():
            count = min(self.batch_size, self.total_frames)
            
            # This is the blocking call we moved out of the main loop
            tensor = extract_frames_batch(
                self.video_path, 
                self.current_frame, 
                count, 
                self.config, 
                frame_type='data'
            )
            
            if tensor is None:
                logging.error("FrameProducerThread failed to extract batch.")
                break
                
            self.out_queue.put(tensor)
            self.current_frame += count
            self.total_frames -= count
        
        self.out_queue.put(None)
        logging.info("FrameProducerThread finished.")

    def stop(self):
        self.stop_event.set()

class FFmpegConsumerThread(threading.Thread):
    def __init__(self, frame_queue: queue.Queue, output_base_path: Path, config: Dict):
        super().__init__(daemon=True)
        self.frame_queue, self.config = frame_queue, config
        self.stop_event = threading.Event()
        self.sub_batch_size = 32
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.frames_written_in_segment = 0
        self.output_paths: List[Path] = []
        self.output_parent = output_base_path.parent
        self.output_stem = output_base_path.stem if output_base_path.suffix else output_base_path.name

        width = self.config["VIDEO_WIDTH"]
        height = self.config["VIDEO_HEIGHT"]
        crf = self.config["X264_CRF"]
        fps = self.config["VIDEO_FPS"]
        keyint = self.config.get("KEYINT_MAX", 1)
        
        # Explicitly enforce output resolution via filters.
        # libx264 often enforces mod-16 dimensions by default if not told otherwise.
        # We also enforce yuv420p for standard H.264 compatibility.
        
        # GPU Encoding Switch
        if self.config.get("ENABLE_NVENC", False):
            # GPU Settings (h264_nvenc)
            logging.info(f"Using GPU Encoding (h264_nvenc) with CRF/CQ={crf}")
            codec_args = [
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',      # Fastest preset
                '-rc', 'vbr',         # Variable Bitrate to allow quality focus
                '-cq', str(crf),      # Constant Quality
                '-spatial-aq', '1',   # Help retain spatial details (edges)
                '-temporal-aq', '1'   
            ]
        else:
            # CPU Settings (libx264)
            logging.info(f"Using CPU Encoding (libx264) with CRF={crf}")
            codec_args = [
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'stillimage',
                '-crf', str(crf),
            ]

        self.ffmpeg_command_base = [
            self.config["FFMPEG_PATH"], '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-',
            *codec_args,
            '-g', str(keyint),
            '-keyint_min', str(keyint),
            '-sc_threshold', '0',
            '-vf', f'scale={width}:{height}', 
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart'
        ]

        max_hours = self.config.get("MAX_VIDEO_SEGMENT_HOURS", 11)
        if max_hours and max_hours > 0:
            self.max_frames_per_segment = int(max_hours * 3600 * fps)
            logging.info(f"FFmpeg consumer will rotate files every {self.max_frames_per_segment} frames (~{max_hours}h @ {fps}fps)")
        else:
            self.max_frames_per_segment = None

        override_frames = self.config.get("MAX_VIDEO_SEGMENT_FRAMES_OVERRIDE")
        if override_frames and override_frames > 0:
            override_frames = int(override_frames)
            if self.max_frames_per_segment is None:
                self.max_frames_per_segment = override_frames
            else:
                self.max_frames_per_segment = min(self.max_frames_per_segment, override_frames)
            logging.info(f"Segment frame override active: {self.max_frames_per_segment} frame(s) per MP4 segment.")

    def _build_output_path(self, segment_index: int) -> Path:
        if self.max_frames_per_segment is None:
            return self.output_parent / f"{self.output_stem}.mp4"
        return self.output_parent / f"{self.output_stem}_part{segment_index:03d}.mp4"

    def _start_new_segment(self):
        segment_index = len(self.output_paths) + 1
        output_path = self._build_output_path(segment_index)
        command = self.ffmpeg_command_base + [str(output_path)]
        logging.info(f"Starting FFmpeg segment #{segment_index}: {output_path}")
        self.ffmpeg_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.output_paths.append(output_path)
        self.frames_written_in_segment = 0

    def _finalize_current_segment(self):
        if self.ffmpeg_process and self.ffmpeg_process.stdin:
            try:
                self.ffmpeg_process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
        if self.ffmpeg_process:
            _, stderr = self.ffmpeg_process.communicate()
            if self.ffmpeg_process.returncode != 0 and not self.stop_event.is_set():
                logging.error(
                    f"FFmpeg process exited with error code {self.ffmpeg_process.returncode}:\n{stderr.decode('utf-8', 'ignore')}"
                )
        self.ffmpeg_process = None
        self.frames_written_in_segment = 0

    def _write_frame(self, frame_np: np.ndarray):
        if self.ffmpeg_process is None:
            self._start_new_segment()
        if self.ffmpeg_process is None or self.ffmpeg_process.stdin is None:
            return
        self.ffmpeg_process.stdin.write(frame_np.tobytes())
        self.frames_written_in_segment += 1
        if self.max_frames_per_segment and self.frames_written_in_segment >= self.max_frames_per_segment:
            logging.info("Segment reached maximum duration; rotating to a new MP4 file.")
            self._finalize_current_segment()

    def run(self):
        logging.info("FFmpegConsumerThread started.")
        width = self.config["VIDEO_WIDTH"]
        height = self.config["VIDEO_HEIGHT"]
        try:
            while not self.stop_event.is_set():
                frame_batch = self.frame_queue.get()
                if frame_batch is None:
                    break
                for sub_batch in torch.split(frame_batch, self.sub_batch_size):
                    if sub_batch.shape[1] != height or sub_batch.shape[2] != width:
                         sub_batch_resized = torch.nn.functional.interpolate(
                             sub_batch.permute(0, 3, 1, 2).float(), size=(height, width), mode='nearest-exact'
                         ).permute(0, 2, 3, 1).byte()
                    else:
                        sub_batch_resized = sub_batch
                    np_batch = sub_batch_resized.contiguous().cpu().numpy()
                    for frame_np in np_batch:
                        if self.stop_event.is_set():
                            break
                        self._write_frame(frame_np)
        except (BrokenPipeError, OSError):
            logging.warning("FFmpeg pipe closed, likely due to an early exit or shutdown.")
        except Exception as e:
            logging.error(f"Error in FFmpegConsumerThread: {e}", exc_info=True)
        finally:
            self._finalize_current_segment()
            logging.info("FFmpegConsumerThread finished.")

    def stop(self):
        self.stop_event.set()
        if self.ffmpeg_process:
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ffmpeg_process.kill()
        self._finalize_current_segment()

def encode_orchestrator(input_path: Path, output_dir: Path, password: Optional[str], config: Dict, device: torch.device):
    encode_start = time.perf_counter()
    logging.info(f"Starting encoding for '{input_path}'...")
    temp_dir = output_dir / TEMP_ENCODE_DIR; temp_dir.mkdir(exist_ok=True)
    produced_files: List[Path] = []
    files_to_encode: List[Path] = []
    try:
        try:
            derived_params = get_derived_encoding_params(config, device)
        except ValueError as e:
            logging.error(f"Configuration Error: {e}")
            return
        prep_result = prepare_files_for_encoding(input_path, temp_dir, config, password)
        if prep_result is None:
            return
        files_to_encode, file_manifest = prep_result
        config["is_password_protected"] = bool(password)
        info_frames_batch = generate_info_artifacts(file_manifest, config, device, derived_params)
        if info_frames_batch is None:
            return
        barcode_frame_batch = generate_barcode_frame(info_frames_batch.shape[0], config, device)
        if barcode_frame_batch is None:
            return
        output_base_path = output_dir / f"{input_path.stem}_F2YT"
        queue_depth = max(4, int(config.get("PIPELINE_QUEUE_DEPTH", 16)))
        data_queue, frame_queue = queue.Queue(maxsize=queue_depth), queue.Queue(maxsize=queue_depth)
        
        # Revised Producer (Decoupled IO)
        producer = DataProducerThread(files_to_encode, data_queue, derived_params)
        consumer = FFmpegConsumerThread(frame_queue, output_base_path, config)
        producer.start(); consumer.start()
        
        use_cuda = device.type == "cuda"
        overlap_streams = max(1, int(config.get("GPU_OVERLAP_STREAMS", 2))) if use_cuda else 1
        encode_streams = [torch.cuda.Stream(device=device) for _ in range(overlap_streams)] if use_cuda else []
        if use_cuda:
            logging.info(f"Using {overlap_streams} CUDA stream(s) for overlapped encode batches.")
        inflight_batches: Deque[Tuple[Optional[torch.cuda.Stream], torch.Tensor]] = deque()
        max_inflight = len(encode_streams) if encode_streams else 1
        next_stream_idx = 0

        def flush_encoded_batches(force: bool = False) -> None:
            while inflight_batches and (force or len(inflight_batches) >= max_inflight):
                stream, cpu_tensor = inflight_batches.popleft()
                if stream is not None:
                    stream.synchronize()
                if cpu_tensor is None or cpu_tensor.numel() == 0:
                    continue
                ready_tensor = cpu_tensor.contiguous()
                # Pushing to queue might block if FFmpeg is slow; that's unavoidable if disk IO is slow
                frame_queue.put(ready_tensor)

        try:
            logging.info("--- Starting main processing pipeline ---")
            frame_queue.put(barcode_frame_batch)
            frame_queue.put(info_frames_batch)
            while True:
                # This should no longer spike the CPU or wait on disk
                bits_tensor_cpu = data_queue.get()
                if bits_tensor_cpu is None:
                    break
                if use_cuda:
                    pinned_bits = pin_tensor_if_possible(bits_tensor_cpu)
                    stream = encode_streams[next_stream_idx]
                    next_stream_idx = (next_stream_idx + 1) % len(encode_streams)
                    with torch.cuda.stream(stream):
                        bits_tensor_gpu = pinned_bits.to(device, non_blocking=True)
                        pixel_tensor_gpu = encode_data_frames_gpu(bits_tensor_gpu, derived_params)
                        if pixel_tensor_gpu.numel() == 0:
                            continue
                        pixel_tensor_cpu = pixel_tensor_gpu.to("cpu", non_blocking=True)
                    inflight_batches.append((stream, pixel_tensor_cpu))
                else:
                    pixel_tensor_cpu = encode_data_frames_gpu(bits_tensor_cpu.to(device), derived_params)
                    if pixel_tensor_cpu.numel() == 0:
                        continue
                    inflight_batches.append((None, pixel_tensor_cpu.cpu()))
                flush_encoded_batches()
            flush_encoded_batches(force=True)
        except KeyboardInterrupt:
            logging.warning("Keyboard interrupt detected. Shutting down pipeline.")
            producer.stop(); consumer.stop()
        except Exception as e:
            logging.error(f"Error in main processing loop: {e}", exc_info=True)
            producer.stop(); consumer.stop()
        finally:
            frame_queue.put(None)
            logging.info("Waiting for producer thread to finish...")
            producer.join()
            logging.info("Producer thread finished. Waiting for FFmpeg consumer to flush to disk...")
            consumer.join()
            logging.info("FFmpeg consumer confirmed all segments written.")
            if consumer.output_paths:
                produced_files = consumer.output_paths.copy()
            else:
                default_name = output_base_path if output_base_path.suffix else output_base_path.parent / f"{output_base_path.name}.mp4"
                produced_files = [default_name]

            segmentation_enabled = config.get("MAX_VIDEO_SEGMENT_HOURS", 11)
            if segmentation_enabled and len(produced_files) == 1 and produced_files[0].name.endswith("_part001.mp4"):
                legacy_name = output_base_path if output_base_path.suffix else output_base_path.parent / f"{output_base_path.name}.mp4"
                current_part = produced_files[0]
                try:
                    current_part.rename(legacy_name)
                    logging.info(f"Renamed single segment {current_part.name} -> {legacy_name.name} for backwards compatibility.")
                    produced_files = [legacy_name]
                except OSError as e:
                    logging.warning(f"Could not rename {current_part} to {legacy_name}: {e}")

            normalized_files = []
            for video_file in produced_files:
                if video_file.suffix.lower() == ".mp4":
                    normalized_files.append(video_file)
                    continue
                target = video_file.with_suffix(".mp4")
                if target.exists():
                    target = video_file.with_name(video_file.name + ".mp4")
                try:
                    video_file.rename(target)
                    logging.info(f"Normalized output filename '{video_file.name}' -> '{target.name}' (missing .mp4 extension).")
                    normalized_files.append(target)
                except OSError as e:
                    logging.warning(f"Failed to append .mp4 extension to {video_file}: {e}")
                    normalized_files.append(video_file)

            produced_files = normalized_files

            for video_file in produced_files:
                logging.info(f"Encoding pipeline finalized segment: {video_file}")
            logging.info("Encoding pipeline finished.")
        total_payload_size = sum(f.stat().st_size for f in files_to_encode)
        logging.info("--- ENCODING SUMMARY (Overall) ---")
        logging.info(f"Original Input: {input_path}")
        logging.info(f"Total Payload Size (after 7z/par2): {total_payload_size / 1024:.2f} KB")
        logging.info("Output Video Files:")
        
        total_video_size = 0
        for video_file in produced_files:
            if video_file.exists():
                video_size = video_file.stat().st_size
                total_video_size += video_size
                logging.info(f"  - {video_file} ({video_size / (1024*1024):.2f} MB)")
            else:
                logging.info(f"  - {video_file} (missing on disk)")
        
        if total_payload_size > 0:
             ratio = total_video_size / total_payload_size
             logging.info(f"Total Video Size: {total_video_size / (1024*1024):.2f} MB")
             logging.info(f"Overall Storage Ratio: {ratio:.2f}x (Video is {ratio:.2f} times larger than payload)")

        encode_elapsed = time.perf_counter() - encode_start
        logging.info(f"Encoding completed successfully in {encode_elapsed:.1f}s ({encode_elapsed/60:.2f} min).")
    finally:
        cleanup_temp_dir(temp_dir, "encoding temp")

# --- Decoding Pipeline ---

class DataWriterThread(threading.Thread):
    def __init__(self, data_queue: queue.Queue, manifest: Dict, output_dir: Path):
        super().__init__(daemon=True)
        self.data_queue, self.manifest, self.output_dir = data_queue, manifest, output_dir
        self.stop_event = threading.Event()
    def run(self):
        logging.info("DataWriterThread started.")
        try:
            file_handles, current_file_idx, bytes_written_to_current_file = {}, 0, 0
            manifest_files = self.manifest['file_manifest_detailed']
            # Optimization: Use a larger buffer size for file writing (1MB)
            write_buffer_size = 1024 * 1024
            
            while not self.stop_event.is_set():
                data_bytes = self.data_queue.get()
                if data_bytes is None: break
                offset = 0
                while offset < len(data_bytes):
                    if current_file_idx >= len(manifest_files): break
                    target_file = manifest_files[current_file_idx]
                    file_path = self.output_dir / target_file['name']
                    if file_path not in file_handles:
                        file_path.parent.mkdir(parents=True, exist_ok=True)
                        file_handles[file_path] = open(file_path, 'wb', buffering=write_buffer_size)
                    bytes_remaining = target_file['size'] - bytes_written_to_current_file
                    chunk_size = min(bytes_remaining, len(data_bytes) - offset)
                    if chunk_size <= 0:
                        break
                    file_handles[file_path].write(data_bytes[offset : offset + chunk_size])
                    bytes_written_to_current_file += chunk_size
                    offset += chunk_size
                    if bytes_written_to_current_file >= target_file['size']:
                        logging.info(f"Finished writing {file_path.name}")
                        file_handles[file_path].flush()
                        os.fsync(file_handles[file_path].fileno())
                        file_handles[file_path].close()
                        del file_handles[file_path]
                        current_file_idx += 1
                        bytes_written_to_current_file = 0
        except Exception as e:
            logging.error(f"Error in DataWriterThread: {e}", exc_info=True)
        finally:
            for handle in file_handles.values():
                if not handle.closed:
                    try:
                        handle.flush()
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
                    handle.close()
            logging.info("DataWriterThread finished.")
    def stop(self): self.stop_event.set()

def extract_frame_as_tensor(video_path: Path, frame_index: int, temp_dir: Path, config: Dict, frame_type: str = 'data') -> Optional[torch.Tensor]:
    """Extract a specific frame from video using frame-accurate seeking.
    
    frame_type: 'barcode' (720x720), 'info' (16x16), or 'data' (180x180)
    Extracts at the appropriate resolution for each frame type to preserve color precision.
    """
    ffmpeg_path = config["FFMPEG_PATH"]
    fps = config.get("VIDEO_FPS", 60)
    
    # Determine extraction resolution based on frame type
    if frame_type == 'barcode':
        extract_size = 720  # Barcode needs full width
    elif frame_type == 'info':
        extract_size = INFO_K_SIDE  # 16x16 for info frames
    else:  # 'data'
        extract_size = config.get("DATA_K_SIDE", 180)
    
    # Calculate timestamp from frame index
    timestamp = frame_index / fps
    
    # Extract at appropriate size to avoid upscaling/downscaling precision loss
    command = [
        ffmpeg_path, '-hide_banner', '-loglevel', 'error',
        '-accurate_seek',
        '-seek_timestamp', '1',
        '-ss', f'{timestamp:.6f}',
        '-i', str(video_path),
        '-vframes', '1',
        '-vf', f'scale={extract_size}:{extract_size}:force_original_aspect_ratio=decrease,pad={extract_size}:{extract_size}:(ow-iw)/2:(oh-ih)/2',
        '-f', 'image2pipe', '-vcodec', 'png', '-'
    ]
    try:
        proc = subprocess.run(command, capture_output=True, check=True, timeout=10)
        if proc.stdout is None or len(proc.stdout) == 0:
            logging.error(f"FFmpeg extracted empty stdout for frame {frame_index}.")
            return None
        img_bytes = proc.stdout
        with Image.open(io.BytesIO(img_bytes)) as img:
            img_rgb = img.convert('RGB')
            np_frame = np.array(img_rgb)
            if np_frame.shape[:2] != (extract_size, extract_size):
                np_frame = np.array(Image.fromarray(np_frame).resize((extract_size, extract_size), Image.Resampling.NEAREST))
        return torch.from_numpy(np_frame)
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to extract frame {frame_index}. FFmpeg stderr:\n{e.stderr.decode('utf-8', 'ignore') if e.stderr else 'no stderr'}")
        return None
    except subprocess.TimeoutExpired:
        logging.error(f"FFmpeg frame extraction timed out for frame {frame_index}.")
        return None
    except Exception as e:
        logging.error(f"An error occurred loading extracted frame {frame_index}: {e}")
        return None

def extract_frames_batch(video_path: Path, start_frame_index: int, frame_count: int, config: Dict, frame_type: str = 'data') -> Optional[torch.Tensor]:
    """Extract a contiguous batch of frames of the requested type in one ffmpeg invocation."""
    if frame_count <= 0:
        return None

    ffmpeg_path = config["FFMPEG_PATH"]
    fps = config.get("VIDEO_FPS", 60)
    if frame_type == 'info':
        extract_size = INFO_K_SIDE
    elif frame_type == 'barcode':
        extract_size = max(config.get("VIDEO_WIDTH", 720), config.get("VIDEO_HEIGHT", 720))
    else:
        extract_size = config.get("DATA_K_SIDE", 180)

    timestamp = start_frame_index / fps
    scale_filter = (
        f"scale={extract_size}:{extract_size}:flags=neighbor:force_original_aspect_ratio=decrease,"
        f"pad={extract_size}:{extract_size}:(ow-iw)/2:(oh-ih)/2"
    )
    command = [
        ffmpeg_path,
        '-hide_banner', '-loglevel', 'error',
        '-accurate_seek', '-seek_timestamp', '1',
        '-ss', f'{timestamp:.6f}',
        '-i', str(video_path),
        '-vframes', str(frame_count),
        '-vf', scale_filter,
        '-pix_fmt', 'rgb24',
        '-vsync', '0',
        '-f', 'rawvideo',
        '-'
    ]

    try:
        proc = subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        logging.error(
            "Batched ffmpeg extraction failed for frames %d-%d (%s). stderr=%s",
            start_frame_index,
            start_frame_index + frame_count - 1,
            frame_type,
            e.stderr.decode('utf-8', 'ignore') if e.stderr else 'no stderr'
        )
        return None
    except Exception as e:
        logging.error(f"Unexpected error while extracting frame batch starting at {start_frame_index} ({frame_type}): {e}")
        return None

    raw = proc.stdout
    if not raw:
        logging.error("FFmpeg returned no pixel data for frames %d-%d (%s)", start_frame_index, start_frame_index + frame_count - 1, frame_type)
        return None

    bytes_per_frame = extract_size * extract_size * 3
    total_bytes = len(raw)
    actual_frames = total_bytes // bytes_per_frame
    if actual_frames == 0:
        logging.error("FFmpeg output (%d bytes) too small for even a single %dx%dx4 frame (%s)", total_bytes, extract_size, extract_size, frame_type)
        return None
    if actual_frames < frame_count:
        logging.warning(
            "Requested %d %s frame(s) starting at %d but only received %d. Will pad the remainder with zeros.",
            frame_count,
            frame_type,
            start_frame_index,
            actual_frames
        )

    usable_bytes = actual_frames * bytes_per_frame
    np_buffer = np.frombuffer(raw[:usable_bytes], dtype=np.uint8).copy()
    frame_tensor = torch.from_numpy(np_buffer).view(actual_frames, extract_size, extract_size, 3).to(torch.uint8)
    return frame_tensor


class SegmentedDataFrameReader:
    """Iterates through multiple MP4 segments as if they were a single continuous stream."""

    def __init__(self, video_paths: List[Path], config: Dict, temp_dir: Path, initial_frame_offset: int, frame_type: str = 'data'):
        self.video_paths = video_paths
        self.config = config
        self.temp_dir = temp_dir
        self.initial_frame_offset = max(0, initial_frame_offset)
        self.current_video_idx = 0
        self.local_frame_idx = self.initial_frame_offset
        self.offset_consumed = self.initial_frame_offset == 0
        self.frame_type = frame_type

    def _advance_video(self):
        self.current_video_idx += 1
        self.local_frame_idx = 0
        self.offset_consumed = True

    def _current_video(self) -> Optional[Path]:
        if self.current_video_idx >= len(self.video_paths):
            return None
        return self.video_paths[self.current_video_idx]

    def _fallback_extract_frames(self, video_path: Path, start_index: int, frame_count: int) -> Optional[torch.Tensor]:
        frames: List[torch.Tensor] = []
        for offset in range(frame_count):
            actual_frame = start_index + offset
            frame = extract_frame_as_tensor(video_path, actual_frame, self.temp_dir, self.config, frame_type=self.frame_type)
            if frame is None:
                break
            frames.append(frame)
        if not frames:
            return None
        return torch.stack(frames)

    def fetch_frames(self, desired_count: int) -> Optional[torch.Tensor]:
        if desired_count <= 0:
            return torch.empty((0, self.config['DATA_K_SIDE'], self.config['DATA_K_SIDE'], PIXEL_CHANNELS), dtype=torch.uint8)

        collected = []
        remaining = desired_count

        while remaining > 0:
            video_path = self._current_video()
            if video_path is None:
                break

            if not self.offset_consumed:
                self.local_frame_idx = self.initial_frame_offset
                self.offset_consumed = True

            batch_tensor = None
            if self.frame_type == 'data':
                batch_tensor = extract_frames_batch(video_path, self.local_frame_idx, remaining, self.config, frame_type='data')
            elif self.frame_type == 'info':
                batch_tensor = extract_frames_batch(video_path, self.local_frame_idx, remaining, self.config, frame_type='info')
            elif self.frame_type == 'barcode':
                batch_tensor = extract_frames_batch(video_path, self.local_frame_idx, remaining, self.config, frame_type='barcode')
            if batch_tensor is None or batch_tensor.shape[0] == 0:
                logging.warning(f"Failed batched extraction for {video_path} at frame {self.local_frame_idx}. Falling back to single-frame extraction.")
                batch_tensor = self._fallback_extract_frames(video_path, self.local_frame_idx, remaining)

            if batch_tensor is None or batch_tensor.shape[0] == 0:
                logging.warning(f"No more frames available in {video_path}. Moving to next segment.")
                self._advance_video()
                continue

            collected.append(batch_tensor)
            actual = batch_tensor.shape[0]
            remaining -= actual
            self.local_frame_idx += actual

            if remaining > 0:
                self._advance_video()

        if not collected:
            return None

        return torch.cat(collected, dim=0)

def decode_barcode(frame_tensor: torch.Tensor) -> Optional[Tuple[int, int]]:
    h, w, _ = frame_tensor.shape
    num_bars, bar_width, bits = BARCODE_NUM_BITS, w / float(BARCODE_NUM_BITS), []
    for i in range(num_bars):
        center_x = int((i + 0.5) * bar_width)
        center_x = min(max(center_x, 0), w - 1)
        avg_brightness = float(frame_tensor[:, center_x, 0].float().mean())
        bits.append('1' if avg_brightness > 127 else '0')
    try:
        combined_value = int("".join(bits), 2)
    except ValueError:
        logging.error("Failed to parse barcode bits into integer.")
        return None
    num_info_frames, encoder_fps = combined_value & 0xFFFF, (combined_value >> 16) & 0xFF
    if encoder_fps == 0: logging.error("Decoded FPS from barcode is 0, which is invalid."); return None
    logging.info(f"Decoded barcode: {num_info_frames} info frames, {encoder_fps} FPS.")
    return num_info_frames, encoder_fps

def bits_to_bytes(bits: torch.Tensor) -> bytes:
    """Pack a 1-D uint8 bit tensor (MSB-first per byte) into bytes.
    Matches the order used by np.unpackbits (big-endian within each byte).
    """
    bits_np = bits.cpu().numpy().astype(np.uint8)
    rem = bits_np.size % 8
    if rem != 0:
        bits_np = np.concatenate((bits_np, np.zeros(8 - rem, dtype=np.uint8)))
    packed = np.packbits(bits_np)
    return bytes(packed.tolist())

def decode_info_frames(frames, device: torch.device, syndrome_table: Optional[torch.Tensor] = None) -> Optional[Dict]:
    """Decode info frames (Hamming(7,4) encoded JSON) back into a Python dict.

    Accepts either a torch.Tensor of shape (num_frames, H, W, 3) or a list of such tensors.
    Automatically downscales frames to INFO_K_SIDE x INFO_K_SIDE before bit extraction.
    If a syndrome_table is provided it will be used; otherwise it is constructed from the
    built-in INFO_H_MATRIX_TENSOR.
    Returns the decoded JSON object or None on failure.
    """
    if frames is None:
        logging.error("No frames provided to decode_info_frames.")
        return None

    # Normalize to a single tensor on the target device
    if isinstance(frames, list):
        if len(frames) == 0:
            logging.error("Empty frame list passed to decode_info_frames.")
            return None
        try:
            frames = torch.stack([f.to(device) if isinstance(f, torch.Tensor) else torch.from_numpy(np.array(f)).to(device) for f in frames])
        except Exception:
            # Fallback: try to coerce each element
            frames = torch.stack([torch.from_numpy(np.array(f)).to(device) if not isinstance(f, torch.Tensor) else f.to(device) for f in frames])
    else:
        frames = frames.to(device)

    if frames.dim() == 3:
        frames = frames.unsqueeze(0)

    # Downscale frames to INFO_K_SIDE x INFO_K_SIDE to match encoding resolution
    num_frames, h, w, c = frames.shape
    if h != INFO_K_SIDE or w != INFO_K_SIDE:
        frames = torch.nn.functional.interpolate(
            frames.permute(0, 3, 1, 2).float(), 
            size=(INFO_K_SIDE, INFO_K_SIDE), 
            mode='nearest-exact'
        ).permute(0, 2, 3, 1).byte()

    # Convert frames back to bit stream using the same palette/packing used for encoding
    bits = frame_to_bits_batch(frames, INFO_COLOR_PALETTE_TENSOR.to(device), INFO_BITS_PER_PALETTE_COLOR)

    # Use provided syndrome table or build one
    if syndrome_table is None:
        h_matrix = INFO_H_MATRIX_TENSOR.to(device)
        syndrome_table = build_syndrome_lookup_table(h_matrix)
    else:
        h_matrix = INFO_H_MATRIX_TENSOR.to(device)

    n = INFO_HAMMING_N
    k = INFO_HAMMING_K

    # Pad to full codewords if necessary
    rem = bits.numel() % n
    if rem != 0:
        pad_n = n - rem
        bits = torch.cat((bits, torch.zeros(pad_n, dtype=torch.uint8, device=device)))

    # Decode using existing Hamming decoder
    # hamming_decode_gpu expects a flat bit tensor and will reshape internally to codewords
    decoded_bits, num_errors = hamming_decode_gpu(bits, h_matrix, k, syndrome_table)
    logging.info(f"Info frames decoded with {num_errors} corrected Hamming codewords.")

    # Convert decoded bits back to bytes and extract JSON
    try:
        payload_bytes = bits_to_bytes(decoded_bits)
    except Exception as e:
        logging.error(f"Failed to pack bits into bytes: {e}")
        return None

    if len(payload_bytes) < 4:
        logging.error("Decoded payload too small to contain length header.")
        return None

    length = int.from_bytes(payload_bytes[:4], 'big')
    if len(payload_bytes) < 4 + length:
        logging.error("Decoded payload length mismatch or incomplete frames.")
        return None

    json_bytes = payload_bytes[4:4+length]
    try:
        decoded_obj = json.loads(json_bytes.decode('utf-8'))
    except Exception as e:
        logging.error(f"Failed to parse decoded JSON: {e}")
        return None

    return decoded_obj

def decode_data_frames_gpu(
    frame_batch: torch.Tensor,
    derived_params: Dict,
    total_encoded_bits: Optional[int] = None,
    return_error_tensor: bool = False
) -> Tuple[torch.Tensor, Union[int, torch.Tensor]]:
    device = frame_batch.device
    palette = derived_params['data_palette'].to(device)
    bits_per_color = derived_params['bits_per_pixel_data']
    h_matrix = derived_params['h_matrix'].to(device)
    hamming_k = derived_params['hamming_k']
    syndrome_table = derived_params['syndrome_table']
    total_encoded_bits_per_frame = derived_params['total_encoded_bits_to_store_data']
    k_side = derived_params['DATA_K_SIDE']
    
    # Frames are already extracted at DATA_K_SIDE x DATA_K_SIDE resolution (no downscaling needed)
    # Verify size and resize if necessary (should be rare)
    if frame_batch.shape[1:3] != (k_side, k_side):
        logging.warning(f"Frame batch size {frame_batch.shape[1:3]} != expected {k_side}x{k_side}, resizing...")
        frame_batch = torch.nn.functional.interpolate(
            frame_batch.permute(0, 3, 1, 2).float(), size=(k_side, k_side), mode='nearest'
        ).permute(0, 2, 3, 1).byte()
    
    # Extract bits using the same palette as encoding
    # This extracts ALL bits from the frame (k_side^2 * bits_per_color), including padding
    coded_bits = frame_to_bits_batch(frame_batch, palette, bits_per_color)
    
    # Log frame bits for diagnostics
    num_frames = frame_batch.shape[0]
    max_bits_per_frame = k_side * k_side * bits_per_color  # 180*180*2 = 64,800 bits
    expected_bits = num_frames * total_encoded_bits_per_frame if total_encoded_bits is None else total_encoded_bits
    actual_bits = coded_bits.numel()
    bits_per_frame_extracted = actual_bits // num_frames if num_frames > 0 else 0
    
    logging.info(f"Data decode: extracted {actual_bits} bits from {num_frames} frames")
    logging.info(f"  Expected: {expected_bits} bits ({total_encoded_bits_per_frame}/frame nominal, {total_encoded_bits} actual)")
    logging.info(f"  Actual per frame extracted: {bits_per_frame_extracted} bits")
    logging.info(f"  Padding per frame: {bits_per_frame_extracted - total_encoded_bits_per_frame} bits")
    
    # THE FIX: Remove padding from each frame individually, but respect actual total
    # Each frame has max_bits_per_frame bits extracted, but only total_encoded_bits_per_frame are valid per frame
    if total_encoded_bits is not None and total_encoded_bits < expected_bits:
        # Last frame(s) have extra padding beyond the actual payload
        valid_bits = coded_bits[:total_encoded_bits]
        discarded = actual_bits - total_encoded_bits
        logging.info(f"  Discarding {discarded} extra padding bits ({discarded / num_frames:.1f} per frame on average)")
    else:
        # Standard case: each frame has padding_per_frame bits of padding
        valid_bits_list = []
        for i in range(num_frames):
            start_idx = i * bits_per_frame_extracted
            end_idx = start_idx + total_encoded_bits_per_frame
            valid_bits_list.append(coded_bits[start_idx:end_idx])
        valid_bits = torch.cat(valid_bits_list)
        
        # Log the per-frame padding removal
        current_valid_bits = valid_bits.numel()
        if actual_bits != current_valid_bits:
            discarded = actual_bits - current_valid_bits
            logging.info(f"  Discarding {discarded} per-frame padding bits ({discarded / num_frames:.1f} per frame)")
    
    if total_encoded_bits is not None and valid_bits.numel() > total_encoded_bits:
        delta = valid_bits.numel() - total_encoded_bits
        logging.info(f"  Trimming {delta} padding bits after per-frame alignment to match actual payload.")
        valid_bits = valid_bits[:total_encoded_bits]

    return hamming_decode_gpu(
        valid_bits,
        h_matrix,
        hamming_k,
        syndrome_table,
        return_error_tensor=return_error_tensor
    )

def decode_orchestrator(input_path_str: str, output_dir: Path, password: Optional[str], config: Dict, device: torch.device):
    decode_start = time.perf_counter()
    logging.info(f"Starting decoding for '{input_path_str}'...")
    temp_dir = output_dir / TEMP_DECODE_DIR; temp_dir.mkdir(exist_ok=True)
    data_writer: Optional[DataWriterThread] = None
    try:
        raw_inputs = [segment.strip() for segment in input_path_str.split(',') if segment.strip()]
        if not raw_inputs:
            logging.error("No input video paths provided for decoding.")
            return

        video_paths = []
        for segment in raw_inputs:
            candidate = Path(segment).expanduser()
            if not candidate.exists():
                logging.error(f"Input video not found: {candidate}")
                return
            video_paths.append(candidate)

        primary_video = video_paths[0]
        multi_segment = len(video_paths) > 1
        if multi_segment:
            segment_lines = "\n".join(f"  [{idx+1}] {path}" for idx, path in enumerate(video_paths))
            logging.info(f"Decoding will concatenate {len(video_paths)} segments:\n{segment_lines}")
        else:
            logging.info(f"Decoding single video: {primary_video}")

        # Extract barcode frame at full 720x720
        barcode_frame = extract_frame_as_tensor(primary_video, 0, temp_dir, config, frame_type='barcode')
        if barcode_frame is None:
            logging.error("Failed to extract barcode frame.")
            return
        barcode_data = decode_barcode(barcode_frame)
        if barcode_data is None:
            return
        num_info_frames, _ = barcode_data
        
        # Extract info frames at 16x16 (INFO_K_SIDE)
        if multi_segment:
            info_reader = SegmentedDataFrameReader(video_paths, config, temp_dir, initial_frame_offset=1, frame_type='info')
            info_frames_tensor = info_reader.fetch_frames(num_info_frames)
            if info_frames_tensor is None or info_frames_tensor.shape[0] < num_info_frames:
                logging.error("Failed to extract required info frames across provided segments.")
                return
            info_frames = info_frames_tensor[:num_info_frames]
            logging.info(f"Read {info_frames.shape[0]} info frames from barcode ({num_info_frames} reported). Decoding them...")
        else:
            info_frames = []
            for i in range(num_info_frames):
                frame = extract_frame_as_tensor(primary_video, i + 1, temp_dir, config, frame_type='info')
                if frame is None:
                    logging.error(f"Failed to extract info frame {i+1}. Aborting.")
                    return
                info_frames.append(frame)
            info_frames = torch.stack(info_frames)
            logging.info(f"Read {info_frames.shape[0]} info frames from barcode ({num_info_frames} reported). Decoding them...")
        
        info_syndrome_table = build_syndrome_lookup_table(INFO_H_MATRIX_TENSOR.to(device))
        session_params = decode_info_frames(info_frames, device, info_syndrome_table)
        if session_params is None:
            logging.error("Failed to recover session parameters.")
            return
        logging.info("Successfully decoded info JSON.")
        
        # DIAGNOSTIC: Log what we decoded
        logging.info(f"Decoded manifest fields: info_frame_count={session_params.get('info_frame_count')}, data_frame_count={session_params.get('data_frame_count')}")
        
        # DIAGNOSTIC: Check if decoded info_frame_count matches barcode
        decoded_info_frame_count = session_params.get("info_frame_count", num_info_frames)
        if decoded_info_frame_count != num_info_frames:
            logging.warning(f"⚠ Info frame count mismatch: barcode says {num_info_frames}, but JSON says {decoded_info_frame_count}")
            logging.warning(f"   This may indicate an encoding issue. Using JSON value: {decoded_info_frame_count}")
            num_info_frames = decoded_info_frame_count

        apply_snapshot_to_config(config, session_params.get("encode_config_snapshot"))

        data_output_dir = temp_dir / ASSEMBLED_FILES_DIR; data_output_dir.mkdir(exist_ok=True)
        data_queue = queue.Queue(maxsize=128)
        data_writer = DataWriterThread(data_queue, session_params, data_output_dir)
        data_writer.start()

        derived_params = get_derived_encoding_params(config, device)
        derived_params['h_matrix'] = DATA_H_MATRIX_TENSOR.to(device)
        derived_params['syndrome_table'] = build_syndrome_lookup_table(derived_params['h_matrix'])

        total_payload_bytes = sum(f['size'] for f in session_params['file_manifest_detailed'])
        total_payload_bits = total_payload_bytes * 8
        total_encoded_bits_budget = session_params.get("total_encoded_data_bits")
        if total_encoded_bits_budget is None:
            hamming_k = derived_params['hamming_k']
            hamming_n = derived_params['hamming_n']
            if hamming_k > 0:
                total_encoded_bits_budget = math.ceil(total_payload_bits / hamming_k) * hamming_n
            else:
                total_encoded_bits_budget = 0
            logging.info(f"   Derived encoded bit budget from manifest: {total_encoded_bits_budget} bits")
        else:
            logging.info(f"   Manifest advertised encoded bit budget: {total_encoded_bits_budget} bits")

        # Calculate frame offset (data frames start after barcode + info frames)
        frame_idx_offset = 1 + num_info_frames
        
        if 'data_frame_count' in session_params:
            num_data_frames = session_params['data_frame_count']
            logging.info(f"[OK] Using data_frame_count from manifest: {num_data_frames} frames (padding frames will be ignored)")
        else:
            payload_bytes_per_frame = derived_params['payload_bits_data'] / 8
            total_payload_bytes = sum(f['size'] for f in session_params['file_manifest_detailed'])
            num_data_frames = math.ceil(total_payload_bytes / payload_bytes_per_frame) if payload_bytes_per_frame > 0 else 0
            logging.info(f"[WARN] Calculated data_frame_count (fallback): {num_data_frames} frames")
        
        logging.info(f"Data extraction will process frames {frame_idx_offset} to {frame_idx_offset + num_data_frames - 1} (total {num_data_frames} data frames)")

        frame_producer_queue = None
        if not multi_segment:
            frame_producer_queue = queue.Queue(maxsize=8)
            batch_size = config['GPU_PROCESSOR_BATCH_SIZE']
            producer_thread = FrameProducerThread(
                primary_video, 
                frame_idx_offset, 
                num_data_frames, 
                batch_size, 
                config, 
                frame_producer_queue
            )
            producer_thread.start()
        
        batch_size = config['GPU_PROCESSOR_BATCH_SIZE']
        total_corrected_data_codewords = 0
        use_cuda = device.type == "cuda"
        overlap_streams = max(1, int(config.get("GPU_OVERLAP_STREAMS", 2))) if use_cuda else 1
        decode_streams = [torch.cuda.Stream(device=device) for _ in range(overlap_streams)] if use_cuda else []
        if use_cuda:
            logging.info(f"Using {overlap_streams} CUDA stream(s) for overlapped decode batches.")
        inflight_decodes: Deque[Tuple[Optional[torch.cuda.Stream], torch.Tensor, Union[int, torch.Tensor]]] = deque()
        max_inflight_decodes = len(decode_streams) if decode_streams else 1
        next_decode_stream = 0

        def flush_decoded_batches(force: bool = False) -> None:
            nonlocal total_corrected_data_codewords
            while inflight_decodes and (force or len(inflight_decodes) >= max_inflight_decodes):
                stream, bits_cpu, corrections_value = inflight_decodes.popleft()
                if stream is not None:
                    stream.synchronize()
                if bits_cpu is None or bits_cpu.numel() == 0:
                    continue
                ready_bits = bits_cpu.contiguous()
                if isinstance(corrections_value, torch.Tensor):
                    corrections_int = int(corrections_value.item())
                else:
                    corrections_int = int(corrections_value)
                total_corrected_data_codewords += corrections_int
                data_queue.put(np.packbits(ready_bits.cpu().numpy()).tobytes())

        try:
            logging.info(f"DEBUG: Will extract data frames with indices {frame_idx_offset} to {frame_idx_offset + num_data_frames - 1}")
            
            # If using producer, loop slightly differently
            if frame_producer_queue:
                while True:
                    if data_writer is not None and not data_writer.is_alive():
                        logging.error("DataWriterThread died unexpectedly! Aborting decode.")
                        break
                    
                    batch_tensor_cpu = frame_producer_queue.get()
                    if batch_tensor_cpu is None:
                        break
                    
                    # Calculate bits budget for this specific batch size
                    bits_per_frame_encoded = derived_params['total_encoded_bits_to_store_data']
                    batch_frame_count = batch_tensor_cpu.shape[0]
                    batch_bits = batch_frame_count * bits_per_frame_encoded
                    bits_argument = None
                    if total_encoded_bits_budget is not None:
                        bits_argument = min(total_encoded_bits_budget, batch_bits)
                        total_encoded_bits_budget = max(0, total_encoded_bits_budget - bits_argument)

                    if use_cuda:
                        pinned_batch = pin_tensor_if_possible(batch_tensor_cpu)
                        stream = decode_streams[next_decode_stream]
                        next_decode_stream = (next_decode_stream + 1) % len(decode_streams)
                        with torch.cuda.stream(stream):
                            batch_tensor = pinned_batch.to(device, non_blocking=True)
                            decoded_bits, num_corrections = decode_data_frames_gpu(
                                batch_tensor,
                                derived_params,
                                bits_argument,
                                return_error_tensor=True
                            )
                            decoded_bits_cpu = decoded_bits.to("cpu", non_blocking=True)
                        inflight_decodes.append((stream, decoded_bits_cpu, num_corrections))
                    else:
                        decoded_bits, num_corrections = decode_data_frames_gpu(
                            batch_tensor_cpu.to(device),
                            derived_params,
                            bits_argument
                        )
                        inflight_decodes.append((None, decoded_bits.cpu(), num_corrections))

                    flush_decoded_batches()
            
            else:
                # Fallback to old loop for multi-segment (or if producer failed init)
                reader = SegmentedDataFrameReader(video_paths, config, temp_dir, frame_idx_offset, frame_type='data') if multi_segment else None
                for i in range(0, num_data_frames, batch_size):
                    # Safety check: if writer crashed, don't deadlock
                    if data_writer is not None and not data_writer.is_alive():
                        logging.error("DataWriterThread died unexpectedly! Aborting decode.")
                        break

                    start, end = i, min(i + batch_size, num_data_frames)
                    batch_count = end - start
                    
                    if reader is not None:
                        batch_tensor_cpu = reader.fetch_frames(batch_count)
                        if batch_tensor_cpu is None or batch_tensor_cpu.shape[0] == 0:
                            batch_tensor_cpu = torch.zeros((batch_count, config['DATA_K_SIDE'], config['DATA_K_SIDE'], PIXEL_CHANNELS), dtype=torch.uint8)
                        elif batch_tensor_cpu.shape[0] < batch_count:
                            deficit = batch_count - batch_tensor_cpu.shape[0]
                            padding = torch.zeros((deficit, config['DATA_K_SIDE'], config['DATA_K_SIDE'], PIXEL_CHANNELS), dtype=torch.uint8)
                            batch_tensor_cpu = torch.cat((batch_tensor_cpu, padding), dim=0)
                    else:
                        # Should not be reached if producer is active for single files
                        batch_tensor_cpu = extract_frames_batch(primary_video, frame_idx_offset + start, batch_count, config, frame_type='data')
                        if batch_tensor_cpu is None: break # Error

                    bits_per_frame_encoded = derived_params['total_encoded_bits_to_store_data']
                    batch_frame_count = batch_tensor_cpu.shape[0]
                    batch_bits = batch_frame_count * bits_per_frame_encoded
                    bits_argument = None
                    if total_encoded_bits_budget is not None:
                        bits_argument = min(total_encoded_bits_budget, batch_bits)
                        if bits_argument <= 0: break
                        total_encoded_bits_budget = max(0, total_encoded_bits_budget - bits_argument)

                    if use_cuda:
                        pinned_batch = pin_tensor_if_possible(batch_tensor_cpu)
                        stream = decode_streams[next_decode_stream]
                        next_decode_stream = (next_decode_stream + 1) % len(decode_streams)
                        with torch.cuda.stream(stream):
                            batch_tensor = pinned_batch.to(device, non_blocking=True)
                            decoded_bits, num_corrections = decode_data_frames_gpu(batch_tensor, derived_params, bits_argument, return_error_tensor=True)
                            decoded_bits_cpu = decoded_bits.to("cpu", non_blocking=True)
                        inflight_decodes.append((stream, decoded_bits_cpu, num_corrections))
                    else:
                        decoded_bits, num_corrections = decode_data_frames_gpu(batch_tensor_cpu.to(device), derived_params, bits_argument)
                        inflight_decodes.append((None, decoded_bits.cpu(), num_corrections))

                    flush_decoded_batches()
            
            flush_decoded_batches(force=True)

        except KeyboardInterrupt:
            logging.warning("Keyboard interrupt during data decoding.")
        finally:
            data_queue.put(None)
            if data_writer is not None:
                logging.info("Waiting for data writer thread to finish flushing all files to disk...")
                data_writer.join()
                logging.info("Data writer finished. Disk synchronization complete.")

        logging.info("--- DECODING SUMMARY (Data Frames) ---")
        logging.info(f"Total codewords with errors corrected: {total_corrected_data_codewords}")

        logging.info("Data reconstruction finished. Starting final recovery (par2/7z).")
        par2_path = config["PAR2_PATH"]
        sz_path = config["SEVENZIP_PATH"]
        
        # Iterate over all generated PAR2 index files.
        # Naming convention was: filename.7z.001.recovery.par2
        # We need to find all .par2 files that do NOT contain ".vol"
        par2_indices = [f for f in data_output_dir.glob("*.par2") if ".vol" not in f.name]
        
        if not par2_indices:
            logging.warning("No PAR2 index files found. Skipping recovery step.")
        else:
            for par2_file in sorted(par2_indices):
                logging.info(f"Running PAR2 repair for volume set: {par2_file.name}")
                # Use stream_output=True for large files so PAR2 output isn't buffered in RAM
                run_command([par2_path, "r", "-a", par2_file.name], cwd=str(data_output_dir), stream_output=True)

        # After repair, find the first 7z volume (usually ends in .001 or just .7z)
        # We look for the base archive file
        archive_candidates = list(data_output_dir.glob("*_data_archive.7z*"))
        # We want the one that is either just .7z or .7z.001
        main_archive = None
        for cand in archive_candidates:
            if cand.name.endswith(".7z") or cand.name.endswith(".001"):
                main_archive = cand
                break
        
        if main_archive:
            final_extraction_dir = output_dir / "Decoded_Files"
            final_extraction_dir.mkdir(exist_ok=True)
            current_password = password
            while True:
                # 7-Zip handles split volumes automatically if you point it to the first one (.001)
                cmd = [sz_path, "x", "-y", f"-o{final_extraction_dir}", str(main_archive)]
                if current_password:
                    cmd.insert(2, f"-p{current_password}")
                if run_command(cmd, cwd=str(data_output_dir)):
                    logging.info(f"Extraction successful. Files are in '{final_extraction_dir}'")
                    break
                if session_params.get("is_password_protected"):
                    logging.warning("Extraction failed, likely due to incorrect password.")
                    current_password = getpass.getpass("Please enter the correct password (or press Enter to skip): ")
                    if not current_password:
                        logging.warning("Skipping final extraction.")
                        break
                else:
                    logging.error("Extraction failed for a non-password protected archive.")
                    break
        else:
            logging.error("Could not find the main archive file for extraction.")

        logging.info("Decoding process completed.")
        decode_elapsed = time.perf_counter() - decode_start
        logging.info(f"Decoding completed successfully in {decode_elapsed:.1f}s ({decode_elapsed/60:.2f} min).")
    finally:
        cleanup_temp_dir(temp_dir, "decoding temp")

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Encode files into video or decode them back using PyTorch.")
    parser.add_argument("-mode", required=False, choices=["encode", "decode"], help="The operation to perform.")
    parser.add_argument("-input", required=False, help="Input file/folder (encode) or video file(s) (decode).")
    parser.add_argument("-output", help="Output directory. Defaults to a subdirectory next to the input.")
    parser.add_argument("-p", "--password", help="Password for archive creation/extraction.")
    
    args = parser.parse_args()

    # If not testing, mode and input are required
    if not args.mode or not args.input:
        parser.error("the following arguments are required: -mode, -input")

    setup_logging()
    
    try:
        config = load_config()
        device = setup_pytorch()
        input_path_str = args.input
        
        # Handle comma-separated inputs for default output directory calculation
        if "," in input_path_str:
            # Take the first file path to determine the default output directory
            # We strip whitespace to handle "file1, file2"
            first_input_str = input_path_str.split(",")[0].strip()
            reference_path = Path(first_input_str).resolve()
        else:
            reference_path = Path(input_path_str).resolve()

        if args.output: 
            output_dir = Path(args.output).resolve()
        else: 
            # Create output dir based on the reference path (first input file)
            output_dir = reference_path.parent / f"{reference_path.stem}_F2YT_Output"
        
        output_dir.mkdir(exist_ok=True)
        logging.info(f"Output will be saved to: {output_dir}")
        
        if args.mode == "encode":
            encode_orchestrator(Path(input_path_str).resolve(), output_dir, args.password, config, device)
        elif args.mode == "decode":
            decode_orchestrator(input_path_str, output_dir, args.password, config, device)
            
    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()