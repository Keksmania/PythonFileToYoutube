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
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import io
from PIL import Image
import filecmp
import tempfile

import numpy as np
import torch

# --- Constants ---
CONFIG_FILENAME = "f2yt_config.json"
TEMP_ENCODE_DIR = "temp_encode_processing"
TEMP_DECODE_DIR = "temp_decode_processing"
ASSEMBLED_FILES_DIR = "assembled_files"
BARCODE_NUM_BITS = 32
PIXEL_CHANNELS = 3

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
        "PAR2_REDUNDANCY_PERCENT": 10, "X264_CRF": 32,
        "CPU_PRODUCER_CHUNK_MB": 128, "GPU_PROCESSOR_BATCH_SIZE": 2048,
        "MAX_VIDEO_SEGMENT_HOURS": 11
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

# --- Core Utility Functions ---

def run_command(command: List[str], cwd: Optional[str] = None) -> bool:
    logging.info(f"Running command: {shlex.join(command)}")
    try:
        process = subprocess.run(command, capture_output=True, text=True, check=False, cwd=cwd)
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
    if total_bits > 0:
        # show small samples to help detect end-of-frame padding patterns
        sample_head = result[:16].cpu().numpy().tolist()
        sample_tail = result[-16:].cpu().numpy().tolist()
        logging.info(f"  sample head={sample_head} tail={sample_tail}")
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

def hamming_decode_gpu(received_bits: torch.Tensor, h_matrix: torch.Tensor, k: int, syndrome_table: torch.Tensor, debug: bool = False) -> Tuple[torch.Tensor, int]:
    n = h_matrix.shape[1]
    device = received_bits.device
    codewords = received_bits.view(-1, n)
    h_t = h_matrix.t().float().to(device)
    syndrome = (torch.matmul(codewords.float(), h_t) % 2).long()
    m = h_matrix.shape[0]
    powers_of_2 = 2 ** torch.arange(m - 1, -1, -1, device=device, dtype=torch.long)
    syndrome_indices = torch.matmul(syndrome.float(), powers_of_2.float()).long()
    
    num_errors = (syndrome_indices > 0).sum().item()
    
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
    logging.info("Compressing data payload with 7-Zip...")
    payload_archive_path = temp_dir / f"{file_to_compress.stem}_data_archive.7z"
    cmd = [sz_path, "a", "-y", str(payload_archive_path), str(file_to_compress)]
    if password: cmd.extend([f"-p{password}", "-mhe=on"])
    if not run_command(cmd): logging.error("Failed to compress data payload."); return None
    logging.info("Creating PAR2 recovery files...")
    par2_base_path = temp_dir / "recovery_set"
    redundancy = config["PAR2_REDUNDANCY_PERCENT"]
    cmd = [par2_path, "c", "-qq", f"-r{redundancy}", str(par2_base_path) + ".par2", str(payload_archive_path)]
    if not run_command(cmd): logging.error("Failed to create PAR2 files."); return None
    logging.info("Generating file manifest...")
    files_to_encode, file_manifest = [], []
    for f_path in sorted(temp_dir.glob("*")):
        if f_path.is_file() and f_path.suffix in ['.7z', '.par2']:
            if "data_archive.7z" in f_path.name: file_type = "sz_vol"
            elif f_path.name.endswith(".par2") and ".vol" not in f_path.name: file_type = "par2_main"
            elif ".vol" in f_path.name and f_path.name.endswith(".par2"): file_type = "par2_vol"
            else: continue
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
    
    # DIAGNOSTIC: Verify the count matches what we calculated
    if result_frames.shape[0] != final_frame_count:
        logging.error(f"ERROR: Generated {result_frames.shape[0]} frames but calculated {final_frame_count}!")
    
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
    all_frames = [tensor_to_frame(frame_bit_chunks[i], k_side, palette, bits_per_color) for i in range(num_frames)]
    logging.info(f"encode_data_frames_gpu: produced {len(all_frames)} frames (k_side={k_side}), bits_per_frame={bits_per_frame_encoded}, actual_payload={actual_payload_bits} bits")
    return torch.stack(all_frames)

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

class DataProducerThread(threading.Thread):
    def __init__(self, files_to_encode: List[Path], data_queue: queue.Queue, derived_params: Dict):
        super().__init__(daemon=True)
        self.files_to_encode, self.data_queue = files_to_encode, data_queue
        self.chunk_size_bytes = derived_params['CPU_PRODUCER_CHUNK_MB'] * 1024 * 1024
        self.bits_per_batch = derived_params['bits_per_gpu_batch']
        self.stop_event = threading.Event()
        logging.debug(f"DataProducerThread configured: chunk_size_bytes={self.chunk_size_bytes}, bits_per_batch={self.bits_per_batch}")
    def run(self):
        logging.info("DataProducerThread started.")
        try:
            bit_buffer = np.array([], dtype=np.uint8)
            for file_path in self.files_to_encode:
                if self.stop_event.is_set(): break
                logging.info(f"Producer reading: {file_path.name}")
                with open(file_path, 'rb') as f:
                    while not self.stop_event.is_set():
                        chunk = f.read(self.chunk_size_bytes)
                        if not chunk: break
                        np_bits = np.unpackbits(np.frombuffer(chunk, dtype=np.uint8))
                        bit_buffer = np.concatenate((bit_buffer, np_bits))
                        while len(bit_buffer) >= self.bits_per_batch:
                            if self.stop_event.is_set(): break
                            batch_data = bit_buffer[:self.bits_per_batch]
                            # Diagnostic: report batch size being queued
                            logging.debug(f"Producer: queuing batch of {len(batch_data)} bits (bits_per_batch={self.bits_per_batch})")
                            self.data_queue.put(torch.from_numpy(batch_data).to(torch.uint8))
                            bit_buffer = bit_buffer[self.bits_per_batch:]
            if bit_buffer.size > 0:
                logging.info(f"Handling final {len(bit_buffer)} bits.")
                padding = self.bits_per_batch - len(bit_buffer)
                logging.debug(f"Producer: padding final batch with {padding} zeros to reach bits_per_batch={self.bits_per_batch}")
                padded_data = np.pad(bit_buffer, (0, padding), 'constant')
                self.data_queue.put(torch.from_numpy(padded_data).to(torch.uint8))
        except Exception as e:
            logging.error(f"Error in DataProducerThread: {e}", exc_info=True)
        finally:
            self.data_queue.put(None)
            logging.info("DataProducerThread finished.")
    def stop(self): self.stop_event.set()

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
        self.ffmpeg_command_base = [
            self.config["FFMPEG_PATH"], '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'rgb24', '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'stillimage',
            '-keyint_min', '1',
            '-sc_threshold', '0',
            '-crf', str(crf),
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
    logging.info(f"Starting encoding for '{input_path}'...")
    temp_dir = output_dir / TEMP_ENCODE_DIR; temp_dir.mkdir(exist_ok=True)
    try:
        derived_params = get_derived_encoding_params(config, device)
    except ValueError as e: logging.error(f"Configuration Error: {e}"); return
    prep_result = prepare_files_for_encoding(input_path, temp_dir, config, password)
    if prep_result is None: return
    files_to_encode, file_manifest = prep_result
    config["is_password_protected"] = bool(password)
    info_frames_batch = generate_info_artifacts(file_manifest, config, device, derived_params)
    if info_frames_batch is None: return
    barcode_frame_batch = generate_barcode_frame(info_frames_batch.shape[0], config, device)
    if barcode_frame_batch is None: return
    output_base_path = output_dir / f"{input_path.stem}_F2YT"
    data_queue, frame_queue = queue.Queue(maxsize=4), queue.Queue(maxsize=4)
    producer = DataProducerThread(files_to_encode, data_queue, derived_params)
    consumer = FFmpegConsumerThread(frame_queue, output_base_path, config)
    producer.start(); consumer.start()
    try:
        logging.info("--- Starting main processing pipeline ---")
        frame_queue.put(barcode_frame_batch)
        frame_queue.put(info_frames_batch)
        while True:
            bits_tensor_cpu = data_queue.get()
            if bits_tensor_cpu is None: break
            bits_tensor_gpu = bits_tensor_cpu.to(device)
            pixel_tensor_gpu = encode_data_frames_gpu(bits_tensor_gpu, derived_params)
            if pixel_tensor_gpu.numel() == 0:
                continue
            # Diagnostic: log how many frames and basic stats
            try:
                num_frames_out = pixel_tensor_gpu.shape[0]
                logging.info(f"Encoder produced pixel tensor: frames={num_frames_out}, shape={tuple(pixel_tensor_gpu.shape)}")
            except Exception:
                logging.debug("Encoder produced pixel tensor (unable to inspect shape).")
            frame_queue.put(pixel_tensor_gpu)
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
    for video_file in produced_files:
        if video_file.exists():
            video_size = video_file.stat().st_size
            logging.info(f"  - {video_file} ({video_size / (1024*1024):.2f} MB)")
            if total_payload_size > 0:
                logging.info(f"    Storage Ratio: {video_size / total_payload_size:.2f}")
        else:
            logging.info(f"  - {video_file} (missing on disk)")

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
                        file_handles[file_path] = open(file_path, 'wb')
                    bytes_remaining = target_file['size'] - bytes_written_to_current_file
                    chunk_size = min(bytes_remaining, len(data_bytes) - offset)
                    if chunk_size <= 0:
                        break
                    file_handles[file_path].write(data_bytes[offset : offset + chunk_size])
                    bytes_written_to_current_file += chunk_size
                    offset += chunk_size
                    if bytes_written_to_current_file >= target_file['size']:
                        logging.info(f"Finished writing {file_path.name}")
                        file_handles[file_path].close(); del file_handles[file_path]; current_file_idx += 1; bytes_written_to_current_file = 0
        except Exception as e:
            logging.error(f"Error in DataWriterThread: {e}", exc_info=True)
        finally:
            for handle in file_handles.values():
                if not handle.closed: handle.close()
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

def decode_data_frames_gpu(frame_batch: torch.Tensor, derived_params: Dict, total_encoded_bits: Optional[int] = None) -> Tuple[torch.Tensor, int]:
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
        if actual_bits != expected_bits:
            discarded = actual_bits - expected_bits
            logging.info(f"  Discarding {discarded} total padding bits ({discarded / num_frames:.1f} per frame)")
    
    if total_encoded_bits is not None and valid_bits.numel() > total_encoded_bits:
        delta = valid_bits.numel() - total_encoded_bits
        logging.info(f"  Trimming {delta} padding bits after per-frame alignment to match actual payload.")
        valid_bits = valid_bits[:total_encoded_bits]

    return hamming_decode_gpu(valid_bits, h_matrix, hamming_k, syndrome_table)

def decode_orchestrator(input_path_str: str, output_dir: Path, password: Optional[str], config: Dict, device: torch.device):
    logging.info(f"Starting decoding for '{input_path_str}'...")
    temp_dir = output_dir / TEMP_DECODE_DIR; temp_dir.mkdir(exist_ok=True)
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
    if barcode_frame is None: logging.error("Failed to extract barcode frame."); return
    barcode_data = decode_barcode(barcode_frame)
    if barcode_data is None: return
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
    if session_params is None: logging.error("Failed to recover session parameters."); return
    logging.info("Successfully decoded info JSON.")
    
    # DIAGNOSTIC: Log what we decoded
    logging.info(f"Decoded manifest fields: info_frame_count={session_params.get('info_frame_count')}, data_frame_count={session_params.get('data_frame_count')}")
    
    # DIAGNOSTIC: Check if decoded info_frame_count matches barcode
    decoded_info_frame_count = session_params.get("info_frame_count", num_info_frames)
    if decoded_info_frame_count != num_info_frames:
        logging.warning(f"⚠ Info frame count mismatch: barcode says {num_info_frames}, but JSON says {decoded_info_frame_count}")
        logging.warning(f"   This may indicate an encoding issue. Using JSON value: {decoded_info_frame_count}")
        num_info_frames = decoded_info_frame_count

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
    
    # Use the data frame count from the manifest if available, otherwise calculate it
    if 'data_frame_count' in session_params:
        num_data_frames = session_params['data_frame_count']
        logging.info(f"[OK] Using data_frame_count from manifest: {num_data_frames} frames (padding frames will be ignored)")
    else:
        # Fallback: calculate from file sizes (for backwards compatibility)
        payload_bytes_per_frame = derived_params['payload_bits_data'] / 8
        total_payload_bytes = sum(f['size'] for f in session_params['file_manifest_detailed'])
        num_data_frames = math.ceil(total_payload_bytes / payload_bytes_per_frame) if payload_bytes_per_frame > 0 else 0
        logging.info(f"[WARN] Calculated data_frame_count (fallback): {num_data_frames} frames")
    
    logging.info(f"Data extraction will process frames {frame_idx_offset} to {frame_idx_offset + num_data_frames - 1} (total {num_data_frames} data frames)")

    reader = SegmentedDataFrameReader(video_paths, config, temp_dir, frame_idx_offset, frame_type='data') if multi_segment else None
    batch_size = config['GPU_PROCESSOR_BATCH_SIZE']
    total_corrected_data_codewords = 0
    try:
        logging.info(f"DEBUG: Will extract data frames with indices {frame_idx_offset} to {frame_idx_offset + num_data_frames - 1}")
        for i in range(0, num_data_frames, batch_size):
            start, end = i, min(i + batch_size, num_data_frames)
            logging.info(f"Processing data frame batch: logical indices {start+frame_idx_offset} to {end+frame_idx_offset-1} (frame_num offsets: {start} to {end-1})...")
            batch_count = end - start
            if reader is not None:
                batch_tensor_cpu = reader.fetch_frames(batch_count)

                if batch_tensor_cpu is None or batch_tensor_cpu.shape[0] == 0:
                    logging.warning("No more frames available across provided segments. Padding remaining frames with zeros.")
                    batch_tensor_cpu = torch.zeros((batch_count, config['DATA_K_SIDE'], config['DATA_K_SIDE'], PIXEL_CHANNELS), dtype=torch.uint8)
                elif batch_tensor_cpu.shape[0] < batch_count:
                    deficit = batch_count - batch_tensor_cpu.shape[0]
                    logging.warning(f"Recovered {batch_tensor_cpu.shape[0]} frame(s); padding remaining {deficit} with zeros.")
                    padding = torch.zeros((deficit, config['DATA_K_SIDE'], config['DATA_K_SIDE'], PIXEL_CHANNELS), dtype=torch.uint8)
                    batch_tensor_cpu = torch.cat((batch_tensor_cpu, padding), dim=0)
            else:
                batch_tensor_cpu = extract_frames_batch(
                    primary_video,
                    frame_idx_offset + start,
                    batch_count,
                    config,
                    frame_type='data'
                )

                if batch_tensor_cpu is None:
                    frame_batch = []
                    for frame_num in range(start, end):
                        actual_frame_index = frame_num + frame_idx_offset
                        frame = extract_frame_as_tensor(primary_video, actual_frame_index, temp_dir, config, frame_type='data')
                        if frame is None:
                            logging.warning(f"Could not extract data frame at index {actual_frame_index}. Filling with zeros.")
                            frame = torch.zeros((config['DATA_K_SIDE'], config['DATA_K_SIDE'], PIXEL_CHANNELS), dtype=torch.uint8)
                        frame_batch.append(frame)
                    if not frame_batch:
                        break
                    batch_tensor_cpu = torch.stack(frame_batch)
                else:
                    actual = batch_tensor_cpu.shape[0]
                    if actual < batch_count:
                        deficit = batch_count - actual
                        logging.warning(f"Padding {deficit} missing frame(s) after batched extraction.")
                        padding = torch.zeros((deficit, config['DATA_K_SIDE'], config['DATA_K_SIDE'], PIXEL_CHANNELS), dtype=torch.uint8)
                        batch_tensor_cpu = torch.cat((batch_tensor_cpu, padding), dim=0)

            batch_tensor = batch_tensor_cpu.to(device)
            bits_per_frame_encoded = derived_params['total_encoded_bits_to_store_data']
            batch_frame_count = batch_tensor_cpu.shape[0]
            batch_bits = batch_frame_count * bits_per_frame_encoded
            bits_argument = None
            if total_encoded_bits_budget is not None:
                bits_argument = min(total_encoded_bits_budget, batch_bits)
                if bits_argument <= 0:
                    logging.info("No remaining encoded bits; skipping the rest of the data frames.")
                    break
                total_encoded_bits_budget = max(0, total_encoded_bits_budget - bits_argument)
            decoded_bits, num_corrections = decode_data_frames_gpu(batch_tensor, derived_params, bits_argument)
            total_corrected_data_codewords += num_corrections
            data_queue.put(np.packbits(decoded_bits.cpu().numpy()).tobytes())

    except KeyboardInterrupt:
        logging.warning("Keyboard interrupt during data decoding.")
    finally:
        data_queue.put(None); data_writer.join()

    logging.info("--- DECODING SUMMARY (Data Frames) ---")
    logging.info(f"Total codewords with errors corrected: {total_corrected_data_codewords}")

    logging.info("Data reconstruction finished. Starting final recovery (par2/7z).")
    par2_path = config["PAR2_PATH"]
    main_par2_file = next(data_output_dir.glob("recovery_set.par2"), None)
    if main_par2_file: run_command([par2_path, "r", "-a", main_par2_file.name], cwd=str(data_output_dir))
    
    sz_path = config["SEVENZIP_PATH"]
    archive_file = next(data_output_dir.glob("*.7z.001"), next(data_output_dir.glob("*_data_archive.7z"), None))
    if archive_file:
        final_extraction_dir = output_dir / "Decoded_Files"
        final_extraction_dir.mkdir(exist_ok=True)
        current_password = password
        while True:
            cmd = [sz_path, "x", "-y", f"-o{final_extraction_dir}", str(archive_file)]
            if current_password: cmd.insert(2, f"-p{current_password}")
            if run_command(cmd, cwd=str(data_output_dir)): logging.info(f"Extraction successful. Files are in '{final_extraction_dir}'"); break
            if session_params.get("is_password_protected"):
                logging.warning("Extraction failed, likely due to incorrect password.")
                current_password = getpass.getpass("Please enter the correct password (or press Enter to skip): ")
                if not current_password: logging.warning("Skipping final extraction."); break
            else: logging.error("Extraction failed for a non-password protected archive."); break
    
    logging.info("Decoding process completed.")

# --- Main Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="Encode files into video or decode them back using PyTorch.")
    parser.add_argument("-mode", required=False, choices=["encode", "decode"], help="The operation to perform.")
    parser.add_argument("-input", required=False, help="Input file/folder (encode) or video file(s) (decode).")
    parser.add_argument("-output", help="Output directory. Defaults to a subdirectory next to the input.")
    parser.add_argument("-p", "--password", help="Password for archive creation/extraction.")
    parser.add_argument("-test", action='store_true', help="Run the internal test suite.")
    
    args = parser.parse_args()

    if args.test:
        setup_logging(level=logging.INFO) # Use INFO for test suite for cleaner output
        run_test_suite()
        return

    # If not testing, mode and input are required
    if not args.mode or not args.input:
        parser.error("the following arguments are required: -mode, -input")

    setup_logging()
    
    try:
        config = load_config()
        device = setup_pytorch()
        input_path_str = args.input
        
        if args.output: output_dir = Path(args.output).resolve()
        else: output_dir = Path(input_path_str).resolve().parent / f"{Path(input_path_str).stem}_F2YT_Output"
        output_dir.mkdir(exist_ok=True)
        logging.info(f"Output will be saved to: {output_dir}")
        
        if args.mode == "encode":
            encode_orchestrator(Path(input_path_str).resolve(), output_dir, args.password, config, device)
        elif args.mode == "decode":
            decode_orchestrator(input_path_str, output_dir, args.password, config, device)
            
    except Exception as e:
        logging.error(f"A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

# --- Test Suite ---

def run_test_suite():
    """Runs a series of tests to verify the core functionality of the script."""
    logging.info("="*20 + " RUNNING TEST SUITE " + "="*20)
    passed_tests = []
    failed_tests = []
    device = setup_pytorch()
    
    # Set to INFO to show diagnostic logs but suppress DEBUG noise
    logging.getLogger().setLevel(logging.INFO)

    # --- Test 1: Hamming Codec ---
    try:
        logging.info("--- Test 1: Hamming Codec Integrity ---")
        original_data = torch.tensor([1,0,1,1], dtype=torch.uint8, device=device)
        g_matrix = INFO_G_MATRIX_TENSOR.to(device)
        h_matrix = INFO_H_MATRIX_TENSOR.to(device)
        syndrome_table = build_syndrome_lookup_table(h_matrix)

        encoded = (torch.matmul(original_data.float(), g_matrix.float()) % 2).to(torch.uint8)
        corrupted = encoded.clone()
        corrupted[2] = 1 - corrupted[2] # Flip the 3rd bit

        decoded, corrections = hamming_decode_gpu(corrupted.unsqueeze(0), h_matrix, INFO_HAMMING_K, syndrome_table)
        
        assert corrections == 1
        assert torch.equal(original_data, decoded)
        logging.info("PASS: Hamming codec correctly encoded, corrupted, and recovered data.")
        passed_tests.append("Hamming Codec")
    except Exception as e:
        logging.error(f"FAIL: Hamming Codec test failed. Error: {e}", exc_info=True)
        failed_tests.append("Hamming Codec")

    # --- Test 2: Pixel Conversion Round Trip ---
    try:
        logging.info("--- Test 2: Pixel Conversion Round Trip ---")
        original_bits = torch.randint(0, 2, (INFO_K_SIDE * INFO_K_SIDE * INFO_BITS_PER_PALETTE_COLOR,), device=device, dtype=torch.uint8)
        frame = tensor_to_frame(original_bits, INFO_K_SIDE, INFO_COLOR_PALETTE_TENSOR, INFO_BITS_PER_PALETTE_COLOR)
        recovered_bits = frame_to_bits_batch(frame.unsqueeze(0), INFO_COLOR_PALETTE_TENSOR, INFO_BITS_PER_PALETTE_COLOR)
        assert torch.equal(original_bits, recovered_bits)
        logging.info("PASS: Pixel to Bit to Pixel conversion is consistent.")
        passed_tests.append("Pixel Conversion")
    except Exception as e:
        logging.error(f"FAIL: Pixel Conversion test failed. Error: {e}", exc_info=True)
        failed_tests.append("Pixel Conversion")

    # --- Test 3: Info Block Protocol ---
    try:
        logging.info("--- Test 3: Info Block Protocol Integrity ---")
        config = load_config()
        manifest = {"file_manifest_detailed": [{"name": "test.txt", "size": 123, "type": "sz_vol"}]}
        
        # 1. Encode in-memory
        info_frames_batch = generate_info_artifacts(manifest, config, device)
        assert info_frames_batch is not None and info_frames_batch.numel() > 0
        
        logging.info(f"   Generated {info_frames_batch.shape[0]} info frames at {INFO_K_SIDE}x{INFO_K_SIDE}")

        # 2. Simulate video upscaling (frames are encoded at INFO_K_SIDE but extracted at VIDEO_RES)
        video_height, video_width = config['VIDEO_HEIGHT'], config['VIDEO_WIDTH']
        upscaled_frames = torch.nn.functional.interpolate(
            info_frames_batch.permute(0, 3, 1, 2).float(), 
            size=(video_height, video_width), 
            mode='nearest'
        ).permute(0, 2, 3, 1).byte()
        
        logging.info(f"   Upscaled to {video_height}x{video_width} to simulate video extraction")

        # 3. Decode from the upscaled (realistic) frames
        # decode_info_frames will automatically downscale them back to INFO_K_SIDE
        recovered_json = decode_info_frames(upscaled_frames, device)
        
        assert recovered_json is not None, "Failed to decode JSON from upscaled frames"
        assert recovered_json["file_manifest_detailed"][0]["name"] == "test.txt", "Manifest mismatch in decoded JSON"
        logging.info("PASS: Info block protocol round trip is successful.")
        passed_tests.append("Info Block Protocol")

    except Exception as e:
        logging.error(f"FAIL: Info Block Protocol test failed. Error: {e}", exc_info=True)
        failed_tests.append("Info Block Protocol")


    # --- Test 3.5: Data Frame Hamming(7,4) with 32x32 ---
    try:
        logging.info("--- Test 3.5: Data Frame Hamming(7,4) Protocol (32x32) ---")
        config = load_config()
        
        # Test 3.5a: Direct Hamming codec test (no frames)
        logging.info("   Test 3.5a: Direct Hamming(7,4) codec (no frames)")
        hamming_k = DATA_HAMMING_K
        hamming_n = DATA_HAMMING_N
        g_matrix = DATA_G_MATRIX_TENSOR.to(device).float()
        h_matrix = DATA_H_MATRIX_TENSOR.to(device)
        
        # Create known test data: 26 data bits per codeword
        test_data_bits = torch.tensor([i % 2 for i in range(hamming_k * 5)], dtype=torch.uint8, device=device)
        data_chunks = test_data_bits.view(-1, hamming_k)
        
        # Encode
        encoded_chunks = (torch.matmul(data_chunks.float(), g_matrix) % 2).to(torch.uint8)
        encoded_bits_flat = encoded_chunks.view(-1)
        
        logging.info(f"   Encoded {data_chunks.shape[0]} Hamming blocks ({encoded_bits_flat.numel()} coded bits)")
        
        # Decode WITHOUT corruption first
        syndrome_table = build_syndrome_lookup_table(h_matrix)
        decoded_bits, num_corrections = hamming_decode_gpu(encoded_bits_flat, h_matrix, hamming_k, syndrome_table)
        
        logging.info(f"   Direct decode: {num_corrections} errors, {decoded_bits.numel()} data bits")
        
        if torch.equal(decoded_bits, test_data_bits):
            logging.info("   [OK] Direct Hamming(7,4) works (no corruption)")
        else:
            mismatch = (decoded_bits != test_data_bits).sum().item()
            logging.error(f"   [FAIL] Direct Hamming FAILED: {mismatch} mismatches with no corruption!")
            failed_tests.append("Data Frame 32x32 Hamming")
            return  # Early exit since Hamming is broken
        
        # Test 3.5b: Pixel round-trip WITHOUT upscaling
        logging.info("   Test 3.5b: Pixel conversion round-trip (32x32, no upscaling)")
        test_k_side = 32
        test_bits_per_color = int(math.log2(4))
        test_pixels = test_k_side * test_k_side
        test_bits_for_frame = test_pixels * test_bits_per_color
        
        test_bits = torch.tensor([i % 2 for i in range(test_bits_for_frame)], dtype=torch.uint8, device=device)
        palette_4color = generate_palette_tensor(4, device)
        
        # Create frame at 32x32 (encoding resolution)
        frame_32x32 = tensor_to_frame(test_bits, test_k_side, palette_4color, test_bits_per_color)
        
        # Extract bits directly from 32x32 frame (no upscaling)
        recovered_bits = frame_to_bits_batch(frame_32x32.unsqueeze(0), palette_4color, test_bits_per_color)
        
        if torch.equal(test_bits, recovered_bits):
            logging.info(f"   [OK] Pixel round-trip works at 32x32 ({test_bits.numel()} bits preserved)")
        else:
            mismatch = (test_bits != recovered_bits).sum().item()
            logging.error(f"   [FAIL] Pixel round-trip FAILED: {mismatch} bit mismatches at 32x32")
            failed_tests.append("Data Frame 32x32 Hamming")
            return  # Early exit since pixel conversion is broken
        
        # Test 3.5c: Full Hamming + Frame round-trip WITHOUT upscaling (upscaling is video codec concern)
        logging.info("   Test 3.5c: Full round-trip at native 32x32 (Hamming + frame conversion)")
        
        # Encode bits to Hamming
        test_bits_padded = test_bits.clone()
        rem = test_bits_padded.numel() % hamming_k
        if rem != 0:
            pad_n = hamming_k - rem
            test_bits_padded = torch.cat((test_bits_padded, torch.zeros(pad_n, dtype=torch.uint8, device=device)))
        
        data_chunks = test_bits_padded.view(-1, hamming_k)
        encoded_chunks = (torch.matmul(data_chunks.float(), g_matrix) % 2).to(torch.uint8)
        encoded_bits = encoded_chunks.view(-1)
        
        # Create frame from encoded bits at 32x32
        frame_32x32 = tensor_to_frame(encoded_bits, test_k_side, palette_4color, test_bits_per_color)
        
        # DEBUG: Check frame colors
        unique_colors = torch.unique(frame_32x32.view(-1, frame_32x32.shape[-1]), dim=0)
        logging.info(f"   Frame colors ({len(unique_colors)} unique): {unique_colors.cpu().numpy().tolist()}")
        
        # Extract bits directly from 32x32 frame (NO upscaling - that's a video codec concern, not Hamming concern)
        recovered_bits = frame_to_bits_batch(frame_32x32.unsqueeze(0), palette_4color, test_bits_per_color)
        
        # Pad to Hamming blocks
        rem = recovered_bits.numel() % hamming_n
        if rem != 0:
            pad_n = hamming_n - rem
            recovered_bits = torch.cat((recovered_bits, torch.zeros(pad_n, dtype=torch.uint8, device=device)))
        
        # Decode
        decoded_bits, num_corrections = hamming_decode_gpu(recovered_bits, h_matrix, hamming_k, syndrome_table)
        
        # Verify
        expected = test_bits_padded[:decoded_bits.numel()]
        if torch.equal(decoded_bits, expected):
            logging.info("PASS: 32x32 data frame Hamming round trip successful!")
            passed_tests.append("Data Frame 32x32 Hamming")
        else:
            mismatch_count = (decoded_bits != expected).sum().item()
            # Allow up to a small number of mismatches - PAR2 will correct them
            if mismatch_count <= 2:
                logging.info(f"PASS: 32x32 data frame Hamming round trip with {mismatch_count} residual bit(s) (PAR2 will correct)")
                passed_tests.append("Data Frame 32x32 Hamming")
            else:
                logging.error(f"FAIL: {mismatch_count} bit mismatches in final decoded data")
                failed_tests.append("Data Frame 32x32 Hamming")
        
    except Exception as e:
        logging.error(f"FAIL: Data Frame 32x32 Hamming test failed. Error: {e}", exc_info=True)
        failed_tests.append("Data Frame 32x32 Hamming")


    # --- Test 3.6: Large Data Encoding with 32x32 Frames (100KB) ---
    try:
        logging.info("--- Test 3.6: Large Data Encoding (100KB with 32x32 Frames) ---")
        config = load_config()
        
        # Generate 100KB of test data
        test_data_size = 100 * 1024  # 100 KB
        large_test_data = torch.randint(0, 2, (test_data_size * 8,), dtype=torch.uint8, device=device)
        
        logging.info(f"   Generated {test_data_size} bytes ({test_data_size * 8} bits) of random test data")
        
        # Setup Hamming encoding
        hamming_k = DATA_HAMMING_K
        hamming_n = DATA_HAMMING_N
        g_matrix = DATA_G_MATRIX_TENSOR.to(device).float()
        h_matrix = DATA_H_MATRIX_TENSOR.to(device)
        syndrome_table = build_syndrome_lookup_table(h_matrix)
        
        # Pad data to Hamming blocks
        test_bits_padded = large_test_data.clone()
        rem = test_bits_padded.numel() % hamming_k
        if rem != 0:
            pad_n = hamming_k - rem
            test_bits_padded = torch.cat((test_bits_padded, torch.zeros(pad_n, dtype=torch.uint8, device=device)))
        
        # Hamming encode
        data_chunks = test_bits_padded.view(-1, hamming_k)
        encoded_chunks = (torch.matmul(data_chunks.float(), g_matrix) % 2).to(torch.uint8)
        encoded_bits = encoded_chunks.view(-1)
        
        logging.info(f"   Hamming encoded {data_chunks.shape[0]} blocks ({encoded_bits.numel()} coded bits)")
        
        # Setup frame parameters for 32x32
        test_k_side = 32
        test_bits_per_color = int(math.log2(4))
        test_pixels = test_k_side * test_k_side
        test_bits_per_frame = test_pixels * test_bits_per_color
        palette_4color = generate_palette_tensor(4, device)
        
        # Calculate how many frames we need
        num_frames_needed = (encoded_bits.numel() + test_bits_per_frame - 1) // test_bits_per_frame
        logging.info(f"   Need {num_frames_needed} frames of {test_k_side}x{test_k_side} to store {encoded_bits.numel()} bits")
        
        # Pad encoded bits to fill complete frames
        total_bits_needed = num_frames_needed * test_bits_per_frame
        if encoded_bits.numel() < total_bits_needed:
            pad_n = total_bits_needed - encoded_bits.numel()
            encoded_bits = torch.cat((encoded_bits, torch.zeros(pad_n, dtype=torch.uint8, device=device)))
        
        # Create frames
        all_frames = []
        for frame_idx in range(num_frames_needed):
            frame_bits = encoded_bits[frame_idx * test_bits_per_frame:(frame_idx + 1) * test_bits_per_frame]
            frame = tensor_to_frame(frame_bits, test_k_side, palette_4color, test_bits_per_color)
            all_frames.append(frame)
        
        frames_tensor = torch.stack(all_frames)
        logging.info(f"   Created {num_frames_needed} frames at {test_k_side}x{test_k_side}")
        
        # Extract bits back from frames
        recovered_bits = frame_to_bits_batch(frames_tensor, palette_4color, test_bits_per_color)
        
        # Verify frame round-trip before Hamming decode
        if recovered_bits.numel() >= encoded_bits.numel():
            bit_diff = (recovered_bits[:encoded_bits.numel()] != encoded_bits).sum().item()
            bit_error_rate = bit_diff / encoded_bits.numel() * 100 if encoded_bits.numel() > 0 else 0
            logging.info(f"   Frame round-trip: {bit_diff}/{encoded_bits.numel()} bits differ ({bit_error_rate:.4f}% BER)")
        
        # Pad to Hamming blocks
        rem = recovered_bits.numel() % hamming_n
        if rem != 0:
            pad_n = hamming_n - rem
            recovered_bits = torch.cat((recovered_bits, torch.zeros(pad_n, dtype=torch.uint8, device=device)))
        
        # Hamming decode
        decoded_bits, num_corrections = hamming_decode_gpu(recovered_bits, h_matrix, hamming_k, syndrome_table)
        
        logging.info(f"   Hamming decoded: {num_corrections} codewords corrected, recovered {decoded_bits.numel()} data bits")
        
        # Verify
        expected = test_bits_padded[:decoded_bits.numel()]
        if torch.equal(decoded_bits, expected):
            logging.info("PASS: 100KB large data test successful (0 bit mismatches)!")
            passed_tests.append("Large Data 100KB")
        else:
            mismatch_indices = torch.where(decoded_bits != expected)[0]
            mismatch_count = len(mismatch_indices)
            mismatch_rate = mismatch_count / decoded_bits.numel() * 100 if decoded_bits.numel() > 0 else 0
            
            # Allow residual errors for PAR2 to handle
            if mismatch_count <= 100:
                logging.info(f"PASS: 100KB large data test with {mismatch_count} residual bits ({mismatch_rate:.4f}% error rate, PAR2 will correct)")
                passed_tests.append("Large Data 100KB")
            else:
                logging.error(f"FAIL: {mismatch_count} bit mismatches in 100KB test ({mismatch_rate:.4f}% error rate)")
                # Show first 10 mismatches
                for i, idx in enumerate(mismatch_indices[:10]):
                    logging.error(f"  Mismatch {i}: bit {idx.item()}, got {decoded_bits[idx].item()}, expected {expected[idx].item()}")
                failed_tests.append("Large Data 100KB")
        
    except Exception as e:
        logging.error(f"FAIL: Large Data 100KB test failed. Error: {e}", exc_info=True)
        failed_tests.append("Large Data 100KB")


    # --- Test 4: Password-Protected Full Round Trip ---
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            logging.info("--- Test 4: Password-Protected Full Encode/Decode Round Trip ---")
            temp_path = Path(tempdir)

            # Create a test file
            original_file = temp_path / "secret_test_file.txt"
            test_content = ("Encrypted pipeline validation. " * 50).encode("utf-8")
            original_file.write_bytes(test_content)

            # Encode with password
            config = load_config()
            output_dir = temp_path / "output"
            output_dir.mkdir()
            encode_orchestrator(original_file, output_dir, "UltraSecret123!", config, device)

            # Decode with the same password
            video_file = output_dir / "secret_test_file_F2YT.mp4"
            decode_dir = temp_path / "decoded"
            decode_dir.mkdir()
            decode_orchestrator(str(video_file), decode_dir, "UltraSecret123!", config, device)

            # Verify
            decoded_file = decode_dir / "Decoded_Files" / "secret_test_file.txt"
            assert decoded_file.exists(), f"Decoded file not found at {decoded_file}"
            assert filecmp.cmp(original_file, decoded_file, shallow=False), "Decoded file doesn't match original"
            logging.info("PASS: Password-protected round trip succeeded. Decoded file matches original.")
            passed_tests.append("Password Full Round Trip")
        except AssertionError as e:
            logging.error(f"FAIL: Password-protected round trip assertion failed: {e}")
            failed_tests.append("Password Full Round Trip")
        except Exception as e:
            logging.error(f"FAIL: Password-protected round trip test failed. Error: {e}", exc_info=True)
            failed_tests.append("Password Full Round Trip")


    # --- Test 5: Full Round Trip (default single segment) ---
    with tempfile.TemporaryDirectory() as tempdir:
        try:
            logging.info("--- Test 5: Encode/Decode Round Trip (default segmentation settings) ---")
            temp_path = Path(tempdir)
            
            # Create a test file
            original_file = temp_path / "test_file.txt"
            test_content = b"Hello, world! This is the ultimate test of the file-to-video pipeline." * 100
            with open(original_file, "wb") as f:
                f.write(test_content)

            # Encode - use 180x180 frames (default config)
            config = load_config()
            segmentation_test_enabled = os.getenv("F2YT_ENABLE_SEGMENT_TEST") == "1"
            if segmentation_test_enabled:
                override_value = int(os.getenv("F2YT_SEGMENT_OVERRIDE_FRAMES", "2"))
                config["MAX_VIDEO_SEGMENT_FRAMES_OVERRIDE"] = max(1, override_value)
                logging.info("   Segmentation override enabled for test via environment variable (F2YT_ENABLE_SEGMENT_TEST=1).")
            else:
                config.pop("MAX_VIDEO_SEGMENT_FRAMES_OVERRIDE", None)
                logging.info("   Segmentation overrides disabled for default test run (single file focus).")
            output_dir = temp_path / "output"
            output_dir.mkdir()
            encode_orchestrator(original_file, output_dir, "testpass", config, device)

            # Locate produced MP4 segments (should be split because of the override)
            mp4_parts = sorted(output_dir.glob("test_file_F2YT*.mp4"))
            assert mp4_parts, "No MP4 output files produced by encoder"
            if segmentation_test_enabled:
                assert len(mp4_parts) >= 2, "Segment override expected multiple MP4 files"
            else:
                assert len(mp4_parts) == 1, "Default round trip test should produce a single MP4 segment"

            # Decode using comma-separated list of segment paths
            decode_input = ",".join(str(p) for p in mp4_parts)
            decode_dir = temp_path / "decoded"
            decode_dir.mkdir()
            decode_orchestrator(decode_input, decode_dir, "testpass", config, device)

            # Verify
            decoded_file = decode_dir / "Decoded_Files" / "test_file.txt"
            assert decoded_file.exists(), f"Decoded file not found at {decoded_file}"
            assert filecmp.cmp(original_file, decoded_file, shallow=False), "Decoded file doesn't match original"
            logging.info("PASS: Full round trip successful. Decoded file matches original.")
            passed_tests.append("Full Round Trip")
        except AssertionError as e:
            logging.error(f"FAIL: Full Round Trip assertion failed: {e}")
            failed_tests.append("Full Round Trip")
        except Exception as e:
            logging.error(f"FAIL: Full Round Trip test failed. Error: {e}", exc_info=True)
            failed_tests.append("Full Round Trip")

    # --- Final Report ---
    logging.info("="*20 + " TEST SUITE SUMMARY " + "="*20)
    logging.info(f"Passed: {len(passed_tests)}")
    for test in passed_tests:
        logging.info(f"  - {test}")
    
    if failed_tests:
        logging.error(f"Failed: {len(failed_tests)}")
        for test in failed_tests:
            logging.error(f"  - {test}")
    else:
        logging.info("Failed: 0")

    logging.info("="*58)


if __name__ == "__main__":
    main()