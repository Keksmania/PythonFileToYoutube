#!/usr/bin/env bash
# Dependency checker for Linux/macOS environments
set -u

print_header() {
  printf '\033[36m%s\033[0m\n' "PythonFileToYoutube dependency check (Unix)"
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

add_missing() {
  local name="$1"
  local msg="$2"
  missing+=("- ${name}: ${msg}")
}

missing=()
print_header

PY_BIN=${PYTHON_BIN:-python3}
if ! command_exists "$PY_BIN"; then
  add_missing "Python 3.11" "Install Python 3.11 and make sure '${PY_BIN}' (or set PYTHON_BIN) is available."
else
  py_version=$($PY_BIN -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || true)
  if [[ -z "$py_version" ]]; then
    add_missing "Python 3.11" "Unable to determine version."
  else
    IFS='.' read -r py_major py_minor _ <<<"$py_version"
    if (( py_major < 3 || (py_major == 3 && py_minor < 11) )); then
      add_missing "Python >= 3.11" "Detected ${py_version}. Upgrade recommended."
    else
      printf '✔ Python %s (%s)\n' "$py_version" "$PY_BIN"
    fi
  fi
fi

if command_exists "$PY_BIN"; then
  torch_json=$($PY_BIN -c 'import json
state = {"installed": False, "cuda": False, "version": None}
try:
    import torch
except ModuleNotFoundError:
    pass
else:
    state["installed"] = True
    state["version"] = getattr(torch, "__version__", "unknown")
    try:
        state["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        state["cuda"] = False
print(json.dumps(state))' 2>/dev/null || true)
  if [[ -z "$torch_json" ]]; then
    add_missing "PyTorch" "Install the CUDA-enabled PyTorch build."
  else
    has_torch=$(printf '%s' "$torch_json" | $PY_BIN -c 'import json,sys; data=json.load(sys.stdin); print("1" if data.get("installed") else "0"); print("1" if data.get("cuda") else "0"); print(data.get("version","unknown"))')
  fi
  if [[ -n "${has_torch:-}" ]]; then
    IFS=$'\n' read -r torch_installed torch_cuda torch_version <<<"$has_torch"
    if [[ "$torch_installed" != "1" ]]; then
      add_missing "PyTorch" "Install the CUDA build from pytorch.org."
    else
      if [[ "$torch_cuda" == "1" ]]; then
        printf '✔ PyTorch %s (CUDA available)\n' "$torch_version"
      else
        printf '⚠ PyTorch %s (CUDA not detected)\n' "$torch_version"
        add_missing "CUDA for PyTorch" "torch.cuda.is_available() returned False. Install NVIDIA drivers/CUDA toolkit."
      fi
    fi
  fi
fi

for entry in ffmpeg "7z" par2 nvidia-smi; do
  if command_exists "$entry"; then
    printf '✔ %s\n' "$entry"
  else
    case "$entry" in
      ffmpeg) add_missing "FFmpeg" "Install ffmpeg with libx264 support." ;;
      7z) add_missing "7-Zip CLI" "Install p7zip-full and ensure '7z' is on PATH." ;;
      par2) add_missing "PAR2" "Install par2cmdline." ;;
      nvidia-smi) add_missing "CUDA / NVIDIA driver" "Install NVIDIA drivers + CUDA toolkit." ;;
    esac
  fi
done

if [[ ${#missing[@]} -eq 0 ]]; then
  printf '\033[32mAll dependencies satisfied.\033[0m\n'
  exit 0
fi

printf '\n\033[33mMissing dependencies:\033[0m\n'
for item in "${missing[@]}"; do
  printf '%s\n' "$item"
done
exit 1
