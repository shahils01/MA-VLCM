#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-ma-vlcm}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Load your Conda module first (example: module load anaconda3)."
  exit 1
fi

echo "[1/4] Creating or updating Conda env: ${ENV_NAME}"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f palmetto_env.yml --prune
else
  conda env create -n "${ENV_NAME}" -f palmetto_env.yml
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "[2/4] Detecting torch + CUDA for PyG wheel URL"
TORCH_VER="$(python - <<'PY'
import torch
print(torch.__version__.split("+")[0])
PY
)"
CUDA_RAW="$(python - <<'PY'
import torch
print(torch.version.cuda or "cpu")
PY
)"

if [[ "${CUDA_RAW}" == "cpu" ]]; then
  CUDA_TAG="cpu"
else
  CUDA_TAG="cu${CUDA_RAW/./}"
fi

PYG_URL="https://data.pyg.org/whl/torch-${TORCH_VER}+${CUDA_TAG}.html"
echo "Using PyG wheel index: ${PYG_URL}"

echo "[3/4] Installing PyTorch Geometric compiled deps"
pip install --upgrade pip
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "${PYG_URL}"
pip install torch_geometric

echo "[4/4] Verifying imports used by MA-VLCM"
python - <<'PY'
mods = [
    "torch",
    "transformers",
    "accelerate",
    "webdataset",
    "datasets",
    "peft",
    "torch_geometric",
    "torch_scatter",
]
for m in mods:
    __import__(m)
print("Environment check passed.")
PY

echo "Done. Activate with: conda activate ${ENV_NAME}"
