#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-ma-vlcm}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Load your Conda module first (example: module load anaconda3)."
  exit 1
fi

create_minimal_env_and_install() {
  echo "[fallback] Creating minimal env to avoid Conda OOM during solve"
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Env ${ENV_NAME} already exists; reusing it."
  else
    conda create -y -n "${ENV_NAME}" python=3.10 pip
  fi

  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"

  pip install --upgrade pip
  # Install CUDA-enabled PyTorch wheels directly to avoid heavy Conda SAT solve.
  pip install --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

  pip install -r requirements.txt
  pip install --no-cache-dir peft sentencepiece protobuf safetensors torch-geometric
}

echo "[1/4] Creating or updating Conda env: ${ENV_NAME}"
set +e
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update -n "${ENV_NAME}" -f palmetto_env.yml --prune --solver libmamba
  CONDA_STATUS=$?
else
  conda env create -n "${ENV_NAME}" -f palmetto_env.yml --solver libmamba
  CONDA_STATUS=$?
fi
set -e

if [[ ${CONDA_STATUS} -ne 0 ]]; then
  echo "Conda yaml solve failed (often OOM on clusters). Switching to fallback installer."
  create_minimal_env_and_install
else
  eval "$(conda shell.bash hook)"
  conda activate "${ENV_NAME}"
fi

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

# Older cluster images often have glibc < 2.29, which breaks prebuilt
# wheels for pyg_lib/torch_sparse. Install a compatible subset in that case.
GLIBC_VER="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')"
if [[ -z "${GLIBC_VER:-}" ]]; then
  GLIBC_VER="0.0"
fi
echo "Detected glibc: ${GLIBC_VER}"

if [[ "$(printf '%s\n' "2.29" "${GLIBC_VER}" | sort -V | head -n1)" == "2.29" ]]; then
  echo "glibc >= 2.29: installing full PyG optional stack"
  pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f "${PYG_URL}"
else
  echo "glibc < 2.29: skipping pyg_lib and torch_sparse to avoid GLIBC mismatch warnings"
  pip install --no-cache-dir torch_scatter torch_cluster torch_spline_conv -f "${PYG_URL}"
fi

pip install --no-cache-dir torch_geometric

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
