#!/usr/bin/env bash
# Creates the LiveSplat venv in WSL. Run from WSL: bash install_venv.sh
set -euo pipefail
export HF_HUB_DISABLE_XET=1
V=~/venvs/livesplat
[ -x "$V/bin/python" ] || uv venv --python 3.10 "$V"
export VIRTUAL_ENV="$V"
uv pip install --python "$V/bin/python" torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$V/bin/python" numpy scipy plyfile scikit-learn trimesh mujoco viser opencv-python-headless pillow requests pytest huggingface_hub
# SAM 2: no CUDA extension (it is optional, only used for mask post-processing)
SAM2_BUILD_CUDA=0 uv pip install --python "$V/bin/python" "git+https://github.com/facebookresearch/sam2.git"
mkdir -p ~/livesplat_data/models
"$V/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt", local_dir="/home/wasif/livesplat_data/models")
print("ckpt", p)
PY
"$V/bin/python" -c "import torch,sam2,mujoco,viser,trimesh,sklearn;print('OK torch',torch.__version__,torch.cuda.is_available(),'mujoco',mujoco.__version__,'viser',viser.__version__)"
